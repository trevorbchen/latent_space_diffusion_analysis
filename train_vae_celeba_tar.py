from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image

from lib.vae import ConvVAE, VAEConfig, vae_loss


class TarImageDataset(Dataset):
    def __init__(self, tar_path: str, *, image_size: int = 32):
        self.tar_path = str(tar_path)
        with tarfile.open(self.tar_path, "r") as tar:
            self.members = [
                m.name
                for m in tar.getmembers()
                if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        self._tar: tarfile.TarFile | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.members)

    def _handle(self) -> tarfile.TarFile:
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path, "r")
        return self._tar

    def __getitem__(self, idx: int):
        member = self._handle().getmember(self.members[idx])
        stream = self._handle().extractfile(member)
        if stream is None:
            raise RuntimeError(f"Could not read {member.name} from {self.tar_path}")
        image = Image.open(io.BytesIO(stream.read())).convert("RGB")
        return self.transform(image), 0


def maybe_subset(dataset: Dataset, n: int | None) -> Dataset:
    if n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, range(n))


def train(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = maybe_subset(TarImageDataset(args.train_tar, image_size=args.image_size), args.n_train_subset)
    val_ds = maybe_subset(TarImageDataset(args.val_tar, image_size=args.image_size), args.n_val_subset)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cfg = VAEConfig(
        image_channels=1,
        image_size=args.image_size,
        base_channels=32,
        n_down=3,
        d_latent=args.d_latent,
    )
    model = ConvVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = out / "training.log"
    log = open(log_path, "w")
    log.write(
        json.dumps(
            {
                "event": "config",
                "dataset": "celeba_hq_tar",
                "train_tar": args.train_tar,
                "val_tar": args.val_tar,
                "cfg": cfg.__dict__,
                "lr": args.lr,
                "epochs": args.epochs,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_tol": args.early_stop_tol,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }
        )
        + "\n"
    )

    best_test_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    epochs_diverging = 0
    stop_reason = None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            x_recon, mu, logvar = model(x)
            loss, parts = vae_loss(x, x_recon, mu, logvar, kl_weight=args.kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            running["loss"] += loss.item() * bs
            running["recon"] += parts["recon"] * bs
            running["kl"] += parts["kl"] * bs
            running["n"] += bs

        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                x_recon, mu, logvar = model(x)
                loss, _ = vae_loss(x, x_recon, mu, logvar, kl_weight=args.kl_weight)
                test_loss += loss.item() * x.size(0)
                n_test += x.size(0)
        test_loss /= n_test

        if best_state is None:
            is_improvement = True
        else:
            is_improvement = test_loss < best_test_loss * (1 - args.early_stop_tol)

        if is_improvement:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
            epochs_diverging = 0
        else:
            epochs_since_improve += 1
            if test_loss > best_test_loss * (1 + args.early_stop_tol):
                epochs_diverging += 1
            else:
                epochs_diverging = 0

        row = {
            "epoch": epoch,
            "wall_time": time.time() - t0,
            "train_loss": running["loss"] / running["n"],
            "train_recon": running["recon"] / running["n"],
            "train_kl": running["kl"] / running["n"],
            "test_loss": test_loss,
            "best_test_loss": best_test_loss,
            "best_epoch": best_epoch,
            "epochs_since_improve": epochs_since_improve,
            "epochs_diverging": epochs_diverging,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(
            f"  epoch {epoch:3d} | train={row['train_loss']:.3f} "
            f"recon={row['train_recon']:.3f} kl={row['train_kl']:.3f} "
            f"test={test_loss:.3f} best={best_test_loss:.3f}@{best_epoch}",
            flush=True,
        )

        if args.early_stop_patience > 0:
            if epochs_diverging >= args.early_stop_patience:
                stop_reason = f"diverging for {epochs_diverging} epochs"
                break
            if epochs_since_improve >= args.early_stop_patience:
                stop_reason = f"plateau: no improvement for {epochs_since_improve} epochs"
                break

    if stop_reason is not None:
        log.write(json.dumps({"event": "early_stop", "reason": stop_reason, "epoch": epoch}) + "\n")
        print(f"  stopped early at epoch {epoch}: {stop_reason}", flush=True)
    log.close()

    save_state = best_state if best_state is not None else {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": save_state,
            "cfg": cfg.__dict__,
            "best_epoch": best_epoch,
            "best_test_loss": best_test_loss,
            "dataset": "celeba_hq_tar",
            "train_tar": args.train_tar,
            "val_tar": args.val_tar,
        },
        out / "vae.pt",
    )

    model.load_state_dict({k: v.to(device) for k, v in save_state.items()})
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))[0][:32].to(device)
        x_recon, _, _ = model(sample_batch)
        grid = torch.cat([sample_batch, x_recon], dim=0)
        save_image((grid * 0.5 + 0.5).clamp(0, 1), out / "recon_grid.png", nrow=8)
    print(f"Saved VAE (best epoch {best_epoch}, test_loss={best_test_loss:.3f}) to {out / 'vae.pt'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d_latent", type=int, required=True)
    p.add_argument("--train_tar", default="/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar")
    p.add_argument("--val_tar", default="/overflow/data/cerberus/celeba_hq_256_partitioned_ws/val/celeba_hq_256-val-0000.tar")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--early_stop_tol", type=float, default=0.002)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_train_subset", type=int, default=None)
    p.add_argument("--n_val_subset", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.out is None:
        args.out = f"vae_checkpoints/celeba_d{args.d_latent}"
    train(args)


if __name__ == "__main__":
    main()
