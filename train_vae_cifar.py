from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from standard_vae import StandardConvVAE, StandardVAEConfig, standard_vae_loss
from train_vae_celeba_standard_tar import (
    ModelEMA,
    beta_at_step,
    capacity_at_step,
    evaluate,
    parse_hidden_dims,
)


class CIFAR10ImageDataset(Dataset):
    def __init__(self, root: str = "data", *, train: bool = True, download: bool = True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def maybe_subset(dataset: Dataset, n: int | None) -> Dataset:
    if n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, range(n))


def train(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = maybe_subset(
        CIFAR10ImageDataset(args.data_root, train=True, download=True),
        args.n_train_subset,
    )
    val_ds = maybe_subset(
        CIFAR10ImageDataset(args.data_root, train=False, download=True),
        args.n_val_subset,
    )
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

    cfg = StandardVAEConfig(
        image_channels=3,
        image_size=32,
        hidden_dims=tuple(args.hidden_dims),
        d_latent=args.d_latent,
        arch=args.arch,
    )
    model = StandardConvVAE(cfg).to(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    total_steps = max(1, args.epochs * len(train_loader))
    step = 0
    best_test_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    stop_reason = None
    t0 = time.time()

    log = open(out / "training.log", "w")
    log.write(
        json.dumps(
            {
                "event": "config",
                "dataset": "cifar10",
                "cfg": cfg.__dict__ | {"hidden_dims": list(cfg.hidden_dims)},
                "data_root": args.data_root,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": args.beta,
                "beta_warmup_frac": args.beta_warmup_frac,
                "gamma": args.gamma,
                "max_capacity": args.max_capacity,
                "capacity_warmup_frac": args.capacity_warmup_frac,
                "mse_weight": args.mse_weight,
                "l1_weight": args.l1_weight,
                "free_bits": args.free_bits,
                "ema_decay": args.ema_decay,
                "grad_clip": args.grad_clip,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_tol": args.early_stop_tol,
            }
        )
        + "\n"
    )

    for epoch in range(args.epochs):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "mse": 0.0, "l1": 0.0, "kl": 0.0, "kl_objective": 0.0, "n": 0}
        for batch in train_loader:
            step += 1
            x = batch[0].to(device, non_blocking=True)
            capacity = capacity_at_step(step, total_steps, args.max_capacity, args.capacity_warmup_frac)
            beta = beta_at_step(step, total_steps, args.beta, args.beta_warmup_frac)
            x_recon, mu, logvar = model(x)
            loss, parts = standard_vae_loss(
                x,
                x_recon,
                mu,
                logvar,
                beta=beta,
                gamma=args.gamma,
                capacity=capacity,
                mse_weight=args.mse_weight,
                l1_weight=args.l1_weight,
                free_bits=args.free_bits,
            )
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            bs = x.size(0)
            sums["loss"] += float(loss.detach()) * bs
            sums["recon"] += parts["recon"] * bs
            sums["mse"] += parts["mse"] * bs
            sums["l1"] += parts["l1"] * bs
            sums["kl"] += parts["kl"] * bs
            sums["kl_objective"] += parts["kl_objective"] * bs
            sums["n"] += bs

        train_loss = sums["loss"] / sums["n"]
        train_recon = sums["recon"] / sums["n"]
        train_mse = sums["mse"] / sums["n"]
        train_l1 = sums["l1"] / sums["n"]
        train_kl = sums["kl"] / sums["n"]
        if ema is not None:
            eval_model = copy.deepcopy(model).to(device)
            ema.copy_to(eval_model)
            val = evaluate(eval_model, val_loader, device, args, step, total_steps)
            del eval_model
        else:
            val = evaluate(model, val_loader, device, args, step, total_steps)
        scheduler.step(val["loss"])

        if val["loss"] < best_test_loss * (1 - args.early_stop_tol):
            best_test_loss = val["loss"]
            best_epoch = epoch
            best_state = (
                ema.materialize(model)
                if ema is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        row = {
            "epoch": epoch,
            "step": step,
            "wall_time": time.time() - t0,
            "lr": opt.param_groups[0]["lr"],
            "beta": beta_at_step(step, total_steps, args.beta, args.beta_warmup_frac),
            "capacity": capacity_at_step(step, total_steps, args.max_capacity, args.capacity_warmup_frac),
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_mse": train_mse,
            "train_l1": train_l1,
            "train_kl": train_kl,
            "train_kl_objective": sums["kl_objective"] / sums["n"],
            "test_loss": val["loss"],
            "test_recon": val["recon"],
            "test_mse": val["mse"],
            "test_l1": val["l1"],
            "test_kl": val["kl"],
            "test_kl_objective": val["kl_objective"],
            "best_test_loss": best_test_loss,
            "best_epoch": best_epoch,
            "epochs_since_improve": epochs_since_improve,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(
            f"  epoch {epoch:3d} | train={train_loss:.3f} "
            f"recon={train_recon:.3f} mse={train_mse:.3f} "
            f"l1={train_l1:.3f} kl={train_kl:.3f} "
            f"test={val['loss']:.3f} test_recon={val['recon']:.3f} "
            f"test_kl={val['kl']:.3f} best={best_test_loss:.3f}@{best_epoch} "
            f"lr={opt.param_groups[0]['lr']:.2e} cap={row['capacity']:.2f}",
            flush=True,
        )

        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            stop_reason = f"plateau: no improvement for {epochs_since_improve} epochs"
            break

    if stop_reason is not None:
        log.write(json.dumps({"event": "early_stop", "reason": stop_reason, "epoch": epoch}) + "\n")
        print(f"  stopped early at epoch {epoch}: {stop_reason}", flush=True)
    log.close()

    save_state = best_state if best_state is not None else {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    ckpt = {
        "state_dict": save_state,
        "cfg": {
            "image_channels": cfg.image_channels,
            "image_size": cfg.image_size,
            "hidden_dims": list(cfg.hidden_dims),
            "d_latent": cfg.d_latent,
            "arch": cfg.arch,
        },
        "model_type": "standard_conv_vae",
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "dataset": "cifar10",
        "data_root": args.data_root,
        "ema_decay": args.ema_decay,
        "loss_config": {
            "beta": args.beta,
            "beta_warmup_frac": args.beta_warmup_frac,
            "gamma": args.gamma,
            "max_capacity": args.max_capacity,
            "capacity_warmup_frac": args.capacity_warmup_frac,
            "mse_weight": args.mse_weight,
            "l1_weight": args.l1_weight,
            "free_bits": args.free_bits,
        },
    }
    torch.save(ckpt, out / "vae.pt")

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
    p.add_argument("--data_root", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta_warmup_frac", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=30.0)
    p.add_argument("--max_capacity", type=float, default=25.0)
    p.add_argument("--capacity_warmup_frac", type=float, default=0.35)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--l1_weight", type=float, default=0.0)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.5)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_tol", type=float, default=0.002)
    p.add_argument("--hidden_dims", type=parse_hidden_dims, default=parse_hidden_dims("32,64,128,256,512"))
    p.add_argument("--arch", choices=["standard", "resnet"], default="resnet")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_train_subset", type=int, default=None)
    p.add_argument("--n_val_subset", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = f"vae_checkpoints/cifar10_resnet_modernloss_d{args.d_latent}"
    train(args)


if __name__ == "__main__":
    main()
