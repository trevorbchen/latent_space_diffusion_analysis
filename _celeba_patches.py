"""Apply local patches to use CelebAFromFolder instead of torchvision.datasets.CelebA.

Idempotent: re-running is a no-op.
"""
from pathlib import Path

REPO = Path.home() / "latent_space_diffusion_analysis" / "code" / "v3"

TRAIN_VAE = REPO / "train_vae.py"
DATA_REAL = REPO / "lib" / "data_real.py"

# --- train_vae.py patch -----------------------------------------------------
old1 = '''    train = datasets.CelebA(root, split='train', download=True, transform=tfm)
    test = datasets.CelebA(root, split='valid', download=True, transform=tfm)'''

new1 = '''    from lib.celeba_folder import CelebAFromFolder  # local mirror; GDrive rate-limited
    train = CelebAFromFolder(root, split='train', transform=tfm)
    test = CelebAFromFolder(root, split='valid', transform=tfm)'''

src = TRAIN_VAE.read_text()
if old1 in src:
    TRAIN_VAE.write_text(src.replace(old1, new1))
    print(f"patched: {TRAIN_VAE}")
elif new1 in src:
    print(f"already patched: {TRAIN_VAE}")
else:
    raise SystemExit(f"OLD pattern not found in {TRAIN_VAE}")

# --- lib/data_real.py patch -------------------------------------------------
old2 = '''    if name == 'celeba':
        return datasets.CelebA(data_root, split=split, download=True,
                               transform=_celeba_transform(image_size))'''

new2 = '''    if name == 'celeba':
        from .celeba_folder import CelebAFromFolder  # local mirror; GDrive rate-limited
        return CelebAFromFolder(data_root, split=split,
                                transform=_celeba_transform(image_size))'''

src = DATA_REAL.read_text()
if old2 in src:
    DATA_REAL.write_text(src.replace(old2, new2))
    print(f"patched: {DATA_REAL}")
elif new2 in src:
    print(f"already patched: {DATA_REAL}")
else:
    raise SystemExit(f"OLD pattern not found in {DATA_REAL}")

print("done.")
