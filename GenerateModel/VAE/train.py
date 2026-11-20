from pathlib import Path

import torch

from data import IMAGE_SIZE, get_dataloader
from model import LATENT_DIM, VAE, count_params, loss_function


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"

EPOCHS = 20
LR = 1e-3
BETA = 1.0            # KL 项的权重。调小会让重建更清晰但隐空间更乱
LOG_EVERY = 200
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, device, epoch, beta=BETA,
                    log_every=LOG_EVERY):
    model.train()

    total, total_rec, total_kl = 0.0, 0.0, 0.0
    n = 0

    for i, (images, _) in enumerate(loader):
        x = images.to(device).view(images.size(0), -1)

        recon, mu, logvar = model(x)
        loss, rec, kl = loss_function(recon, x, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        total_rec += rec.item()
        total_kl += kl.item()
        n += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"[train] epoch {epoch} step {i + 1}/{len(loader)}  "
                  f"loss={loss.item():.1f}  recon={rec.item():.1f}  kl={kl.item():.1f}")

    return total / n, total_rec / n, total_kl / n


def run_learn(*args, **kwargs):
    """Train VAE on MNIST"""
    torch.manual_seed(SEED)

    loader = get_dataloader()
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[train] device={DEVICE}  batches={len(loader)}  epochs={EPOCHS}")
    print(f"[train] params={count_params(model):,}  latent_dim={LATENT_DIM}")

    for epoch in range(1, EPOCHS + 1):
        loss, rec, kl = train_one_epoch(model, loader, optimizer, DEVICE, epoch)
        print(f"[train] === epoch {epoch}/{EPOCHS}  loss={loss:.1f}  "
              f"recon={rec:.1f}  kl={kl:.1f} ===")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "vae_mnist.pth")
    print(f"[train] saved to {CKPT_DIR / 'vae_mnist.pth'}")
