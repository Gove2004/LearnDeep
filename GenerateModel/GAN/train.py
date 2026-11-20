from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloader
from model import LATENT_DIM, build_models, count_params


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"

EPOCHS = 30
LR = 2e-4
BETA1 = 0.5
LOG_EVERY = 200
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(G, D, loader, opt_g, opt_d, criterion, latent_dim,
                    device, epoch, log_every=LOG_EVERY):
    """跑一个 epoch。

    GAN 每步要更新两个网络：
      1. 先训判别器 D：让它能分辨真图和假图
      2. 再训生成器 G：让它骗过 D
    """
    G.train()
    D.train()

    total_d, total_g = 0.0, 0.0
    n = 0

    for i, (real, _) in enumerate(loader):
        real = real.to(device)
        batch = real.size(0)

        # ---------------- 1. 训练判别器 ----------------
        # 真图标签为 1，假图标签为 0
        z = torch.randn(batch, latent_dim, device=device)
        fake = G(z).detach()             # detach：这一步不更新 G

        pred_real = D(real)
        pred_fake = D(fake)

        loss_d_real = criterion(pred_real, torch.ones_like(pred_real))
        loss_d_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # ---------------- 2. 训练生成器 ----------------
        # 生成器的目标：让 D 认为假图是真的（标签给 1）
        z = torch.randn(batch, latent_dim, device=device)
        fake = G(z)
        pred_fake = D(fake)

        loss_g = criterion(pred_fake, torch.ones_like(pred_fake))

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        total_d += loss_d.item()
        total_g += loss_g.item()
        n += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"[train] epoch {epoch} step {i + 1}/{len(loader)}  "
                  f"D={loss_d.item():.4f}  G={loss_g.item():.4f}")

    return total_d / n, total_g / n


def run_learn(*args, **kwargs):
    """Train GAN on MNIST"""
    torch.manual_seed(SEED)

    loader = get_dataloader()
    G, D = build_models()
    G, D = G.to(DEVICE), D.to(DEVICE)

    print(f"[train] device={DEVICE}  batches={len(loader)}  epochs={EPOCHS}")
    print(f"[train] G params={count_params(G):,}  D params={count_params(D):,}")

    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        loss_d, loss_g = train_one_epoch(G, D, loader, opt_g, opt_d, criterion,
                                         LATENT_DIM, DEVICE, epoch)
        print(f"[train] === epoch {epoch}/{EPOCHS}  D={loss_d:.4f}  G={loss_g:.4f} ===")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), CKPT_DIR / "gan_mnist_generator.pth")
    torch.save(D.state_dict(), CKPT_DIR / "gan_mnist_discriminator.pth")
    print(f"[train] saved to {CKPT_DIR}")
