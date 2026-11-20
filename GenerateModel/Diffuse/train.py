from pathlib import Path

import torch
import torch.nn.functional as F

from data import get_dataloader
from diffusion import TIMESTEPS, make_schedule, q_sample
from model import UNet, count_params


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"

EPOCHS = 20
LR = 2e-4
LOG_EVERY = 200
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, schedule, device, epoch,
                    log_every=LOG_EVERY):
    """训练一个 epoch。

    训练出奇地简单：
      1. 随机采一个时间步 t
      2. 用闭式公式直接算出加噪后的 x_t
      3. 让 UNet 预测加进去的噪声
      4. MSE 损失

    不需要对抗、不需要 ELBO、不需要逐步展开 —— 就一个回归任务。
    """
    model.train()

    total, n = 0.0, 0

    for i, (images, _) in enumerate(loader):
        x_0 = images.to(device)
        batch = x_0.size(0)

        # 每个样本随机采一个时间步
        t = torch.randint(0, TIMESTEPS, (batch,), device=device).long()

        # 加噪，同时拿到真实噪声
        x_t, noise = q_sample(x_0, t, schedule)

        # 让模型预测噪声
        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        n += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"[train] epoch {epoch} step {i + 1}/{len(loader)}  "
                  f"mse={loss.item():.4f}")

    return total / n


def run_learn(*args, **kwargs):
    """Train DDPM on MNIST"""
    torch.manual_seed(SEED)

    loader = get_dataloader()
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = make_schedule(TIMESTEPS, device=DEVICE)

    print(f"[train] device={DEVICE}  batches={len(loader)}  epochs={EPOCHS}")
    print(f"[train] params={count_params(model):,}  timesteps={TIMESTEPS}")
    print(f"[train] beta: {schedule['betas'][0]:.5f} -> {schedule['betas'][-1]:.5f}")
    print(f"[train] alpha_bar: {schedule['alphas_cumprod'][0]:.5f} -> "
          f"{schedule['alphas_cumprod'][-1]:.2e}")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, schedule, DEVICE, epoch)
        print(f"[train] === epoch {epoch}/{EPOCHS}  mse={loss:.4f} ===")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "ddpm_mnist.pth")
    print(f"[train] saved to {CKPT_DIR / 'ddpm_mnist.pth'}")
