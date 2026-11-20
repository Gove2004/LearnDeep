from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import get_dataloader
from diffusion import TIMESTEPS, make_schedule, q_sample, sample
from model import IMAGE_SIZE, UNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"
PLOT_DIR = PROJECT_ROOT / "Logs" / "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = UNet()
    model.load_state_dict(torch.load(CKPT_DIR / "ddpm_mnist.pth",
                                     map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def plot_samples(model, schedule, save_path=None, n=8):
    """从纯噪声采样生成。要跑 1000 步去噪，比较慢。"""
    print(f"[plot] sampling {n}x{n} images ({TIMESTEPS} denoising steps)...")
    fake = sample(model, n * n, schedule, DEVICE)

    # 反归一化 [-1,1] -> [0,1]
    fake = (fake + 1) / 2

    grid = torch.zeros(n * IMAGE_SIZE, n * IMAGE_SIZE)
    for i in range(n):
        for j in range(n):
            grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                 j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = fake[i * n + j, 0].cpu()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"DDPM generated digits ({n}x{n})")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "ddpm_samples.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forward(schedule, save_path=None, steps=10):
    """展示前向加噪过程：一张真实图像如何逐步变成噪声。"""
    loader = get_dataloader(batch_size=1)
    x_0, _ = next(iter(loader))
    x_0 = x_0.to(DEVICE)

    ts = torch.linspace(0, TIMESTEPS - 1, steps).long()

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.3, 1.8))
    for k, t in enumerate(ts):
        x_t, _ = q_sample(x_0, t.unsqueeze(0), schedule)
        img = (x_t[0, 0] + 1) / 2
        axes[k].imshow(img.cpu(), cmap="gray", vmin=0, vmax=1)
        axes[k].axis("off")
        axes[k].set_title(f"t={t.item()}", fontsize=9)
    fig.suptitle("forward diffusion: image -> noise", fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "ddpm_forward.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reverse(model, schedule, save_path=None, steps=10):
    """展示反向去噪过程：噪声如何逐步变成数字。

    这是扩散模型最直观的展示 —— 能看到图像从混沌中"浮现"。
    """
    model.eval()
    x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    # 记录若干时间点的快照
    record_at = set(torch.linspace(0, TIMESTEPS - 1, steps).long().tolist())
    snapshots = []

    from diffusion import p_sample
    for t in reversed(range(TIMESTEPS)):
        x = p_sample(model, x, t, schedule)
        if t in record_at:
            snapshots.append((t, (x[0, 0].cpu() + 1) / 2))

    snapshots = snapshots[::-1]

    fig, axes = plt.subplots(1, len(snapshots), figsize=(len(snapshots) * 1.3, 1.8))
    for k, (t, img) in enumerate(snapshots):
        axes[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[k].axis("off")
        axes[k].set_title(f"t={t}", fontsize=9)
    fig.suptitle("reverse diffusion: noise -> image", fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "ddpm_reverse.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_real_vs_fake(model, schedule, save_path=None, n=6):
    """上排真实，下排生成。"""
    loader = get_dataloader(batch_size=n)
    real, _ = next(iter(loader))
    real = (real + 1) / 2

    fake = (sample(model, n, schedule, DEVICE) + 1) / 2

    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    for j in range(n):
        axes[0, j].imshow(real[j, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        axes[1, j].imshow(fake[j, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off")
    axes[0, 0].set_title("real", fontsize=11, loc="left")
    axes[1, 0].set_title("generated", fontsize=11, loc="left")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "ddpm_real_vs_fake.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_learn(*args, **kwargs):
    """DDPM Samples & Diffusion Process"""
    model = load_model()
    schedule = make_schedule(TIMESTEPS, device=DEVICE)

    print(f"[plot] saved {plot_forward(schedule)}")
    print(f"[plot] saved {plot_reverse(model, schedule)}")
    print(f"[plot] saved {plot_samples(model, schedule)}")
    print(f"[plot] saved {plot_real_vs_fake(model, schedule)}")
