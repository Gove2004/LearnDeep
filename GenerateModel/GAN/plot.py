from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import get_dataloader
from model import LATENT_DIM, build_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"
PLOT_DIR = PROJECT_ROOT / "Logs" / "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_generator():
    G, _ = build_models()
    G.load_state_dict(torch.load(CKPT_DIR / "gan_mnist_generator.pth",
                                 map_location=DEVICE))
    G.to(DEVICE).eval()
    return G


@torch.no_grad()
def make_grid(G, n=8, seed=0):
    """生成 n×n 张图，返回一个可直接 imshow 的大图。"""
    torch.manual_seed(seed)
    z = torch.randn(n * n, LATENT_DIM, device=DEVICE)
    fake = G(z).cpu()

    # 反归一化：[-1, 1] -> [0, 1]
    fake = (fake + 1) / 2

    grid = torch.zeros(n * 28, n * 28)
    for i in range(n):
        for j in range(n):
            grid[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = fake[i * n + j, 0]
    return grid


def plot_samples(G, save_path=None, n=8):
    grid = make_grid(G, n=n)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray")
    ax.axis("off")
    ax.set_title(f"GAN generated digits ({n}x{n})")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "gan_samples.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_real_vs_fake(G, save_path=None, n=6):
    """上排真实，下排生成，方便直接对比。"""
    loader = get_dataloader(batch_size=n * n)
    real, _ = next(iter(loader))
    real = (real + 1) / 2

    torch.manual_seed(1)
    z = torch.randn(n * n, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake = (G(z).cpu() + 1) / 2

    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    for j in range(n):
        axes[0, j].imshow(real[j, 0], cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(fake[j, 0], cmap="gray")
        axes[1, j].axis("off")
    axes[0, 0].set_title("real", fontsize=11, loc="left")
    axes[1, 0].set_title("generated", fontsize=11, loc="left")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "gan_real_vs_fake.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_interpolation(G, save_path=None, steps=10):
    """在两个随机噪声之间插值，看生成的数字如何平滑过渡。"""
    torch.manual_seed(3)
    z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
    z2 = torch.randn(1, LATENT_DIM, device=DEVICE)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.3, 1.8))
    with torch.no_grad():
        for i, t in enumerate(torch.linspace(0, 1, steps)):
            z = z1 * (1 - t) + z2 * t
            img = (G(z).cpu()[0, 0] + 1) / 2
            axes[i].imshow(img, cmap="gray")
            axes[i].axis("off")
    fig.suptitle("latent space interpolation", fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "gan_interpolation.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_learn(*args, **kwargs):
    """GAN Generated Samples"""
    G = load_generator()
    print(f"[plot] saved {plot_samples(G)}")
    print(f"[plot] saved {plot_real_vs_fake(G)}")
    print(f"[plot] saved {plot_interpolation(G)}")
