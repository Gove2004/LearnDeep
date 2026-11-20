from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import IMAGE_SIZE, get_dataloader
from model import LATENT_DIM, VAE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "Checkpoints"
PLOT_DIR = PROJECT_ROOT / "Logs" / "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = VAE()
    model.load_state_dict(torch.load(CKPT_DIR / "vae_mnist.pth",
                                     map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def plot_reconstruction(model, save_path=None, n=8):
    """上排原图，下排重建。看 VAE 能否还原输入。"""
    loader = get_dataloader(batch_size=n)
    images, _ = next(iter(loader))
    x = images.to(DEVICE).view(n, -1)

    recon, _, _ = model(x)
    recon = recon.view(n, 1, IMAGE_SIZE, IMAGE_SIZE).cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    for j in range(n):
        axes[0, j].imshow(images[j, 0], cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(recon[j, 0], cmap="gray")
        axes[1, j].axis("off")
    axes[0, 0].set_title("original", fontsize=11, loc="left")
    axes[1, 0].set_title("reconstructed", fontsize=11, loc="left")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "vae_reconstruction.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def plot_samples(model, save_path=None, n=8):
    """直接从先验 N(0, I) 采样生成，这是 VAE 作为生成模型的能力。"""
    torch.manual_seed(0)
    fake = model.sample(n * n, DEVICE).cpu()

    grid = torch.zeros(n * IMAGE_SIZE, n * IMAGE_SIZE)
    for i in range(n):
        for j in range(n):
            img = fake[i * n + j].view(IMAGE_SIZE, IMAGE_SIZE)
            grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                 j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = img

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray")
    ax.axis("off")
    ax.set_title(f"VAE generated digits ({n}x{n})")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "vae_samples.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def plot_latent_traversal(model, save_path=None, dims=8, steps=11):
    """逐个改变隐向量的某一维，看它控制图像的什么特征。

    这是 VAE 相比 GAN 的独特优势——隐空间是有结构的、可解释的。
    """
    fig, axes = plt.subplots(dims, steps, figsize=(steps * 1.2, dims * 1.2))

    for d in range(dims):
        # 固定其他维，只让第 d 维从 -2 变到 2
        z = torch.zeros(1, LATENT_DIM, device=DEVICE)
        for j, v in enumerate(torch.linspace(-2, 2, steps)):
            z[0, d] = v
            img = model.decoder(z).cpu()[0].view(IMAGE_SIZE, IMAGE_SIZE)
            axes[d, j].imshow(img, cmap="gray")
            axes[d, j].axis("off")
        axes[d, 0].set_ylabel(f"z{d}", fontsize=9, rotation=0, labelpad=14)

    fig.suptitle("latent traversal: each row varies one dimension from -2 to 2",
                 fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "vae_traversal.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def plot_interpolation(model, save_path=None, steps=10):
    """两个真实图像的隐向量之间插值，看数字如何渐变。"""
    loader = get_dataloader(batch_size=2)
    images, _ = next(iter(loader))
    x = images.to(DEVICE).view(2, -1)

    mu, _ = model.encoder(x)
    z1, z2 = mu[0], mu[1]

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.3, 1.8))
    for i, t in enumerate(torch.linspace(0, 1, steps)):
        z = (z1 * (1 - t) + z2 * t).unsqueeze(0)
        img = model.decoder(z).cpu()[0].view(IMAGE_SIZE, IMAGE_SIZE)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    fig.suptitle("interpolation between two real digits", fontsize=11)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_path or (PLOT_DIR / "vae_interpolation.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_learn(*args, **kwargs):
    """VAE Reconstruction & Samples"""
    model = load_model()
    print(f"[plot] saved {plot_reconstruction(model)}")
    print(f"[plot] saved {plot_samples(model)}")
    print(f"[plot] saved {plot_latent_traversal(model)}")
    print(f"[plot] saved {plot_interpolation(model)}")
