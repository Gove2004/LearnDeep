from pathlib import Path

import torch
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Datasets"
BATCH_SIZE = 128
IMAGE_SIZE = 28


def get_dataloader(batch_size=BATCH_SIZE):
    """MNIST，像素值缩放到 [-1, 1]（和扩散过程的约定一致）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform,
    )

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )


def run_learn(*args, **kwargs):
    """Inspect MNIST Data"""
    loader = get_dataloader()
    images, labels = next(iter(loader))

    print(f"[data] batches={len(loader)}  batch_size={images.shape[0]}")
    print(f"[data] image shape={tuple(images.shape)}  dtype={images.dtype}")
    print(f"[data] pixel range=[{images.min():.2f}, {images.max():.2f}]")
    print(f"[data] label range=[{labels.min()}, {labels.max()}]")
