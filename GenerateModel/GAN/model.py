import torch
import torch.nn as nn


LATENT_DIM = 64
IMAGE_SIZE = 28
HIDDEN_DIM = 256


class Generator(nn.Module):
    """噪声 -> 图像。输入 (N, 64)，输出 (N, 1, 28, 28)。"""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                 image_size=IMAGE_SIZE):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, image_size * image_size),
            nn.Tanh(),                      # 输出范围 [-1, 1]
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(z.size(0), 1, self.image_size, self.image_size)


class Discriminator(nn.Module):
    """图像 -> 真假概率。输入 (N, 1, 28, 28)，输出 (N, 1)。"""

    def __init__(self, hidden_dim=HIDDEN_DIM, image_size=IMAGE_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),                   # 输出真假概率
        )

    def forward(self, x):
        return self.net(x)


def build_models(latent_dim=LATENT_DIM):
    G = Generator(latent_dim=latent_dim)
    D = Discriminator()
    return G, D


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_learn(*args, **kwargs):
    """GAN Model Architecture"""
    G, D = build_models()

    print(f"[model] Generator params={count_params(G):,}")
    print(f"[model] Discriminator params={count_params(D):,}")

    z = torch.randn(4, LATENT_DIM)
    fake = G(z)
    print(f"[model] G: {tuple(z.shape)} -> {tuple(fake.shape)}  "
          f"range=[{fake.min():.2f}, {fake.max():.2f}]")

    prob = D(fake)
    print(f"[model] D: {tuple(fake.shape)} -> {tuple(prob.shape)}  "
          f"range=[{prob.min():.3f}, {prob.max():.3f}]")

    print("[model] Generator:")
    print(G)
