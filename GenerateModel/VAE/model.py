import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 16
HIDDEN_DIM = 400
IMAGE_SIZE = 28
PIXELS = IMAGE_SIZE * IMAGE_SIZE


class Encoder(nn.Module):
    """图像 -> 隐分布的均值和对数方差。

    VAE 和普通自编码器的关键区别：编码器不输出一个点，
    而是输出一个分布 N(mu, sigma^2)，用 logvar 表示方差（不用 exp 保证正数）。
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(PIXELS, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """隐向量 -> 重建图像。输出用 sigmoid 压到 [0, 1]。"""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, PIXELS)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + sigma * eps，eps ~ N(0, 1)。

        直接采样的话梯度传不回去；写成这个形式后，
        随机性转移到 eps 上，mu 和 logvar 就能正常求导了。
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu                        # 推理时直接用均值

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n, device):
        """从先验 N(0, I) 采样再解码，用于生成新图。"""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def loss_function(recon, x, mu, logvar, beta=1.0):
    """VAE 的损失 = 重建误差 + KL 散度。

    重建项：让输出像输入
    KL 项：让隐分布接近标准正态 N(0, I)，
           这样从 N(0, I) 采样才能生成有意义的图
    """
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_learn(*args, **kwargs):
    """VAE Model Architecture"""
    model = VAE()

    print(f"[model] params={count_params(model):,}")
    print(f"[model] latent_dim={LATENT_DIM}  hidden_dim={HIDDEN_DIM}")

    x = torch.rand(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    flat = x.view(4, -1)
    recon, mu, logvar = model(flat)
    print(f"[model] input {tuple(x.shape)} -> flat {tuple(flat.shape)}")
    print(f"[model] encoder: mu {tuple(mu.shape)}  logvar {tuple(logvar.shape)}")
    print(f"[model] decoder: recon {tuple(recon.shape)}  "
          f"range=[{recon.min():.3f}, {recon.max():.3f}]")

    z = model.reparameterize(mu, logvar)
    print(f"[model] reparameterize: z {tuple(z.shape)}")

    total, rec, kl = loss_function(recon, flat, mu, logvar)
    print(f"[model] loss={total.item():.1f}  recon={rec.item():.1f}  kl={kl.item():.1f}")
