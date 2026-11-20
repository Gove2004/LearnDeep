"""扩散过程（DDPM）。

两个方向：

    前向（加噪）：x_0 -> x_1 -> ... -> x_T，逐步加高斯噪声，
                  T 步之后图像变成纯噪声。这个过程是固定的，不用学。

    反向（去噪）：x_T -> ... -> x_0，从纯噪声逐步还原出图像。
                  这个过程要学，用 UNet 预测每步的噪声。

关键公式（前向可以直接跳到任意 t 步，不用真的迭代）：

    x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε

其中 ᾱ_t = ∏ α_i，α_i = 1 - β_i。

训练目标就是让 UNet 预测这个 ε。
"""

import torch


TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02


def make_schedule(timesteps=TIMESTEPS, beta_start=BETA_START,
                  beta_end=BETA_END, device="cpu"):
    """生成 β、α、ᾱ 的查找表。

    β 从 1e-4 线性增长到 0.02：
    初期加很少的噪声（保留细节），后期加很多（彻底打乱）。
    """
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device),
                                     alphas_cumprod[:-1]])

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        # 反向过程用的后验方差
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev)
                              / (1.0 - alphas_cumprod),
    }


def q_sample(x_0, t, schedule, noise=None):
    """前向加噪：直接算出 x_t。

    x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε

    一次公式就跳到第 t 步，不用循环 t 次。
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_a = schedule["sqrt_alphas_cumprod"][t][:, None, None, None]
    sqrt_1ma = schedule["sqrt_one_minus_alphas_cumprod"][t][:, None, None, None]

    return sqrt_a * x_0 + sqrt_1ma * noise, noise


@torch.no_grad()
def p_sample(model, x_t, t, schedule):
    """反向一步：从 x_t 采样出 x_{t-1}。

    模型预测噪声 ε_θ，然后：
        x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ) + σ_t·z

    其中 z 是标准正态噪声（t=0 时不加）。
    """
    t_batch = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)

    # 模型预测这一步的噪声
    eps_pred = model(x_t, t_batch)

    beta_t = schedule["betas"][t]
    alpha_t = schedule["alphas"][t]
    sqrt_1ma = schedule["sqrt_one_minus_alphas_cumprod"][t]
    sqrt_a = schedule["sqrt_alphas_cumprod"][t]

    # 算出 x_{t-1} 的均值
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / sqrt_1ma) * eps_pred)

    if t == 0:
        return mean

    # 加上后验方差对应的噪声
    var = schedule["posterior_variance"][t]
    return mean + torch.sqrt(var) * torch.randn_like(x_t)


@torch.no_grad()
def sample(model, n, schedule, device, image_size=28):
    """完整采样：从纯噪声开始，逐步去噪 T 次。"""
    model.eval()
    x = torch.randn(n, 1, image_size, image_size, device=device)

    for t in reversed(range(len(schedule["betas"]))):
        x = p_sample(model, x, t, schedule)

    return x
