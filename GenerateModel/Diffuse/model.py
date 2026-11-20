import math

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 28
BASE_CHANNELS = 32
TIME_EMB = 128


class SinusoidalPosEmb(nn.Module):
    """把时间步 t 编码成向量。

    和 Transformer 的位置编码同样的思路：
    用不同频率的 sin/cos 表示 t，让网络能区分不同的噪声强度。
    """

    def __init__(self, dim=TIME_EMB):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Block(nn.Module):
    """卷积 + 归一化 + 激活，并把时间信息注入进去。"""

    def __init__(self, in_ch, out_ch, time_emb_dim=TIME_EMB):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.conv(x)
        h = self.norm(h)
        # 把时间嵌入加到每个通道上
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        return F.silu(h)


class UNet(nn.Module):
    """去噪网络：输入 (噪声图, 时间步)，输出预测的噪声。

    结构是经典的编码器-解码器 + 跳跃连接：
        编码器逐层下采样，提取语义
        解码器逐层上采样，恢复分辨率
        跳跃连接把编码器的细节直接送到解码器

    MNIST 只有 28x28，所以下采样两次就到 7x7 了。
    """

    def __init__(self, base=BASE_CHANNELS, time_emb=TIME_EMB):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb),
            nn.Linear(time_emb, time_emb * 2),
            nn.SiLU(),
            nn.Linear(time_emb * 2, time_emb),
        )

        # 编码器：28 -> 14 -> 7
        self.enc1 = Block(1, base, time_emb)             # 28x28
        self.enc2 = Block(base, base * 2, time_emb)      # 14x14
        self.enc3 = Block(base * 2, base * 4, time_emb)  # 7x7

        self.down = nn.MaxPool2d(2)

        # 中间层
        self.mid = Block(base * 4, base * 4, time_emb)

        # 解码器：7 -> 14 -> 28
        self.up2 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec2 = Block(base * 4 + base * 2, base * 2, time_emb)

        self.up1 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec1 = Block(base * 2 + base, base, time_emb)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # 编码
        e1 = self.enc1(x, t_emb)                  # (N, base, 28, 28)
        e2 = self.enc2(self.down(e1), t_emb)      # (N, base*2, 14, 14)
        e3 = self.enc3(self.down(e2), t_emb)      # (N, base*4, 7, 7)

        # 中间
        m = self.mid(e3, t_emb)

        # 解码 + 跳跃连接
        d2 = self.up2(m)                          # (N, base*4, 14, 14)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)

        d1 = self.up1(d2)                         # (N, base*2, 28, 28)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out(d1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_learn(*args, **kwargs):
    """UNet Denoising Network"""
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

    model = UNet()

    print(f"[model] params={count_params(model):,}")
    print(f"[model] base_channels={BASE_CHANNELS}  time_emb={TIME_EMB}")

    x = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print(f"[model] input {tuple(x.shape)}, t {tuple(t.shape)} -> out {tuple(out.shape)}")

    # 时间嵌入检查
    emb = SinusoidalPosEmb()(torch.tensor([0, 250, 500, 999]))
    print(f"[model] 时间嵌入 shape={tuple(emb.shape)}")
    print(f"[model] 不同 t 的嵌入差异:")
    print(f"    t=0   vs t=999: {(emb[0] - emb[3]).abs().mean():.4f}")
    print(f"    t=0   vs t=250: {(emb[0] - emb[1]).abs().mean():.4f}")

    print(f"\n[model] 各层输出尺寸:")
    with torch.no_grad():
        t_emb = model.time_mlp(t)
        e1 = model.enc1(x, t_emb)
        e2 = model.enc2(model.down(e1), t_emb)
        e3 = model.enc3(model.down(e2), t_emb)
        print(f"    编码器: {tuple(e1.shape)} -> {tuple(e2.shape)} -> {tuple(e3.shape)}")
        m = model.mid(e3, t_emb)
        d2 = model.dec2(torch.cat([model.up2(m), e2], dim=1), t_emb)
        d1 = model.dec1(torch.cat([model.up1(d2), e1], dim=1), t_emb)
        print(f"    解码器: {tuple(m.shape)} -> {tuple(d2.shape)} -> {tuple(d1.shape)}")
