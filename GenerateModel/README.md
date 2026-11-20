# GenerateModel —— 生成模型

从零实现四类主流生成模型，用 MNIST 和自定义二维数据跑通。

---

## 目录

| 模块 | 方法 | 数据 | 参数量 | 状态 |
|---|---|---|---|---|
| [GMM](#一gmm--概念热身) | 高斯混合模型 + EM | 自定义二维分布 | 18 个 | ✅ |
| [GAN](#二gan--对抗思想) | 生成对抗网络 | MNIST | 1.08M | ✅ |
| [VAE](#三vae--隐变量模型) | 变分自编码器 | MNIST | 648K | ✅ |
| [Diffuse](#四diffuse--当前主流) | DDPM + UNet | MNIST | 585K | ✅ |

**统一运行方式**：

```powershell
cd GenerateModel\GAN
python main.py
```

菜单里选编号执行，`0` 退出。

---

## 生成模型的分类

按"如何建模概率分布"划分：

```
显式建模 p(x)
├── 精确似然
│   └── Autoregressive   逐像素分解联合概率（本项目未实现）
└── 近似似然
    └── VAE              ELBO 下界

不显式建模
├── GAN                  对抗博弈，隐式分布
└── Diffuse              去噪过程，隐式分布

经典方法（非神经网络）
└── GMM                  隐变量 + EM 算法
```

---

## 一、GMM —— 概念热身

### 为什么放第一个

**它不是神经网络** —— 用 EM 算法拟合，不涉及梯度下降、不需要 GPU。

放这里是因为它是最简生成模型：参数少到**可以直接画在图上**，
但已经包含了生成模型的核心思想 —— **用隐变量描述数据是怎么产生的**。

### 模型

数据假设由 K 个二维高斯混合而成：

```
p(x) = Σ_k π_k · N(x | μ_k, Σ_k)
```

| 符号 | 含义 | 维度 |
|---|---|---|
| `π_k` | 第 k 个成分的权重（和为 1） | K |
| `μ_k` | 第 k 个成分的均值 | K × 2 |
| `Σ_k` | 第 k 个成分的协方差 | K × 2 × 2 |

**要拟合的就是这三组参数。**

### EM 算法

难点：数据没告诉我们"这个点属于哪个成分"。这个未知归属就是**隐变量** `z`。

**E 步**（Expectation）—— 固定参数，算责任度 `γ`：

```
γ_nk = π_k · N(x_n | μ_k, Σ_k) / Σ_j π_j · N(x_n | μ_j, Σ_j)
```

`γ_nk` 的含义是"第 n 个点有多大概率来自第 k 个成分"，每行和为 1。

**M 步**（Maximization）—— 固定 `γ`，用加权平均重新估计：

```
N_k  = Σ_n γ_nk                    ← 第 k 个成分的"有效样本数"
π_k  = N_k / N
μ_k  = (1/N_k) Σ_n γ_nk · x_n
Σ_k  = (1/N_k) Σ_n γ_nk · (x_n - μ_k)(x_n - μ_k)ᵀ
```

两步交替迭代到收敛。

### 实现要点

**1. 用 Cholesky 分解代替求逆**

算高斯密度要算 `(x-μ)ᵀ Σ⁻¹ (x-μ)` 和 `log|Σ|`，直接求逆数值不稳定：

```python
L = np.linalg.cholesky(cov)
z = np.linalg.solve(L, diff.T)          # z = L⁻¹ diffᵀ
mahalanobis = np.sum(z ** 2, axis=0)    # 马氏距离² = |z|²
log_det = 2 * np.sum(np.log(np.diag(L)))  # 顺带得到 log|Σ|
```

**一次分解算两样东西。**

**2. log-sum-exp 防止下溢**

二维高斯密度可能小到 `1e-300` 以下，直接相乘会变 0。
所有计算在对数域做，最后用 log-sum-exp 归一化。

**3. 多次随机初始化是必须的**

EM 只保证收敛到**局部最优**。实测 10 次初始化里有 1-2 次卡住：

```
init  1: loglik=-3.2102   ← 全局最优
init  6: loglik=-3.2703   ← 陷入局部最优
```

所以 `N_INIT = 10`，取似然最高的那次。

### 实测结果

数据：2000 个点，3 个成分，真实参数写在 `data.py`（模型看不到）。

| 成分 | 估计均值 | 真实均值 | 估计权重 | 真实权重 |
|---|---|---|---|---|
| 0 | ( 2.061, 1.429) | ( 2.000, 1.500) | 0.3699 | 0.4000 |
| 1 | (-2.037, -0.997) | (-2.000, -1.000) | 0.3510 | 0.3500 |
| 2 | ( 0.083, 3.004) | ( 0.000, 3.000) | 0.2791 | 0.2500 |

协方差也吻合，如成分 1：估计 `[0.679 0.273; 0.273 0.678]` vs 真实 `[0.700 0.300; 0.300 0.700]`。

**与 sklearn 对照**：

| 手写 EM 均值 | sklearn 均值 | 距离 |
|---|---|---|
| ( 2.061, 1.429) | ( 2.064, 1.427) | 0.0035 |
| (-2.037, -0.997) | (-2.038, -0.997) | 0.0005 |
| ( 0.083, 3.004) | ( 0.086, 3.002) | 0.0036 |

**最大偏差 0.0036** → 手写实现正确 ✓

> 注意：两个模型的**成分编号是随机的**（顺序取决于初始化），必须先配对才能比较。

**验证了 EM 的核心性质**：

```
对数似然: -4.2749 -> -3.2102
单调不降: True  (最小增量 4.43e-07)
```

**似然单调不降是 EM 的理论保证**，也是检查实现对不对的最好方法。

### 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `gmm.py` | 150 | 手写 EM（核心） |
| `plot.py` | 111 | 绘图 |
| `compare.py` | 66 | sklearn 对照 |
| `data.py` | 32 | 造数据 |
| `main.py` | 32 | 菜单 |

---

## 二、GAN —— 对抗思想

### 原理

生成器 G 把噪声变成图像，判别器 D 分辨真假，两者对抗。

```
G: z (噪声) → 图像
D: 图像 → 真假概率
```

最终达到纳什均衡：D 分辨不出真假（准确率≈50%），G 生成的图像足以乱真。

### 训练循环

GAN 每步要更新两个网络：

```python
# ---------------- 1. 训练判别器 ----------------
z = torch.randn(batch, latent_dim)
fake = G(z).detach()             # detach：这一步不更新 G
pred_real = D(real)
pred_fake = D(fake)
loss_d = (BCE(pred_real, 1) + BCE(pred_fake, 0)) / 2
opt_d.zero_grad(); loss_d.backward(); opt_d.step()

# ---------------- 2. 训练生成器 ----------------
z = torch.randn(batch, latent_dim)
pred_fake = D(G(z))
loss_g = BCE(pred_fake, 1)       # 让 D 认为假图是真的
opt_g.zero_grad(); loss_g.backward(); opt_g.step()
```

**`.detach()` 是关键** —— 训 D 时不能让梯度流回 G。

### 网络结构

```python
class Generator(nn.Module):
    # 64 维噪声 → 256 → 512 → 784 → reshape 成 28x28
    nn.Linear(latent_dim, hidden_dim)
    nn.Linear(hidden_dim, hidden_dim * 2)
    nn.Linear(hidden_dim * 2, 784)
    nn.Tanh()                       # 输出 [-1, 1]

class Discriminator(nn.Module):
    # 784 → 512 → 256 → 1
    nn.Flatten()
    nn.Linear(784, hidden_dim * 2)
    nn.Dropout(0.3)                 # 防止 D 过强
    nn.Linear(hidden_dim * 2, hidden_dim)
    nn.Dropout(0.3)
    nn.Linear(hidden_dim, 1)
    nn.Sigmoid()
```

### 两个稳定训练的细节

| 设置 | 值 | 原因 |
|---|---|---|
| Adam `beta1` | **0.5**（默认 0.9） | DCGAN 论文的建议，GAN 上低动量更稳 |
| 判别器 Dropout | 0.3 | 防止 D 过强，避免 G 梯度消失 |

### 实测结果

训练 30 epoch（约 7 分钟）：

```
epoch 22:  D=0.5112  G=1.2701
epoch 23:  D=0.5163  G=1.2492
```

**D 稳定在 0.5 附近是关键信号** —— 判别器分辨不出真假，说明生成器追上了判别器。

如果 D 掉到 0.1（判别器完胜）或升到 0.9，都说明训练失衡。

**生成效果**：8×8 采样能清楚辨认出 0、1、4、7、8、9，笔画结构和粗细都正常。

**隐空间插值**：两个随机噪声之间线性插值，图像平滑过渡（如 `6 → 7`），
说明生成器学到了**连续的隐空间**。

### 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `plot.py` | 111 | 三种可视化 |
| `train.py` | 98 | 训练循环 |
| `model.py` | 79 | G / D |
| `data.py` | 42 | MNIST 加载 |
| `main.py` | 33 | 菜单 |

---

## 三、VAE —— 隐变量模型

### 原理

编码器输出**分布**（均值和方差），不是点；解码器从分布里采样重建。

```
x → Encoder → (μ, σ²) → 采样 z → Decoder → x̂
```

### 三个核心点

**1. 用 `logvar` 而不是 `var`**

```python
self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值
self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差
```

方差必须为正，而 `logvar` 可以是任意实数，无需额外约束。

**2. 重参数化技巧** —— VAE 能训练的关键

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)      # 随机性隔离在 eps 上
    return mu + std * eps            # mu 和 logvar 可正常求导
```

直接 `torch.normal(mu, std)` 采样的话**梯度传不回去**。
写成 `mu + std * eps` 后，随机性转移到 `eps`，`mu` 和 `logvar` 就有梯度了。

**3. 损失 = 重建 + KL**

```python
recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
return recon_loss + beta * kl
```

**两项在互相拉扯**：

- 重建项要输出像输入 → 希望隐空间**散开**（每个样本独立编码）
- KL 项要隐分布接近 `N(0, I)` → 希望隐空间**规整**

如果只有重建项，VAE 会退化成普通自编码器（隐空间散乱，无法采样生成）。
**KL 项才是"生成"能力的来源。**

### 实测结果

训练 20 epoch（约 3 分钟）：

```
epoch  1:  loss=21115.9  recon=19262.1  kl=1853.8
epoch 20:  loss=13351.2  recon=10343.2  kl=3007.9
```

**单调下降**。其中 KL 项从 1853 → 3008 **持续上升**，这是正常且正确的 ——
训练初期模型倾向"记住"每个样本（KL 小），后期学会用规范的隐空间表示（KL 增大到接近先验）。

**四种可视化**：

**1. 重建** —— 数字忠实还原，只是略模糊。

**2. 采样生成** —— 从先验 `N(0, I)` 采样，生成清晰可辨的数字。

**3. 隐空间遍历**（最能说明问题）—— 每行固定其他维、只让某一维从 -2 变到 2：

```
第 1 行:  8 → 9 → 7
第 2 行:  9 → 8 → 2
第 3 行:  1 → 8 → 0 → 9
```

**数字是平滑渐变的**，说明隐空间**连续且有结构** —— 这是 VAE 相比 GAN 的独特优势。

**4. 真实图像插值** —— 两个真实数字的隐向量之间插值，`6` 平滑变成 `8`。

### 为什么 VAE 生成比 GAN 模糊

| | GAN | VAE |
|---|---|---|
| 优化目标 | 判别器判断真假 | 逐像素重建 + KL |
| 损失 | 对抗（关注整体真实性） | MSE/BCE（关注每个像素） |
| 结果 | **清晰锐利** | **偏模糊** |

**原因**：VAE 用逐像素误差，模型倾向输出"平均值"来降低误差 ——
而多个合理数字的平均就是模糊的。GAN 不逐像素比对，只问"整体像不像"，所以更锐利。

**这也解释了为什么 VAE 训练更稳**：没有对抗博弈，不会模式崩塌。

### 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `plot.py` | 134 | 四种可视化 |
| `model.py` | 99 | Encoder / Decoder / VAE |
| `train.py` | 68 | 训练循环 |
| `data.py` | 40 | MNIST 加载 |
| `main.py` | 33 | 菜单 |

---

## 四、Diffuse —— 当前主流

### 原理

**前向**：逐步给图像加高斯噪声，T 步后变成纯噪声（固定过程，不用学）。

**反向**：从纯噪声逐步去噪还原出图像（要学）。

### 前向加噪

**关键公式**（可以直接跳到任意 t 步，不用真的迭代）：

```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

其中 `ᾱ_t = ∏ α_i`，`α_i = 1 - β_i`。

`ᾱ` 表示"还剩多少原始信号"：

| t | ᾱ | 含义 |
|---|---|---|
| 0 | 1.000 | 完全原图 |
| 500 | 0.078 | 信号所剩无几 |
| 999 | 4.04e-05 | 几乎纯噪声 |

**实测验证**：`t=999` 时输出 `mean=-0.02, std=1.01`，精确等于标准正态 ✓

### 反向去噪

```
x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ) + σ_t·z
```

模型预测噪声 `ε_θ`，然后算出上一步的均值和方差。

### 训练出奇地简单

```python
t = torch.randint(0, 1000, (batch,))     # 随机采时间步
x_t, noise = q_sample(x_0, t, schedule)  # 闭式加噪
loss = mse(model(x_t, t), noise)         # 预测噪声
```

**没有对抗、没有 ELBO、没有逐步展开** —— 就一个回归任务。

### UNet 结构

写在 `Diffuse/` 内部，不单独成模块（UNet 是**网络结构**，不是生成模型）。

```
编码器: 28×28 → 14×14 → 7×7     （逐层下采样，提取语义）
中间层: 7×7
解码器: 7×7 → 14×14 → 28×28     （逐层上采样，恢复细节）
跳跃连接: 把编码器的细节直接送到解码器
```

**时间嵌入**用正弦编码（和 Transformer 位置编码同思路）：

```python
class SinusoidalPosEmb(nn.Module):
    emb = torch.exp(torch.arange(half) * -(log(10000) / (half - 1)))
    emb = t[:, None] * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)
```

每个卷积块把时间嵌入加到所有通道上，让网络知道当前的噪声强度。

### 实测结果

训练 20 epoch（约 15 分钟）：

```
epoch  1:  mse=0.0347
epoch 10:  mse=0.0288
epoch 20:  mse=0.0266
```

**平稳下降，没有任何震荡** —— 这是扩散模型最大的优点。

**四种可视化**：

**1. 前向加噪** —— 真实数字"3"在 1000 步内逐步被噪声淹没。

**2. 反向去噪**（最能说明原理）—— 从 `t=999` 的纯噪声开始，
到 `t=0` 时一个清晰的"0"浮现出来。**能看到图像从混沌中逐渐成形。**

**3. 采样生成** —— 8×8 采样中 0-9 都清晰可辨，**笔画连贯锐利**。

**4. 真实 vs 生成对比** —— 生成数字的形态与真实数字一致。

### 为什么效果好

**585K 参数，比 GAN（1.08M）还小，但效果更好。**

原因：扩散模型把"生成"这个难题分解成 1000 个"小去噪"步骤，
每一步只需要预测一点点噪声 —— 比 GAN 一步到位容易得多。

**代价是采样慢**：1000 步，每步一次完整前向。

### 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `plot.py` | 141 | 四种可视化 |
| `model.py` | 137 | UNet + 时间嵌入 |
| `diffusion.py` | 94 | **扩散过程** |
| `train.py` | 84 | 训练循环 |
| `data.py` | 40 | MNIST 加载 |
| `main.py` | 33 | 菜单 |

---

## 四个模型的对比

| | GMM | GAN | VAE | Diffuse |
|---|---|---|---|---|
| 类型 | 经典统计 | 神经网络 | 神经网络 | 神经网络 |
| 显式似然 | ✅ 精确 | ❌ | ⚠️ 下界 | ❌ |
| 参数量 | 18 个 | 1.08M | 648K | **585K** |
| 生成质量 | — | 高 | 中 | **高** |
| 训练稳定性 | 局部最优 | 需调参 | 稳定 | **最稳** |
| 采样速度 | 快 | **快**（1 次前向） | **快**（1 次前向） | 慢（1000 步） |
| 训练时间 | 秒 | 7 分钟 | 3 分钟 | 15 分钟 |

**核心权衡**：

- **GAN**：快 + 清晰，但训练不稳，容易模式崩塌
- **VAE**：稳 + 隐空间可解释，但模糊
- **Diffuse**：稳 + 清晰，但采样慢 —— 所以成为当前主流

**一句话总结**：扩散模型用"慢采样"换来了"稳定训练 + 高质量"。

---

## 环境

现有依赖足够，**不需要额外安装**：

```
torch 2.11.0+cu128
torchvision 0.26.0+cu128
numpy / matplotlib / scikit-learn
```

**注意**：`diffusers` 未安装，但本项目是**从零实现**，不需要它。

**硬件**：RTX 5060 Laptop 8GB，所有实验都能跑。

---

## 数据

| 数据集 | 位置 | 用途 |
|---|---|---|
| 自定义二维分布 | 代码生成 | GMM |
| MNIST | `Datasets/MNIST` | GAN / VAE / Diffuse |

**MNIST 加载的坑**：

```python
DATA_DIR = PROJECT_ROOT / "Datasets"          # ✅
# 不是 PROJECT_ROOT / "Datasets" / "MNIST"   # ❌ 会找不到
```

`torchvision.datasets.MNIST(root=...)` 会在 `root` 下**自己建 `MNIST/raw/`**，
所以传入的应该是 `Datasets` 而不是 `Datasets/MNIST`。

---

## 统一结构

每个子模块都是同样的文件组织：

```
<模块>/
├── data.py      数据加载
├── model.py     模型定义
├── train.py     训练循环
├── plot.py      可视化
└── main.py      菜单入口
```

菜单用 `run_learn()` 的 docstring 作为显示文字：

```python
RUN_LEARNS = [data, model, train, plot]

for i, learning in enumerate(RUN_LEARNS, start=1):
    print(f"    {i} - {learning.__doc__}")
```

---

## 踩过的坑

### 1. 掩码卷积不能原地改权重

（Autoregressive 模块遇到，代码已删除，但值得记录）

```python
# ❌ 错误
def forward(self, x):
    self.weight.data *= self.mask    # 原地修改！
    return super().forward(x)
```

`optimizer.step()` 之后掩码位置的权重会变成非零（实测到 **4.58**），
下一轮 `backward` 用的就是被污染的权重，**梯度全程算错**。

```python
# ✅ 正确
def forward(self, x):
    return F.conv2d(x, self.weight * self.mask, ...)
```

### 2. 输出饱和导致采样坍缩

不加正则化时 logits 会饱和到 `[-608, 134]`，softmax 变成精确的 one-hot，
采样时永远取同一个值，**生成的图全黑**。

用 `label_smoothing` 缓解，但值要调好：太大（0.1）会让模型过于不确定，
采样出来是椒盐噪声。

### 3. 像素值范围必须和输出层匹配

| 模型 | 数据范围 | 输出层 |
|---|---|---|
| GAN | `[-1, 1]` | `tanh` |
| VAE | `[0, 1]` | `sigmoid` |
| Diffuse | `[-1, 1]` | 线性（预测噪声） |

不匹配的话生成图像会灰蒙蒙的。

### 4. Windows 控制台的 Unicode 陷阱

`ᾱ`（U+1FB1）无法用 GBK 编码，会让程序**在训练开始前就崩溃**：

```
'gbk' codec can't encode character '\u1fb1'
```

控制台输出统一用 ASCII，或者显式设置 UTF-8。

### 5. 扩散模型的 Unicode 与范围问题

- 数据必须缩放到 `[-1, 1]`（前向加噪公式的假设）
- 采样后要反归一化：`fake = (fake + 1) / 2`

---

## 未实现的模块

### Autoregressive（已删除）

曾尝试用 PixelCNN 做逐像素自回归，**实现正确但效果不理想**：

- ✅ 因果性严格验证通过（Gated 双栈修复了盲点）
- ✅ 模型确实学会了分布（笔画检测 recall 0.994, precision 0.9998）
- ❌ 采样慢（784 次前向）且效果差（椒盐噪声）

**失败原因**：

1. **像素空间自回归本身就难** —— 784 个像素的依赖关系极其复杂
2. **类别极度不平衡** —— MNIST 80% 是纯黑，剩下 20% 摊在 255 个灰度等级上
3. **自回归误差累积** —— 训练时看真实上下文，采样时看自己的输出

**正确的方向**是在 **latent 空间**做自回归（如 VQ-VAE + Transformer），
而不是在像素空间 —— 序列更短、token 更抽象、效果更好。

### Flow（已删除）

曾尝试用 RealNVP 做标准化流，**数学正确但高维训练失败**：

- ✅ 耦合层雅可比解析 vs 数值：误差 `0.00e+00`
- ✅ 可逆性：误差 `1e-6`
- ✅ 2D 密度积分：`0.9941 ≈ 1`
- ❌ 784 维训练时 NLL 持续变负（应为正数 ~576）

**失败原因**：全连接耦合层在高维不稳定 —— `s` 饱和后每层放大 `e^1 = 2.72` 倍，
16 层累计放大 **2996 倍**，`log|det J|` 冲到 3587（理论上界 3136）。

**正确的方向**是用**卷积耦合层**（按通道劈分），保留空间局部性，
论文在 MNIST 上能做到 ~1.06 bits/dim。

---

## 参考

| 模型 | 论文 |
|---|---|
| GMM | Dempster et al., *Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977 |
| GAN | Goodfellow et al., *Generative Adversarial Networks*, NeurIPS 2014 |
| DCGAN | Radford et al., *Unsupervised Representation Learning with Deep Convolutional GANs*, ICLR 2016 |
| VAE | Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014 |
| DDPM | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 |
| UNet | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015 |
| RealNVP | Dinh et al., *Density Estimation using Real NVP*, ICLR 2017 |
| PixelCNN | van den Oord et al., *Conditional Image Generation with PixelCNN Decoders*, NeurIPS 2016 |

**参考教材**：Bishop, *Pattern Recognition and Machine Learning*, Chapter 9（混合模型与 EM）
