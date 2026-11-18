# GenerateModel

生成模型学习项目。

> **当前状态：计划阶段，尚未开始编码。**
> 各子目录只有空的 `__init__.py`。本文档用于临时存放计划，开工后会被正式文档替换。

---

## 子模块

```
GenerateModel/
├── GMM             高斯混合模型（经典方法，非神经网络）
├── GAN             生成对抗网络
├── VAE             变分自编码器
├── Flow            标准化流
├── Diffuse         扩散模型（内部实现 UNet）
└── Autoregressive  自回归生成
```

这 6 个覆盖了生成模型的**主流范式**：

```
显式建模概率密度
├── 精确似然
│   ├── Autoregressive   逐像素/逐token 分解联合概率
│   └── Flow             可逆变换，变量替换公式
├── 近似似然
│   └── VAE              ELBO，隐变量 + 重构
└── 不显式建模
    ├── GAN              对抗博弈，隐式分布
    └── Diffuse          去噪过程，隐式分布
```

GMM 不属于神经网络，作为**概念热身**放在最前面。

---

## 实施顺序

按**难度递增**排列，且每一步都为下一步铺垫：

| # | 模块 | 预计难度 | 为什么这个顺序 |
|---|---|---|---|
| 1 | **GMM** | ★ | 参数极少、可手算，先建立"生成模型在做什么"的直觉 |
| 2 | **GAN** | ★★ | 结构最简单（生成器 + 判别器），能最快看到图像生成效果 |
| 3 | **VAE** | ★★ | 引入隐变量与重构，是 Diffuse 的思想基础 |
| 4 | **Flow** | ★★★ | 可逆变换，精确似然，数学稍重 |
| 5 | **Diffuse** | ★★★★ | 需要 UNet 骨干网，训练最慢，放最后 |
| 6 | **Autoregressive** | ★★ | 已在 Transformer 里接触过（逐 token），这里换成逐像素 |

**铺垫关系**：

```
GMM 的隐变量 ──→ VAE 的隐空间
VAE 的加噪思想 ──→ Diffuse 的前向扩散
Autoregressive（Transformer 已做过）──→ 这里换成 PixelCNN 逐像素
```

---

## 关键设计决定

### 1. UNet 写在 `Diffuse/` 内部

不单独成子模块。

**原因**：UNet 是**网络结构**（2015 年为医学图像分割提出），不是生成模型。
它出现在扩散模型里是**作为去噪网络**。单独成模块会让人误以为"UNet 是一种生成模型"。

**代价**：`Diffuse/` 的代码会长一些（UNet 约 100 行）。
**好处**：概念清晰，且 Diffuse 自包含。

### 2. GMM 保留，但明确标注"非神经网络"

它不是深度学习模型 —— 用 EM 算法拟合，不涉及梯度下降、不需要 GPU。

**保留的理由**：
- 是最简生成模型，参数少到可以**手算验证**
- EM 算法是 VAE 的 ELBO 的思想来源
- 调整成分个数能**直观看到模式崩塌**（为 GAN 做铺垫）

### 3. 自回归不重复造 Transformer

`Autoregressive/` 用 **PixelCNN**（逐像素生成图像），
而不是再做一遍 GPT —— 那个在 `NaturalLanguageProcessing/Transformer/` 已完成。

---

## 数据

| 数据集 | 大小 | 用途 |
|---|---|---|
| **MNIST** | 63.5 MB | GMM / GAN / VAE / Flow 的首选（28×28 灰度，训练快） |
| **CIFAR-10** | 129.3 MB | Diffuse / Autoregressive（32×32 彩色，更接近真实场景） |

**先用 MNIST 跑通，再上 CIFAR-10。**

---

## 环境

现有依赖已足够，**不需要额外安装**：

```
torch 2.11.0+cu128
torchvision 0.26.0+cu128
numpy / matplotlib / pandas
```

**注意**：`diffusers` 未安装。但本项目是**从零实现**扩散模型，不需要它。

**硬件**：RTX 5060 Laptop 8GB —— 跑 MNIST 级别完全够用。

---

## 待定事项

开工前需要确认：

| 项 | 说明 |
|---|---|
| **代码风格** | 沿用 `Transformer/` 的精简风格（config.py + 中文注释），还是 `ImageClassification/` 的工程结构（ABC 基类 + 配置驱动）？ |
| **推进节奏** | 每个模块做完停下来 review，还是连着做几个？ |
| **评估指标** | 除目视采样结果外，是否要加定量指标（如 FID / IS）？ |

---

## 参考

**论文**

| 模型 | 论文 |
|---|---|
| GMM | Dempster et al., *Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977 |
| GAN | Goodfellow et al., *Generative Adversarial Networks*, NeurIPS 2014 |
| VAE | Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014 |
| Flow | Rezende & Mohamed, *Variational Inference with Normalizing Flows*, ICML 2015 |
| Diffuse | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 |
| PixelCNN | van den Oord et al., *Conditional Image Generation with PixelCNN Decoders*, NeurIPS 2016 |
| UNet | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015 |
