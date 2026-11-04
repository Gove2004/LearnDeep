# NaturalLanguageProcessing

自然语言处理学习项目。对应鱼书②《深度学习进阶：自然语言处理》，分三个子包。

```
NaturalLanguageProcessing/
├── Word2Vec/    分布式表示：基于计数 + 基于推理
├── Seq2Seq/     序列到序列：GRU / GRU+Attention
└── Transformer/ Transformer：中译英
```

---

## 前置条件

```powershell
# 在项目根目录 LearnDeep/ 下运行
.\.venv\Scripts\Activate.ps1
```

依赖：`torch`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`、`pandas`、`gensim`、`datasets`、`sentencepiece`。

**网络**：部分数据需要联网下载。若下载超时，先开启代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
```

---

# 一、Word2Vec

## 文件

| 文件 | 说明 |
|---|---|
| `corpus.py` | PTB 数据集下载与加载、词表构建 |
| `counter.py` | 基于计数：共现矩阵 → PPMI → SVD 降维 |
| `word2vec.py` | 基于推理：手写 CBOW + Negative Sampling |
| `mygensim.py` | 基于推理：gensim 库实现 |
| `main.py` | 菜单启动器 |

## 语料

PTB（Penn Treebank），下载到 `Datasets/PTB/`：

| 文件 | 大小 |
|---|---|
| `ptb.train.txt` | 4.9 MB |
| `ptb.valid.txt` | 0.4 MB |
| `ptb.test.txt` | 0.4 MB |

## 使用

```powershell
cd NaturalLanguageProcessing\Word2Vec
python main.py
```

| 选项 | 内容 |
|---|---|
| 1 | 基于计数（PPMI + SVD） |
| 2 | 基于推理（CBOW + 负采样） |
| 3 | gensim |

## 三种方法对比

| 方法 | 原理 | 特点 |
|---|---|---|
| **计数** | 共现矩阵 → PPMI → SVD | 一次矩阵分解，无需训练 |
| **推理（手写）** | CBOW + Negative Sampling | 需要训练，可增量 |
| **推理（gensim）** | 同上，工业级实现 | 快、稳定，作为基准 |

## 两个实现要点

### 1. `nn.Embedding` 等价于 `one_hot @ W`

实测 `torch.allclose` 为 True。嵌入层只是把"查表"写成了矩阵乘法的高效形式。

### 2. 两张嵌入表

`in_embed`（上下文角色）与 `out_embed`（目标角色）是**两张独立的表**。
输出时用 `in_embed`。

### 3. 负采样把 V 分类变成 1+K 二分类

原本要对整个词表做 softmax（V 类），负采样后只需区分
"真实上下文词" 和 K 个随机负样本，计算量从 O(V) 降到 O(K)。

### 4. `f^0.75` 压缩词频偏斜

实测效果：

| 词 | 原始概率 | `^0.75` 后 |
|---|---|---|
| the | 0.329981 | 0.144231 |
| 稀有词/the 比值 | 1290 : 1 | **215 : 1** |

高频词被压制，稀有词被相对提升，负采样更均衡。

## 已知限制

- **未实现 subsampling**（高频词下采样）
- `corpus.py` 的 `urlretrieve` 没有设 timeout

---

# 二、Seq2Seq

## 文件

| 文件 | 说明 |
|---|---|
| `dataset.py` | 数据集基类 + 两个语料 + 词表构建 |
| `model.py` | `GRUSeq2Seq` / `AttentionSeq2Seq` |
| `util.py` | `decode` / `train_once` / `test_once` / `Checkpoint` / `EpochLogger` |
| `train.py` | 统一训练器（`Config` 驱动） |
| `main.py` | 菜单启动器 |
| `draw_attention.py` | 注意力热力图 |
| `架构图.html` | 模型结构图 |

## 数据

| 文件 | 行数 | 问题长度 | 答案长度 | 唯一问题数 |
|---|---|---|---|---|
| `addition.txt` | 50,000 | 7 | 4 | 50,000 |
| `date.txt` | 50,000 | 29（右填充） | 10 | 41,578 |

词表：加法 13（`' +0123456789_'`）、日期 59。

**`reverse=True`**：加法任务需要**反转输入**，让最低位对齐（鱼书 7.4.1）。
日期任务不需要。

## 使用

```powershell
# 在 Seq2Seq/ 目录下运行
python main.py
```

| 选项 | 内容 |
|---|---|
| 1 | 加法（GRU） |
| 2 | 日期（GRU，无 attention） |
| 3 | 日期（GRU + Attention） |
| 4 | 注意力热力图 |

## 两个模型的差异

| | `GRUSeq2Seq` | `AttentionSeq2Seq` |
|---|---|---|
| 参数量 | 164,155 | **253,755**（+54.6%） |
| 编码器输出 | 只传最后隐状态 `h` | 传**所有时刻**的隐状态 |
| 解码器输入 | 单向量 | 每步动态加权 |

attention 多出的 **32,896** 参数 = `128² + 128² + 128`（W_enc、W_dec、v）。

## Attention 参数如何更新

attention 只有 3 个普通 `nn.Linear` 层，**没有独立的监督信号**。
对齐关系是从交叉熵损失中**自监督涌现**的。

## Bahdanau 是什么

加性注意力：

```
score = v · tanh(W_enc · encoder_out + W_dec · decoder_h)
```

对比 Luong 的乘性注意力（直接点积），Transformer 用的是 Luong 风格。

## 实测结果

| 任务 | 模型 | 准确率 |
|---|---|---|
| 加法 | GRU | **100%**（30 epoch） |
| 日期 | GRU | **100%** |
| 日期 | GRU + Attention | **100%**（早停于 epoch 17，收敛更快） |

两个日期模型打平 —— 因为 `date.txt` 太小，`HIDDEN_SIZE=128` 足够宽裕。

## ⚠️ 加法数据集的已知问题（重要）

`addition.txt` 里有 **9,077 / 50,000（18.15%）** 的算式**算术上是错的** ——
答案字段只有 3 位宽，4 位数的和被截断了。

### 原因

```
题目: 123+456  →  正确答案 579
题目: 900+900  →  正确答案 1800，但字段只有 3 位 → 存成 800（截断）
```

### 为什么还能得 100%

模型**学的是数据集里的标签，不是真正的加法**。指标高只说明模型拟合了你给的标签。

### 教训

**标签本身要先验证。** 18% 的错误标签不会被指标暴露出来。

### 处理选项

| 方案 | 说明 |
|---|---|
| A | 修正数据集 |
| B | 自己生成一致的加法数据（`000..999 + 000..999`）—— 数学上完全正确 |
| **C（本项目采用）** | 保持原样，并在本文档中说明 |

选 C 的原因：这个数据集的价值在于**教学**（理解 seq2seq 骨架），不在于"能不能真的算加法"。

---

# 三、Transformer

Transformer（encoder-decoder）中译英实现，参考 CSDN 教程《Transformer 论文实现》的 9 模块拆解。

```
Transformer/
├── model.py      9 个模块 + Block 封装 + Translate
├── masks.py      padding mask / causal mask
├── dataset.py    OPUS-100 加载 + 并行分词
├── train.py      训练循环 + greedy_decode
├── tokenizer.py  sentencepiece 封装
└── translate.py  交互式翻译
```

## 数据

数据放在 `Datasets/opus100_en_zh/`（`Datasets/` 已在 `.gitignore` 中，避免 Git 追踪大文件）：

```
Datasets/opus100_en_zh/
├── train/           191.1 MB   978,815 句对
├── validation/        0.5 MB     1,974 句对
├── test/              0.5 MB     2,000 句对
└── processed/                   处理后产物
    ├── spm.model / spm.vocab          16000 词表
    ├── pairs_raw.pkl        176.5 MB  原始句对缓存
    ├── pairs_128.pkl        221.3 MB  分词结果缓存
    └── pairs_valid*.pkl               验证集缓存
```

**目录分层的意义**：原始数据和处理产物是**同一份数据的不同阶段**，放在一起便于分类管理。
`processed/` 里的东西全部可以由原始 arrow 重新生成。

### 数据来源

`Helsinki-NLP/opus-100`（config `en-zh`）—— **混合领域**语料，来自 OPUS 的多个来源
（字幕、网页、书籍、新闻），因此同时覆盖日常口语和正式文体。

```python
# 需先设代理
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-zh")
ds.save_to_disk("Datasets/opus100_en_zh")
```

字段：`translation = {"en": ..., "zh": ...}`。

### 长度分布（20 万条采样）

| 统计量 | 中文 token | 英文 token |
|---|---|---|
| median | 11 | 16 |
| p95 | 51 | 66 |
| max | 462 | 532 |

覆盖率：`≤96` 源 99.3% / 目标 98.6%、**`≤128` 源 99.8% / 目标 99.6%**。
本项目取 `MAX_LENGTH = 128`。

**注意**：OPUS-100 是句级语料，但有约 2,000 行是整段联合国文件（最长 4.9 万字符）。
`dataset.build_pairs()` 用 `max_chars=500` 过滤掉这些，避免它们主导分词器。

## 使用

```powershell
cd NaturalLanguageProcessing\Transformer

# 训练（首次会加载数据并分词，结果缓存；之后直接命中缓存）
python train.py

# 交互式翻译
python translate.py
```

## 模型

**encoder-decoder Transformer**，参数量 **19,677,824**。

```python
HEAD_NUM = 4
LAYER_NUM = 4
D_MODEL = 256
D_FF = 1024
MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 3e-4
VOCAB_SIZE = 16000
```

### 9 个模块对应关系

| # | 模块 | 代码 |
|---|---|---|
| 1 | Positional Encoding | `PositionalEncoding` |
| 2 | Multi-Head Attention | `MultiHeadAttention` |
| 3 | ADD（残差） | 内联在 `BlockOne`/`BlockTwo` |
| 4 | Layer Normalization | `LayerNormalization` |
| 5 | Feed Forward Network | `FFN` |
| 6 | Masked Multi-Head Attention | `MultiHeadAttention` + mask |
| 7 | Multi-Head Attention（cross） | `MultiHeadAttention` + memory |
| 8 | Linear | `Generator.proj` |
| 9 | Softmax | `Generator` |

**教程的关键设计**：模块 2、6、7 **共用一个类** —— 传入 mask 就是 masked self-attention，
传入 memory 作 key/value 就是 cross-attention。所以只需实现 4 个基础模块。

### 封装层级

```
BlockOne = MHA -> ADD -> LN            （模块 2/6/7 + 3 + 4）
BlockTwo = FFN -> ADD -> LN            （模块 5 + 3 + 4）
    ↓
EncoderLayer = BlockOne(self) + BlockTwo
DecoderLayer = BlockOne(masked self) + BlockOne(cross) + BlockTwo
    ↓
EncoderStack / DecoderStack = layer_num 份拷贝 + 最终 LN
    ↓
Translate = Embedding + PE + EncoderStack + DecoderStack + Generator
```

### 两个 mask 的区别

| | 形状 | 作用 |
|---|---|---|
| `src_mask` | `(N, 1, T_src)` | 屏蔽 `<pad>` |
| `tgt_mask` | `(N, T_tgt, T_tgt)` | 屏蔽 `<pad>` **且** 屏蔽未来位置 |

`subsequent_mask(4)` 实测输出下三角：

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**这是 decoder 自回归的关键** —— 预测第 i 个词时不能偷看后面的词。
`tgt_mask` 是 padding mask 与 causal mask 按位与的结果。

## 训练细节

### 损失：labels 错位一位

```python
tgt   = labels[:, :-1]    # 解码器输入：<bos> ... 倒数第二个
tgt_y = labels[:, 1:]     # 监督目标：  第二个 ... <eos>
```

### 忽略 padding

```python
loss_fn = CrossEntropyLoss(ignore_index=0)   # PAD_ID = 0
```

否则模型会去学"预测 `<pad>`"，浪费容量。

### AdamW + 分组权重衰减

```python
decay    = [p for p in params if p.dim() >= 2]   # 权重矩阵
no_decay = [p for p in params if p.dim() < 2]    # bias / LayerNorm
```

LayerNorm 的 a/b 和所有 bias 不做权重衰减。

### Warmup + Cosine

前 1000 步线性升温，之后余弦衰减。Transformer 对初期学习率敏感。

### 混合精度

```python
scaler = torch.amp.GradScaler("cuda", enabled=True)
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss = ...
```

8GB 显存下 AMP 基本是必需的。

## 推理：`greedy_decode`

和训练**完全不同**：

| | 训练 | 推理 |
|---|---|---|
| 解码器输入 | 完整目标序列（teacher forcing） | 模型自己上一步的输出 |
| 调用次数 | 1 次前向 | `max_len` 次循环 |
| 停止条件 | 固定长度 | 遇到 `<eos>` |

## 实测结果

训练 2 个 epoch（30,586 步，约 3.5 小时，RTX 5060 8GB）：

| epoch | step | train loss | valid loss |
|---|---|---|---|
| 1 | 15,293 | 4.010 | 3.037 |
| 2 | 30,586 | 2.970 | **2.528** |

**valid 低于 train**，说明完全没有过拟合，继续训练仍有空间。

翻译结果：

```
你好              →  Hello.                    ✅
是的              →  Yes.                      ✅
这是一个测试。     →  This is a test.           ✅
我不知道。         →  I don't know.             ✅
飞机              →  aircraft                  ✅
请帮我一个忙。     →  Please help me a favor.   ✅
自然语言处理       →  Natural language          ⚠️ 缺 processing
今天天气很好。     →  Today's good day.         ⚠️ 语义对，表达不地道
```

## 性能实测（RTX 5060 8GB）

| 阶段 | 耗时 |
|---|---|
| 加载 98.6 万句对 | 约 5 分钟 |
| **并行分词** | **约 30 秒** |
| 训练 1 epoch | 约 1.7 小时 |

## 踩过的坑

### 单线程分词慢 4 个数量级

sentencepiece 是**单线程**的，98.6 万句对最初跑了 **52 分钟还没完成**（只用了 0.32 核）。

改成 `multiprocessing` 20 进程并行后：

```
之前：单线程，预计数小时
之后：32,860 pairs/s  →  98.6 万句对仅需 30 秒
```

**这台机器有 24 个逻辑核，之前完全浪费了。**

### `-1e9` 在 fp16 下溢出

```python
scores = scores.masked_fill(mask == 0, -1e9)   # ❌ fp16 崩溃
```

```
RuntimeError: value cannot be converted to type c10::Half without overflow
```

fp16 最大约 65504。改成：

```python
neg = torch.finfo(scores.dtype).min          # ✅ 两种精度都安全
```

**这个 bug 在 fp32 + 小数据下不会暴露，一开 AMP 就崩。**

### 领域不匹配比模型大小更致命

第一版用 `Helsinki-NLP/news_commentary`（6.9 万句对，纯新闻评论），训练 10 epoch 后：

```
你好    →  you't you't you
是的    →  The Right
飞机    →  The Rest
```

**句子越短越糟** —— 因为新闻语料的中文字符中位长度是 **113**，
`你好`（2 字）这类输入在训练分布之外。

换成 OPUS-100（97.9 万句对，混合领域，中位长度 **18**）后，**仅 1 epoch**：

```
你好    →  Hello.       ✅
是的    →  Yes.         ✅
飞机    →  aircraft     ✅
```

**教训**：语料的**领域和句长分布**比语料规模更重要。
一个在新闻上表现良好的模型，换个领域可能完全失效。

---

## 参考

**鱼书②官方代码**
- [oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)

**数据集来源**
- `addition.txt` / `date.txt` — 上述仓库的 `dataset/` 目录
- PTB — [wojzaremba/lstm](https://github.com/wojzaremba/lstm) 的 `data/`
- OPUS-100 — [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)（混合领域平行语料）

**参考**
- CSDN《Transformer 论文实现》系列（nocml）— 9 模块拆解与 encoder-decoder 实现

**论文**
- Mikolov et al., *Distributed Representations of Words and Phrases*, 2013 — word2vec、negative sampling、0.75 指数
- Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015 — 加性 attention
- Luong et al., *Effective Approaches to Attention-based Neural Machine Translation*, 2015 — 乘性 attention
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — Transformer
