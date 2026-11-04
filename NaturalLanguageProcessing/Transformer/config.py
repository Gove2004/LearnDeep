"""所有配置集中在这里。

改参数只需要动这一个文件。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "Datasets" / "opus100_en_zh"
PROCESSED_DIR = DATA_DIR / "processed"

CKPT_DIR = PROJECT_ROOT / "Checkpoints"
CKPT_NAME = "transformer_opus_zh_en.pth"
CKPT_PATH = CKPT_DIR / CKPT_NAME

TOKENIZER_NAME = "spm.model"

# ---------------------------------------------------------------- 数据

MAX_LEN = 128          # 实测覆盖 99.8% 的句子
MIN_CHARS = 2          # 过滤过短的句子
MAX_CHARS = 500        # 过滤整段联合国文件（最长 4.9 万字符）
LIMIT = None           # 限制训练句对数，调试时可设成 5000
VALID_LIMIT = 2000     # 验证集句对数

# 特殊 token 编号，必须与训练分词器时一致
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ---------------------------------------------------------------- 模型

VOCAB_SIZE = 16000     # 词表大小（训练分词器时用）
HEAD_NUM = 4           # 多头个数
LAYER_NUM = 4          # 编码器/解码器层数
D_MODEL = 256          # 词向量维度
D_FF = 1024            # 前馈网络中间层维度
DROPOUT = 0.1

# ---------------------------------------------------------------- 训练

BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 3e-4
WARMUP_STEPS = 1000    # 前 1000 步线性升温
LOG_EVERY = 500        # 每多少步打印一次
SEED = 42

# ---------------------------------------------------------------- 其他

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True             # 混合精度，8GB 显存下基本必需
