"""数据加载、分词，以及注意力掩码。

掩码放在这里是因为它属于"把原始文本变成模型输入"这一环。
"""

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from config import (BOS_ID, DATA_DIR, EOS_ID, MAX_CHARS, MAX_LEN, MIN_CHARS,
                    PAD_ID, PROCESSED_DIR, TOKENIZER_NAME, UNK_ID, VOCAB_SIZE)


# ---------------------------------------------------------------- 掩码


def make_src_mask(src):
    """源语言掩码：(N, T) -> (N, 1, 1, T)。

    真实词为 True，<pad> 为 False，让注意力忽略填充符。

    多出来的两个 1 是为了能直接和注意力分数 (N, head, T_q, T_k) 广播：
    倒数第二维给 T_q（所有查询位置都一样），最后一维对应 T_k。
    """
    return (src != PAD_ID)[:, None, None, :]


def make_tgt_mask(tgt):
    """目标语言掩码：(N, T) -> (N, 1, T, T)。

    两件事合一：
      1. 屏蔽 <pad>
      2. 屏蔽未来的位置（下三角）

    第 2 点是解码器能自回归的关键：预测第 i 个词时看不到 i 之后的词。
    维度 (N, 1, T, T) 同样是为了和 (N, head, T_q, T_k) 广播。
    """
    T = tgt.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
    return (tgt != PAD_ID)[:, None, None, :] & causal


# ---------------------------------------------------------------- 分词器


def get_tokenizer():
    """加载分词器；如果还没有，就先训练一个。"""
    path = PROCESSED_DIR / TOKENIZER_NAME
    if path.exists():
        return spm.SentencePieceProcessor(model_file=str(path))

    print(f"分词器不存在，正在训练（词表 {VOCAB_SIZE}）...")
    pairs = load_pairs("train", limit=200000)
    corpus = PROCESSED_DIR / "corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus, "w", encoding="utf-8") as f:
        for zh, en in pairs:
            f.write(zh + "\n" + en + "\n")

    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(path.with_suffix("")),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,   # 中文常用取值
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
    )
    return spm.SentencePieceProcessor(model_file=str(path))


# ---------------------------------------------------------------- 数据


def load_pairs(split="train", limit=None):
    """读出 (中文, 英文) 句对。

    个别行是整段联合国文件而不是单句，用 MAX_CHARS 过滤掉，
    否则它们会主导分词器。
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_DIR))[split]
    n = len(ds) if limit is None else min(limit, len(ds))

    pairs = []
    for i in range(n):
        t = ds[i]["translation"]
        zh = str(t["zh"]).strip().replace("\n", " ")
        en = str(t["en"]).strip().replace("\n", " ")
        if MIN_CHARS <= len(zh) <= MAX_CHARS and MIN_CHARS <= len(en) <= MAX_CHARS:
            pairs.append((zh, en))
    return pairs


class ParallelData(Dataset):
    """平行语料。每条样本是 (源 id 列表, 目标 id 列表)。

    目标序列两端加上 <bos> 和 <eos>。
    """

    def __init__(self, pairs, tokenizer):
        self.samples = []
        for zh, en in pairs:
            src = tokenizer.Encode(zh, out_type=int)[:MAX_LEN]
            tgt = ([BOS_ID] + tokenizer.Encode(en, out_type=int) + [EOS_ID])[:MAX_LEN]
            if len(src) >= 2 and len(tgt) >= 3:
                self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    """把一个 batch 补齐（右侧填 PAD）。"""
    max_src = max(len(s) for s, _ in batch)
    max_tgt = max(len(t) for _, t in batch)
    src = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src, tgt
