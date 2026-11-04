"""Transformer（编码器-解码器），中译英。

对应教程的 9 个模块：

    1  PositionalEncoding    位置编码
    2  MultiHeadAttention    多头注意力
    3  残差相加              写在 BlockOne / BlockTwo 里
    4  LayerNormalization    层归一化
    5  FFN                   前馈网络
    6  带掩码的多头注意力      就是模块 2 传入掩码
    7  交叉注意力             就是模块 2 传入 memory 作 key/value
    8  Linear                Generator 里的 proj
    9  Softmax               Generator 里
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from config import D_FF, D_MODEL, DROPOUT, HEAD_NUM, LAYER_NUM, MAX_LEN


class PositionalEncoding(nn.Module):
    """模块 1：位置编码。

    Transformer 没有循环，词序信息必须显式加进去。
    这里用固定的 sin/cos 模式。
    """

    def __init__(self, d_model, dropout, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        # buffer 不是参数，但会跟着 .to(device) 一起移动
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (N, T, D)，加上表的前 T 行
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    """模块 2、6、7：多头注意力。

    一个类服务三种用途：
      不传 mask        -> 自注意力（编码器）
      传 mask          -> 带掩码的自注意力（解码器第一层）
      传 memory 作 kv  -> 交叉注意力（解码器第二层）
    """

    def __init__(self, head_num, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // head_num
        self.head_num = head_num
        # 4 个投影：W_q, W_k, W_v, W_o
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        N = query.size(0)

        # 投影后拆成多头：(N, T, D) -> (N, head, T, d_k)
        def split(x, linear):
            return linear(x).view(N, -1, self.head_num, self.d_k).transpose(1, 2)

        q = split(query, self.w_q)
        k = split(key, self.w_k)
        v = split(value, self.w_v)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 掩码形状是 (N, 1, T_q, T_k)，可以直接和分数广播。
            # 为 0 的位置填成极小值，softmax 后趋近于 0。
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), v)

        # 多头拼回去：(N, head, T, d_k) -> (N, T, D)
        out = out.transpose(1, 2).contiguous().view(N, -1, self.head_num * self.d_k)
        return self.w_o(out)


class LayerNormalization(nn.Module):
    """模块 4：层归一化。"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class FFN(nn.Module):
    """模块 5：前馈网络。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class BlockOne(nn.Module):
    """注意力 + 残差 + 层归一化。对应模块 2/6/7 + 3 + 4。"""

    def __init__(self, head_num, d_model, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(head_num, d_model, dropout)
        self.ln = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        return self.ln(query + self.dropout(self.mha(query, key, value, mask)))


class BlockTwo(nn.Module):
    """前馈 + 残差 + 层归一化。对应模块 5 + 3 + 4。"""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ln(x + self.dropout(self.ffn(x)))


class EncoderLayer(nn.Module):
    """编码器的一层：自注意力 + 前馈。"""

    def __init__(self, head_num, d_model, d_ff, dropout):
        super().__init__()
        self.attn = BlockOne(head_num, d_model, dropout)
        self.ffn = BlockTwo(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        return self.ffn(self.attn(x, x, x, mask))


class DecoderLayer(nn.Module):
    """解码器的一层：带掩码自注意力 + 交叉注意力 + 前馈。"""

    def __init__(self, head_num, d_model, d_ff, dropout):
        super().__init__()
        self.self_attn = BlockOne(head_num, d_model, dropout)
        self.cross_attn = BlockOne(head_num, d_model, dropout)
        self.ffn = BlockTwo(d_model, d_ff, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 第一层：带掩码的自注意力，不能看到未来的词
        x = self.self_attn(x, x, x, tgt_mask)
        # 第二层：交叉注意力，query 来自解码器，key/value 来自编码器
        x = self.cross_attn(x, memory, memory, src_mask)
        return self.ffn(x)


class Generator(nn.Module):
    """模块 8 + 9：线性层 + log_softmax。"""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Translate(nn.Module):
    """完整模型。结构参数默认从 config.py 读取。"""

    def __init__(self, vocab_size, head_num=HEAD_NUM, layer_num=LAYER_NUM,
                 d_model=D_MODEL, d_ff=D_FF, dropout=DROPOUT,
                 max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model

        # 用 nn.ModuleList 堆叠多层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(head_num, d_model, d_ff, dropout)
            for _ in range(layer_num)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(head_num, d_model, d_ff, dropout)
            for _ in range(layer_num)
        ])

        # 源语言和目标语言各一张嵌入表。
        # 虽然共用同一个分词器，但训练后两张表并不相同，不能合并。
        self.src_embedd = nn.Embedding(vocab_size, d_model)
        self.tgt_embedd = nn.Embedding(vocab_size, d_model)
        self.pe_encode = PositionalEncoding(d_model, dropout, max_len)
        self.pe_decode = PositionalEncoding(d_model, dropout, max_len)
        # 每层输出后做一次归一化（论文里叫 final layer norm）
        self.encoder_norm = LayerNormalization(d_model)
        self.decoder_norm = LayerNormalization(d_model)
        self.generator = Generator(d_model, vocab_size)

    def encode(self, src, src_mask):
        """编码：源语言 -> memory。"""
        x = self.pe_encode(self.src_embedd(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """解码：返回 (输出向量, log 概率)。"""
        x = self.pe_decode(self.tgt_embedd(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.decoder_norm(x)
        return x, self.generator(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
