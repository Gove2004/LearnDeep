from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSeq2SeqModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.vocab_size = None
        self.start_id = 0

    @abstractmethod
    def forward(self, x, t):
        ...

    @abstractmethod
    def generate(self, x, max_len):
        ...

    def get_vocab_size(self) -> int:
        return self.vocab_size


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (N, T) -> embed: (N, T, D) -> out: (N, T, H), h: (1, N, H)
        embedded = self.embed(x)
        out, h = self.gru(embedded)
        return out, h


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, t, h):
        # t: (N, T) -> embed: (N, T, D) -> out: (N, T, H) -> logits: (N, T, V)
        embedded = self.embed(t)
        out, h = self.gru(embedded, h)
        logits = self.fc(out)
        return logits, h


class GRUSeq2Seq(BaseSeq2SeqModel):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embed_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_size)

    def forward(self, x, t):
        # x: (N, T_in), t: (N, T_out) -> logits: (N, T_out, V)
        _, h = self.encoder(x)
        logits, _ = self.decoder(t, h)
        return logits

    @torch.no_grad()
    def generate(self, x, max_len, return_attention=False):
        # x: (N, T_in) -> ids: (N, max_len)
        self.eval()
        _, h = self.encoder(x)
        batch = x.size(0)
        current = torch.full((batch, 1), self.start_id, dtype=torch.long, device=x.device)
        outputs = []
        for _ in range(max_len):
            logits, h = self.decoder(current, h)
            current = logits[:, -1, :].argmax(dim=1, keepdim=True)
            outputs.append(current)
        return torch.cat(outputs, dim=1)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_encoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_out):
        # decoder_hidden: (N, 1, H)
        # encoder_out: (N, T_in, H)
        # returns context: (N, 1, H), weights: (N, T_in, 1)
        score = self.v(torch.tanh(self.w_encoder(encoder_out) + self.w_decoder(decoder_hidden)))
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.transpose(1, 2), encoder_out)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, t, h, encoder_out):
        # t: (N, T) -> embed: (N, T, D)
        # h: (1, N, H), encoder_out: (N, T_in, H)
        embedded = self.embed(t)
        outputs, weights_all = [], []
        for step in range(embedded.size(1)):
            query = h.transpose(0, 1)
            context, weights = self.attention(query, encoder_out)
            gru_in = torch.cat([embedded[:, step:step + 1, :], context], dim=2)
            out, h = self.gru(gru_in, h)
            combined = torch.cat([out, context], dim=2)
            outputs.append(self.fc(combined))
            weights_all.append(weights)
        logits = torch.cat(outputs, dim=1)
        weights_all = torch.stack(weights_all, dim=1).squeeze(-1)
        return logits, h, weights_all


class AttentionSeq2Seq(BaseSeq2SeqModel):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embed_dim, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_size)

    def forward(self, x, t):
        encoder_out, h = self.encoder(x)
        logits, _, _ = self.decoder(t, h, encoder_out)
        return logits

    @torch.no_grad()
    def generate(self, x, max_len, return_attention=False):
        self.eval()
        encoder_out, h = self.encoder(x)
        batch = x.size(0)
        current = torch.full((batch, 1), self.start_id, dtype=torch.long, device=x.device)
        outputs, attentions = [], []
        for _ in range(max_len):
            logits, h, weights = self.decoder(current, h, encoder_out)
            current = logits[:, -1, :].argmax(dim=1, keepdim=True)
            outputs.append(current)
            attentions.append(weights)
        ids = torch.cat(outputs, dim=1)
        if return_attention:
            return ids, torch.stack(attentions, dim=1).squeeze(-1)
        return ids


MODELS = {
    "gru": GRUSeq2Seq,
    "attention": AttentionSeq2Seq,
}


def build_model(name, vocab_size, embed_dim=64, hidden_size=128):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {list(MODELS)}")
    return MODELS[name](vocab_size, embed_dim, hidden_size)


if __name__ == "__main__":
    for name in MODELS:
        model = build_model(name, vocab_size=59, embed_dim=64, hidden_size=128)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{name}] {type(model).__name__}  params={total:,}")
        for pname, p in model.named_parameters():
            print(f"    {pname:38s} {tuple(p.shape)}")
