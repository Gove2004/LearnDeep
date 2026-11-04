"""训练中译英 Transformer。

数据：Helsinki-NLP/opus-100（en-zh），约 98 万句对。
运行：python train.py
"""

import math

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from config import (AMP, BATCH_SIZE, BOS_ID, CKPT_PATH, DEVICE, EOS_ID, EPOCHS,
                    LEARNING_RATE, LIMIT, LOG_EVERY, MAX_LEN, PAD_ID, SEED,
                    VALID_LIMIT, WARMUP_STEPS)
from dataset import (ParallelData, collate, get_tokenizer, load_pairs,
                     make_src_mask, make_tgt_mask)
from model import Translate


def lr_at(step, total_steps):
    """前 WARMUP_STEPS 步线性升温，之后余弦衰减。"""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(model, loader, loss_fn, optimizer, vocab_size, step, total_steps,
              train=True):
    """跑一个 epoch，返回平均 loss 和更新后的 step。"""
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # 目标序列错位一位：输入 tgt[:, :-1]，监督 tgt[:, 1:]
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            _, log_probs = model(src, tgt_in,
                                 make_src_mask(src), make_tgt_mask(tgt_in))
            loss = loss_fn(log_probs.reshape(-1, vocab_size), tgt_out.reshape(-1))

        if train:
            lr = lr_at(step, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：Transformer 对学习率敏感
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % LOG_EVERY == 0:
                avg = total_loss / max(total_tokens, 1)
                print(f"  step {step}/{total_steps}  loss={avg:.4f}  "
                      f"ppl={math.exp(min(avg, 20)):.1f}  lr={lr:.1e}", flush=True)

        # padding 不计入平均 loss
        n = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * n
        total_tokens += n

    return total_loss / max(total_tokens, 1), step


@torch.no_grad()
def translate(model, tokenizer, text, max_len=100):
    """贪心解码：一次生成一个词，遇到 <eos> 停止。"""
    model.eval()
    ids = torch.tensor([tokenizer.Encode(text, out_type=int)[:MAX_LEN]],
                       device=DEVICE)
    src_mask = make_src_mask(ids)
    memory = model.encode(ids, src_mask)

    # 从 <bos> 开始，逐个生成
    out = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len - 1):
        logits = model.decode(memory, src_mask, out, make_tgt_mask(out))[1]
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS_ID:
            break
        out = torch.cat([out, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    return tokenizer.Decode(out[0, 1:].tolist())


def main():
    torch.manual_seed(SEED)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size()
    print(f"词表大小 = {vocab_size}")

    print("加载数据 ...")
    train_pairs = load_pairs("train", LIMIT)
    valid_pairs = load_pairs("validation", VALID_LIMIT)
    print(f"训练 {len(train_pairs):,} 句对，验证 {len(valid_pairs):,} 句对")

    train_loader = DataLoader(ParallelData(train_pairs, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(ParallelData(valid_pairs, tokenizer),
                              batch_size=BATCH_SIZE, collate_fn=collate)

    model = Translate(vocab_size).to(DEVICE)
    print(f"参数量 = {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss(ignore_index=PAD_ID)   # padding 不算损失
    total_steps = len(train_loader) * EPOCHS
    print(f"总步数 = {total_steps:,}\n")

    step = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, step = run_epoch(model, train_loader, loss_fn, optimizer,
                                     vocab_size, step, total_steps, train=True)
        valid_loss, _ = run_epoch(model, valid_loader, loss_fn, optimizer,
                                  vocab_size, step, total_steps, train=False)
        print(f"=== epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  "
              f"valid={valid_loss:.4f} ===\n", flush=True)

        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"已保存 -> {CKPT_PATH}\n")

    print("试翻几句：")
    for s in ["你好", "这是一个测试。", "我喜欢学习新事物。"]:
        print(f"  {s}  ->  {translate(model, tokenizer, s)}")


if __name__ == "__main__":
    main()
