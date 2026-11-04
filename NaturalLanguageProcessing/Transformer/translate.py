"""用训练好的模型做中译英（交互式）。

运行：python translate.py
"""

import io
import sys

import torch

from config import BOS_ID, CKPT_PATH, DEVICE, EOS_ID, MAX_LEN, TOKENIZER_NAME
from dataset import get_tokenizer, make_src_mask, make_tgt_mask
from model import Translate


def translate(model, tokenizer, text, max_len=100):
    """把一句中文翻译成英文。"""
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tokenizer.Encode(text, out_type=int)[:MAX_LEN]],
                           device=DEVICE)
        src_mask = make_src_mask(ids)
        memory = model.encode(ids, src_mask)

        # 从 <bos> 开始，一次生成一个词
        out = torch.tensor([[BOS_ID]], device=DEVICE)
        for _ in range(max_len - 1):
            logits = model.decode(memory, src_mask, out, make_tgt_mask(out))[1]
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS_ID:
                break
            out = torch.cat([out, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    return tokenizer.Decode(out[0, 1:].tolist())


def main():
    tokenizer = get_tokenizer()
    model = Translate(tokenizer.vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    print(f"模型已加载：{CKPT_PATH.name}")
    print("输入中文，回车翻译；输入 q 退出\n")

    while True:
        try:
            text = input("中文 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            break

        print(f"  英文: {translate(model, tokenizer, text)}\n")

    print("bye")


if __name__ == "__main__":
    # 控制台是 GBK，强制 UTF-8 避免中文乱码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    main()
