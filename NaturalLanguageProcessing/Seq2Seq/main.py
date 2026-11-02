from pathlib import Path

import draw_attention
import train
from train import Config


CKPT_DIR = Path("Checkpoints")
LOG_DIR = Path("Logs")
PLOT_DIR = LOG_DIR / "plots"


class RunConfig(Config):
    """A fresh copy of Config per run.

    Subclassing gives each run its own attribute namespace, so setting
    attributes on one run never leaks into another (or into Config itself).
    """


def make_config(dataset, model, name):
    cfg = RunConfig()
    cfg.DATASET = dataset
    cfg.MODEL = model
    cfg.BEST_CKPT_PATH = CKPT_DIR / f"best_seq2seq_{name}.pth"
    cfg.LAST_CKPT_PATH = CKPT_DIR / f"last_seq2seq_{name}.pth"
    cfg.LOG_PATH = LOG_DIR / f"seq2seq_{name}_log.csv"
    return cfg


def run_addition():
    """Seq2Seq on Addition (GRU)"""
    train.run(make_config("addition", "gru", "addition"))


def run_date_gru():
    """Seq2Seq on Date Conversion (GRU, no attention)"""
    train.run(make_config("date", "gru", "date_gru"))


def run_date_attention():
    """Seq2Seq on Date Conversion (GRU + Attention)"""
    train.run(make_config("date", "attention", "date_attention"))


def run_attention_heatmap():
    """Attention Heatmap"""
    draw_attention.run(
        ckpt_path=CKPT_DIR / "best_seq2seq_date_attention.pth",
        save_path=PLOT_DIR / "attention_heatmap.png",
    )


RUN_LEARNS = [
    run_addition,
    run_date_gru,
    run_date_attention,
    run_attention_heatmap,
]


def main():
    is_running = True
    while is_running:
        print(f"[main] Select a learn to run (1-{len(RUN_LEARNS)}) or 0 to quit:")
        for i, learning in enumerate(RUN_LEARNS, start=1):
            print(f"    {i} - {learning.__doc__}")
        choice = input(">>> ").strip()
        if choice.lower() == "0":
            is_running = False
            continue
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(RUN_LEARNS):
                learning = RUN_LEARNS[choice_num - 1]
                print(f"=====  Start   {choice_num} - {learning.__doc__} =====")
                learning()
                print(f"===== Finished {choice_num} - {learning.__doc__} =====")
            else:
                print(f"[main] Error: number between 1 and {len(RUN_LEARNS)}.")
        except ValueError as e:
            print(f"[main] Error: invalid input. {e}")


if __name__ == "__main__":
    main()
