from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class Config:
    LOG_PATH = Path("./Logs/image_classification_log.csv")
    SAVE_PATH = Path("")


def main():
    if not Config.LOG_PATH.exists():
        raise FileNotFoundError(f"[draw] Not found log file: {Config.LOG_PATH}")
    df = pd.read_csv(Config.LOG_PATH)
    df["epoch"] = pd.to_numeric(df["epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="train", color="#1f77b4", lw=2)
    ax.plot(df["epoch"], df["test_loss"], label="test", color="#d62728", lw=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"] * 100, label="train", color="#2ca02c", lw=2)
    ax.plot(df["epoch"], df["test_acc"] * 100, label="test", color="#ff7f0e", lw=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df["epoch"], df["lr"], color="#9467bd", lw=2)
    ax.set_title("Learning Rate (cosine)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if Config.SAVE_PATH.absolute() != Path("").absolute():
        plt.savefig(Config.SAVE_PATH, dpi=150)
        print(f"[draw] image saved -> {Config.SAVE_PATH}")


if __name__ == "__main__":
    main()
