import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_corpus
from model import build_model
from util import decode


def run(ckpt_path, save_path, num_samples=3, device=None, embed_dim=64,
        hidden_size=128):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_corpus("date")
    id_to_char = dataset.get_id_to_char()
    answer_len = dataset.get_answer_len()

    model = build_model("attention", dataset.get_vocab_size(),
                        embed_dim, hidden_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.start_id = dataset.get_start_id()
    model.eval()

    test_set = dataset.get_test_dataset()
    x = torch.stack([test_set[i][0] for i in range(num_samples)]).to(device)

    with torch.no_grad():
        ids, weights = model.generate(x, answer_len, return_attention=True)

    weights = weights.cpu().numpy().squeeze(2)
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        question = decode(x[i].tolist(), id_to_char, dataset.reverse)
        answer = decode(ids[i].tolist(), id_to_char, dataset.reverse)
        ax.imshow(weights[i], aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(question)))
        ax.set_xticklabels(list(question), rotation=90, fontsize=7)
        ax.set_yticks(range(len(answer)))
        ax.set_yticklabels(list(answer), fontsize=9)
        ax.set_xlabel("input")
        ax.set_ylabel("output")
        ax.set_title(f"{question.strip()} -> {answer}", fontsize=9)

    plt.tight_layout()
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[attention] heatmap saved -> {save_path}")


if __name__ == "__main__":
    run("Checkpoints/best_seq2seq_date_attention.pth",
        "Logs/plots/attention_heatmap.png")
