import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_corpus
from model import build_model
from util import Checkpoint, EpochLogger, decode, test_once, train_once


class Config:
    DATASET = "addition"
    MODEL = "gru"

    DATASET_LIMIT = None
    TEST_RATE = 0.1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    PATIENCE = 5
    EMBED_DIM = 64
    HIDDEN_SIZE = 128

    SEED = 42
    LOAD_CKPT = False

    BEST_CKPT_PATH = None
    LAST_CKPT_PATH = None
    LOG_PATH = None


def build_loaders(dataset, config):
    train_loader = DataLoader(dataset.get_train_dataset(),
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              generator=torch.Generator().manual_seed(config.SEED))
    test_loader = DataLoader(dataset.get_test_dataset(),
                             batch_size=config.BATCH_SIZE, shuffle=False,
                             generator=torch.Generator().manual_seed(config.SEED))
    return train_loader, test_loader


def run(config):
    print(f"[config] dataset={config.DATASET}  model={config.MODEL}  "
          f"device={config.DEVICE}  seed={config.SEED}")

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

    dataset = build_corpus(config.DATASET,
                           test_rate=config.TEST_RATE,
                           seed=config.SEED,
                           limit=config.DATASET_LIMIT)
    train_loader, test_loader = build_loaders(dataset, config)

    id_to_char = dataset.get_id_to_char()
    answer_len = dataset.get_answer_len()

    model = build_model(config.MODEL, dataset.get_vocab_size(),
                        config.EMBED_DIM, config.HIDDEN_SIZE).to(config.DEVICE)
    model.start_id = dataset.get_start_id()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = Checkpoint(config.BEST_CKPT_PATH, config.LAST_CKPT_PATH, config.PATIENCE)
    if config.LOAD_CKPT:
        ckpt.load(model)

    print(f"[main] epochs={config.EPOCHS}  batch={config.BATCH_SIZE}  "
          f"lr={config.LEARNING_RATE}  vocab={dataset.get_vocab_size()}  "
          f"answer_len={answer_len}")

    logger = EpochLogger(config.LOG_PATH, config.EPOCHS)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_once(model, train_loader, criterion, optimizer, config.DEVICE)
        test_loss, test_acc = test_once(model, test_loader, criterion, config.DEVICE)

        is_improved, is_stopped = ckpt.update(model, test_loss, epoch)
        logger.log(epoch, train_loss, test_loss, test_acc,
                   optimizer.param_groups[0]["lr"], is_improved)

        if is_stopped:
            print(f"[main] early stopping @ epoch {epoch} "
                  f"(no improvement for {ckpt.patience} epochs, "
                  f"best was epoch {ckpt.best_epoch})")
            break

    print(f"[main] done. best test loss = {ckpt.best_loss:.4f} @ epoch {ckpt.best_epoch}")
    show_examples(model, dataset, id_to_char, answer_len, config)


def show_examples(model, dataset, id_to_char, answer_len, config, n=10):
    model.eval()
    test_set = dataset.get_test_dataset()
    x = torch.stack([test_set[i][0] for i in range(n)]).to(config.DEVICE)
    t = torch.stack([test_set[i][1] for i in range(n)])
    with torch.no_grad():
        pred = model.generate(x, answer_len)
    width = max(len(decode(x[i].tolist(), id_to_char, dataset.reverse).strip())
                for i in range(n))
    print("[main] sample predictions:")
    for i in range(n):
        question = decode(x[i].tolist(), id_to_char, dataset.reverse).strip()
        target = decode(t[i, 1:].tolist(), id_to_char, dataset.reverse).strip()
        got = decode(pred[i].tolist(), id_to_char, dataset.reverse).strip()
        flag = "OK " if got == target else "BAD"
        print(f"    {flag} {question:>{width}} -> {got:<12} (expect {target})")


if __name__ == "__main__":
    run(Config)
