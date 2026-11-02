from pathlib import Path

import pandas as pd
import torch


def decode(ids, id_to_char, reverse=False):
    text = "".join(id_to_char[int(i)] for i in ids)
    if reverse:
        text = text[0] + text[1:][::-1]
    return text


def train_once(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, total = 0.0, 0

    for x, t in train_loader:
        x, t = x.to(device), t.to(device)
        t_in, t_out = t[:, :-1], t[:, 1:]

        optimizer.zero_grad()
        logits = model(x, t_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), t_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += x.size(0)

    return running_loss / total


def test_once(model, test_loader, criterion, device, max_len=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            t_in, t_out = t[:, :-1], t[:, 1:]

            logits = model(x, t_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), t_out.reshape(-1))
            running_loss += loss.item() * x.size(0)

            ids = model.generate(x, max_len if max_len is not None else t_out.size(1))
            pred = ids[:, :t_out.size(1)]
            correct += (pred == t_out).all(dim=1).sum().item()
            total += x.size(0)

    return running_loss / total, correct / total


class Checkpoint:
    def __init__(self, best_path, last_path, patience):
        self.best_path = Path(best_path) if best_path else None
        self.last_path = Path(last_path) if last_path else None
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = -1

    def load(self, model):
        if self.best_path and self.best_path.exists():
            model.load_state_dict(torch.load(self.best_path, map_location="cpu"))
            print(f"[ckpt] Loaded best checkpoint from {self.best_path}")
        elif self.last_path and self.last_path.exists():
            model.load_state_dict(torch.load(self.last_path, map_location="cpu"))
            print(f"[ckpt] Loaded last checkpoint from {self.last_path}")
        else:
            print("[ckpt] No checkpoint found. Starting from scratch.")

    def update(self, model, test_loss, epoch):
        if self.last_path:
            self.last_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.last_path)
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            if self.best_path:
                self.best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.best_path)
            return True, False
        elif epoch - self.best_epoch >= self.patience:
            return False, True
        return False, False


class EpochLogger:
    def __init__(self, log_path, epochs, save_epoch_rate=5):
        self.epoch_logs = []
        self.log_path = Path(log_path) if log_path else None
        self.epochs = epochs
        self.save_epoch_rate = save_epoch_rate

    def log(self, epoch, train_loss, test_loss, test_acc, lr, is_improved):
        self.epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
        })
        print(f"[logger] Epoch {epoch}/{self.epochs} (lr = {lr:.6f}): "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
              f"{' [new best]' if is_improved else ''}")
        if self.log_path and (epoch % self.save_epoch_rate == 0 or epoch == self.epochs):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.epoch_logs).to_csv(self.log_path, index=False)
