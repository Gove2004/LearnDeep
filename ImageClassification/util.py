from pathlib import Path
import torch
import pandas as pd


def train_once(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def test_once(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


class Checkpoint:
    def __init__(self, best_path, last_path, patience):
        self.best_path = best_path
        self.last_path = last_path
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = -1

    def load(self, model):
        if self.best_path.exists():
            model.load_state_dict(torch.load(self.best_path, map_location="cpu"))
            print(f"[ckpt] Loaded best checkpoint from {self.best_path}")
        elif self.last_path.exists():
            model.load_state_dict(torch.load(self.last_path, map_location="cpu"))
            print(f"[ckpt] Loaded last checkpoint from {self.last_path}")
        else:
            print("[ckpt] No checkpoint found. Starting from scratch.")

    def update(self, model, test_loss, epoch):
        self.last_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.last_path)
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.best_path)
            return True, False
        elif epoch - self.best_epoch >= self.patience:
            return False, True
        return False, False


class EpochLogger:
    def __init__(self, log_path, epochs, save_epoch_rate=5):
        self.epoch_logs = []
        self.log_path = log_path
        self.epochs = epochs
        self.save_epoch_rate = save_epoch_rate

    def log(self, epoch, train_loss, train_acc, test_loss, test_acc, lr, is_improved):
        self.epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "lr": lr,
            "test_acc": test_acc,
        })
        print(f"[logger] Epoch {epoch}/{self.epochs} (lr = {lr:.6f}): "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
              f"{' [new best]' if is_improved else ''}")
        if epoch % self.save_epoch_rate == 0 or epoch == self.epochs:
            logs = pd.DataFrame(self.epoch_logs)
            logs.to_csv(self.log_path, index=False)
