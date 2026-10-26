import os
from pathlib import Path
import random
import numpy
import torch
from torchvision import transforms
from dataset import GarbageClassificationImageClassificationDataset
from model import ResNet18
from util import Checkpoint, EpochLogger, train_once, test_once
from torch.utils.data import DataLoader


class Config:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MNIST_DATASET_PATH = Path("./Datasets")
    CIFAR_10_DATASET_PATH = Path("./Datasets/CIFAR-10")
    CIFAR_100_DATASET_PATH = Path("./Datasets/CIFAR-100")
    OCT_DATASET_PATH = Path("./Datasets/OCT")
    GARBAGECLASSIFICATION_DATASET_PATH = Path("./Datasets/GarbageClassification")
    
    BEST_CKPT_PATH = Path("./Checkpoints/best_image_classification_model.pth")
    LAST_CKPT_PATH = Path("./Checkpoints/last_image_classification_model.pth")

    LOG_PATH = Path("./Logs/image_classification_log.csv")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] Using device: {DEVICE}")

    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5

    SEED = 42

    LOAD_CKPT = False
    print(f"[config] Using seed: {SEED}")

def main():
    random.seed(Config.SEED)
    numpy.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])

    dataset = GarbageClassificationImageClassificationDataset(
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        data_dir=Config.GARBAGECLASSIFICATION_DATASET_PATH,
        seed=Config.SEED,
    )
    train_loader = DataLoader(dataset.get_train_dataset(),
                              batch_size=Config.BATCH_SIZE,
                              shuffle=True,
                              generator=torch.Generator().manual_seed(Config.SEED))
    test_loader = DataLoader(dataset.get_test_dataset(),
                             batch_size=Config.BATCH_SIZE, shuffle=False,
                             generator=torch.Generator().manual_seed(Config.SEED))

    channels = dataset.get_channel_nums()
    num_classes = len(dataset.get_class_names())
    model = ResNet18(input_size=(channels, 224, 224), num_classes=num_classes).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=Config.LEARNING_RATE,
                                   weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = Checkpoint(Config.BEST_CKPT_PATH, Config.LAST_CKPT_PATH, Config.PATIENCE)
    if Config.LOAD_CKPT:
        ckpt.load(model)

    print(f"[main] device={Config.DEVICE}  epochs={Config.EPOCHS}  "
          f"batch={Config.BATCH_SIZE}  lr={Config.LEARNING_RATE}  "
          f"wd={Config.WEIGHT_DECAY}  patience={Config.PATIENCE}")

    logger = EpochLogger(Config.LOG_PATH, Config.EPOCHS)

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_once(model, train_loader, criterion, optimizer, Config.DEVICE)
        test_loss, test_acc = test_once(model, test_loader, criterion, Config.DEVICE)
        scheduler.step()

        is_improved, is_stopped = ckpt.update(model, test_loss, epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch, train_loss, train_acc, test_loss, test_acc, lr_now, is_improved)

        if is_stopped:
            print(f"[main] early stopping @ epoch {epoch} "
                  f"(no improvement for {ckpt.patience} epochs, "
                  f"best was epoch {ckpt.best_epoch})")
            break

    print(f"[main] done. best test loss = {ckpt.best_loss:.4f} @ epoch {ckpt.best_epoch}")
    print(f"[main] best ckpt -> {ckpt.best_path}\nlast ckpt -> {ckpt.last_path}")


if __name__ == "__main__":
    main()
