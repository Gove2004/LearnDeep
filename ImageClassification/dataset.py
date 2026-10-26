from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms


class BaseImageClassificationDataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        """returns the training dataset (with self.train_transforms)."""
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        """returns the test dataset (with self.test_transforms)."""
        pass

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """returns the class names."""
        pass

    @abstractmethod
    def get_channel_nums(self) -> int:
        """returns the number of image channels (1 for grayscale, 3 for RGB)."""
        pass


# implementations


class MNISTImageClassificationDataset(BaseImageClassificationDataset):
    def __init__(self, train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 data_dir: Path):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir

    def get_train_dataset(self) -> Dataset:
        return datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transforms
        )

    def get_test_dataset(self) -> Dataset:
        return datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transforms
        )

    def get_class_names(self) -> list[str]:
        return [str(i) for i in range(10)]

    def get_channel_nums(self) -> int:
        return 1


class CIFAR10ImageClassificationDataset(BaseImageClassificationDataset):
    def __init__(self, train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 data_dir: Path):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir

    def get_train_dataset(self) -> Dataset:
        return datasets.CIFAR10(
            root=self.data_dir / "trainval",
            train=True,
            download=True,
            transform=self.train_transforms
        )

    def get_test_dataset(self) -> Dataset:
        return datasets.CIFAR10(
            root=self.data_dir / "test",
            train=False,
            download=True,
            transform=self.test_transforms
        )

    def get_class_names(self) -> list[str]:
        test_data_dir = self.data_dir / "test"
        names = []
        for folder in test_data_dir.iterdir():
            if folder.is_dir():
                names.append(folder.name)
        return names

    def get_channel_nums(self) -> int:
        return 3


class CIFAR100ImageClassificationDataset(BaseImageClassificationDataset):
    def __init__(self, train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 data_dir: Path):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir

    def get_train_dataset(self) -> Dataset:
        return datasets.CIFAR100(
            root=self.data_dir / "trainval",
            train=True,
            download=True,
            transform=self.train_transforms
        )

    def get_test_dataset(self) -> Dataset:
        return datasets.CIFAR100(
            root=self.data_dir / "test",
            train=False,
            download=True,
            transform=self.test_transforms
        )

    def get_class_names(self) -> list[str]:
        test_data_dir = self.data_dir / "test"
        names = []
        for folder in test_data_dir.iterdir():
            if folder.is_dir():
                names.append(folder.name)
        return names

    def get_channel_nums(self) -> int:
        return 3


class OCTImageClassificationDataset(BaseImageClassificationDataset):
    def __init__(self, train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 data_dir: Path):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir

    def get_train_dataset(self) -> Dataset:
        train_data_dir = self.data_dir / "train"
        val_data_dir = self.data_dir / "val"
        train_dataset = datasets.ImageFolder(
            root=train_data_dir,
            transform=self.train_transforms
        )
        val_dataset = datasets.ImageFolder(
            root=val_data_dir,
            transform=self.test_transforms
        )
        return ConcatDataset([train_dataset, val_dataset])

    def get_test_dataset(self) -> Dataset:
        test_data_dir = self.data_dir / "test"
        return datasets.ImageFolder(
            root=test_data_dir,
            transform=self.test_transforms
        )

    def get_class_names(self) -> list[str]:
        test_data_dir = self.data_dir / "test"
        names = []
        for folder in test_data_dir.iterdir():
            if folder.is_dir():
                names.append(folder.name)
        return names

    def get_channel_nums(self) -> int:
        return 1


class GarbageClassificationImageClassificationDataset(BaseImageClassificationDataset):
    def __init__(self, train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 data_dir: Path, test_split_rate: float = 0.2, seed: int = 42):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir
        self.test_split_rate = test_split_rate
        self.seed = seed

    def get_train_dataset(self) -> Dataset:
        all_data_dir = self.data_dir
        full_dataset = datasets.ImageFolder(
            root=all_data_dir,
            transform=self.train_transforms
        )
        train_size = int((1 - self.test_split_rate) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        torch.manual_seed(self.seed)
        train_dataset, _ = random_split(full_dataset, [train_size, val_size])
        return train_dataset

    def get_test_dataset(self) -> Dataset:
        all_data_dir = self.data_dir
        full_dataset = datasets.ImageFolder(
            root=all_data_dir,
            transform=self.test_transforms
        )
        train_size = int((1 - self.test_split_rate) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        torch.manual_seed(self.seed)
        _, val_dataset = random_split(full_dataset, [train_size, val_size])
        return val_dataset

    def get_class_names(self) -> list[str]:
        all_data_dir = self.data_dir
        full_dataset = datasets.ImageFolder(
            root=all_data_dir,
            transform=self.test_transforms
        )
        return full_dataset.classes

    def get_channel_nums(self) -> int:
        return 3


if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = GarbageClassificationImageClassificationDataset(
        train_transforms, test_transforms, Path("./Dataset/GarbageClassification"))
    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()
    class_names = dataset.get_class_names()

    print(f"channels: {dataset.get_channel_nums()}")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")
    print(f"class names ({len(class_names)}): {class_names}")
