from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from dataset import *
from model import *


class Config:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    IMAGE_PATH = Path(r".\Datasets\GarbageClassification\cardboard\cardboard1.jpg")
    CKPT_PATH = Path(r".\Checkpoints\best_image_classification_model.pth")
    MODEL = ResNet18

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] Using device: {DEVICE}")

def main():
    dataset = GarbageClassificationImageClassificationDataset(
        train_transforms=None,
        test_transforms=None,
        data_dir=Path(r".\Datasets\GarbageClassification"),
    )
    channel_nums = dataset.get_channel_nums()
    class_names = dataset.get_class_names()

    mode = "RGB" if channel_nums == 3 else "L"
    image = Image.open(Config.IMAGE_PATH).convert(mode)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])
    image = transform(image).unsqueeze(0).to(Config.DEVICE)

    model = Config.MODEL(input_size=(channel_nums, 224, 224), num_classes=len(class_names))
    model.load_state_dict(torch.load(Config.CKPT_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1)

    print(f"[infer] Predicted class: {class_names[predicted.item()]}")


if __name__ == "__main__":
    main()
