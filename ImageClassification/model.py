from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torchsummary import summary


class BaseImageClassificationModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.input_size = None
        self.output_size = None

    @abstractmethod
    def get_input_size(self) -> tuple:
        ...

    @abstractmethod
    def get_output_size(self) -> tuple:
        ...


class LeNet5(BaseImageClassificationModel):
    def __init__(self, input_size=(1, 28, 28), num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = (num_classes,)
        in_ch = input_size[0]
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def get_input_size(self) -> tuple:
        return self.input_size

    def get_output_size(self) -> tuple:
        return self.output_size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet8(BaseImageClassificationModel):
    def __init__(self, input_size=(3, 227, 227), num_classes=1000):
        super().__init__()
        self.input_size = input_size
        self.output_size = (num_classes,)
        in_ch = input_size[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def get_input_size(self) -> tuple:
        return self.input_size

    def get_output_size(self) -> tuple:
        return self.output_size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGGNet16(BaseImageClassificationModel):
    def __init__(self, input_size=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.input_size = input_size
        self.output_size = (num_classes,)
        in_ch = input_size[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def get_input_size(self) -> tuple:
        return self.input_size

    def get_output_size(self) -> tuple:
        return self.output_size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[5], kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class GoogLeNet(BaseImageClassificationModel):
    def __init__(self, input_size=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.input_size = input_size
        self.output_size = (num_classes,)
        in_ch = input_size[0]
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = _InceptionBlock(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = _InceptionBlock(256, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = _InceptionBlock(480, [192, 96, 208, 16, 48, 64])
        self.inception4b = _InceptionBlock(512, [160, 112, 224, 24, 64, 64])
        self.inception4c = _InceptionBlock(512, [128, 128, 256, 24, 64, 64])
        self.inception4d = _InceptionBlock(512, [112, 144, 288, 32, 64, 64])
        self.inception4e = _InceptionBlock(528, [256, 160, 320, 32, 128, 128])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = _InceptionBlock(832, [256, 160, 320, 32, 128, 128])
        self.inception5b = _InceptionBlock(832, [384, 192, 384, 48, 128, 128])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def get_input_size(self) -> tuple:
        return self.input_size

    def get_output_size(self) -> tuple:
        return self.output_size

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = torch.relu(out)

        return out


class ResNet18(BaseImageClassificationModel):
    def __init__(self, input_size=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.input_size = input_size
        self.output_size = (num_classes,)
        in_ch = input_size[0]
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(_ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(_ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def get_input_size(self) -> tuple:
        return self.input_size

    def get_output_size(self) -> tuple:
        return self.output_size

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ResNet18(input_size=(3, 224, 224), num_classes=6).to(device)
    print(f"input_size  = {model.get_input_size()}")
    print(f"output_size = {model.get_output_size()}")
    summary(model, model.get_input_size(), device=device)