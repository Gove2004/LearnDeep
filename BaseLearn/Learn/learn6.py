import torch
import torchvision
import torch.utils.data as Data
from matplotlib import pyplot as plt

def run_learn(*args, **kwargs):
    """CNN and MNIST"""

    MNIST_DATASET_PATH = kwargs.get("mnist_dataset_path")
    EPOCH = 1
    BATCH_SIZE = 50
    LR = 0.0001
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn4] Using device: {DEVICES}")

    train_data = torchvision.datasets.MNIST(
        root=MNIST_DATASET_PATH,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_data = torchvision.datasets.MNIST(
        root=MNIST_DATASET_PATH,
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title(f'Label: {train_data.targets[0]}')
    plt.show()

    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2)
            )
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2)
            )
            self.out = torch.nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output

    net = ConvNet().to(DEVICES)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(DEVICES), batch_y.to(DEVICES)
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                accuracy = (output.argmax(dim=1) == batch_y).float().mean().item()
                print(f"[learn6] Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICES), batch_y.to(DEVICES)
            output = net(batch_x)
            predictions = output.argmax(dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

        test_accuracy = total_correct / total_samples
        print(f"[learn6] Test Accuracy: {test_accuracy:.4f}")