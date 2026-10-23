
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

def run_learn(*args, **kwargs):
    """Encoder and Decoder"""

    MNIST_DATASET_PATH = kwargs.get("mnist_dataset_path")
    EPOCH = 5
    BATCH_SIZE = 64
    LR = 0.001
    VIEW_SIZE = 10
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn8] Using device: {DEVICES}")

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

    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 12),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 28 * 28),
                nn.Sigmoid(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    autoencoder = AutoEncoder().to(DEVICES)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    fig, axis = plt.subplots(2, VIEW_SIZE, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    view_data = train_data.data[:VIEW_SIZE].view(-1, 28 * 28).type(torch.FloatTensor) / 255.0
    for i in range(VIEW_SIZE):
        axis[0][i].imshow(view_data[i].numpy().reshape(28, 28), cmap='gray')
        axis[0][i].set_xticks(())
        axis[0][i].set_yticks(())
    
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28*28).to(DEVICES)
            b_y = x.view(-1, 28*28).to(DEVICES)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                plt.suptitle(f'Epoch: {epoch+1}/{EPOCH}, Step: {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                with torch.no_grad():
                    _, decode_view_data = autoencoder(view_data.to(DEVICES))
                for i in range(VIEW_SIZE):
                    axis[1][i].clear()
                    axis[1][i].imshow(decode_view_data[i].cpu().numpy().reshape(28, 28), cmap='gray')
                    axis[1][i].set_xticks(())
                    axis[1][i].set_yticks(())
                plt.pause(0.05)

            if not plt.fignum_exists(fig.number):
                print("[learn3] Window closed.")
                break

        if not plt.fignum_exists(fig.number):
            print("[learn3] Window closed.")
            break