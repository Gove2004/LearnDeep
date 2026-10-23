import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def run_learn(*args, **kwargs):
    """Dropout, Normalization, More Optimizer and Batch DataLoader"""

    EPOCHS = 5
    BATCH_SIZE = 250
    LR = 0.01
    MOMENTUM = 0.9
    N_HIDDEN = 4
    HIDDEN_SIZE = 16
    TEST_SPLIT_RATE = 0.2
    DROPOUT_P = 0.2
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn4] Using device: {DEVICES}")

    def make_data(n=5000, seed=42):
        rng = np.random.default_rng(seed)
        x_all = np.sort(rng.uniform(0, 1, size=(n, 1)), axis=0)
        y_clean = np.tanh(10 * (x_all - 0.5)) + np.sin(x_all) + x_all**3 - x_all
        y_all = y_clean + rng.normal(0, 1, x_all.shape) * 0.2
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=TEST_SPLIT_RATE, random_state=seed)
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32).to(DEVICES)
        return to_tensor(x_train), to_tensor(y_train), to_tensor(x_test), to_tensor(y_test), to_tensor(x_all), to_tensor(y_clean)

    x_train, y_train, x_test, y_test, x_all, y_clean = make_data()
    torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    class BigNet(torch.nn.Module):
        def __init__(self):
            super(BigNet, self).__init__()
            self.fcs = []
            self.bns = []
            self.drops = []
            for i in range(N_HIDDEN):
                fc = torch.nn.Linear(1 if i == 0 else HIDDEN_SIZE, HIDDEN_SIZE)
                bn = torch.nn.BatchNorm1d(HIDDEN_SIZE)
                drop = torch.nn.Dropout(p=DROPOUT_P)
                self.add_module(f'fc{i}', fc)
                self.add_module(f'bn{i}', bn)
                self.add_module(f'drop{i}', drop)
                self.fcs.append(fc)
                self.bns.append(bn)
                self.drops.append(drop)
            self.out = torch.nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, x):
            for i, (fc, bn, drop) in enumerate(zip(self.fcs, self.bns, self.drops)):
                x = fc(x)
                x = bn(x)
                x = torch.relu(x)
                x = drop(x)
            x = self.out(x)
            return x

    name_list = ['Adam', 'SGD', 'Momentum', 'RMSprop']
    net_list = [BigNet().to(DEVICES), BigNet().to(DEVICES), BigNet().to(DEVICES), BigNet().to(DEVICES)]
    optimizer_list = [
        torch.optim.Adam(net_list[0].parameters(), lr=LR),
        torch.optim.SGD(net_list[1].parameters(), lr=LR),
        torch.optim.SGD(net_list[2].parameters(), lr=LR, momentum=MOMENTUM),
        torch.optim.RMSprop(net_list[3].parameters(), lr=LR)
    ]
    loss_func = torch.nn.MSELoss()
    loss_list = {name: [] for name in name_list}

    for epoch in range(EPOCHS):
        for step, (batch_x, batch_y) in enumerate(loader):
            for name, net, optimizer in zip(name_list, net_list, optimizer_list):
                prediction = net(batch_x)
                loss = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list[name].append(loss.item())
                if name == 'Adam' and step % 10 == 0:
                    print(f"[learn5] Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for n in name_list:
        axes[0].plot(loss_list[n], label=n)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid()

    xs = x_all.cpu().numpy().ravel()
    axes[1].plot(xs, y_clean.cpu().numpy().ravel(), 'k--', label='Ground Truth')
    axes[1].scatter(
        x_test.cpu().numpy().ravel(), 
        y_test.cpu().numpy().ravel(),
        alpha=0.1, c='gray', label='Test data')
    for name, net in zip(name_list, net_list):
        net.eval()
        with torch.no_grad():
            axes[1].plot(xs, net(x_all).cpu().numpy().ravel(), label=name)
    axes[1].set_title('Fitted Curves')
    axes[1].legend()
    axes[1].grid()
    plt.show()

