import torch
import matplotlib.pyplot as plt

def run_learn(*args, **kwargs):
    """Simple Regression and Neural Network"""

    EPOCHS = 250
    LR = 0.1
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn3] Using device: {DEVICES}")

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1).to(DEVICES)
    y = x.pow(2) + 0.2 * torch.rand(x.size()).to(DEVICES)

    class SimpleNet1(torch.nn.Module):
        def __init__(self):
            super(SimpleNet1, self).__init__()
            self.hidden = torch.nn.Linear(1, 10)
            self.predict = torch.nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = self.predict(x)
            return x

    class SimpleNet2(torch.nn.Module):
        def __init__(self):
            super(SimpleNet2, self).__init__()
            self.hidden1 = torch.nn.Linear(1, 10)
            self.predict = torch.nn.Linear(10, 1)

        def forward(self, x):
            x = torch.tanh(self.hidden1(x))
            x = self.predict(x)
            return x

    net1 = SimpleNet1().to(DEVICES)
    net2 = SimpleNet2().to(DEVICES)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=LR)
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for epoch in range(EPOCHS):
        if not plt.fignum_exists(fig.number):
            print("[learn3] Window closed.")
            break

        prediction1 = net1(x)
        loss1 = loss_func(prediction1, y)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        ax1.cla()
        ax1.scatter(x.cpu().numpy(), y.cpu().numpy(), label='Data', color='blue')
        ax1.plot(x.cpu().numpy(), prediction1.detach().cpu().numpy(), label='Prediction 1', color='red')
        ax1.text(0.02, 0.95, f'Epoch={epoch+1}', fontsize=10, color='red', transform=ax1.transAxes)
        ax1.text(0.02, 0.88, f'Loss={loss1.item():.4f}', fontsize=10, color='red', transform=ax1.transAxes)
        ax1.set_title('ReLU')
        ax1.legend()

        prediction2 = net2(x)
        loss2 = loss_func(prediction2, y)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        ax2.cla()
        ax2.scatter(x.cpu().numpy(), y.cpu().numpy(), label='Data', color='blue')
        ax2.plot(x.cpu().numpy(), prediction2.detach().cpu().numpy(), label='Prediction 2', color='green')
        ax2.text(0.02, 0.95, f'Epoch={epoch+1}', fontsize=10, color='green', transform=ax2.transAxes)
        ax2.text(0.02, 0.88, f'Loss={loss2.item():.4f}', fontsize=10, color='green', transform=ax2.transAxes)
        ax2.set_title('Tanh')
        ax2.legend()

        plt.pause(0.1)
