import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from matplotlib import pyplot as plt

def run_learn(*args, **kwargs):
    """RNN and LSTM"""

    LR = 0.001
    TIME_STEP = 10
    TIME_COUNT = 200
    VIEW_SIZE = 100
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn6] Using device: {DEVICES}")

    class RNN(torch.nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = torch.nn.RNN(
                input_size=1,
                hidden_size=32,
                num_layers=1,
                batch_first=True
            )
            self.out = torch.nn.Linear(32, 1)

        def forward(self, x, hidden_state):
            r_out, h_n = self.rnn(x, hidden_state)
            out = self.out(r_out)
            return out, h_n

    net = RNN().to(DEVICES)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    hidden_state = None
    fig = plt.figure(figsize=(12, 4))
    for time in range(TIME_COUNT):
        if not plt.fignum_exists(fig.number):
            print("[learn3] Window closed.")
            break
        
        start, end = time * np.pi, (time + 1) * np.pi
        step = np.random.randint(1, 4) * TIME_STEP
        # step = 2 * TIME_STEP 
        steps = np.linspace(start, end, step, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(steps)
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).to(DEVICES)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).to(DEVICES)
        
        prediction, hidden_state = net(x, hidden_state)
        hidden_state = hidden_state.detach()
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.detach().cpu().numpy().flatten(), 'b-')
        plt.ylim(-1.2, 1.2)
        plt.xlim(max(0, end - VIEW_SIZE), end)
        plt.draw()
        plt.pause(0.05)
