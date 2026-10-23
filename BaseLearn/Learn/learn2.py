import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def run_learn(*args, **kwargs):
    """Activation Functions"""

    x = torch.linspace(-5, 5, 100)
    y_relu = F.relu(x)
    y_sigmoid = torch.sigmoid(x)
    y_tanh = torch.tanh(x)
    y_softplus = F.softplus(x)

    plt.figure(figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', color='blue')
    plt.title('ReLU')
    plt.legend()
    plt.grid()
    plt.subplot(222)
    plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid', color='orange')
    plt.title('Sigmoid')
    plt.legend()
    plt.grid()
    plt.subplot(223)
    plt.plot(x.numpy(), y_tanh.numpy(), label='Tanh', color='green')
    plt.title('Tanh')
    plt.legend()
    plt.grid()
    plt.subplot(224)
    plt.plot(x.numpy(), y_softplus.numpy(), label='Softplus', color='red')
    plt.title('Softplus')
    plt.legend()
    plt.grid()
    plt.show()
