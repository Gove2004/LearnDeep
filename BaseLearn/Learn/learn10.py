import torch
import numpy as np
from matplotlib import pyplot as plt

def run_learn(*args, **kwargs):
    """GAN"""

    EPOCH = 10000
    BATCH_SIZE = 64
    LR_G = 0.0001
    LR_D = 0.0001
    N_IDEAS = 5
    ART_COMPLEXITY = 100
    PAINT_POINTS = np.linspace(-1, 1, 100)
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn10] Using device: {DEVICES}")

    def artist_works():
        a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
        paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
        return torch.from_numpy(paintings).float().to(DEVICES)

    G = torch.nn.Sequential(
        torch.nn.Linear(N_IDEAS, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, ART_COMPLEXITY),
    ).to(DEVICES)

    D = torch.nn.Sequential(
        torch.nn.Linear(ART_COMPLEXITY, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid(),
    ).to(DEVICES)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    loss_func = torch.nn.BCELoss()

    fig, ax = plt.subplots(figsize=(8, 5))

    for step in range(EPOCH):
        if not plt.fignum_exists(fig.number):
            print("[learn10] Window closed.")
            break

        artist_paintings = artist_works()

        G_ideas = torch.rand(BATCH_SIZE, N_IDEAS, device=DEVICES)
        G_paintings = G(G_ideas)

        prob_artist1 = D(G_paintings)
        G_loss = loss_func(prob_artist1, torch.ones_like(prob_artist1))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        prob_artist0 = D(artist_paintings)
        prob_artist1 = D(G_paintings.detach())
        D_loss = loss_func(prob_artist0, torch.ones_like(prob_artist0)) + \
                 loss_func(prob_artist1, torch.zeros_like(prob_artist1))
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        if step % 100 == 0:
            ax.cla()
            ax.plot(PAINT_POINTS.ravel(), G_paintings[0].cpu().detach().numpy(), c='#4AD631', lw=3, label='Generated')
            ax.plot(PAINT_POINTS.ravel(), 2 * np.power(PAINT_POINTS.ravel(), 2) + 1, c='#74BCFF', lw=3, label='Upper bound')
            ax.plot(PAINT_POINTS.ravel(), 1 * np.power(PAINT_POINTS.ravel(), 2) + 0, c='#FF9359', lw=3, label='Lower bound')
            ax.text(-0.5, 2.3, f'D accuracy={prob_artist0.mean().item():.2f}', fontsize=13)
            ax.text(-0.5, 2.0, f'D score={-D_loss.item():.2f}', fontsize=13)
            ax.set_ylim(0, 3)
            ax.legend(loc='upper right', fontsize=10)
            plt.pause(0.05)

    plt.show()
