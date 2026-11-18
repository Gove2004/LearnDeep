import numpy as np


N_SAMPLES = 2000
SEED = 42

# (权重, 均值, 协方差)，权重会自动归一化
TRUE_COMPONENTS = [
    (0.35, (-2.0, -1.0), ((0.7, 0.3), (0.3, 0.7))),
    (0.40, (2.0, 1.5), ((0.6, -0.4), (-0.4, 0.6))),
    (0.25, (0.0, 3.0), ((0.4, 0.0), (0.0, 0.9))),
]


def make_data(n_samples=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)

    weights = np.array([w for w, _, _ in TRUE_COMPONENTS])
    weights = weights / weights.sum()
    means = np.array([m for _, m, _ in TRUE_COMPONENTS])
    covs = np.array([c for _, _, c in TRUE_COMPONENTS])

    which = rng.choice(len(weights), size=n_samples, p=weights)

    X = np.zeros((n_samples, 2))
    for k in range(len(weights)):
        mask = which == k
        cnt = int(mask.sum())
        if cnt:
            X[mask] = rng.multivariate_normal(means[k], covs[k], size=cnt)

    truth = {"weights": weights, "means": means, "covs": covs}
    return X, truth
