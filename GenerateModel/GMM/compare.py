import numpy as np
from sklearn.mixture import GaussianMixture

from data import make_data
from gmm import N_COMPONENTS, report


SEED = 42


def fit_sklearn(X, n_components=N_COMPONENTS, seed=SEED):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        max_iter=100,
        tol=1e-6,
        random_state=seed,
    )
    model.fit(X)
    return {
        "weights": model.weights_,
        "means": model.means_,
        "covs": model.covariances_,
        "history": [],
        "model": model,
        "log_likelihood": float(model.score(X)),
    }


def match_components(means_a, means_b):
    used = set()
    pairs = []
    for i, m in enumerate(means_a):
        best_j, best_d = None, np.inf
        for j, n in enumerate(means_b):
            if j in used:
                continue
            d = float(np.linalg.norm(m - n))
            if d < best_d:
                best_j, best_d = j, d
        used.add(best_j)
        pairs.append((i, best_j, best_d))
    return pairs


def run_learn(*args, **kwargs):
    """Gaussian Mixture Model (sklearn)"""
    from gmm import fit as fit_handwritten

    X, truth = make_data()

    sk = fit_sklearn(X)
    print(f"[compare] sklearn loglik={sk['log_likelihood']:.4f}")
    report(sk, truth, "sklearn estimated")

    hw = fit_handwritten(X, seed=SEED)
    print(f"[compare] handwritten loglik={hw['history'][-1]:.4f}")

    print("[compare] component matching (order is random, must pair first):")
    max_dist = 0.0
    for i, j, d in match_components(hw["means"], sk["means"]):
        max_dist = max(max_dist, d)
        print(f"    handwritten=({hw['means'][i][0]:7.3f}, {hw['means'][i][1]:7.3f})  "
              f"sklearn=({sk['means'][j][0]:7.3f}, {sk['means'][j][1]:7.3f})  dist={d:.4f}")
    print(f"[compare] max deviation={max_dist:.4f}")
