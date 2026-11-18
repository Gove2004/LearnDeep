import numpy as np

from data import make_data


N_COMPONENTS = 3
MAX_ITER = 100
TOL = 1e-6
N_INIT = 10
REG_COVAR = 1e-6
TOP_K = 3
EPS = 1e-300


def log_gaussian(X, mean, cov):
    D = X.shape[1]
    diff = X - mean
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, diff.T)
    mahalanobis = np.sum(z ** 2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + log_det + mahalanobis)


def log_sum_exp(a):
    a_max = np.max(a, axis=1, keepdims=True)
    return (a_max + np.log(np.sum(np.exp(a - a_max), axis=1, keepdims=True))).ravel()


def e_step(X, weights, means, covs):
    N, K = X.shape[0], len(weights)
    weighted = np.zeros((N, K))
    for k in range(K):
        weighted[:, k] = np.log(weights[k] + EPS) + log_gaussian(X, means[k], covs[k])
    log_norm = log_sum_exp(weighted)
    gamma = np.exp(weighted - log_norm[:, None])
    return gamma, float(np.mean(log_norm))


def m_step(X, gamma):
    N, D = X.shape
    K = gamma.shape[1]

    Nk = gamma.sum(axis=0) + 1e-10
    weights = Nk / N
    means = (gamma.T @ X) / Nk[:, None]

    covs = np.zeros((K, D, D))
    for k in range(K):
        diff = X - means[k]
        covs[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
        covs[k] += np.eye(D) * REG_COVAR

    return weights, means, covs


def init_params(X, K, rng):
    N, D = X.shape
    idx = rng.choice(N, K, replace=False)
    means = X[idx].copy()
    weights = np.full(K, 1.0 / K)
    covs = np.array([np.cov(X.T) + np.eye(D) * REG_COVAR] * K)
    return weights, means, covs


def fit(X, n_components=N_COMPONENTS, max_iter=MAX_ITER, tol=TOL,
        n_init=N_INIT, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    best = None

    for run in range(n_init):
        weights, means, covs = init_params(X, n_components, rng)
        history = []
        prev_ll = -np.inf
        it = 0

        for it in range(max_iter):
            gamma, ll = e_step(X, weights, means, covs)
            history.append(ll)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            weights, means, covs = m_step(X, gamma)

        if verbose:
            print(f"[gmm] init {run + 1:>2}: iters={it + 1:>3}  loglik={history[-1]:.4f}")

        if best is None or history[-1] > best["history"][-1]:
            best = {"weights": weights, "means": means, "covs": covs, "history": history}

    return best


def score_samples(model, X):
    N, K = X.shape[0], len(model["weights"])
    weighted = np.zeros((N, K))
    for k in range(K):
        weighted[:, k] = (np.log(model["weights"][k] + EPS)
                          + log_gaussian(X, model["means"][k], model["covs"][k]))
    return log_sum_exp(weighted)


def sample(model, n, seed=0):
    rng = np.random.default_rng(seed)
    K = len(model["weights"])

    which = rng.choice(K, size=n, p=model["weights"])
    X = np.zeros((n, 2))
    for k in range(K):
        mask = which == k
        cnt = int(mask.sum())
        if cnt:
            X[mask] = rng.multivariate_normal(model["means"][k], model["covs"][k], size=cnt)
    return X


def report(model, truth, title):
    print(f"[gmm] {title}")
    for k in range(len(model["weights"])):
        m, c = model["means"][k], model["covs"][k]
        print(f"    {k}  w={model['weights'][k]:.4f}  mean=({m[0]:7.3f}, {m[1]:7.3f})  "
              f"cov=[{c[0, 0]:6.3f} {c[0, 1]:6.3f}; {c[1, 0]:6.3f} {c[1, 1]:6.3f}]")
    if truth is None:
        return
    print(f"[gmm] {title} (true values)")
    for k in range(len(truth["weights"])):
        m, c = truth["means"][k], truth["covs"][k]
        print(f"    {k}  w={truth['weights'][k]:.4f}  mean=({m[0]:7.3f}, {m[1]:7.3f})  "
              f"cov=[{c[0, 0]:6.3f} {c[0, 1]:6.3f}; {c[1, 0]:6.3f} {c[1, 1]:6.3f}]")


def run_learn(*args, **kwargs):
    """Gaussian Mixture Model (hand-written EM)"""
    X, truth = make_data()
    print(f"[gmm] samples={len(X)}  dim={X.shape[1]}  true components={len(truth['weights'])}")

    model = fit(X, verbose=True)

    history = np.array(model["history"])
    diffs = np.diff(history)
    print(f"[gmm] converged in {len(history)} iters")
    print(f"[gmm] loglik {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"[gmm] monotonic non-decreasing={bool((diffs >= -1e-9).all())}  "
          f"min delta={diffs.min():.2e}")

    report(model, truth, "estimated")

    print("[gmm] sampling 5 points from the fitted model:")
    for row in sample(model, 5, seed=7):
        print(f"    ({row[0]:7.3f}, {row[1]:7.3f})")
