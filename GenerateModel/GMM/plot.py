from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from data import make_data
from gmm import fit, sample, score_samples


PLOT_DIR = Path(__file__).resolve().parents[2] / "Logs" / "plots"
GRID_SIZE = 200
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle,
                         facecolor="none", edgecolor=color, linewidth=2.2,
                         linestyle="--"))


def plot_fit(X, truth, model):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.35, c="#7f8c8d", label=f"samples (n={len(X)})")
    for k in range(len(truth["means"])):
        draw_ellipse(ax, truth["means"][k], truth["covs"][k], COLORS[k % len(COLORS)])
        ax.plot(*truth["means"][k], "X", color=COLORS[k % len(COLORS)],
                markersize=11, markeredgecolor="black")
    ax.set_title("true distribution (dashed=2σ, X=true mean)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.35, c="#7f8c8d")
    for k in range(len(model["means"])):
        draw_ellipse(ax, model["means"][k], model["covs"][k], COLORS[k % len(COLORS)])
        ax.plot(*model["means"][k], "o", color=COLORS[k % len(COLORS)],
                markersize=11, markeredgecolor="black")
    ax.set_title("EM estimate (dashed=2σ, dot=estimated mean)")

    ax = axes[2]
    ax.plot(model["history"], color="#e67e22", linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean log-likelihood")
    ax.set_title("EM convergence (must be non-decreasing)")
    ax.grid(alpha=0.3)

    for ax in axes[:2]:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / "gmm_fit.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_density(X, model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pad = 2.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, GRID_SIZE),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, GRID_SIZE))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = score_samples(model, grid).reshape(xx.shape)

    ax = axes[0]
    ax.contour(xx, yy, zz, levels=14, cmap="viridis", linewidths=1.3)
    ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.25, c="black")
    ax.set_title("fitted density (black=training data)")

    ax = axes[1]
    ax.contour(xx, yy, zz, levels=14, cmap="viridis", linewidths=1.3)
    X_new = sample(model, len(X), seed=123)
    ax.scatter(X_new[:, 0], X_new[:, 1], s=6, alpha=0.35, c="#e74c3c")
    ax.set_title("samples generated from the model (red)")

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / "gmm_density.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_learn(*args, **kwargs):
    """GMM Plots"""
    X, truth = make_data()
    model = fit(X, seed=42)
    print(f"[plot] saved {plot_fit(X, truth, model)}")
    print(f"[plot] saved {plot_density(X, model)}")
