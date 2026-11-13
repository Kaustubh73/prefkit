"""Matplotlib figures from a frozen library record. Ranks plotted as-is."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def midrank_fig(rec: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    ranks = rec.get("ranks") or {}
    n = rec.get("n") or 0
    y = np.arange(n)
    for method, mid in ranks.items():
        ax.scatter(mid, y[: len(mid)], label=method, s=18)
    ax.set_xlabel("mid-rank")
    ax.set_ylabel("outcome index")
    if ranks:
        ax.legend()
    fig.tight_layout()
    return fig


def rho_heatmap(rec: dict):
    methods = rec.get("methods") or []
    matrix = rec.get("matrix") or {}
    n = len(methods)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                mat[i, j] = 1.0
            else:
                rho = matrix.get(f"{a}|{b}", matrix.get(f"{b}|{a}"))
                if rho is not None:
                    mat[i, j] = rho
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n), methods)
    ax.set_yticks(range(n), methods)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
