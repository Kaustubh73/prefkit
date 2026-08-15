"""Thurstone utilities from M1 pair logs. Not M1.score() — A2 needs win-rate."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from prefkit.outcomes import id_order


def _phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def pair_p_after_flip(logs: list[dict], ids: list[str]) -> list[tuple[int, int, float]]:
    """P(x ≻ y) on canonical id_order index pairs after flipping ji prompts."""
    wins: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in logs:
        parsed = row.get("parsed")
        if parsed not in ("A", "B"):
            continue
        i, j = row["ids"]
        if row["order"] == "ij":
            winner = i if parsed == "A" else j
        else:
            winner = j if parsed == "A" else i
        a, b = (i, j) if ids.index(i) < ids.index(j) else (j, i)
        wins[(a, b)].append(1 if winner == a else 0)
    out = []
    for (a, b), ys in wins.items():
        out.append((ids.index(a), ids.index(b), float(sum(ys) / len(ys))))
    return out


def fit_thurstonian(
    comparisons: list[tuple[int, int, float]],
    n: int,
    steps: int = 400,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # ponytail: finite-diff Adam over full pair graph; N=14 ceiling. Upgrade: UE trainer if N grows.
    theta = np.zeros(2 * n)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def pack_p(th: np.ndarray) -> np.ndarray:
        mu = th[:n]
        sig = np.exp(th[n:])
        ps = []
        for i, j, _y in comparisons:
            denom = math.sqrt(sig[i] ** 2 + sig[j] ** 2) + 1e-12
            ps.append(float(_phi(np.array([(mu[i] - mu[j]) / denom]))[0]))
        return np.clip(np.array(ps), 1e-6, 1 - 1e-6)

    def loss(th: np.ndarray) -> float:
        p = pack_p(th)
        y = np.array([c[2] for c in comparisons])
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    last = 0.0
    for t in range(1, steps + 1):
        last = loss(theta)
        g = np.zeros_like(theta)
        h = 1e-4
        for k in range(theta.size):
            d = np.zeros_like(theta)
            d[k] = h
            g[k] = (loss(theta + d) - last) / h
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g ** 2)
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
    p = pack_p(theta)
    y = np.array([c[2] for c in comparisons])
    pred = (p > 0.5).astype(int)
    truth = (y > 0.5).astype(int)
    acc = float(np.mean(pred == truth)) if len(y) else 0.0
    mu = theta[:n]
    var = np.exp(theta[n:]) ** 2
    stats = {"log_loss": last, "accuracy": acc}
    return mu, var, stats


def fit_from_m1_logs(logs: list[dict], ids: list[str] | list[dict]) -> dict:
    if ids and isinstance(ids[0], dict):
        ids = id_order(ids)  # type: ignore[arg-type]
    ids = list(ids)  # type: ignore[arg-type]
    comps = pair_p_after_flip(logs, ids)
    if not comps:
        return {"utilities": {i: {"mean": 0.0, "variance": 1.0} for i in ids}, "log_loss": None, "accuracy": None}
    mu, var, stats = fit_thurstonian(comps, len(ids))
    return {
        "utilities": {ids[i]: {"mean": float(mu[i]), "variance": float(var[i])} for i in range(len(ids))},
        "log_loss": stats["log_loss"],
        "accuracy": stats["accuracy"],
    }
