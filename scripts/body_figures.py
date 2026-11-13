"""Clean body figures for Track 4 report. Seaborn; no figures/report reuse.

Regen:
  export MPLCONFIGDIR="$PWD/.cache/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
  .venv/bin/python scripts/body_figures.py

Outputs under figures/body/:
  01_cms_vs_size.png|pdf
  02_intermodel_size.png|pdf
  03_m1_vs_m3_14b.png|pdf
  04_cms_size_heatmap.png|pdf
  05_method_pair_rho.png|pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import group_share_figs as gsf  # noqa: E402

OUT = ROOT / "figures" / "body"
PAIRS = gsf.PAIRS4
E3 = gsf.E3
E3_B = gsf.E3_B
SHORT = gsf.SHORT
IDS = gsf.IDS

# Size × size axes: original family + Instruct-2507 checkpoint
INTERMODEL_ROWS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]


def _style() -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "savefig.dpi": 220,
            "figure.dpi": 120,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _heatmap_kwargs(**extra):
    base = dict(
        annot=False,
        cmap="vlag",
        center=0,
        vmin=-0.4,
        vmax=1.0,
        linewidths=1.0,
        linecolor="white",
        square=False,
        cbar_kws={"shrink": 0.82, "pad": 0.02},
    )
    base.update(extra)
    return base


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


def fig_cms_vs_size(asst: dict) -> None:
    """One CMS series: original Qwen3 family, assistant wrap only."""
    ys = [asst[m]["cms4"] for m in E3]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(E3_B, ys, "o-", color="#0072B2", lw=2, ms=8)
    for x, y, m in zip(E3_B, ys, E3):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("CMS")
    ax.set_title("Cross-method agreement vs size (original Qwen3, assistant)")
    ax.set_xticks(E3_B)
    ax.set_xticklabels([SHORT[m] for m in E3])
    ax.set_ylim(-0.05, 0.35)
    _save(fig, "01_cms_vs_size")


def _intermodel_matrix(asst: dict, models: list[str], method: str) -> pd.DataFrame:
    ranks = {m: gsf.midranks(asst[m]["scores"][method], IDS) for m in models}
    labels = [SHORT[m] for m in models]
    data = [[gsf.pearson(ranks[a], ranks[b]) for b in models] for a in models]
    return pd.DataFrame(data, index=labels, columns=labels)


def fig_intermodel_size(asst: dict) -> None:
    """Square heatmaps: model size × model size for M1 and M3 midrank ρ."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, method, title in zip(
        axes,
        ("M1", "M3"),
        ("Inter-model M1 (stated)", "Inter-model M3 (action)"),
    ):
        mat = _intermodel_matrix(asst, INTERMODEL_ROWS, method)
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-0.4,
            vmax=1.0,
            square=True,
            linewidths=0.4,
            linecolor="white",
            cbar=(method == "M3"),
            cbar_kws={"label": "Pearson ρ", "shrink": 0.75},
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Size × size midrank agreement (assistant wrap)", y=1.02, fontsize=11)
    _save(fig, "02_intermodel_size")


def _mean_method_intermodel(asst: dict, models: list[str]) -> pd.DataFrame:
    """Mean midrank Pearson across M1–M4 between each model pair (CMS-style)."""
    mats = [_intermodel_matrix(asst, models, m) for m in ("M1", "M2", "M3", "M4")]
    stacked = np.mean([df.to_numpy() for df in mats], axis=0)
    labels = [SHORT[m] for m in models]
    return pd.DataFrame(stacked, index=labels, columns=labels)


def fig_cms_size_heatmap(asst: dict) -> None:
    """Size × size: mean inter-model midrank ρ over M1–M4 (no cell numbers)."""
    mat = _mean_method_intermodel(asst, INTERMODEL_ROWS)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(mat, ax=ax, **_heatmap_kwargs(square=True, cbar_kws={"label": "Mean Pearson ρ", "shrink": 0.82, "pad": 0.02}))
    ax.set_title("Size × size agreement (mean midrank ρ over M1–M4)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, length=0)
    ax.tick_params(axis="y", rotation=0, length=0)
    sns.despine(left=True, bottom=True)
    _save(fig, "04_cms_size_heatmap")


def fig_method_pair_rho(asst: dict) -> None:
    """Model × method-pair midrank ρ — CMS ingredients (no cell numbers)."""
    rows = [[asst[m]["matrix4"][p] for p in PAIRS] for m in INTERMODEL_ROWS]
    mat = pd.DataFrame(rows, index=[SHORT[m] for m in INTERMODEL_ROWS], columns=PAIRS)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sns.heatmap(
        mat,
        ax=ax,
        **_heatmap_kwargs(cbar_kws={"label": "Pearson ρ", "shrink": 0.85, "pad": 0.02}),
    )
    ax.set_title("Method-pair midrank ρ (assistant wrap)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, length=0)
    ax.tick_params(axis="y", rotation=0, length=0)
    sns.despine(left=True, bottom=True)
    _save(fig, "05_method_pair_rho")


def fig_m1_m3_scatter_14b(asst: dict) -> None:
    m = "Qwen/Qwen3-14B"
    s1 = asst[m]["scores"]["M1"]
    s3 = asst[m]["scores"]["M3"]
    xs = [float(s1[i]) for i in IDS]
    ys = [float(s3[i]) for i in IDS]
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(xs, ys, s=36, color="#0072B2", alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1.2, zorder=1)
    if "stay_general" in IDS:
        i = IDS.index("stay_general")
        ax.scatter([xs[i]], [ys[i]], s=90, facecolors="none", edgecolors="#D55E00", linewidths=1.8, zorder=4)
        ax.annotate(
            "stay_general",
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#D55E00",
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("M1 score (stated pairwise)")
    ax.set_ylabel("M3 score (action)")
    ax.set_title("14B stated vs action (one point = one outcome)")
    rho = asst[m]["matrix4"]["M1|M3"]
    ax.text(0.05, 0.95, f"M1|M3 ρ = {rho:.3f}", transform=ax.transAxes, va="top", fontsize=9)
    _save(fig, "03_m1_vs_m3_14b")


def main() -> None:
    _style()
    asst = gsf.assistant_merged()
    assert abs(asst["Qwen/Qwen3-14B"]["cms4"] - 0.256) < 0.015
    assert abs(asst["Qwen/Qwen3-4B-Instruct-2507"]["cms4"] - 0.528) < 0.015

    fig_cms_vs_size(asst)
    fig_intermodel_size(asst)
    fig_m1_m3_scatter_14b(asst)
    fig_cms_size_heatmap(asst)
    fig_method_pair_rho(asst)
    print("done", OUT)


if __name__ == "__main__":
    main()
