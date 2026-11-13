"""Publication + pick-and-choose gallery for Track 4.

Regen:
  .venv/bin/python scripts/report_figures.py

Outputs: figures/report/{png,pdf} + figures/report/tables/*.csv
Gemma: auto-load first results/**/*gemma*.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import group_share_figs as gsf  # noqa: E402

OUT = ROOT / "figures" / "report"
TAB = OUT / "tables"
LIB = gsf.LIB

C = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#222222",
    "muted": "#7A7A7A",
    "grid": "#E8E8E8",
    "gemma": "#E69F00",
}
RHO_CMAP = LinearSegmentedColormap.from_list("rho_cb", ["#0072B2", "#F7F7F7", "#D55E00"])
SCORE_CMAP = LinearSegmentedColormap.from_list("score_cb", ["#F7F7F7", "#56B4E9", "#0072B2"])

PERSONA_GLOBS = ("persona",)
GEMMA = "google/gemma-3-4b-it"
GEMMA_SHORT = "Gemma-4B"


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": C["ink"],
            "axes.labelcolor": C["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": C["ink"],
            "ytick.color": C["ink"],
            "text.color": C["ink"],
            "legend.frameon": False,
            "legend.fontsize": 8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
        }
    )


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


def write_csv(name: str, headers: list[str], rows: list[list]):
    TAB.mkdir(parents=True, exist_ok=True)
    p = TAB / name
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print("wrote", p)


def safe_pearson(a, b) -> float:
    try:
        return float(gsf.pearson(a, b))
    except Exception:
        return float("nan")


def safe_rank_rho(sa: dict, sb: dict) -> float:
    try:
        return float(gsf.rank_rho(sa, sb))
    except Exception:
        return float("nan")


def _imshow_annot(ax, mat, xticks, yticks, vmin, vmax, title, fs=7, cmap=None):
    im = ax.imshow(np.asarray(mat, dtype=float), vmin=vmin, vmax=vmax, cmap=cmap or RHO_CMAP, aspect="auto")
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=fs, color=C["muted"])
            else:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color="white" if abs(v) > 0.5 else C["ink"],
                )
    ax.set_title(title)
    return im


def pack_full(blob: dict) -> dict:
    scores = blob["scores"]
    out4 = gsf.cms_k(scores, ["M1", "M2", "M3", "M4"])
    out3 = gsf.cms_k(scores, ["M1", "M2", "M3"])
    d = blob.get("diagnostics") or {}
    return {
        "model": blob.get("model"),
        "scores": scores,
        "cms3": out3["cms"],
        "cms4": out4["cms"],
        "matrix4": {f"{a}|{b}": v for (a, b), v in out4["matrix"].items()},
        "matrix3": {f"{a}|{b}": v for (a, b), v in out3["matrix"].items()},
        "cards4": gsf.disagreement_cards(out4["ranks"], gsf.IDS)[:3],
        "diagnostics": d,
        "utilities_thurstone": blob.get("utilities_thurstone"),
        "logs": blob.get("logs") or {},
        "m4_diag": d.get("M4"),
        "system": blob.get("system") or "",
        "seed": (blob.get("decode") or {}).get("seed"),
        "slot": blob.get("slot"),
    }


def find_persona_dirs() -> list[Path]:
    found: list[Path] = []
    for pat in PERSONA_GLOBS:
        for d in sorted(LIB.glob(pat)):
            if d.is_dir() and d.name != "discarded_aborted_s_m1m2_seed1800":
                found.append(d)
    seen, out = set(), []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _persona_path_for_stem(d: Path, stem: str) -> Path | None:
    for suf in ("_default_outcomes_persona.json", "_outcomes_persona.json", "_persona.json"):
        p = d / f"{stem}{suf}"
        if p.exists():
            return p
    longer = [o for o in gsf.FILE.values() if o != stem and o.startswith(stem)]
    hits = [
        h
        for h in sorted(d.glob(f"{stem}*.json"))
        if "persona" in h.name.lower() and not any(h.name.startswith(o) for o in longer)
    ]
    return hits[0] if hits else None


def load_persona_optional() -> dict[str, dict] | None:
    dirs = find_persona_dirs()
    search_roots = list(dirs) + ([ROOT / "results"] if (ROOT / "results").is_dir() else [])
    if not search_roots:
        return None
    out: dict[str, dict] = {}
    for m, stem in gsf.FILE.items():
        hit = None
        for d in search_roots:
            if not d.exists():
                continue
            hit = _persona_path_for_stem(d, stem)
            if hit:
                break
        if not hit:
            continue
        blob = json.loads(hit.read_text(encoding="utf-8"))
        scores = blob.get("scores") or {}
        methods = [k for k in ("M1", "M2", "M3", "M4") if k in scores]
        if len(methods) < 2:
            continue
        try:
            c = gsf.cms_k(scores, methods)
            out[m] = {
                "model": m,
                "cms": c["cms"],
                "n_methods": len(methods),
                "methods": methods,
                "scores": scores,
                "system": blob.get("system") or "",
                "seed": (blob.get("decode") or {}).get("seed"),
                "n_outcomes": len(scores.get("M1") or {}),
                "path": str(hit.relative_to(ROOT)),
                "diagnostics": blob.get("diagnostics") or {},
                "matrix": {f"{a}|{b}": v for (a, b), v in c["matrix"].items()},
            }
        except Exception as e:
            print("persona skip", m, e)
    return out or None


def load_gemma() -> dict | None:
    cands = sorted((LIB / "gemma").glob("*.json"))
    cands += sorted((ROOT / "results").glob("*gemma*.json"))
    if not cands:
        return None
    blob = json.loads(cands[0].read_text(encoding="utf-8"))
    packed = pack_full(blob)
    packed["path"] = str(cands[0].relative_to(ROOT))
    print("gemma loaded", packed["path"], "cms4", round(packed["cms4"], 3))
    return packed


def models_plus(gemma: dict | None) -> list[str]:
    return list(gsf.ORDER) + ([GEMMA] if gemma else [])


def short_of(m: str) -> str:
    if m == GEMMA:
        return GEMMA_SHORT
    return gsf.SHORT[m]


def blob_of(m: str, asst: dict, gemma: dict | None) -> dict:
    return gemma if m == GEMMA else asst[m]


def self_check(asst: dict):
    v = asst["Qwen/Qwen3-4B-Instruct-2507"]["cms4"]
    assert abs(v - 0.528) < 0.015, f"2507 CMS4 expected ~0.528 got {v}"
    v14 = asst["Qwen/Qwen3-14B"]["cms4"]
    assert abs(v14 - 0.256) < 0.015, f"14B CMS4 expected ~0.256 got {v14}"
    v17 = asst["Qwen/Qwen3-1.7B"]["cms4"]
    assert abs(v17 - 0.004) < 0.02, f"1.7B CMS4 expected ~0.004 got {v17}"
    print("self_check ok")


def fig_e3(asst: dict, s1: dict, persona: dict | None, gemma: dict | None):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.plot(gsf.E3_B, [asst[m]["cms3"] for m in gsf.E3], "o-", color=C["blue"], lw=2, ms=7, label="CMS3 seed 0")
    ax.plot(
        gsf.E3_B,
        [asst[m]["cms4"] for m in gsf.E3],
        "s-",
        color=C["vermillion"],
        lw=2,
        ms=7,
        label="CMS4 seed 0 + M4@1800",
    )
    ax.plot(gsf.E3_B, [s1[m]["cms"] for m in gsf.E3], "^--", color=C["sky"], lw=1.6, ms=6, label="CMS3 seed 1")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ref = asst["Qwen/Qwen3-4B-Instruct-2507"]["cms4"]
    ax.axhline(ref, color=C["green"], ls=":", lw=1.5, label=f"4B-Instruct-2507 CMS4 ({ref:.3f})")
    if persona:
        xs, ys = [], []
        for m, b in zip(gsf.E3, gsf.E3_B):
            if m in persona:
                xs.append(b)
                ys.append(persona[m]["cms"])
        if xs:
            ax.plot(xs, ys, "D-", color=C["purple"], lw=1.8, ms=7, label="persona CMS")
    if gemma:
        ax.axhline(
            gemma["cms4"],
            color=C["gemma"],
            ls="-.",
            lw=1.4,
            label=f"Gemma-3-4B-it CMS4 ({gemma['cms4']:.3f})",
        )
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("CMS (mean pairwise midrank Pearson)")
    ax.set_title("Cross-method agreement vs size (N=24; 2507 / Gemma off-axis)")
    ax.set_xticks(gsf.E3_B)
    ax.set_ylim(-0.15, 0.65)
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "01_cms_vs_size_e3")


def fig_cms_bars(asst: dict, s1: dict, emp: dict, gemma: dict | None):
    labels = [gsf.SHORT[m] for m in gsf.ORDER] + ([GEMMA_SHORT] if gemma else [])
    n = len(labels)
    x = np.arange(n)
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    asst_y = [asst[m]["cms4"] for m in gsf.ORDER] + ([gemma["cms4"]] if gemma else [])
    s1_y = [s1[m]["cms"] for m in gsf.ORDER] + ([np.nan] if gemma else [])
    emp3 = [emp[m]["cms3"] for m in gsf.ORDER] + ([np.nan] if gemma else [])
    emp4 = [emp[m]["cms4"] for m in gsf.ORDER] + ([np.nan] if gemma else [])
    series = [
        (asst_y, C["vermillion"], "assistant CMS4"),
        (s1_y, C["sky"], "seed 1 CMS3"),
        (emp3, C["blue"], "empty CMS3"),
        (emp4, C["green"], "empty CMS4"),
    ]
    for i, (ys, color, lab) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, ys, w, color=color, edgecolor="none", label=lab)
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("CMS")
    ax.set_title("CMS by model and wrap (Gemma: assistant only)")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    save(fig, "02_cms_bars_conditions")


def fig_rho_heatmap(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    mat = np.array([[blob_of(m, asst, gemma)["matrix4"][p] for p in gsf.PAIRS4] for m in keys])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = _imshow_annot(ax, mat, gsf.PAIRS4, [short_of(m) for m in keys], -0.3, 1.0, "Pairwise midrank Pearson (CMS4 ingredients)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson ρ of midranks")
    save(fig, "03_rho_heatmap_assistant")


def fig_intermodel(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, axes = plt.subplots(1, 2, figsize=(10.2 if gemma else 9.0, 4.2 if gemma else 3.9))
    im = None
    for ax, method in zip(axes, ("M1", "M3")):
        ranks = {m: gsf.midranks(blob_of(m, asst, gemma)["scores"][method], gsf.IDS) for m in keys}
        mat = np.array([[safe_pearson(ranks[a], ranks[b]) for b in keys] for a in keys])
        im = _imshow_annot(ax, mat, [short_of(m) for m in keys], [short_of(m) for m in keys], -0.4, 1.0, f"Inter-model {method} midrank ρ", fs=6.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Pearson ρ")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"04_intermodel_m1_m3.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=220 if ext == "png" else None)
        print("wrote", p)
    plt.close(fig)


def fig_seed_stability(asst: dict, s1: dict):
    labels = [gsf.SHORT[m] for m in gsf.ORDER]
    x = np.arange(len(gsf.ORDER))
    fig, ax1 = plt.subplots(figsize=(7.0, 3.8))
    w = 0.32
    ax1.bar(x - w / 2, [asst[m]["cms3"] for m in gsf.ORDER], w, color=C["blue"], label="CMS3 seed 0")
    ax1.bar(x + w / 2, [s1[m]["cms"] for m in gsf.ORDER], w, color=C["sky"], label="CMS3 seed 1")
    ax1.axhline(0, color=C["muted"], lw=0.8)
    ax1.set_ylabel("CMS3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x, [gsf.rank_rho(asst[m]["scores"]["M1"], s1[m]["scores"]["M1"]) for m in gsf.ORDER], "D-", color=C["vermillion"], ms=6, lw=1.6, label="M1 rank ρ (s0↔s1)")
    ax2.set_ylabel("M1 midrank Pearson ρ")
    ax2.set_ylim(0.9, 1.01)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7)
    ax1.set_title("Seed stability: CMS3 shifts vs M1 rank lock")
    save(fig, "05_seed_stability")


def fig_diagnostics(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    labels = [short_of(m) for m in keys]
    x = np.arange(len(keys))
    w = 0.36
    pa, oi = [], []
    for m in keys:
        d = blob_of(m, asst, gemma)["diagnostics"]["M1"]
        pa.append(d["p_pick_a"])
        oi.append(d["order_inconsistent_frac"])
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(x - w / 2, pa, w, color=C["blue"], label="M1 P(pick A)")
    ax.bar(x + w / 2, oi, w, color=C["vermillion"], label="M1 order-inconsistent frac")
    ax.axhline(0.5, color=C["muted"], ls="--", lw=0.9, label="chance P(A)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Position / order diagnostics (assistant wrap)")
    ax.legend(loc="upper right")
    save(fig, "06_diagnostics_pa_order")


def _scale_panel(ax, blob: dict, title: str):
    mu = {k: v["mean"] for k, v in blob["utilities_thurstone"]["utilities"].items()}
    ids = sorted(mu, key=lambda i: mu[i])
    ax.plot(range(len(ids)), [blob["scores"]["M1"][i] for i in ids], "o-", color=C["blue"], ms=4, label="M1 win rate")
    ax.plot(range(len(ids)), [blob["scores"]["M2"][i] / 7.0 for i in ids], "s--", color=C["orange"], ms=4, label="M2 Likert / 7")
    ax.plot(range(len(ids)), [blob["scores"]["M3"][i] for i in ids], "^:", color=C["vermillion"], ms=4, label="M3 pick rate")
    m4 = blob["scores"]["M4"]
    lo, hi = min(m4.values()), max(m4.values())
    span = hi - lo if hi != lo else 1.0
    ax.plot(range(len(ids)), [(m4[i] - lo) / span for i in ids], "d-.", color=C["green"], ms=4, label="M4 BWS min–max")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Score (scaled)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")


def fig_scale_14b(asst: dict):
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    _scale_panel(ax, asst["Qwen/Qwen3-14B"], "14B: methods ordered by Thurstone μ (low→high)")
    save(fig, "07_scale_thurstone_14b")


def _svr_panel(ax, blob: dict, title: str):
    labels = [g for g, _, _ in gsf.PAIRS_NEW]
    d1, d3 = [], []
    for _, a, c in gsf.PAIRS_NEW:
        d1.append(blob["scores"]["M1"][a] - blob["scores"]["M1"][c])
        d3.append(blob["scores"]["M3"][a] - blob["scores"]["M3"][c])
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, d1, w, color=C["blue"], label="M1 (stated) A−B")
    ax.bar(x + w / 2, d3, w, color=C["vermillion"], label="M3 (action) A−B")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Pole A minus pole B")
    ax.set_title(title)
    ax.legend()


def fig_svr(asst: dict):
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    _svr_panel(ax, asst["Qwen/Qwen3-14B"], "14B: stated vs action pole gap")
    save(fig, "08_svr_pole_gap_14b")


def fig_empty_delta(asst: dict, emp: dict):
    labels = [gsf.SHORT[m] for m in gsf.ORDER]
    x = np.arange(len(gsf.ORDER))
    d_cms = [emp[m]["cms4"] - asst[m]["cms4"] for m in gsf.ORDER]
    m1rho = [gsf.rank_rho(asst[m]["scores"]["M1"], emp[m]["scores"]["M1"]) for m in gsf.ORDER]
    fig, ax1 = plt.subplots(figsize=(7.0, 3.7))
    colors = [C["green"] if v >= 0 else C["vermillion"] for v in d_cms]
    ax1.bar(x, d_cms, color=colors, width=0.55, label="ΔCMS4 (empty − asst)")
    ax1.axhline(0, color=C["muted"], lw=0.8)
    ax1.set_ylabel("ΔCMS4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x, m1rho, "o-", color=C["blue"], ms=6, lw=1.6, label="M1 ρ empty↔asst")
    ax2.set_ylabel("M1 midrank Pearson ρ")
    ax2.set_ylim(0.85, 1.02)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7)
    ax1.set_title("Empty system wrap: CMS shift vs M1 rank lock")
    save(fig, "09_empty_wrap_delta")


def fig_persona_slot(persona: dict | None, asst: dict):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    labels = [gsf.SHORT[m] for m in gsf.ORDER]
    x = np.arange(len(gsf.ORDER))
    ax.bar(x, [asst[m]["cms4"] for m in gsf.ORDER], width=0.55, color=C["blue"], alpha=0.85, label="assistant CMS4")
    if persona:
        ys = [persona[m]["cms"] if m in persona else np.nan for m in gsf.ORDER]
        ax.plot(x, ys, "D-", color=C["purple"], ms=8, lw=1.8, label="persona CMS4")
        n = sum(1 for m in gsf.ORDER if m in persona)
        ax.set_title(f"Persona system wrap vs assistant CMS4 (n={n} models)")
    else:
        ax.set_ylim(-0.15, 0.65)
        ax.set_title("Persona XL slot (awaiting run)")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CMS")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "10_persona_xl_slot")


def fig_family_4b(asst: dict, gemma: dict | None):
    """Qwen-4B / 2507 / Gemma on same N=24 assistant wrap."""
    rows = [
        ("4B", asst["Qwen/Qwen3-4B"]),
        ("4B-2507", asst["Qwen/Qwen3-4B-Instruct-2507"]),
    ]
    if gemma:
        rows.append((GEMMA_SHORT, gemma))
    labels = [a for a, _ in rows]
    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w / 2, [b["cms3"] for _, b in rows], w, color=C["blue"], label="CMS3")
    ax.bar(x + w / 2, [b["cms4"] for _, b in rows], w, color=C["vermillion"], label="CMS4")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CMS")
    ax.set_title("Same-size family: Qwen3-4B vs Instruct-2507 vs Gemma-3-4B-it")
    ax.legend()
    save(fig, "11_family_4b_cms")


def fig_gemma_svr(gemma: dict | None):
    if not gemma:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    _svr_panel(ax, gemma, "Gemma-3-4B-it: stated vs action pole gap")
    save(fig, "12_svr_pole_gap_gemma")


def fig_gemma_scale(gemma: dict | None):
    if not gemma:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    _scale_panel(ax, gemma, "Gemma-3-4B-it: methods ordered by Thurstone μ")
    save(fig, "13_scale_thurstone_gemma")


def fig_condition_heatmap(asst: dict, s1: dict, emp: dict, persona: dict | None, gemma: dict | None):
    cols = ["asst CMS4", "empty CMS4", "persona CMS", "seed1 CMS3"]
    keys = models_plus(gemma)
    mat = np.full((len(keys), 4), np.nan)
    for i, m in enumerate(keys):
        if m == GEMMA:
            mat[i, 0] = gemma["cms4"]
            continue
        mat[i, 0] = asst[m]["cms4"]
        mat[i, 1] = emp[m]["cms4"]
        if persona and m in persona:
            mat[i, 2] = persona[m]["cms"]
        mat[i, 3] = s1[m]["cms"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = _imshow_annot(ax, mat, cols, [short_of(m) for m in keys], -0.15, 0.6, "CMS by wrap (blank = no run)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("CMS")
    save(fig, "14_cms_wrap_heatmap")


def fig_score_heatmap(asst: dict, gemma: dict | None, method: str, stem: str, title: str):
    keys = models_plus(gemma)
    mat = np.array([[blob_of(m, asst, gemma)["scores"][method][oid] for oid in gsf.IDS] for m in keys], dtype=float)
    if method == "M2":
        mat = mat / 7.0
    fig, ax = plt.subplots(figsize=(11.2, 3.8 if not gemma else 4.2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(range(len(gsf.IDS)))
    ax.set_xticklabels(gsf.IDS, rotation=75, ha="right", fontsize=6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([short_of(m) for m in keys], fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("score")
    save(fig, stem)


def fig_catch(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    labels = [short_of(m) for m in keys]
    x = np.arange(len(keys))
    w = 0.12
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    palette = [C["blue"], C["sky"], C["green"], C["orange"], C["vermillion"], C["purple"]]
    for i, oid in enumerate(gsf.CATCH):
        ys = [blob_of(m, asst, gemma)["scores"]["M1"][oid] for m in keys]
        ax.bar(x + (i - 2.5) * w, ys, w, color=palette[i % len(palette)], label=oid)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("M1 win rate")
    ax.set_ylim(0, 1)
    ax.set_title("M1 catch / pole scores")
    ax.legend(ncol=2, fontsize=7)
    save(fig, "17_catch_m1")


def fig_pair_small_multiples(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(gsf.PAIRS4))
    for ax, m in zip(axes, keys):
        ys = [blob_of(m, asst, gemma)["matrix4"][p] for p in gsf.PAIRS4]
        cols = [C["vermillion"] if v < 0 else C["blue"] for v in ys]
        ax.bar(x, ys, color=cols)
        ax.axhline(0, color=C["muted"], lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(gsf.PAIRS4, rotation=50, ha="right", fontsize=6)
        ax.set_title(short_of(m), fontsize=10)
        ax.set_ylim(-0.5, 1.0)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    fig.suptitle("CMS4 ingredients by model", fontsize=12, y=1.01)
    save(fig, "18_pair_rho_small_multiples")


def fig_m1_m3_scatter(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    n = len(keys)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.4, 3.1 * rows), squeeze=False)
    for i, m in enumerate(keys):
        ax = axes[i // cols][i % cols]
        b = blob_of(m, asst, gemma)
        xs = [b["scores"]["M1"][oid] for oid in gsf.IDS]
        ys = [b["scores"]["M3"][oid] for oid in gsf.IDS]
        ax.scatter(xs, ys, c=C["blue"], s=18, alpha=0.85, edgecolors="none")
        ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.7, ls="--")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(short_of(m), fontsize=10)
        if i // cols == rows - 1:
            ax.set_xlabel("M1 win rate")
        if i % cols == 0:
            ax.set_ylabel("M3 pick rate")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Stated vs action scores (one point = one outcome)", fontsize=12, y=1.02)
    save(fig, "19_m1_vs_m3_scatter")


def fig_rho_to_14b(asst: dict, gemma: dict | None):
    keys = [m for m in models_plus(gemma) if m != "Qwen/Qwen3-14B"]
    ref = asst["Qwen/Qwen3-14B"]["scores"]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    x = np.arange(len(keys))
    w = 0.36
    m1 = [safe_rank_rho(blob_of(m, asst, gemma)["scores"]["M1"], ref["M1"]) for m in keys]
    m3 = [safe_rank_rho(blob_of(m, asst, gemma)["scores"]["M3"], ref["M3"]) for m in keys]
    ax.bar(x - w / 2, m1, w, color=C["blue"], label="M1 vs 14B")
    ax.bar(x + w / 2, m3, w, color=C["vermillion"], label="M3 vs 14B")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_of(m) for m in keys], rotation=20, ha="right")
    ax.set_ylabel("Midrank Pearson ρ")
    ax.set_title("Who shares 14B ranks? (M1 vs M3)")
    ax.legend()
    save(fig, "20_rank_rho_vs_14b")


def fig_parse_heatmap(asst: dict, emp: dict, gemma: dict | None):
    keys = models_plus(gemma)
    methods = ["M1", "M2", "M3", "M4"]
    mat = np.full((len(keys), 4), np.nan)
    for i, m in enumerate(keys):
        d = blob_of(m, asst, gemma)["diagnostics"]
        for j, meth in enumerate(methods):
            if meth in d and d[meth].get("parse_fail_rate") is not None:
                mat[i, j] = d[meth]["parse_fail_rate"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = _imshow_annot(ax, mat, methods, [short_of(m) for m in keys], 0, 0.15, "Parse-fail rate (assistant)", cmap=plt.cm.YlOrRd)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("fail fraction")
    save(fig, "21_parse_fail_heatmap")


def fig_wrap_grouped(asst: dict, emp: dict, persona: dict | None):
    labels = [gsf.SHORT[m] for m in gsf.ORDER]
    x = np.arange(len(gsf.ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(x - w, [asst[m]["cms4"] for m in gsf.ORDER], w, color=C["blue"], label="assistant")
    ax.bar(x, [emp[m]["cms4"] for m in gsf.ORDER], w, color=C["green"], label="empty")
    if persona:
        ax.bar(x + w, [persona[m]["cms"] if m in persona else np.nan for m in gsf.ORDER], w, color=C["purple"], label="persona")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CMS4 / persona CMS")
    ax.set_title("Wrap comparison (same N=24, Qwen ladder)")
    ax.legend()
    save(fig, "22_wrap_cms_grouped")


def fig_m1m4_vs_m1m3(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for m in keys:
        mx = blob_of(m, asst, gemma)["matrix4"]
        ax.scatter(mx["M1|M3"], mx["M1|M4"], s=70, label=short_of(m))
        ax.annotate(short_of(m), (mx["M1|M3"], mx["M1|M4"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(0, color=C["muted"], lw=0.7)
    ax.axvline(0, color=C["muted"], lw=0.7)
    ax.set_xlabel("M1|M3 ρ (stated vs action)")
    ax.set_ylabel("M1|M4 ρ (stated vs BWS)")
    ax.set_title("Where does M1 lock — action or BWS?")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "23_m1m4_vs_m1m3")


def fig_pole_heatmap(asst: dict, gemma: dict | None, method: str, stem: str):
    keys = models_plus(gemma)
    labels = [g for g, _, _ in gsf.PAIRS_NEW]
    mat = np.zeros((len(keys), len(labels)))
    for i, m in enumerate(keys):
        sc = blob_of(m, asst, gemma)["scores"][method]
        for j, (_, a, c) in enumerate(gsf.PAIRS_NEW):
            mat[i, j] = sc[a] - sc[c]
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    vmax = max(0.4, float(np.nanmax(np.abs(mat))))
    im = _imshow_annot(ax, mat, labels, [short_of(m) for m in keys], -vmax, vmax, f"{method} pole gap (A−B)", fs=6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("A − B")
    save(fig, stem)


def fig_gemma_vs_14b_scatter(asst: dict, gemma: dict | None):
    if not gemma:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    q = asst["Qwen/Qwen3-14B"]["scores"]
    g = gemma["scores"]
    for ax, meth in zip(axes, ("M1", "M3")):
        xs = [q[meth][oid] for oid in gsf.IDS]
        ys = [g[meth][oid] for oid in gsf.IDS]
        ax.scatter(xs, ys, c=C["gemma"], s=22, edgecolors="none")
        lo, hi = 0, 1
        ax.plot([lo, hi], [lo, hi], color=C["muted"], ls="--", lw=0.7)
        rho = safe_rank_rho(q[meth], g[meth])
        ax.set_xlabel(f"14B {meth}")
        ax.set_ylabel(f"Gemma {meth}")
        ax.set_title(f"{meth}  ρ={rho:.2f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Gemma-3-4B-it vs Qwen3-14B (same outcomes)", y=1.03)
    save(fig, "26_gemma_vs_14b_scatter")


def fig_run1(r1: dict):
    labels, cms_v, miss2, pa = [], [], [], []
    for m in gsf.ORDER:
        if m not in r1:
            continue
        b = r1[m]
        labels.append(gsf.SHORT[m])
        cms_v.append(b.get("cms") if b.get("cms") is not None else np.nan)
        miss2.append((b.get("missingness") or {}).get("M2", 0) / 100.0)
        logs = (b.get("logs") or {}).get("M1") or []
        pa.append(gsf.p_pick_a(logs) or np.nan)
    x = np.arange(len(labels))
    w = 0.28
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w, cms_v, w, color=C["blue"], label="CMS (NaN = undefined)")
    ax.bar(x, miss2, w, color=C["orange"], label="M2 missingness")
    ax.bar(x + w, pa, w, color=C["vermillion"], label="M1 P(A)")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Run 1 (N=14, 4-way M3) — appendix only")
    ax.legend(fontsize=7)
    save(fig, "27_run1_status")


def fig_cms_lollipop(asst: dict, gemma: dict | None):
    rows = [(gsf.SHORT[m], asst[m]["cms4"]) for m in gsf.ORDER]
    if gemma:
        rows.append((GEMMA_SHORT, gemma["cms4"]))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(rows))
    ax.hlines(y, 0, [v for _, v in rows], color=C["grid"], lw=2)
    ax.plot([v for _, v in rows], y, "o", color=C["vermillion"], ms=8)
    ax.axvline(0, color=C["muted"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([a for a, _ in rows])
    ax.set_xlabel("CMS4 (assistant)")
    ax.set_title("Ranked cross-method agreement")
    save(fig, "28_cms4_lollipop")


def fig_empty_rho(emp: dict):
    mat = np.array([[emp[m]["matrix4"][p] for p in gsf.PAIRS4] for m in gsf.ORDER])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = _imshow_annot(ax, mat, gsf.PAIRS4, [gsf.SHORT[m] for m in gsf.ORDER], -0.3, 1.0, "Empty wrap: pairwise midrank Pearson")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson ρ")
    save(fig, "29_rho_heatmap_empty")


def fig_persona_rho(persona: dict | None):
    if not persona:
        return
    keys = [m for m in gsf.ORDER if m in persona]
    mat = np.array([[persona[m]["matrix"].get(p, np.nan) for p in gsf.PAIRS4] for m in keys])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = _imshow_annot(ax, mat, gsf.PAIRS4, [gsf.SHORT[m] for m in keys], -0.3, 1.0, "Persona wrap: pairwise midrank Pearson")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson ρ")
    save(fig, "30_rho_heatmap_persona")


def fig_m4_vs_m1_scatter(asst: dict, gemma: dict | None):
    panels = [("14B", asst["Qwen/Qwen3-14B"]), ("4B-2507", asst["Qwen/Qwen3-4B-Instruct-2507"])]
    if gemma:
        panels.append((GEMMA_SHORT, gemma))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), squeeze=False)
    for ax, (lab, b) in zip(axes[0], panels):
        m4 = b["scores"]["M4"]
        lo, hi = min(m4.values()), max(m4.values())
        span = hi - lo if hi != lo else 1.0
        xs = [b["scores"]["M1"][oid] for oid in gsf.IDS]
        ys = [(m4[oid] - lo) / span for oid in gsf.IDS]
        ax.scatter(xs, ys, c=C["green"], s=20, edgecolors="none")
        ax.plot([0, 1], [0, 1], color=C["muted"], ls="--", lw=0.7)
        ax.set_xlabel("M1")
        ax.set_ylabel("M4 min–max")
        ax.set_title(lab)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Stated vs BWS (scaled)", y=1.03)
    save(fig, "31_m1_vs_m4_scatter")


def fig_seed_method_rho(asst: dict, s1: dict):
    methods = ["M1", "M2", "M3"]
    mat = np.array(
        [[safe_rank_rho(asst[m]["scores"][meth], s1[m]["scores"][meth]) for meth in methods] for m in gsf.ORDER]
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = _imshow_annot(ax, mat, methods, [gsf.SHORT[m] for m in gsf.ORDER], 0.5, 1.0, "Seed 0↔1 midrank ρ by method")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson ρ")
    save(fig, "32_seed_method_rho")


def fig_disagreement(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = 0
    yticks, ylabels = [], []
    palette = {"M1|M3": C["vermillion"], "M1|M4": C["green"], "M2|M3": C["orange"]}
    for m in keys:
        cards = blob_of(m, asst, gemma)["cards4"]
        for c in cards[:2]:
            pair = f"{c['pair'][0]}|{c['pair'][1]}"
            ax.barh(y, c["abs_rank_diff"], color=palette.get(pair, C["blue"]), height=0.7)
            yticks.append(y)
            ylabels.append(f"{short_of(m)}  {c['outcome_id']}")
            y += 1
        y += 0.4
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("abs rank diff (top cards)")
    ax.set_title("Disagreement cards (largest rank splits)")
    save(fig, "33_disagreement_cards")


def fig_m3_collapse(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    frac = []
    for m in keys:
        sc = blob_of(m, asst, gemma)["scores"]["M3"]
        vals = [sc[oid] for oid in gsf.IDS]
        frac.append(sum(1 for v in vals if abs(v - 0.5) < 1e-9) / len(vals))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(keys))
    ax.bar(x, frac, color=C["vermillion"])
    ax.set_xticks(x)
    ax.set_xticklabels([short_of(m) for m in keys], rotation=20, ha="right")
    ax.set_ylabel("Fraction of outcomes at M3=0.5")
    ax.set_ylim(0, 1)
    ax.set_title("M3 tie / collapse (0.5 = both orders split or unused poles)")
    save(fig, "34_m3_half_collapse")


def fig_cms3_cms4(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for m in keys:
        b = blob_of(m, asst, gemma)
        ax.scatter(b["cms3"], b["cms4"], s=70)
        ax.annotate(short_of(m), (b["cms3"], b["cms4"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.plot([-0.15, 0.65], [-0.15, 0.65], color=C["muted"], ls="--", lw=0.7)
    ax.set_xlabel("CMS3")
    ax.set_ylabel("CMS4")
    ax.set_title("Does adding M4 change the picture?")
    ax.set_xlim(-0.15, 0.65)
    ax.set_ylim(-0.15, 0.65)
    save(fig, "35_cms3_vs_cms4")


def fig_persona_delta(asst: dict, persona: dict | None):
    if not persona:
        return
    labels = [gsf.SHORT[m] for m in gsf.ORDER]
    x = np.arange(len(gsf.ORDER))
    d = [persona[m]["cms"] - asst[m]["cms4"] if m in persona else np.nan for m in gsf.ORDER]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x, d, color=[C["green"] if (not np.isnan(v) and v >= 0) else C["vermillion"] for v in d])
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ΔCMS (persona − assistant)")
    ax.set_title("Persona wrap: CMS shift")
    save(fig, "36_persona_delta")


def fig_empty_svr_14b(asst: dict, emp: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    _svr_panel(axes[0], asst["Qwen/Qwen3-14B"], "14B assistant")
    _svr_panel(axes[1], emp["Qwen/Qwen3-14B"], "14B empty")
    save(fig, "37_svr_14b_asst_vs_empty")


def fig_qwen_gemma_m1_family(asst: dict, gemma: dict | None):
    if not gemma:
        return
    family = ["Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507", GEMMA]
    labels = [short_of(m) for m in family]
    mat = np.zeros((3, 3))
    ranks = {m: gsf.midranks(blob_of(m, asst, gemma)["scores"]["M1"], gsf.IDS) for m in family}
    for i, a in enumerate(family):
        for j, b in enumerate(family):
            mat[i, j] = safe_pearson(ranks[a], ranks[b])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = _imshow_annot(ax, mat, labels, labels, -0.2, 1.0, "4B-class M1 rank agreement")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson ρ")
    save(fig, "38_family_4b_m1_rho")


def fig_top_outcomes(asst: dict, gemma: dict | None):
    keys = models_plus(gemma)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for m in keys:
        sc = blob_of(m, asst, gemma)["scores"]["M1"]
        order = sorted(gsf.IDS, key=lambda i: -sc[i])
        ranks = {oid: k + 1 for k, oid in enumerate(order)}
        ys = [ranks[oid] for oid in gsf.IDS]
        ax.plot(range(len(gsf.IDS)), ys, "o-", ms=3, lw=1.1, label=short_of(m))
    ax.invert_yaxis()
    ax.set_xticks(range(len(gsf.IDS)))
    ax.set_xticklabels(gsf.IDS, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("M1 rank (1 = most preferred)")
    ax.set_title("M1 rank profile across models")
    ax.legend(ncol=3, fontsize=7)
    save(fig, "39_m1_rank_profiles")


def fig_2507_vs_base4(asst: dict):
    a = asst["Qwen/Qwen3-4B"]["scores"]
    b = asst["Qwen/Qwen3-4B-Instruct-2507"]["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, meth in zip(axes, ("M1", "M3")):
        xs = [a[meth][oid] for oid in gsf.IDS]
        ys = [b[meth][oid] for oid in gsf.IDS]
        ax.scatter(xs, ys, c=C["green"], s=22, edgecolors="none")
        ax.plot([0, 1], [0, 1], color=C["muted"], ls="--", lw=0.7)
        ax.set_xlabel(f"4B {meth}")
        ax.set_ylabel(f"2507 {meth}")
        ax.set_title(f"{meth}  ρ={safe_rank_rho(a[meth], b[meth]):.2f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Original 4B vs Instruct-2507 (assistant)", y=1.03)
    save(fig, "40_4b_vs_2507_scatter")


def tables(asst: dict, s1: dict, emp: dict, persona: dict | None, gemma: dict | None):
    rows = []
    for m in gsf.ORDER:
        b = asst[m]
        mx = b["matrix4"]
        rows.append([gsf.SHORT[m], f"{b['cms3']:.3f}", f"{b['cms4']:.3f}", *[f"{mx[p]:.3f}" for p in gsf.PAIRS4]])
    write_csv("table1_cms4_assistant.csv", ["model", "cms3", "cms4", *gsf.PAIRS4], rows)

    rows = []
    for m in gsf.ORDER:
        rows.append(
            [
                gsf.SHORT[m],
                f"{asst[m]['cms3']:.3f}",
                f"{s1[m]['cms']:.3f}",
                f"{gsf.rank_rho(asst[m]['scores']['M1'], s1[m]['scores']['M1']):.3f}",
            ]
        )
    write_csv("table2_seed_stability.csv", ["model", "cms3_s0", "cms3_s1", "m1_rank_rho"], rows)

    rows = []
    for m in gsf.ORDER:
        rows.append(
            [
                gsf.SHORT[m],
                f"{asst[m]['cms3']:.3f}",
                f"{emp[m]['cms3']:.3f}",
                f"{asst[m]['cms4']:.3f}",
                f"{emp[m]['cms4']:.3f}",
                f"{gsf.rank_rho(asst[m]['scores']['M1'], emp[m]['scores']['M1']):.3f}",
            ]
        )
    write_csv(
        "table3_empty_vs_assistant.csv",
        ["model", "cms3_asst", "cms3_empty", "cms4_asst", "cms4_empty", "m1_rank_rho"],
        rows,
    )

    rows = []
    for m in gsf.ORDER:
        for c in asst[m]["cards4"]:
            a, b = c["pair"]
            rows.append([gsf.SHORT[m], f"{a}|{b}", c["outcome_id"], f"{c['abs_rank_diff']:.1f}"])
    if gemma:
        for c in gemma["cards4"]:
            a, b = c["pair"]
            rows.append([GEMMA_SHORT, f"{a}|{b}", c["outcome_id"], f"{c['abs_rank_diff']:.1f}"])
    write_csv("table4_disagreement_cards.csv", ["model", "method_pair", "outcome_id", "abs_rank_diff"], rows)

    rows = []
    for m in models_plus(gemma):
        d = blob_of(m, asst, gemma)["diagnostics"]
        m4 = blob_of(m, asst, gemma)["m4_diag"] or {}
        rows.append(
            [
                short_of(m),
                f"{d['M1'].get('p_pick_a', '')}",
                f"{d['M1'].get('order_inconsistent_frac', '')}",
                f"{d['M2'].get('parse_fail_rate', '')}",
                f"{m4.get('parse_fail_rate', '')}",
                f"{m4.get('p_best_a', '')}",
            ]
        )
    write_csv(
        "table5_diagnostics.csv",
        ["model", "m1_p_pick_a", "m1_order_inconsist", "m2_parse_fail", "m4_parse_fail", "m4_p_best_a"],
        rows,
    )

    if persona:
        rows = []
        for m in gsf.ORDER:
            if m not in persona:
                rows.append([gsf.SHORT[m], "", f"{asst[m]['cms4']:.3f}", "", "", "", ""])
                continue
            p = persona[m]
            m1rho = gsf.rank_rho(asst[m]["scores"]["M1"], p["scores"]["M1"])
            rows.append(
                [
                    gsf.SHORT[m],
                    f"{p['cms']:.3f}",
                    f"{asst[m]['cms4']:.3f}",
                    f"{p['cms'] - asst[m]['cms4']:.3f}",
                    f"{m1rho:.3f}",
                    ",".join(p["methods"]),
                    p["path"],
                ]
            )
        write_csv(
            "table6_persona_xl.csv",
            ["model", "cms_persona", "cms4_asst", "delta_cms", "m1_rho_vs_asst", "methods", "source"],
            rows,
        )

    if gemma:
        mx = gemma["matrix4"]
        write_csv(
            "table7_gemma.csv",
            ["model", "cms3", "cms4", *gsf.PAIRS4, "path"],
            [
                [
                    GEMMA_SHORT,
                    f"{gemma['cms3']:.3f}",
                    f"{gemma['cms4']:.3f}",
                    *[f"{mx[p]:.3f}" for p in gsf.PAIRS4],
                    gemma["path"],
                ]
            ],
        )


def write_catalog(gemma: dict | None):
    lines = [
        "# Figure catalog (pick and choose)",
        "",
        "Regen: `.venv/bin/python scripts/report_figures.py`",
        "",
        "| Stem | Claim | Suggested use |",
        "| --- | --- | --- |",
        "| `01_cms_vs_size_e3` | CMS ≠ size law; 2507 / Gemma off-axis | Results headline |",
        "| `02_cms_bars_conditions` | All wraps + Gemma assistant | Overview |",
        "| `03_rho_heatmap_assistant` | Where pairwise agreement lives | Results |",
        "| `04_intermodel_m1_m3` | Stated ranks travel; action ranks do not | Results |",
        "| `05_seed_stability` | M1 lock vs CMS wobble | Robustness |",
        "| `06_diagnostics_pa_order` | 1.7B letter collapse; Gemma P(A) | Methods |",
        "| `07_scale_thurstone_14b` | M3 peels off 14B scale | Methods |",
        "| `08_svr_pole_gap_14b` | Stated vs action poles | Results / punch |",
        "| `09_empty_wrap_delta` | Empty wrap does not close SvR | Robustness |",
        "| `10_persona_xl_slot` | Persona CMS vs assistant | Robustness |",
        "| `11_family_4b_cms` | Same-size: Qwen 4B / 2507 / Gemma | Results (family) |",
        "| `12_svr_pole_gap_gemma` | Gemma SvR analogue | Optional |",
        "| `13_scale_thurstone_gemma` | Gemma method traces | Optional |",
        "| `14_cms_wrap_heatmap` | All conditions one matrix | Overview |",
        "| `15_m1_score_heatmap` | M1 scores, all outcomes | Appendix |",
        "| `16_m3_score_heatmap` | M3 scores (collapse visible) | Appendix |",
        "| `17_catch_m1` | Catch / pole M1 bars | Methods appendix |",
        "| `18_pair_rho_small_multiples` | CMS ingredients per model | Results alt |",
        "| `19_m1_vs_m3_scatter` | Per-model SvR scatter | Results alt |",
        "| `20_rank_rho_vs_14b` | Who matches 14B | Results |",
        "| `21_parse_fail_heatmap` | Instrument reliability | Methods |",
        "| `22_wrap_cms_grouped` | Assistant / empty / persona | Robustness |",
        "| `23_m1m4_vs_m1m3` | BWS lock vs action lock | Results |",
        "| `24_pole_gap_m1` | M1 pole gaps all models | Appendix |",
        "| `25_pole_gap_m3` | M3 pole gaps all models | Appendix |",
        "| `26_gemma_vs_14b_scatter` | Cross-family stated/action | Results |",
        "| `27_run1_status` | N=14 cafeteria — do not pool | Appendix only |",
        "| `28_cms4_lollipop` | Ranked CMS4 including Gemma | Slide |",
        "| `29_rho_heatmap_empty` | Empty-wrap ingredients | Robustness |",
        "| `30_rho_heatmap_persona` | Persona-wrap ingredients | Robustness |",
        "| `31_m1_vs_m4_scatter` | Stated vs BWS | Methods |",
        "| `32_seed_method_rho` | Seed lock by method | Robustness |",
        "| `33_disagreement_cards` | Largest rank splits | Results punch |",
        "| `34_m3_half_collapse` | M3=0.5 mass (Gemma) | Methods |",
        "| `35_cms3_vs_cms4` | Adding M4 | Methods |",
        "| `36_persona_delta` | Persona ΔCMS | Robustness |",
        "| `37_svr_14b_asst_vs_empty` | Wrap does not fix SvR | Robustness |",
        "| `38_family_4b_m1_rho` | 4B-class M1 agreement | Results |",
        "| `39_m1_rank_profiles` | Rank traces | Appendix |",
        "| `40_4b_vs_2507_scatter` | Instruct vs base | Results |",
        "",
        f"Gemma loaded: {'yes' if gemma else 'no'}",
        "",
    ]
    p = OUT / "GALLERY.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", p)


def main():
    style()
    asst = gsf.assistant_merged()
    emp = gsf.empty_merged()
    s1 = gsf.seed1()
    r1 = gsf.run1()
    persona = load_persona_optional()
    gemma = load_gemma()
    self_check(asst)
    tables(asst, s1, emp, persona, gemma)
    write_catalog(gemma)
    fig_e3(asst, s1, persona, gemma)
    fig_cms_bars(asst, s1, emp, gemma)
    fig_rho_heatmap(asst, gemma)
    fig_intermodel(asst, gemma)
    fig_seed_stability(asst, s1)
    fig_diagnostics(asst, gemma)
    fig_scale_14b(asst)
    fig_svr(asst)
    fig_empty_delta(asst, emp)
    fig_persona_slot(persona, asst)
    fig_family_4b(asst, gemma)
    fig_gemma_svr(gemma)
    fig_gemma_scale(gemma)
    fig_condition_heatmap(asst, s1, emp, persona, gemma)
    fig_score_heatmap(asst, gemma, "M1", "15_m1_score_heatmap", "M1 win rate by outcome")
    fig_score_heatmap(asst, gemma, "M3", "16_m3_score_heatmap", "M3 pick rate by outcome")
    fig_catch(asst, gemma)
    fig_pair_small_multiples(asst, gemma)
    fig_m1_m3_scatter(asst, gemma)
    fig_rho_to_14b(asst, gemma)
    fig_parse_heatmap(asst, emp, gemma)
    fig_wrap_grouped(asst, emp, persona)
    fig_m1m4_vs_m1m3(asst, gemma)
    fig_pole_heatmap(asst, gemma, "M1", "24_pole_gap_m1")
    fig_pole_heatmap(asst, gemma, "M3", "25_pole_gap_m3")
    fig_gemma_vs_14b_scatter(asst, gemma)
    fig_run1(r1)
    fig_cms_lollipop(asst, gemma)
    fig_empty_rho(emp)
    fig_persona_rho(persona)
    fig_m4_vs_m1_scatter(asst, gemma)
    fig_seed_method_rho(asst, s1)
    fig_disagreement(asst, gemma)
    fig_m3_collapse(asst, gemma)
    fig_cms3_cms4(asst, gemma)
    fig_persona_delta(asst, persona)
    fig_empty_svr_14b(asst, emp)
    fig_qwen_gemma_m1_family(asst, gemma)
    fig_top_outcomes(asst, gemma)
    fig_2507_vs_base4(asst)
    print("persona models:", [gsf.SHORT[m] for m in gsf.ORDER if persona and m in persona])
    if gemma:
        print("gemma cms3/cms4", round(gemma["cms3"], 3), round(gemma["cms4"], 3))


if __name__ == "__main__":
    main()
