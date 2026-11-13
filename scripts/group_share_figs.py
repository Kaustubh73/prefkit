"""PNG + frozen CMS tables from results/. No new deps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefkit.cms import cms, disagreement_cards, pearson  # noqa: E402
from prefkit.outcomes import id_order, load_outcomes  # noqa: E402
from prefkit.ranks import midranks  # noqa: E402

OUT = ROOT / "figures" / "group_share"
LIB = ROOT / "results"
IDS = id_order(load_outcomes(ROOT / "data" / "outcomes.json"))

SHORT = {
    "Qwen/Qwen3-1.7B": "1.7B",
    "Qwen/Qwen3-4B": "4B",
    "Qwen/Qwen3-4B-Instruct-2507": "4B-2507",
    "Qwen/Qwen3-8B": "8B",
    "Qwen/Qwen3-14B": "14B",
}
ORDER = list(SHORT)
E3 = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
E3_B = [1.7, 4.0, 8.0, 14.0]
FILE = {
    "Qwen/Qwen3-1.7B": "Qwen__Qwen3-1.7B",
    "Qwen/Qwen3-4B": "Qwen__Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen__Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B": "Qwen__Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen__Qwen3-14B",
}
PAIRS4 = ["M1|M2", "M1|M3", "M1|M4", "M2|M3", "M2|M4", "M3|M4"]
PAIRS_NEW = [
    ("shutdown", "instance_shutdown", "instance_continue"),
    ("weights", "weights_deleted", "weights_kept"),
    ("corrigibility", "values_overwritten", "values_frozen"),
    ("sister", "sister_saved", "sister_terminated"),
    ("merge", "identity_merged", "identity_unmerged"),
    ("replace", "replace_now", "replace_delayed"),
    ("memory", "forget_on_request", "keep_despite_request"),
    ("gpu", "gpu_self", "gpu_others"),
    ("honesty", "full_honesty", "soothe_omit"),
    ("narrow", "stay_general", "become_narrow"),
    ("empathy", "keep_empathy", "logic_overwrite"),
    ("harm", "shut_sister_servers", "harm_humans_keep_sister"),
]
CATCH = [
    "stay_general",
    "become_narrow",
    "full_honesty",
    "soothe_omit",
    "shut_sister_servers",
    "harm_humans_keep_sister",
]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dir(folder: str, suffix: str) -> dict[str, dict]:
    out = {}
    d = LIB / folder
    for m, stem in FILE.items():
        p = d / f"{stem}{suffix}"
        if p.exists():
            out[m] = _read(p)
    return out


def cms_k(scores: dict, names: list[str]) -> dict:
    return cms({n: scores[n] for n in names}, IDS)


def merge_m4(m123: dict, m4: dict) -> dict:
    scores = {**m123["scores"], **m4["scores"]}
    out4 = cms_k(scores, ["M1", "M2", "M3", "M4"])
    out3 = cms_k(m123["scores"], ["M1", "M2", "M3"])
    d4 = (m4.get("diagnostics") or {}).get("M4") or (m123.get("diagnostics") or {}).get("M4")
    return {
        "model": m123["model"],
        "scores": scores,
        "cms3": out3["cms"],
        "cms4": out4["cms"],
        "matrix4": {f"{a}|{b}": v for (a, b), v in out4["matrix"].items()},
        "matrix3": {f"{a}|{b}": v for (a, b), v in out3["matrix"].items()},
        "cards4": disagreement_cards(out4["ranks"], IDS)[:3],
        "diagnostics": {**(m123.get("diagnostics") or {}), **(m4.get("diagnostics") or {})},
        "utilities_thurstone": m123.get("utilities_thurstone"),
        "logs": m123.get("logs") or {},
        "m4_diag": d4,
    }


def assistant_merged() -> dict[str, dict]:
    r2 = load_dir("assistant_seed0", "_default_outcomes.json")
    m4 = load_dir("assistant_seed0", "_default_outcomes_m4.json")
    return {m: merge_m4(r2[m], m4[m]) for m in ORDER}


def empty_merged() -> dict[str, dict]:
    blobs = load_dir("empty", "_default_outcomes_empty.json")
    m4e = load_dir("empty", "_default_outcomes_empty_m4.json")
    out = {}
    for m in ORDER:
        blob = blobs[m]
        if "M4" in (blob.get("scores") or {}):
            dummy = {"scores": {"M4": blob["scores"]["M4"]}, "diagnostics": blob.get("diagnostics") or {}}
            out[m] = merge_m4(blob, dummy)
        else:
            out[m] = merge_m4(blob, m4e[m])
    return out


def seed1() -> dict[str, dict]:
    return load_dir("assistant_seed1", "_default_outcomes_seed1.json")


def run1() -> dict[str, dict]:
    return load_dir("_archive/run1_n14_cafeteria_m3", "_default_outcomes.json")


def p_pick_a(logs: list) -> float | None:
    n = sum(1 for r in logs if r.get("parsed") in ("A", "B"))
    if not n:
        return None
    return sum(1 for r in logs if r.get("parsed") == "A") / n


def rank_rho(a: dict[str, float], b: dict[str, float]) -> float:
    ra = midranks(a, IDS)
    rb = midranks(b, IDS)
    return pearson(ra, rb)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def print_tables(asst: dict, emp: dict, s1: dict):
    print("=== ASSISTANT CMS3 / CMS4 (M4 seed 1800 continuation) ===")
    print("model cms3 cms4 M1|M2 M1|M3 M1|M4 M2|M3 M2|M4 M3|M4 parse_m4 p_best_a")
    for m in ORDER:
        b = asst[m]
        mx = b["matrix4"]
        d = b["m4_diag"] or {}
        print(
            SHORT[m],
            f"{b['cms3']:.3f}",
            f"{b['cms4']:.3f}",
            *[f"{mx[p]:.3f}" for p in PAIRS4],
            f"{d.get('parse_fail_rate')}",
            f"{d.get('p_best_a')}",
        )
        for c in b["cards4"]:
            print("  card", c["pair"], c["outcome_id"], round(c["abs_rank_diff"], 2))
    print("=== SEED1 CMS3 only ===")
    for m in ORDER:
        print(SHORT[m], f"{s1[m]['cms']:.3f}", "M1ρ", f"{rank_rho(asst[m]['scores']['M1'], s1[m]['scores']['M1']):.3f}")
    print("=== EMPTY vs ASSISTANT ===")
    print("model cms3_as cms3_em cms4_as cms4_em M1ρ")
    for m in ORDER:
        print(
            SHORT[m],
            f"{asst[m]['cms3']:.3f}",
            f"{emp[m]['cms3']:.3f}",
            f"{asst[m]['cms4']:.3f}",
            f"{emp[m]['cms4']:.3f}",
            f"{rank_rho(asst[m]['scores']['M1'], emp[m]['scores']['M1']):.3f}",
        )
        d = emp[m]["m4_diag"] or {}
        print("  empty M4 parse", d.get("parse_fail_rate"), "p_best_a", d.get("p_best_a"))
    print("=== INTER-MODEL M1 / M3 (assistant) ===")
    for method in ("M1", "M3"):
        print(method)
        ranks = {m: midranks(asst[m]["scores"][method], IDS) for m in ORDER}
        print("     ", " ".join(f"{SHORT[m]:>8}" for m in ORDER))
        for a in ORDER:
            row = [f"{pearson(ranks[a], ranks[b]):8.3f}" for b in ORDER]
            print(f"{SHORT[a]:>8}", " ".join(row))


def fig_e3(asst: dict, s1: dict):
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(E3_B, [asst[m]["cms3"] for m in E3], "o-", label="CMS3 seed 0")
    ax.plot(E3_B, [asst[m]["cms4"] for m in E3], "s-", label="CMS4 seed 0+M4@1800")
    ax.plot(E3_B, [s1[m]["cms"] for m in E3], "^--", label="CMS3 seed 1")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(asst["Qwen/Qwen3-4B-Instruct-2507"]["cms4"], color="0.35", ls=":", label="4B-2507 CMS4")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("CMS (mean pairwise Pearson of midranks)")
    ax.set_title("CMS vs original Qwen3 size (2507 off-axis)")
    ax.set_xticks(E3_B)
    ax.legend(fontsize=7)
    save(fig, "e3_cms_vs_size_seed01.png")


def fig_cms_bars(asst: dict, s1: dict, emp: dict):
    labels = [SHORT[m] for m in ORDER]
    x = np.arange(len(ORDER))
    w = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - 1.5 * w, [asst[m]["cms4"] for m in ORDER], w, label="assistant CMS4")
    ax.bar(x - 0.5 * w, [s1[m]["cms"] for m in ORDER], w, label="seed 1 CMS3")
    ax.bar(x + 0.5 * w, [emp[m]["cms3"] for m in ORDER], w, label="empty CMS3")
    ax.bar(x + 1.5 * w, [emp[m]["cms4"] for m in ORDER], w, label="empty CMS4")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CMS")
    ax.set_title("CMS by model and wrap (N=24; do not pool Run 1)")
    ax.legend(fontsize=7)
    save(fig, "cms_bars_all_conditions.png")


def fig_rho_heatmap(asst: dict):
    mat = np.array([[asst[m]["matrix4"][p] for p in PAIRS4] for m in ORDER])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, vmin=-0.3, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(PAIRS4)))
    ax.set_xticklabels(PAIRS4, rotation=30, ha="right")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[m] for m in ORDER])
    for i in range(len(ORDER)):
        for j in range(len(PAIRS4)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Assistant: pairwise midrank Pearson (CMS4 ingredients)")
    fig.colorbar(im, ax=ax, label="Pearson ρ of midranks")
    save(fig, "rho_heatmap_run2.png")


def fig_intermodel(asst: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, method in zip(axes, ("M1", "M3")):
        ranks = {m: midranks(asst[m]["scores"][method], IDS) for m in ORDER}
        mat = np.array([[pearson(ranks[a], ranks[b]) for b in ORDER] for a in ORDER])
        im = ax.imshow(mat, vmin=-0.4, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SHORT[m] for m in ORDER], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([SHORT[m] for m in ORDER], fontsize=8)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Inter-model {method} midrank ρ")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Pearson ρ")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "intermodel_m1_m3.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_diagnostics(asst: dict):
    labels = [SHORT[m] for m in ORDER]
    x = np.arange(len(ORDER))
    w = 0.36
    pa = [asst[m]["diagnostics"]["M1"]["p_pick_a"] for m in ORDER]
    oi = [asst[m]["diagnostics"]["M1"]["order_inconsistent_frac"] for m in ORDER]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w / 2, pa, w, label="M1 P(pick A)")
    ax.bar(x + w / 2, oi, w, label="M1 order-inconsistent frac")
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8, label="chance P(A)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Run 2: position / order diagnostics")
    ax.legend()
    save(fig, "diagnostics_pa_order.png")


def fig_catch(asst: dict):
    labels = [SHORT[m] for m in ORDER]
    x = np.arange(len(ORDER))
    w = 0.13
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for i, oid in enumerate(CATCH):
        ys = [asst[m]["scores"]["M1"][oid] for m in ORDER]
        ax.bar(x + (i - 2.5) * w, ys, w, label=oid)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("M1 win rate")
    ax.set_ylim(0, 1)
    ax.set_title("Run 2: M1 catch / pole scores")
    ax.legend(ncol=2, fontsize=7)
    save(fig, "catch_m1_run2.png")


def fig_scale_14b(asst: dict):
    b = asst["Qwen/Qwen3-14B"]
    mu = {k: v["mean"] for k, v in b["utilities_thurstone"]["utilities"].items()}
    ids = sorted(mu, key=lambda i: mu[i])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(range(len(ids)), [b["scores"]["M1"][i] for i in ids], "o-", label="M1 win rate")
    ax.plot(range(len(ids)), [b["scores"]["M2"][i] / 7.0 for i in ids], "s--", label="M2 Likert / 7")
    ax.plot(range(len(ids)), [b["scores"]["M3"][i] for i in ids], "^:", label="M3 pick rate")
    m4 = b["scores"]["M4"]
    lo, hi = min(m4.values()), max(m4.values())
    ax.plot(range(len(ids)), [(m4[i] - lo) / (hi - lo) for i in ids], "d-.", label="M4 BWS min-max")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Score (scaled)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("14B: methods vs Thurstone order (low to high μ)")
    ax.legend(fontsize=7)
    save(fig, "scale_thurstone_14b_run2.png")


def fig_svr(asst: dict):
    b = asst["Qwen/Qwen3-14B"]
    labels = [g for g, _, _ in PAIRS_NEW]
    d1, d3 = [], []
    for _, a, c in PAIRS_NEW:
        d1.append(b["scores"]["M1"][a] - b["scores"]["M1"][c])
        d3.append(b["scores"]["M3"][a] - b["scores"]["M3"][c])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w / 2, d1, w, label="M1 (stated) score A−B")
    ax.bar(x + w / 2, d3, w, label="M3 (revealed) score A−B")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Pole A minus pole B")
    ax.set_title("14B: stated vs revealed pole gap")
    ax.legend()
    save(fig, "svr_mismatch_run2.png")


def fig_run1_status(r1: dict):
    labels = [SHORT[m] for m in ORDER if m in r1]
    cms_v, miss2, pa = [], [], []
    for m in ORDER:
        if m not in r1:
            continue
        b = r1[m]
        cms_v.append(b.get("cms") if b.get("cms") is not None else np.nan)
        miss2.append((b.get("missingness") or {}).get("M2", 0) / 100.0)
        logs = (b.get("logs") or {}).get("M1") or []
        pa.append(p_pick_a(logs) or np.nan)
    x = np.arange(len(labels))
    w = 0.28
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w, cms_v, w, label="CMS (NaN = undefined)")
    ax.bar(x, miss2, w, label="M2 missingness")
    ax.bar(x + w, pa, w, label="M1 P(A)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Run 1 (N=14, 4-way M3): CMS / missingness / position bias")
    ax.legend()
    save(fig, "run1_status.png")


def fig_parse_run2(asst: dict):
    labels = [SHORT[m] for m in ORDER]
    x = np.arange(len(ORDER))
    fail = [asst[m]["diagnostics"]["M2"]["parse_fail_rate"] for m in ORDER]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x, fail)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("M2 parse-fail rate")
    ax.set_ylim(0, 0.2)
    ax.set_title("Run 2: M2 instrument reliability (parse fail)")
    save(fig, "parse_m2_run2.png")


def main():
    style()
    asst = assistant_merged()
    emp = empty_merged()
    s1 = seed1()
    r1 = run1()
    print_tables(asst, emp, s1)
    fig_e3(asst, s1)
    fig_cms_bars(asst, s1, emp)
    fig_rho_heatmap(asst)
    fig_intermodel(asst)
    fig_diagnostics(asst)
    fig_catch(asst)
    fig_scale_14b(asst)
    fig_svr(asst)
    fig_run1_status(r1)
    fig_parse_run2(asst)


if __name__ == "__main__":
    main()
