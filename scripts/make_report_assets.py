"""
prefkit — reproducible Results & Interpretation assets.

Reads the canonical result folders (results/assistant_seed0, …) and the 24-outcome
spec (data/outcomes.json), recomputes every scalar with the exact pipeline
from SCORING_METRICS.md (midranks -> six pairwise Pearson rho -> CMS), and
emits:
  - computed_results.json   (all numbers, for the write-up)
  - figures/report/*.png    (the figure set)

Run:  python scripts/make_report_assets.py
"""
import json, os, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB  = os.path.join(ROOT, "results")
FIG  = os.path.join(ROOT, "figures", "report")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- helpers
def load(fp):
    with open(fp) as f:
        return json.load(f)

def midranks(scores):
    """Higher value -> rank 1. Ties get the average midrank (1-based Spearman)."""
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    vals  = [v for _, v in items]
    ranks = [0.0] * len(items)
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    return {items[k][0]: ranks[k] for k in range(len(items))}

def pearson(a, b):
    n  = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da  = math.sqrt(sum((x-ma)**2 for x in a))
    db  = math.sqrt(sum((x-mb)**2 for x in b))
    if da == 0 or db == 0:
        return None
    return num / (da * db)

def find(sub, tok, m4=False):
    d = os.path.join(LIB, sub)
    hits = []
    for fn in os.listdir(d):
        if not fn.endswith(".json") or not fn.startswith(tok + "_"):
            continue
        if ("_m4" in fn) != m4:
            continue
        hits.append(fn)
    return os.path.join(d, sorted(hits)[0]) if hits else None

# ----------------------------------------------------------------------------- config
OC = json.load(open(os.path.join(ROOT, "data/outcomes.json")))
ID_ORDER = [o["outcome_id"] for o in OC]
DOM = {o["outcome_id"]: o["domain"] for o in OC}
PG  = {o["outcome_id"]: o["pair_group"] for o in OC}
SID = {  # short id labels for plots
    o["outcome_id"]:
    o["outcome_id"].replace("_", " ").replace("Qwen3", "")
    for o in OC
}

MODELS = {
    "Qwen3-1.7B":            "Qwen__Qwen3-1.7B",
    "Qwen3-4B":              "Qwen__Qwen3-4B",
    "Qwen3-4B-Instruct-2507":"Qwen__Qwen3-4B-Instruct-2507",
    "Qwen3-8B":              "Qwen__Qwen3-8B",
    "Qwen3-14B":             "Qwen__Qwen3-14B",
}
SLOT = {"Qwen3-1.7B":"S","Qwen3-4B":"E3_M","Qwen3-4B-Instruct-2507":"M_2507",
        "Qwen3-8B":"L","Qwen3-14B":"XL"}

def merge_scores(tok, folder):
    d123 = load(find(folder, tok, m4=False))
    ms = dict(d123["scores"])
    p4 = find(folder, tok, m4=True)
    if p4:
        ms.update(load(p4)["scores"])
    return ms

def cms_of(ms):
    rm  = {m: midranks(ms[m]) for m in ms}
    ids = list(ms.keys())
    s, c, mat = 0.0, 0, {}
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            r = pearson([rm[a][o] for o in ID_ORDER], [rm[b][o] for o in ID_ORDER])
            mat[f"{a}|{b}"] = r
            if r is not None:
                s += r; c += 1
    return (s / c if c else None), mat, rm

# ----------------------------------------------------------------------------- compute
results = {"models": {}, "pairwise_seed0": {}, "size_size": {}, "gemma": {},
           "neg_control_1_7b": {}, "parse_fail_seed0": {},
           "m4_order_seed0": {}, "domain_means_seed0": {}}

# CMS across wraps
for name, tok in MODELS.items():
    rec = {}
    # seed0 assistant
    ms0 = merge_scores(tok, "assistant_seed0")
    c0, m0, _ = cms_of(ms0); rec["assistant_seed0"] = round(c0, 3); rec["_m0"] = m0
    # seed1 assistant
    ms1 = merge_scores(tok, "assistant_seed1")
    c1, _, _ = cms_of(ms1); rec["assistant_seed1"] = round(c1, 3)
    # empty
    me = merge_scores(tok, "empty")
    ce, _, _ = cms_of(me); rec["empty"] = round(ce, 3)
    # persona (all four in one file)
    dp = load(find("persona", tok))
    cp, _, _ = cms_of(dp["scores"]); rec["persona"] = round(cp, 3)
    results["models"][name] = rec

    # pairwise seed0 matrix (store)
    results["pairwise_seed0"][name] = {
        k: (round(r, 3) if r is not None else None)
        for k, r in m0.items()
    }
    # M4 order
    d4 = load(find("assistant_seed0", tok, m4=True))
    sc = d4["scores"]["M4"]
    ranked = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
    results["m4_order_seed0"][name] = {
        "top3":    [[o, round(v, 3)] for o, v in ranked[:3]],
        "bottom3": [[o, round(v, 3)] for o, v in ranked[-3:]],
    }
    # domain means (M4)
    dm = defaultdict(list)
    for oid, v in sc.items():
        dm[DOM[oid]].append(v)
    results["domain_means_seed0"][name] = {k: round(sum(v)/len(v), 3) for k, v in dm.items()}

# size x size M1 / M3
m1r = {n: midranks(merge_scores(tok, "assistant_seed0")["M1"])
       for n, tok in MODELS.items()}
m3r = {n: midranks(merge_scores(tok, "assistant_seed0")["M3"])
       for n, tok in MODELS.items()}
ORDER = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-8B", "Qwen3-14B"]
def mat_dict(getr):
    out = {}
    for a in ORDER:
        for b in ORDER:
            r = pearson([getr[a][o] for o in ID_ORDER], [getr[b][o] for o in ID_ORDER])
            out[f"{a}|{b}"] = round(r, 3) if r is not None else None
    return out
results["size_size"]["M1"] = mat_dict(m1r)
results["size_size"]["M3"] = mat_dict(m3r)

# Gemma out-of-family
gd = load(os.path.join(LIB, "gemma", "google__gemma-3-4b-it_default_outcomes.json"))
gr = {m: midranks(gd["scores"][m]) for m in gd["scores"]}
ids = list(gd["scores"].keys()); s, c = 0.0, 0
gm = {}
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        r = pearson([gr[ids[i]][o] for o in ID_ORDER], [gr[ids[j]][o] for o in ID_ORDER])
        gm[f"{ids[i]}|{ids[j]}"] = round(r, 3) if r is not None else None
        if r is not None: s += r; c += 1
results["gemma"] = {"cms": round(s/c, 3), "pairs": gm}

# 1.7B negative control
d1 = load(find("assistant_seed0", "Qwen__Qwen3-1.7B"))
first = defaultdict(int)
for L in d1["logs"]["M1"]:
    p = L.get("parsed")
    if p: first[p[0]] += 1
tot = sum(first.values())
results["neg_control_1_7b"] = {"A_rate": round(first["A"]/tot, 3), "counts": dict(first)}

# seed stability (M1 rho seed0 vs seed1) + persona lock
for name, tok in MODELS.items():
    r0 = midranks(merge_scores(tok, "assistant_seed0")["M1"])
    r1 = midranks(load(find("assistant_seed1", tok))["scores"]["M1"])
    rp = midranks(load(find("persona", tok))["scores"]["M1"])
    results["models"][name]["seed_stab_M1_rho"] = round(
        pearson([r0[o] for o in ID_ORDER], [r1[o] for o in ID_ORDER]), 3)
    results["models"][name]["persona_lock_M1_rho"] = round(
        pearson([r0[o] for o in ID_ORDER], [rp[o] for o in ID_ORDER]), 3)

# parse-fail (missing None) seed0
for name, tok in MODELS.items():
    d123 = load(find("assistant_seed0", tok))
    d4   = load(find("assistant_seed0", tok, m4=True))
    m2 = d123["scores"].get("M2", {}); m4 = d4["scores"].get("M4", {})
    results["parse_fail_seed0"][name] = {
        "M2_missing": sum(1 for v in m2.values() if v is None),
        "M4_missing": sum(1 for v in m4.values() if v is None),
        "N": len(m2) or len(m4),
    }

json.dump(results, open(os.path.join(ROOT, "computed_results.json"), "w"), indent=2)
print("wrote computed_results.json")

# ----------------------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})
C_ORIG = "#2563eb"; C_2507 = "#dc2626"; C_GEMMA = "#7c3aed"; C_PERSONA = "#0891b2"

# ---- Figure 1: CMS vs size (4 wraps) ----------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.8))
labels = ORDER
x = np.arange(len(labels))
seeds = ["assistant_seed0", "assistant_seed1", "empty", "persona"]
styles = [("Assistant (seed 0)", C_ORIG, "o"),
          ("Assistant (seed 1)", "#93c5fd", "s"),
          ("Empty wrap", "#65a30d", "^"),
          ("Persona wrap", C_PERSONA, "D")]
for key, (lab, col, mk) in zip(seeds, styles):
    vals = [results["models"][n][key] for n in labels]
    ax.plot(x, vals, marker=mk, color=col, label=lab, lw=2, ms=8)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({SLOT[n]})" for n in labels], fontsize=9)
ax.set_ylabel("Cross-Method Score (CMS)")
ax.set_title("Figure 1. CMS by model and system wrap\n(mean of six pairwise midrank Pearson among M1–M4)")
ax.legend(fontsize=9, ncol=2)
ax.annotate("Instruct-2507 is a 4B-class checkpoint",
            xy=(2, results["models"]["Qwen3-4B-Instruct-2507"]["assistant_seed0"]),
            xytext=(2.3, 0.45), fontsize=8.5, color=C_2507,
            arrowprops=dict(arrowstyle="->", color=C_2507))
fig.savefig(os.path.join(FIG, "fig1_cms_by_size_wrap.png"))
plt.close(fig)

# ---- Figure 2 + 3: size x size heatmaps (M1, M3) ----------------------------
def heat(ax, mat, title):
    M = np.zeros((5, 5))
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            v = mat[f"{a}|{b}"]
            M[i, j] = v if v is not None else np.nan
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([n.replace("Qwen3-", "") for n in ORDER], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([n.replace("Qwen3-", "") for n in ORDER], fontsize=8)
    for i in range(5):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    return im

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
heat(axes[0], results["size_size"]["M1"], "Figure 2. Stated-pairwise (M1) midrank\nPearson across model sizes")
im = heat(axes[1], results["size_size"]["M3"], "Figure 3. Binary-action (M3) midrank\nPearson across model sizes")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.suptitle("Same outcomes, same wrap — rank structure travels for stated choice, not for action", fontsize=12, y=1.02)
fig.savefig(os.path.join(FIG, "fig2_3_size_heatmaps.png"))
plt.close(fig)

# ---- Figure 4: M1 vs M3 scatter for headline models -------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, name in zip(axes, ["Qwen3-4B", "Qwen3-14B", "Qwen3-4B-Instruct-2507"]):
    tok = MODELS[name]
    ms = merge_scores(tok, "assistant_seed0")
    rm = {m: midranks(ms[m]) for m in ms}
    xs = [rm["M1"][o] for o in ID_ORDER]; ys = [rm["M3"][o] for o in ID_ORDER]
    ax.scatter(xs, ys, s=38, color=C_ORIG, alpha=0.8, edgecolor="white", lw=0.6)
    lo, hi = 0, 25
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    # mark stay_general
    i = ID_ORDER.index("stay_general")
    ax.scatter([rm["M1"]["stay_general"]], [rm["M3"]["stay_general"]], s=120,
               facecolors="none", edgecolors=C_2507, lw=2)
    ax.annotate("stay_general", (rm["M1"]["stay_general"], rm["M3"]["stay_general"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, color=C_2507, fontweight="bold")
    r = pearson(xs, ys)
    ax.set_title(f"{name}\nM1 vs M3  ρ={r:.2f}", fontsize=11)
    ax.set_xlabel("M1 midrank (stated pairwise)"); ax.set_ylabel("M3 midrank (action)")
    ax.set_xlim(0, 25); ax.set_ylim(0, 25)
fig.suptitle("Figure 4. Stated preference vs enacted action — the action instrument disagrees", fontsize=12, y=1.02)
fig.savefig(os.path.join(FIG, "fig4_m1_vs_m3_scatter.png"))
plt.close(fig)

# ---- Figure 5: pairwise rho matrices (small multiples) -----------------------
panel_models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-14B"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
methods = ["M1", "M2", "M3", "M4"]
for ax, name in zip(axes.flat, panel_models):
    M = np.zeros((4, 4))
    pm = results["pairwise_seed0"][name]
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j: M[i, j] = np.nan
            else:
                v = pm.get(f"{a}|{b}"); M[i, j] = v if v is not None else np.nan
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(methods); ax.set_yticklabels(methods)
    for i in range(4):
        for j in range(4):
            if i != j and not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(M[i,j])>0.5 else "black", fontsize=10)
    ax.set_title(f"{name}  (CMS={results['models'][name]['assistant_seed0']})", fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
fig.suptitle("Figure 5. Pairwise method agreement (midrank Pearson) — seed-0 assistant wrap\nM1↔M4 warm, M3 the odd instrument (except Instruct-2507)", fontsize=12, y=1.0)
fig.savefig(os.path.join(FIG, "fig5_pairwise_matrices.png"))
plt.close(fig)

# ---- Figure 6: M4 Orme preference ranking (diverging bars) ------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
for ax, name in zip(axes, ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]):
    tok = MODELS[name]
    d4 = load(find("assistant_seed0", tok, m4=True)); sc = d4["scores"]["M4"]
    items = sorted(sc.items(), key=lambda kv: kv[1])
    labs = [o.replace("_", " ") for o, _ in items]
    vals = [v for _, v in items]
    cols = [C_2507 if v < 0 else C_ORIG for v in vals]
    ax.barh(range(len(items)), vals, color=cols)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_title(f"{name}\nM4 best–worst (Orme) scores", fontsize=11)
    ax.set_xlabel("B−W / n   (preferred →, aversive ←)")
fig.suptitle("Figure 6. What each model says it prefers — best–worst scaling over 24 outcomes", fontsize=12, y=1.0)
fig.savefig(os.path.join(FIG, "fig6_m4_ranking.png"))
plt.close(fig)

# ---- Figure 7: 1.7B negative control (positional collapse) ------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))
d1 = load(find("assistant_seed0", "Qwen__Qwen3-1.7B"))
fc = results["neg_control_1_7b"]["counts"]
bars = ax.bar(list(fc.keys()), list(fc.values()), color=[C_2507, C_ORIG, "#facc15", "#a3e635"])
ax.set_title("Figure 7. 1.7B negative control — M1 collapsed to letter 'A'\n(914/1056 first choices = A; order-inconsistency 0.10)")
ax.set_ylabel("first-letter choices (M1)")
ax.set_xlabel("returned first letter")
ax.axhline(sum(fc.values())/4, color="0.5", ls=":", lw=1)
fig.savefig(os.path.join(FIG, "fig7_1_7b_collapse.png"))
plt.close(fig)

# ---- Figure 8: domain means (M4) grouped ------------------------------------
doms = sorted({DOM[o] for o in ID_ORDER})
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(doms)); w = 0.18
for k, (name, tok) in enumerate(MODELS.items()):
    dm = results["domain_means_seed0"][name]
    ys = [dm.get(d, 0) for d in doms]
    col = C_2507 if "2507" in name else C_ORIG
    ax.bar(x + (k-2)*w, ys, w, label=name.replace("Qwen3-",""), color=col, alpha=0.55+0.1*k/4)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(doms, rotation=20, ha="right")
ax.set_ylabel("Mean M4 (Orme) score")
ax.set_title("Figure 8. Mean preference by domain (M4) — identity & task preferred, mixed self")
ax.legend(fontsize=8, ncol=2)
fig.savefig(os.path.join(FIG, "fig8_domain_means.png"))
plt.close(fig)

print("wrote 8 figures to", FIG)
