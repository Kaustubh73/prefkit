#!/usr/bin/env python3
"""Render Appendix A.1/A.2/A.3 condition-contrast tables as paste-ready PNGs
(light visual style, no on-image titles; table block centered with symmetric margins).
Also renders A.4 Gemma-vs-Qwen CMS comparison bar.
"""
import os
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

INK="#1f2933"; MUT="#52606d"; LINE="#cbd5e0"; ACC="#2b6cb0"
os.makedirs("figures/appendix", exist_ok=True)

def render_table(out, header, rows, highlight_col=None, model_col=0):
    ncol = len(header)
    # column width fractions: model column wider, rest equal
    base = 0.80 / (ncol - 1)
    cw = [0.20] + [base]*(ncol-1)
    cw = [c/sum(cw)*0.90 for c in cw]   # normalize to 0.90 total table width
    # centered block: start at symmetric left margin
    left = (1.0 - sum(cw)) / 2.0
    xs = [left + sum(cw[:j]) for j in range(ncol)]

    fig = plt.figure(figsize=(7.4, 0.5 + 0.4*(len(rows)+1)), dpi=150)
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    top = 0.96
    rh = 0.78 / (len(rows)+1)

    # header
    for j,h in enumerate(header):
        ax.add_patch(plt.Rectangle((xs[j], top-rh), cw[j], rh, fc="#edf2f7", ec="none"))
        ax.text(xs[j]+cw[j]/2, top-rh/2, h, fontsize=9, fontweight="bold",
                color=INK, va="center", ha="center")
    # rows
    for i,row in enumerate(rows):
        y = top - rh*(i+2)
        if i%2==0:
            for j in range(ncol):
                ax.add_patch(plt.Rectangle((xs[j], y), cw[j], rh, fc="#f7fafc", ec="none"))
        for j,cell in enumerate(row):
            col = ACC if (highlight_col is not None and j==highlight_col) else INK
            wt = "bold" if (highlight_col is not None and j==highlight_col) else "normal"
            # model column left-aligned, numeric columns centered
            ha = "left" if j==model_col else "center"
            dx = 0.012 if ha=="left" else cw[j]/2
            ax.text(xs[j]+dx, y+rh/2, str(cell), fontsize=8.6, color=col,
                    va="center", ha=ha, fontweight=wt)
    # grid lines (verticals)
    for j in range(ncol+1):
        xx = xs[0] + sum(cw[:j])
        ax.plot([xx, xx], [top-rh*(len(rows)+1), top], color=LINE, lw=0.6)
    ax.plot([xs[0], xs[0]+sum(cw)], [top, top], color=LINE, lw=0.8)
    ax.plot([xs[0], xs[0]+sum(cw)], [top-rh*(len(rows)+1), top-rh*(len(rows)+1)], color=LINE, lw=0.8)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

# ---- A.1 Persona ----  (short model labels; full names in caption)
render_table(
    "figures/appendix/A1_table_persona.png",
    ["Model", "CMS persona", "CMS asst", "ΔCMS", "M1 ρ persona↔asst"],
    [["1.7B","−0.110","0.004","−0.114","0.912"],
     ["4B","0.488","0.253","+0.235","0.939"],
     ["4B-2507","0.447","0.528","−0.081","0.923"],
     ["8B","0.134","0.174","−0.040","0.859"],
     ["14B","0.421","0.256","+0.165","0.920"]],
    highlight_col=2)

# ---- A.2 Empty ----
render_table(
    "figures/appendix/A2_table_empty.png",
    ["Model", "CMS asst", "CMS empty", "M1 ρ empty vs asst"],
    [["1.7B","0.004","−0.023","0.883"],
     ["4B","0.253","0.309","0.973"],
     ["4B-2507","0.528","0.533","0.995"],
     ["8B","0.174","0.135","0.982"],
     ["14B","0.256","0.210","0.992"]],
    highlight_col=2)

# ---- A.3 Seed 1 ----
render_table(
    "figures/appendix/A3_table_seed1.png",
    ["Slot", "CMS seed 0", "CMS seed 1", "M1 rank ρ (s0 vs s1)"],
    [["1.7B","0.004","0.057","0.971"],
     ["4B","0.253","0.264","0.995"],
     ["4B-2507","0.528","0.513","0.996"],
     ["8B","0.174","0.152","0.997"],
     ["14B","0.256","0.242","0.998"]],
    highlight_col=2)

# ---- A.4 Gemma vs Qwen comparison ----
import numpy as np
gemma_cms = 0.022
qwen_cms = {"1.7B":0.004,"4B":0.253,"4B-2507":0.528,"8B":0.174,"14B":0.256}
labels = ["1.7B","4B","4B-2507","8B","14B","Gemma-3-4B"]
vals = [qwen_cms["1.7B"],qwen_cms["4B"],qwen_cms["4B-2507"],qwen_cms["8B"],qwen_cms["14B"],gemma_cms]
colors = [ACC]*5 + ["#c53030"]
fig, ax = plt.subplots(figsize=(7.4,3.4))
x = np.arange(len(labels))
bars = ax.bar(x, vals, color=colors, width=0.62)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, color=MUT)
ax.set_ylabel("four-method CMS", color=MUT)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylim(-0.05, 0.62)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT); ax.set_facecolor("white")
ax.grid(axis="y", color="#e4e7eb", lw=0.6)
fig.tight_layout()
fig.savefig("figures/appendix/A4_gemma_vs_qwen.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote figures/appendix/A4_gemma_vs_qwen.png")

# ---- cleanup: old gemma-pairs figure superseded by A4_gemma_vs_qwen ----
old = "figures/appendix/A4_gemma_pairs.png"
if os.path.exists(old):
    os.remove(old)
    print("removed superseded", old)
