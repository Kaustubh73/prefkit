#!/usr/bin/env python3
"""Appendix A4 figure: Gemma-3-4B-it method-pair agreement (6 Pearson rho bars),
diverging style to match A1. Source: computed_results.json -> gemma.pairs.
"""
import json, os
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

C = json.load(open("computed_results.json"))
pairs = C["gemma"]["pairs"]           # M1|M2, M1|M3, M1|M4, M2|M3, M2|M4, M3|M4
cms = C["gemma"]["cms"]

INK="#1f2933"; MUT="#52606d"; POS="#2f855a"; NEG="#c53030"
os.makedirs("figures/appendix", exist_ok=True)

items = list(pairs.items())           # (label, rho)
labels = [k for k,_ in items]
vals = [v for _,v in items]

fig, ax = plt.subplots(figsize=(7.4, 3.4))
y = np.arange(len(vals))
colors = [POS if v >= 0 else NEG for v in vals]
ax.barh(y, vals, color=colors, height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, color=MUT)
ax.axvline(0, color=INK, lw=0.9)
ax.set_xlim(-0.6, 0.95)
ax.set_xlabel("Pearson rho between methods", color=MUT, fontsize=9)
for yi, v in zip(y, vals):
    ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT)
ax.set_facecolor("white")
ax.grid(axis="x", color="#e4e7eb", lw=0.6)
ax.set_title(f"Appendix A4 · Gemma-3-4B-it (out-of-family) — method-pair agreement\nfour-method CMS = {cms:.3f} (appendix only, not in size line)",
             fontsize=11.5, color=INK, fontweight="bold")

fig.tight_layout()
out = "figures/appendix/A4_gemma_pairs.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
