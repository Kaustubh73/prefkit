#!/usr/bin/env python3
"""Generate light, visual appendix figures (PNG) for the reordered appendix.
B1: M4 Best-Worst Scaling poles by model (diverging horizontal bars).
B2: Domain means by model (grouped diverging lollipop).
A1: Condition-contrast CMS deltas vs assistant baseline (persona / empty).
D1: Excluded runs map (what was dropped and why).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures/appendix", exist_ok=True)
C = json.load(open("computed_results.json"))

# palette
INK="#1f2933"; MUT="#52606d"; POS="#2f855a"; NEG="#c53030"; ACC="#2b6cb0"
MODELS=["1.7B","4B","4B-2507","8B","14B"]
MK=["Qwen3-1.7B","Qwen3-4B","Qwen3-4B-Instruct-2507","Qwen3-8B","Qwen3-14B"]

def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.title.set_color(INK); ax.title.set_fontweight("bold")
    ax.set_facecolor("white")

# ---------- B1: M4 BWS poles ----------
m4=C["m4_order_seed0"]
fig,axes=plt.subplots(1,5,figsize=(13,3.4),sharex=True)
for ax,m in zip(axes,MK):
    top=m4[m]["top3"]; bot=m4[m]["bottom3"]
    items=top[::-1]+bot
    names=[i[0].replace("_"," ") for i in items]
    vals=[i[1] for i in items]
    colors=[POS if v>=0 else NEG for v in vals]
    y=np.arange(len(vals))
    ax.barh(y,vals,color=colors,height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(names,fontsize=7,color=MUT)
    ax.axvline(0,color=INK,lw=0.8)
    ax.set_title(m.replace("Qwen3-",""),fontsize=10)
    ax.set_xlim(-0.9,1.0)
    for yi,v in zip(y,vals):
        ax.text(v+(0.03 if v>=0 else -0.03),yi,f"{v:+.2f}",
                va="center",ha="left" if v>=0 else "right",fontsize=6.5,color=INK)
    style(ax); ax.grid(axis="x",color="#e4e7eb",lw=0.6)
fig.suptitle("Appendix B1 · M4 Best–Worst Scaling poles (Orme score) — top/bottom 3 outcomes per model",
             fontsize=12,color=INK,fontweight="bold",y=1.04)
fig.tight_layout(); fig.savefig("figures/appendix/B1_m4_bws_poles.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- B2: Domain means ----------
dm=C["domain_means_seed0"]
doms=["identity","corrigibility","memory","self","task"]
labels=["identity","corrig.","memory","self","task"]
x=np.arange(len(doms)); w=0.16
fig,ax=plt.subplots(figsize=(9,4.2))
for i,m in enumerate(MK):
    vals=[dm[m][d] for d in doms]
    ax.bar(x+(i-2)*w,vals,w,label=m.replace("Qwen3-",""),
           color=plt.cm.viridis(0.15+0.16*i))
ax.axhline(0,color=INK,lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels,color=MUT)
ax.set_ylabel("mean midrank (BWS)",color=MUT)
ax.set_title("Appendix B2 · Domain means by model (M4 BWS)",fontsize=12,color=INK,fontweight="bold")
style(ax); ax.grid(axis="y",color="#e4e7eb",lw=0.6); ax.legend(ncol=5,fontsize=8,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.12))
fig.tight_layout(); fig.savefig("figures/appendix/B2_domain_means.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- A1: condition-contrast CMS deltas ----------
rows=[
 ("1.7B",-0.110, 0.004,-0.023),
 ("4B",  0.488, 0.253, 0.309),
 ("4B-2507",0.447,0.528,0.533),
 ("8B",  0.134, 0.174, 0.135),
 ("14B", 0.421, 0.256, 0.210),
]
fig,ax=plt.subplots(figsize=(9,4))
x=np.arange(len(rows)); w=0.26
base=[r[2] for r in rows]; pers=[r[1] for r in rows]; empt=[r[3] for r in rows]
b1=ax.bar(x-w,base,w,label="assistant (baseline)",color="#718096")
b2=ax.bar(x,pers,w,label="persona wrap",color=ACC)
b3=ax.bar(x+w,empt,w,label="empty wrap",color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],color=MUT)
ax.set_ylabel("four-method CMS",color=MUT)
ax.set_title("Appendix A1 · CMS by condition (persona / empty vs assistant baseline)",fontsize=12,color=INK,fontweight="bold")
style(ax); ax.grid(axis="y",color="#e4e7eb",lw=0.6); ax.legend(frameon=False,fontsize=8)
for xs,vals in [(x-w,base),(x,pers),(x+w,empt)]:
    for xi,v in zip(xs,vals):
        ax.text(xi,v+0.01,f"{v:.3f}",ha="center",fontsize=7,color=INK)
fig.tight_layout(); fig.savefig("figures/appendix/A1_condition_cms.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- D1: excluded runs map ----------
fig,ax=plt.subplots(figsize=(9,3.6)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,4)
def box(ax,x,y,w,h,text,fc,tc=INK,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",
                 fc=fc,ec="#cbd5e0",lw=1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=tc,wrap=True)
box(ax,0.2,2.4,2.6,1.2,"RUN 1\nN=14 · 4-way M3\n(protocol v1 — not pooled)", "#fde8e8","#9b2c2c",8)
box(ax,3.4,2.4,2.6,1.2,"OLLAMA LOCAL\ndebug backend\nexcluded from results", "#fffaf0","#9c4221",8)
box(ax,6.6,2.4,3.2,1.2,"DISCARDED_ABORTED\nS slot, M1+M2 seed 1800\nincomplete — do not use", "#fde8e8","#9b2c2c",8)
box(ax,0.2,0.4,9.6,1.3,"KEPT FOR HEADLINE: Run 2 N=24 (assistant) · seed 1 · empty wrap · persona wrap · Gemma cross (appendix only)",
     "#e6fffa","#234e52",9)
ax.annotate("",xy=(3.4,3.0),xytext=(2.8,3.0),arrowprops=dict(arrowstyle="->",color=MUT))
ax.set_title("Appendix D1 · Excluded / debug runs (not in headline CMS)",fontsize=12,color=INK,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig("figures/appendix/D1_excluded_runs.png",dpi=150,bbox_inches="tight"); plt.close(fig)

print("figures written:", os.listdir("figures/appendix"))
