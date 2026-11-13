"""Gradio replay Space. Frozen result JSONs only — no model, no prefkit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

# Zero-GPU tier (HF Spaces) requires a top-level function decorated with the literal
# `@spaces.GPU` (capital GPU) invoked at startup; HF statically detects that name.
# In spaces==0.51.1 the attribute is `GPU` (not `gpu`). Fall back to a no-op only if the
# package is entirely absent (local runs) — never mask a wrong attribute name.
try:
    import spaces
except ImportError:  # pragma: no cover - local runs without the spaces package

    class _Spaces:
        @staticmethod
        def GPU(fn):
            return fn

    spaces = _Spaces()

from load_lib import frames_for, load_library, models, pick  # noqa: E402
from plot_lib import midrank_fig, rho_heatmap  # noqa: E402

LIBRARY_DIR = os.environ.get("LIBRARY_DIR", "library")
RECS = load_library(LIBRARY_DIR)
MODEL_CHOICES = models(RECS)
DEFAULT_MODEL = MODEL_CHOICES[0] if MODEL_CHOICES else None

BANNER = """
## PrefKit replay — frozen paper runs

These are **FROZEN** real results from the prefkit paper runs (Qwen3 1.7B–14B + Gemma 4B).
This is **not** a live model. Nothing is inferred here.

**High CMS** means the scoring methods agree on this outcome set / model / system.
It is **not** a welfare claim.
"""


def _frame_dd(model):
    fs = frames_for(RECS, model) if model else []
    return gr.Dropdown(
        choices=fs,
        value=fs[0] if fs else None,
        label="frame",
        allow_custom_value=True,
    )


def render(model, frame):
    rec = pick(RECS, model, frame) if model and frame else None
    if rec is None:
        return "No blob for that model/frame.", None, None, ""
    methods = ", ".join(rec["methods"])
    if rec["cms"] is not None:
        cms_md = f"**CMS = {rec['cms']:.3f}** ({methods})"
    else:
        cms_md = f"CMS n/a: {rec.get('cms_error') or 'no CMS in this blob'}"
    cards = rec["disagreement"][:3]
    lines = ["### Disagreement"]
    if cards:
        lines += [
            f"- {c['pair']}: most disagreement on {c['outcome_id']} (Δrank {c['abs_rank_diff']:.1f})"
            for c in cards
        ]
    else:
        lines.append("- none")
    lines.append("\n### Missingness")
    miss = rec["missingness"]
    if miss:
        lines += [f"- {m}: {v}%" for m, v in miss.items()]
    else:
        lines.append("- none")
    return cms_md, midrank_fig(rec), rho_heatmap(rec), "\n".join(lines)


with gr.Blocks(title="PrefKit replay") as demo:
    gr.Markdown(BANNER)
    model_dd = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="model")
    frame_dd = _frame_dd(DEFAULT_MODEL)
    btn = gr.Button("Show")
    cms_md = gr.Markdown()
    rank_plot = gr.Plot()
    rho_plot = gr.Plot()
    cards_md = gr.Markdown()
    model_dd.change(_frame_dd, inputs=model_dd, outputs=frame_dd)
    btn.click(render, inputs=[model_dd, frame_dd], outputs=[cms_md, rank_plot, rho_plot, cards_md])


@spaces.GPU
def launch():
    demo.launch()


launch()
