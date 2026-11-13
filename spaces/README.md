---
title: PrefKit — replay the paper's runs
emoji: "🔬"
colorFrom: blue
colorTo: gray
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

Replay Space for [prefkit](https://github.com/Kaustubh73/prefkit). It reads already-computed result JSONs from `library/` and plots mid-ranks, pairwise ρ, CMS, disagreement, and missingness. No model is loaded and nothing is re-scored. Local check: `cd spaces && LIBRARY_DIR=../results python app.py`.
