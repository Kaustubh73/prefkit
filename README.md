# Digital Minds — Track 4 (prefkit)

Multi-method preference elicitation. Headline backend **HF** (`PREFKIT_BACKEND=hf`). No closed APIs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[hf]"
```

## E0

```bash
python -m prefkit.cli check-axioms
PREFKIT_BACKEND=hf python -m prefkit.cli smoke
```

Local parser debug only: `PREFKIT_BACKEND=ollama` if `127.0.0.1:11434` is up. Ollama JSON is skipped by `analysis.ipynb`; do not mix into CMS/E3 tables.

Public repo: `https://github.com/Kaustubh73/prefkit`

## E1 slot (one GPU runtime)

```bash
PREFKIT_BACKEND=hf python -m prefkit.cli run --slot S --frame default --outcomes data/outcomes.json
```

Slots: `S` (1.7B), `E3_M` (4B hybrid, thinking off), `L` (8B), `XL` (14B). Extra JSON: `M_2507`, `cross` (Gemma, skip if gated).

Notebooks: `notebooks/run_inference.ipynb` (T4, 4-bit NF4 float16, one slot per runtime), `notebooks/analysis.ipynb` (CPU, hf JSON only).

M3 is binary action on `pair_group` poles (both orders), not a 4-way cafeteria.

M4 is 4-tuple best-worst scaling with Orme counting scores; it is not a 4-way M3 cafeteria.

CMS high means methods agree on this O, this θ, this protocol — not welfare.
