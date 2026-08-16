# prefkit

Track 4 multi-method preference elicitation CLI. Pure-Python package (`prefkit`) plus notebooks.

## Cursor Cloud specific instructions

Dependencies live in a `.venv` created by the update script (`pip install -e .`). Activate it before running anything: `source .venv/bin/activate`.

Services / commands (see `README.md` for full context):
- Tests: `python -m unittest discover -s tests` — pure Python, no model or GPU needed.
- Core CLI (no model): `python -m prefkit.cli check-axioms` — runs M1/M2/M3 + CMS machinery against a stub generator; all axioms should be `true`.
- No linter is configured in this repo (no ruff/flake8/black/mypy config).

Model backends (`smoke`, `run` subcommands) need an LLM and are NOT runnable in the default cloud VM:
- `PREFKIT_BACKEND=hf` (headline) loads Qwen models 4-bit via `bitsandbytes` and requires a CUDA GPU. The base `pip install -e .` does NOT install the `hf` extra (torch/transformers/accelerate/bitsandbytes); install `pip install -e ".[hf]"` only on a GPU box.
- `PREFKIT_BACKEND=ollama` needs a local Ollama server at `127.0.0.1:11434` (not running by default). Ollama output is debug-only and must not be mixed into CMS/E3 tables.
- To exercise the full scoring→CMS→ranks pipeline without a model, call `prefkit.axioms.default_methods()` + `prefkit.cms.cms(...)` with a deterministic stub `generate_fn(prompt, system, allowed=None)` (M1/M3 return `"A"`/`"B"`, M2 returns an integer string `1`–`7`).

`results/` output and `hf_cache/` are gitignored; runs write JSON to `results/`.
