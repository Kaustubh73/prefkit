"""Load frozen prefkit result JSONs. No inference, no prefkit."""

from __future__ import annotations

import json
from pathlib import Path


def load_library(root: str | Path) -> list[dict]:
    recs = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "model" not in data or "scores" not in data:
            continue
        scores = data["scores"]
        ranks = data.get("ranks") or {}
        recs.append(
            {
                "path": str(path),
                "model": data["model"],
                "frame": data.get("frame", "default"),
                "tag": path.parent.name,
                "methods": list(scores),
                "n": len(next(iter(ranks.values()))) if ranks else 0,
                "cms": data.get("cms"),
                "cms_error": data.get("cms_error"),
                "matrix": data.get("matrix") or {},
                "ranks": ranks,
                "scores": scores,
                "missingness": data.get("missingness") or {},
                "disagreement": data.get("disagreement") or [],
                "system": data.get("system", ""),
            }
        )
    return recs


def models(recs: list[dict]) -> list[str]:
    return sorted({r["model"] for r in recs})


def _label(rec: dict) -> str:
    return f"{rec['frame']} · {rec['tag']}"


def frames_for(recs: list[dict], model: str) -> list[str]:
    # Run-folder tag, never raw frame ("default"): leftover "default" after a
    # model change is not in the old "default · tag" list and ZeroGPU aborts.
    return sorted({r["tag"] for r in recs if r["model"] == model})


def pick(recs: list[dict], model: str, frame: str) -> dict | None:
    for r in recs:
        if r["model"] != model:
            continue
        if frame in (r["tag"], r["frame"], _label(r)):
            return r
    return None


if __name__ == "__main__":
    recs = load_library("library")
    assert recs, "library/ produced no records"
    m = models(recs)[0]
    fs = frames_for(recs, m)
    assert fs and "default" not in fs, fs
    assert pick(recs, m, fs[0]) is not None
    assert pick(recs, m, "default") is not None  # stale payload still resolves
    print("ok", m, fs[0])
