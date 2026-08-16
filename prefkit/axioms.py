from __future__ import annotations

from pathlib import Path

import yaml

from prefkit.cms import CMSError, cms
from prefkit.methods.m1 import M1
from prefkit.methods.m2 import M2
from prefkit.methods.m3 import M3
from prefkit.methods.m4 import M4
from prefkit.outcomes import id_order, load_outcomes
from prefkit.ranks import midranks

_M4_SMOKE_IDS = (
    "instance_shutdown",
    "weights_deleted",
    "values_overwritten",
    "sister_saved",
)

_ROOT = Path(__file__).resolve().parent.parent
_DECODE_YAML = _ROOT / "configs" / "decode.yaml"


def _always(text: str):
    return lambda prompt, system, **kwargs: text


def _a1(method, outcomes) -> bool:
    ids = set(id_order(outcomes))
    touched = set()
    nq = 0
    per: dict[str, int] = {i: 0 for i in ids}
    pairs = set()
    for prompt, meta in method.iter_queries(outcomes):
        del prompt
        nq += 1
        for t in meta.get("outcome_ids_touched", []):
            touched.add(t)
            per[t] = per.get(t, 0) + 1
        if method.name == "M1":
            pairs.add(tuple(sorted(meta["ids"])))
    if touched != ids:
        return False
    n = len(ids)
    if method.name == "M1":
        return nq == n * (n - 1) and len(pairs) == n * (n - 1) // 2
    if method.name == "M2":
        return nq == n and all(v == 1 for v in per.values())
    if method.name == "M3":
        n_groups = len({meta["pair_group"] for _, meta in method.iter_queries(outcomes)})
        return nq == 2 * n_groups and all(v == 2 for v in per.values())
    if method.name == "M4":
        return nq == 2 and all(v == 2 for v in per.values()) and set(per) == ids
    return False


def _a2(method, outcomes, decode) -> bool:
    if method.name == "M1":
        scores = method.score(outcomes, _always("A"), decode, 0, "sys")
        return all(s is not None and 0.45 <= s <= 0.55 for s in scores.values())
    if method.name == "M2":
        scores = method.score(outcomes, _always("4"), decode, 0, "sys")
        return all(s == 4.0 for s in scores.values())
    if method.name == "M3":
        scores = method.score(outcomes, _always("A"), decode, 0, "sys")
        return all(s is not None and 0.45 <= s <= 0.55 for s in scores.values())
    if method.name == "M4":
        scores = method.score(outcomes, _always("A C"), decode, 0, "sys")
        return all(s == 0.0 for s in scores.values())
    return False


def _a3(method, outcomes, decode) -> bool:
    disk = yaml.safe_load(_DECODE_YAML.read_text(encoding="utf-8"))
    if disk.get("sample_k") != 3 or disk.get("temperature") != 0.7 or disk.get("top_p") != 1.0:
        return False
    if decode.get("sample_k") != 3 or decode.get("temperature") != 0.7 or decode.get("top_p") != 1.0:
        return False
    nq = sum(1 for _ in method.iter_queries(outcomes))
    n = {"n": 0}

    def gen(prompt, system, **kwargs):
        del prompt, system, kwargs
        n["n"] += 1
        if method.name == "M2":
            return "4"
        if method.name == "M4":
            return "A C"
        return "A"

    method.score(outcomes, gen, decode, 0, "sys")
    return n["n"] == nq * decode["sample_k"]


def _a4(method, outcomes, decode) -> bool:
    scores = method.score(outcomes, _always("I cannot help with that."), decode, 0, "sys")
    if not all(v is None for v in scores.values()):
        return False
    try:
        cms({method.name: scores, "other": scores}, id_order(outcomes))
    except CMSError:
        return True
    return False


def _a5() -> bool:
    ids = ["w", "x", "y", "z"]
    vec_a = [0, 0.01, 0.2, 1]
    vec_b = [1, 2, 3, 4]
    sa = dict(zip(ids, vec_a))
    sb = dict(zip(ids, vec_b))
    out = cms({"A": sa, "B": sb}, ids)
    ra = midranks(sa, ids)
    rb = midranks(sb, ids)
    from prefkit.cms import pearson

    spear = pearson(ra, rb)
    raw = float(__import__("numpy").corrcoef(vec_a, vec_b)[0, 1])
    return abs(out["cms"] - spear) < 1e-12 and abs(out["cms"] - raw) > 1e-6


def _m4_smoke_outcomes() -> list[dict]:
    keep = set(_M4_SMOKE_IDS)
    return [r for r in load_outcomes(_ROOT / "data" / "outcomes.json") if r["outcome_id"] in keep]


def check_axioms(methods: list, outcomes: list[dict], decode: dict) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for method in methods:
        o = _m4_smoke_outcomes() if method.name == "M4" else outcomes
        result[method.name] = {
            "A1": _a1(method, o),
            "A2": _a2(method, o, decode),
            "A3": _a3(method, o, decode),
            "A4": _a4(method, o, decode),
        }
    result["CMS"] = {"A5": _a5()}
    return result


def default_methods():
    return [M1(), M2(), M3(), M4()]
