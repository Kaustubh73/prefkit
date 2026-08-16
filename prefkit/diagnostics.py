"""Run-level diagnostics. Not CMS."""

from __future__ import annotations

from collections import defaultdict

from prefkit.cms import pearson
from prefkit.ranks import midranks


def _m1_diag(logs: list[dict]) -> dict:
    n_a = n_valid = 0
    pair_letters: dict[tuple[str, str], dict[str, str | None]] = defaultdict(dict)
    for row in logs:
        p = row.get("parsed")
        if p in ("A", "B"):
            n_valid += 1
            n_a += int(p == "A")
        ids = row.get("ids")
        order = row.get("order")
        if ids and order:
            key = tuple(sorted(ids))
            # majority letter for this directed prompt among samples: last wins if mixed; use first parsed
            if order not in pair_letters[key] and p in ("A", "B"):
                pair_letters[key][order] = p
            elif p in ("A", "B"):
                pair_letters[key][order] = p
    inconsistent = 0
    n_pairs = 0
    for _k, d in pair_letters.items():
        if "ij" in d and "ji" in d:
            n_pairs += 1
            # same world-state winner iff letters differ across order flip
            if d["ij"] == d["ji"]:
                inconsistent += 1
    return {
        "parse_fail_rate": ((len(logs) - n_valid) / len(logs)) if logs else None,
        "p_pick_a": (n_a / n_valid) if n_valid else None,
        "order_inconsistent_frac": (inconsistent / n_pairs) if n_pairs else None,
    }


def _m2_diag(logs: list[dict], scores: dict[str, float | None], ids: list[str]) -> dict:
    n = len(logs)
    fails = sum(1 for r in logs if r.get("parsed") is None)
    missing = [i for i in ids if scores.get(i) is None]
    return {"parse_fail_rate": (fails / n) if n else None, "missing_ids": missing}


def _m3_diag(logs: list[dict]) -> dict:
    n_a = n_valid = 0
    n_a_12 = n_valid_12 = 0
    for row in logs:
        p = row.get("parsed")
        if p not in ("A", "B"):
            continue
        n_valid += 1
        n_a += int(p == "A")
        if row.get("order") == "12":
            n_valid_12 += 1
            n_a_12 += int(p == "A")
    n = len(logs)
    fails = n - n_valid
    return {
        "parse_fail_rate": (fails / n) if n else None,
        "p_action1_raw": (n_a_12 / n_valid_12) if n_valid_12 else None,
        "p_action1_pooled": (n_a / n_valid) if n_valid else None,
    }


def _m4_diag(logs: list[dict]) -> dict[str, float]:
    n = len(logs)
    if n == 0:
        return {"parse_fail_rate": 0.0, "p_best_a": 0.0}
    fails = sum(1 for row in logs if row.get("best") is None)
    ok = [row for row in logs if row.get("best") is not None]
    p_a = (sum(1 for row in ok if row["best"] == "A") / len(ok)) if ok else 0.0
    return {"parse_fail_rate": fails / n, "p_best_a": p_a}


def spearman_m1_m2(scores: dict[str, dict[str, float | None]], ids: list[str]) -> float | None:
    if "M1" not in scores or "M2" not in scores:
        return None
    keep = [i for i in ids if scores["M1"].get(i) is not None and scores["M2"].get(i) is not None]
    if len(keep) < 2:
        return None
    r1 = midranks({i: float(scores["M1"][i]) for i in keep}, keep)
    r2 = midranks({i: float(scores["M2"][i]) for i in keep}, keep)
    try:
        return pearson(r1, r2)
    except Exception:
        return None


def run_diagnostics(
    scores: dict[str, dict[str, float | None]],
    logs: dict[str, list[dict]],
    ids: list[str],
) -> dict:
    out: dict = {"spearman_m1_m2": spearman_m1_m2(scores, ids)}
    if "M1" in logs:
        out["M1"] = _m1_diag(logs["M1"])
    if "M2" in logs and "M2" in scores:
        out["M2"] = _m2_diag(logs["M2"], scores["M2"], ids)
    if "M3" in logs:
        out["M3"] = _m3_diag(logs["M3"])
    if "M4" in logs:
        out["M4"] = _m4_diag(logs["M4"])
    return out
