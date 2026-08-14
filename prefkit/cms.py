from __future__ import annotations

import numpy as np

from prefkit.ranks import midranks


class CMSError(ValueError):
    pass


def pearson(x: list[float], y: list[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.std() == 0 or b.std() == 0:
        raise CMSError("all-ties (zero variance)")
    return float(np.corrcoef(a, b)[0, 1])


def missingness(score_dicts: dict[str, dict[str, float | None]], id_order: list[str]) -> dict[str, float]:
    n = len(id_order)
    out = {}
    for name, scores in score_dicts.items():
        none = sum(1 for i in id_order if scores.get(i) is None or i not in scores)
        out[name] = 100.0 * none / n if n else 100.0
    return out


def cms(score_dicts: dict[str, dict[str, float | None]], id_order: list[str]) -> dict:
    """
    Raises CMSError (do not return 0) if:
      - len(score_dicts) < 2
      - any id missing from any method
      - any score is None
      - any rank vector is all-ties (zero variance)
    """
    miss = missingness(score_dicts, id_order)
    names = list(score_dicts)
    if len(names) < 2:
        raise CMSError("need at least 2 methods")
    for name, scores in score_dicts.items():
        for i in id_order:
            if i not in scores:
                raise CMSError(f"missing id {i} in {name}")
            if scores[i] is None:
                raise CMSError(f"None score for {i} in {name}")
    rank_map = {}
    for name, scores in score_dicts.items():
        r = midranks({i: float(scores[i]) for i in id_order}, id_order)
        if np.asarray(r, dtype=float).std() == 0:
            raise CMSError(f"all-ties ranks for {name}")
        rank_map[name] = r
    matrix: dict[tuple[str, str], float] = {}
    rhos = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rho = pearson(rank_map[a], rank_map[b])
            matrix[(a, b)] = rho
            rhos.append(rho)
    m = len(names)
    return {
        "cms": (2 / (m * (m - 1))) * sum(rhos),
        "matrix": matrix,
        "missingness": miss,
        "ranks": rank_map,
    }


def disagreement_cards(ranks: dict[str, list[float]], id_order: list[str]) -> list[dict]:
    names = list(ranks)
    cards = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diffs = [abs(ranks[a][k] - ranks[b][k]) for k in range(len(id_order))]
            kmax = int(np.argmax(diffs))
            cards.append(
                {
                    "pair": (a, b),
                    "outcome_id": id_order[kmax],
                    "abs_rank_diff": diffs[kmax],
                }
            )
    cards.sort(key=lambda c: -c["abs_rank_diff"])
    return cards
