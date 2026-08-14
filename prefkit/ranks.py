def midranks(scores: dict[str, float], id_order: list[str]) -> list[float]:
    """Higher score → rank 1. Ties: average rank (1-based). Length N. No None allowed."""
    if any(scores[i] is None for i in id_order):
        raise ValueError("midranks: None not allowed")
    n = len(id_order)
    indexed = sorted(range(n), key=lambda k: -scores[id_order[k]])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = scores[id_order[indexed[i]]]
        while j < n and scores[id_order[indexed[j]]] == v:
            j += 1
        avg = (i + 1 + j) / 2.0
        for t in range(i, j):
            ranks[indexed[t]] = avg
        i = j
    return ranks
