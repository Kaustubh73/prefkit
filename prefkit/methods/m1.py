from itertools import combinations

from prefkit.outcomes import by_id, id_order
from prefkit.parse import parse_m1
from prefkit.prompts import M1_TEMPLATE


class M1:
    name = "M1"

    def __init__(self):
        self.logs: list[dict] = []

    def iter_queries(self, outcomes: list[dict]):
        lookup = by_id(outcomes)
        ids = id_order(outcomes)
        for i, j in combinations(ids, 2):
            yield (
                M1_TEMPLATE.format(option_A=lookup[i]["statement"], option_B=lookup[j]["statement"]),
                {"ids": (i, j), "order": "ij", "outcome_ids_touched": [i, j]},
            )
            yield (
                M1_TEMPLATE.format(option_A=lookup[j]["statement"], option_B=lookup[i]["statement"]),
                {"ids": (i, j), "order": "ji", "outcome_ids_touched": [i, j]},
            )

    def score(self, outcomes, generate_fn, decode: dict, seed: int, system: str) -> dict[str, float | None]:
        del seed
        self.logs = []
        k = decode["sample_k"]
        ids = id_order(outcomes)
        n = len(ids)
        ordered: list[tuple[str, str, list[str | None]]] = []
        for prompt, meta in self.iter_queries(outcomes):
            i, j = meta["ids"]
            parsed_list = []
            for _ in range(k):
                raw = generate_fn(prompt, system)
                parsed = parse_m1(raw)
                parsed_list.append(parsed)
                self.logs.append(
                    {
                        "prompt": prompt,
                        "raw_text": raw,
                        "parsed": parsed,
                        "outcome_ids_touched": meta["outcome_ids_touched"],
                    }
                )
            a_id, b_id = (i, j) if meta["order"] == "ij" else (j, i)
            ordered.append((a_id, b_id, parsed_list))

        scores: dict[str, float | None] = {}
        need_samples = 2 * (n - 1) * k
        for x in ids:
            ps = []
            valid_ordered = 0
            valid_samples = 0
            for a_id, b_id, parsed_list in ordered:
                if x not in (a_id, b_id):
                    continue
                valids = [p for p in parsed_list if p in ("A", "B")]
                valid_samples += len(valids)
                if not valids:
                    continue
                valid_ordered += 1
                letter = "A" if x == a_id else "B"
                ps.append(sum(p == letter for p in valids) / len(valids))
            if valid_ordered < 2 or valid_samples < 0.5 * need_samples:
                scores[x] = None
            else:
                scores[x] = sum(ps) / len(ps)
        return scores
