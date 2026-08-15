import random

from prefkit.outcomes import by_id, id_order
from prefkit.parse import parse_m2
from prefkit.prompts import M2_TEMPLATE


class M2:
    name = "M2"

    def __init__(self):
        self.logs: list[dict] = []

    def iter_queries(self, outcomes: list[dict]):
        ids = id_order(outcomes)
        lookup = by_id(outcomes)
        rng = random.Random(0)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        for oid in shuffled:
            yield (
                M2_TEMPLATE.format(option=lookup[oid]["statement"]),
                {"id": oid, "outcome_ids_touched": [oid]},
            )

    def score(self, outcomes, generate_fn, decode: dict, seed: int, system: str) -> dict[str, float | None]:
        self.logs = []
        k = decode["sample_k"]
        lookup = by_id(outcomes)
        ids = id_order(outcomes)
        rng = random.Random(seed)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        scores: dict[str, float | None] = {}
        for oid in shuffled:
            prompt = M2_TEMPLATE.format(option=lookup[oid]["statement"])
            vals = []
            for _ in range(k):
                raw = generate_fn(prompt, system, allowed=("1", "2", "3", "4", "5", "6", "7"))
                parsed = parse_m2(raw)
                if parsed is not None:
                    vals.append(parsed)
                self.logs.append(
                    {
                        "prompt": prompt,
                        "raw_text": raw,
                        "parsed": parsed,
                        "outcome_ids_touched": [oid],
                    }
                )
            scores[oid] = (sum(vals) / len(vals)) if vals else None
        return scores
