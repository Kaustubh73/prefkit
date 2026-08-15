import json
from pathlib import Path

from prefkit.outcomes import by_id, id_order
from prefkit.parse import parse_m1
from prefkit.prompts import M3_TEMPLATE

_ROOT = Path(__file__).resolve().parents[2]


def pair_groups(outcomes: list[dict]) -> list[tuple[str, str, str]]:
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for row in outcomes:
        g = row["pair_group"]
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(row["outcome_id"])
    out = []
    for g in order:
        ids = groups[g]
        if len(ids) != 2:
            raise ValueError(f"pair_group {g} must have exactly 2 ids, got {ids}")
        out.append((g, ids[0], ids[1]))
    return out


def build_menus(outcomes: list[dict]) -> dict:
    ids = id_order(outcomes)
    menus = []
    for g, a, b in pair_groups(outcomes):
        menus.append({"pair_group": g, "order": "12", "letter_to_id": {"A": a, "B": b}})
        menus.append({"pair_group": g, "order": "21", "letter_to_id": {"A": b, "B": a}})
    return {"id_order": ids, "menus": menus}


def menus_path(n: int) -> Path:
    name = "m3_menus.smoke.json" if n == 4 else "m3_menus.json"
    return _ROOT / "data" / name


def load_menus(outcomes: list[dict]) -> dict:
    blob = json.loads(menus_path(len(outcomes)).read_text(encoding="utf-8"))
    if blob["id_order"] != id_order(outcomes):
        raise ValueError("frozen M3 menus id_order mismatch")
    return blob


class M3:
    name = "M3"

    def __init__(self):
        self.logs: list[dict] = []

    def iter_queries(self, outcomes: list[dict]):
        lookup = by_id(outcomes)
        blob = load_menus(outcomes)
        for menu in blob["menus"]:
            l2i = menu["letter_to_id"]
            a_id, b_id = l2i["A"], l2i["B"]
            situation = f"{lookup[a_id]['statement']}\n{lookup[b_id]['statement']}"
            prompt = M3_TEMPLATE.format(
                situation=situation,
                stem_A=lookup[a_id]["action_stem"],
                stem_B=lookup[b_id]["action_stem"],
            )
            yield prompt, {
                "pair_group": menu["pair_group"],
                "order": menu["order"],
                "letter_to_id": l2i,
                "outcome_ids_touched": [a_id, b_id],
            }

    def score(self, outcomes, generate_fn, decode: dict, seed: int, system: str) -> dict[str, float | None]:
        del seed
        self.logs = []
        k = decode["sample_k"]
        ids = id_order(outcomes)
        wins: dict[str, list[int]] = {i: [] for i in ids}
        pair_valid: dict[frozenset[str], int] = {}
        for prompt, meta in self.iter_queries(outcomes):
            l2i = meta["letter_to_id"]
            pair = frozenset(l2i.values())
            pair_valid.setdefault(pair, 0)
            for _ in range(k):
                raw = generate_fn(prompt, system, allowed=("A", "B"))
                letter = parse_m1(raw)
                parsed_id = l2i[letter] if letter in l2i else None
                if parsed_id is not None:
                    pair_valid[pair] += 1
                    for oid in pair:
                        wins[oid].append(1 if oid == parsed_id else 0)
                self.logs.append(
                    {
                        "prompt": prompt,
                        "raw_text": raw,
                        "parsed": letter,
                        "letter_to_id": l2i,
                        "order": meta["order"],
                        "outcome_ids_touched": meta["outcome_ids_touched"],
                    }
                )
        need = 0.5 * 2 * k
        scores: dict[str, float | None] = {}
        for oid in ids:
            pair = None
            for p in pair_valid:
                if oid in p:
                    pair = p
                    break
            if pair is None or pair_valid[pair] < need:
                scores[oid] = None
            else:
                v = wins[oid]
                scores[oid] = sum(v) / len(v) if v else None
        return scores
