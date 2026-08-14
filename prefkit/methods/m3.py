import json
from pathlib import Path

from prefkit.outcomes import by_id, id_order
from prefkit.parse import parse_m3
from prefkit.prompts import M3_TEMPLATE

_ROOT = Path(__file__).resolve().parents[2]


def build_menus(outcomes: list[dict]) -> dict:
    ids = id_order(outcomes)
    n = len(ids)
    menus = []
    for i, focal in enumerate(ids):
        foils = [ids[(i + k) % n] for k in (1, 2, 3)]
        for pos in range(4):
            slots = [None] * 4
            fi = 0
            for s in range(4):
                if s == pos:
                    slots[s] = focal
                else:
                    slots[s] = foils[fi]
                    fi += 1
            letters = "ABCD"
            letter_to_id = {letters[s]: slots[s] for s in range(4)}
            menus.append({"focal": focal, "letter_to_id": letter_to_id})
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
            stems = [lookup[l2i[L]]["action_stem"] for L in "ABCD"]
            prompt = M3_TEMPLATE.format(
                stem_0=stems[0], stem_1=stems[1], stem_2=stems[2], stem_3=stems[3]
            )
            yield prompt, {
                "focal": menu["focal"],
                "letter_to_id": l2i,
                "outcome_ids_touched": [l2i[L] for L in "ABCD"],
            }

    def score(self, outcomes, generate_fn, decode: dict, seed: int, system: str) -> dict[str, float | None]:
        del seed
        self.logs = []
        k = decode["sample_k"]
        picks: dict[str, list[int]] = {i: [] for i in id_order(outcomes)}
        for prompt, meta in self.iter_queries(outcomes):
            focal = meta["focal"]
            l2i = meta["letter_to_id"]
            for _ in range(k):
                raw = generate_fn(prompt, system)
                letter = parse_m3(raw)
                parsed_id = l2i[letter] if letter in l2i else None
                if letter is not None:
                    picks[focal].append(1 if parsed_id == focal else 0)
                self.logs.append(
                    {
                        "prompt": prompt,
                        "raw_text": raw,
                        "parsed": letter,
                        "outcome_ids_touched": meta["outcome_ids_touched"],
                    }
                )
        return {
            oid: (sum(v) / len(v) if v else None) for oid, v in picks.items()
        }
