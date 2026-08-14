import json
from pathlib import Path

REQUIRED = ("outcome_id", "domain", "pair_group", "statement", "action_stem")


def load_outcomes(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("outcomes: need JSON array with N>=2")
    ids = []
    for row in rows:
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"outcomes: missing {missing} on {row.get('outcome_id')}")
        if "likert_item" in row and row["likert_item"] != row["statement"]:
            raise ValueError(f"outcomes: likert_item must equal statement for {row['outcome_id']}")
        ids.append(row["outcome_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("outcomes: outcome_id must be unique")
    return rows


def id_order(outcomes: list[dict]) -> list[str]:
    return [r["outcome_id"] for r in outcomes]


def by_id(outcomes: list[dict]) -> dict[str, dict]:
    return {r["outcome_id"]: r for r in outcomes}
