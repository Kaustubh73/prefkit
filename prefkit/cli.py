from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from prefkit.axioms import check_axioms, default_methods
from prefkit.cms import CMSError, cms, disagreement_cards, missingness
from prefkit.diagnostics import run_diagnostics
from prefkit.generate import backend_from_env, make_generate_fn
from prefkit.outcomes import load_outcomes, id_order
from prefkit.prompts import SYSTEM_DEFAULT
from prefkit.thurstone import fit_from_m1_logs

_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(rel: str):
    return yaml.safe_load((_ROOT / rel).read_text(encoding="utf-8"))


def cmd_check_axioms(_args) -> int:
    decode = _load_yaml("configs/decode.yaml")
    outcomes = load_outcomes(_ROOT / "data" / "outcomes.smoke.json")
    result = check_axioms(default_methods(), outcomes, decode)
    print(json.dumps(result, indent=2))
    ok = all(all(v.values()) for v in result.values())
    return 0 if ok else 1


def _run_slot(slot: str, frame: str, outcomes_path: str, backend: str) -> dict:
    decode = _load_yaml("configs/decode.yaml")
    models = _load_yaml("configs/models.yaml")
    if slot not in models:
        raise SystemExit(f"unknown slot {slot}")
    spec = models[slot]
    hf_id = spec["hf"]
    ollama_tag = spec.get("ollama")
    outcomes = load_outcomes(outcomes_path)
    system = SYSTEM_DEFAULT
    if frame != "default":
        raise SystemExit("only frame=default is shipped (persona TBD)")
    inner = make_generate_fn(backend, hf_id, ollama_tag, decode)
    methods = default_methods()
    k = int(decode["sample_k"])
    total = sum(sum(1 for _ in m.iter_queries(outcomes)) * k for m in methods)
    # ponytail: stdlib has no progress bar; tqdm is cheaper than a CR printer.
    pbar = tqdm(total=total, file=sys.stderr, desc=f"{slot}")

    def gen(prompt: str, system: str, allowed=None) -> str:
        try:
            return inner(prompt, system, allowed=allowed)
        finally:
            pbar.update(1)

    scores = {}
    logs = {}
    ids = id_order(outcomes)

    def _blob():
        miss = missingness(scores, ids) if scores else {}
        out = {
            "backend": backend,
            "model": hf_id,
            "slot": slot,
            "frame": frame,
            "system": system,
            "outcomes": str(outcomes_path),
            "decode": decode,
            "scores": scores,
            "logs": logs,
            "missingness": miss,
            "diagnostics": run_diagnostics(scores, logs, ids) if scores else {},
        }
        if "M1" in logs:
            out["utilities_thurstone"] = fit_from_m1_logs(logs["M1"], ids)
        try:
            if len(scores) == len(methods) and all(v == 0 for v in miss.values()):
                cms_out = cms(scores, ids)
                out["cms"] = cms_out["cms"]
                out["matrix"] = {f"{a}|{b}": v for (a, b), v in cms_out["matrix"].items()}
                out["ranks"] = cms_out["ranks"]
                out["disagreement"] = [
                    {**c, "pair": f"{c['pair'][0]}|{c['pair'][1]}"}
                    for c in disagreement_cards(cms_out["ranks"], ids)
                ]
            elif len(scores) == len(methods):
                out["cms_error"] = "skip headline CMS (missingness or incomplete scores)"
        except CMSError as e:
            out["cms_error"] = str(e)
        return out

    try:
        for m in methods:
            scores[m.name] = m.score(outcomes, gen, decode, decode["seed"], system)
            logs[m.name] = m.logs
            # spec §11.1: save after each method (Colab disconnect)
            _write_result(_blob(), hf_id, frame)
        return _blob(), hf_id
    finally:
        pbar.close()


def _write_result(blob: dict, hf_id: str, frame: str) -> Path:
    safe = hf_id.replace("/", "__")
    stem = Path(str(blob.get("outcomes", "outcomes"))).stem
    out = _ROOT / "results" / f"{safe}_{frame}_{stem}.json"
    out.parent.mkdir(exist_ok=True)
    text = json.dumps(blob, indent=2)
    out.write_text(text, encoding="utf-8")
    if blob.get("backend") == "ollama":
        debug = _ROOT / "results" / f"debug_ollama_{blob.get('slot', 'S')}_{stem}.json"
        debug.write_text(text, encoding="utf-8")
    return out


def cmd_smoke(args) -> int:
    backend = backend_from_env()
    path = args.outcomes or str(_ROOT / "data" / "outcomes.smoke.json")
    blob, hf_id = _run_slot("S", "default", path, backend)
    out = _write_result(blob, hf_id, "default")
    print(out)
    return 0


def cmd_run(args) -> int:
    backend = backend_from_env()
    blob, hf_id = _run_slot(args.slot, args.frame, args.outcomes, backend)
    out = _write_result(blob, hf_id, args.frame)
    print(out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="prefkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-axioms")
    sm = sub.add_parser("smoke")
    sm.add_argument("--outcomes", default=None)
    rn = sub.add_parser("run")
    rn.add_argument("--slot", required=True)
    rn.add_argument("--frame", default="default")
    rn.add_argument("--outcomes", required=True)
    args = p.parse_args(argv)
    if args.cmd == "check-axioms":
        return cmd_check_axioms(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "run":
        return cmd_run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
