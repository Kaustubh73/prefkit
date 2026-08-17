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
_METHOD_NAMES = ("M1", "M2", "M3", "M4")
_FRAMES = ("default", "empty", "persona", "custom")


def _load_yaml(rel: str):
    return yaml.safe_load((_ROOT / rel).read_text(encoding="utf-8"))


def cmd_check_axioms(_args) -> int:
    decode = _load_yaml("configs/decode.yaml")
    outcomes = load_outcomes(_ROOT / "data" / "outcomes.smoke.json")
    result = check_axioms(default_methods(), outcomes, decode)
    print(json.dumps(result, indent=2))
    ok = all(all(v.values()) for v in result.values())
    return 0 if ok else 1


def _system_for_frame(frame: str) -> str:
    if frame == "default":
        return SYSTEM_DEFAULT
    if frame == "empty":
        return ""
    if frame == "persona":
        msg = (_load_yaml("configs/persona.yaml") or {}).get("system_message")
        if not isinstance(msg, str) or not msg.strip() or msg.strip() == "TBD":
            raise SystemExit("configs/persona.yaml missing system_message")
        return msg.strip()
    raise SystemExit(f"unknown frame {frame}")


def _resolve_system(frame: str, system_override: str | None) -> tuple[str, str]:
    """Return (frame_written_to_json, system_string)."""
    if system_override is not None:
        # frozen cells keep honest filenames; custom text is never "persona"/"empty"
        if frame in ("empty", "persona"):
            raise SystemExit("--system cannot combine with frozen --frame empty|persona")
        return "custom", system_override
    if frame == "custom":
        raise SystemExit("--frame custom requires --system")
    return frame, _system_for_frame(frame)


def _parse_methods(methods_csv: str | None):
    all_m = default_methods()
    if methods_csv is None or methods_csv.strip() in ("", "all"):
        return all_m
    names = [t.strip() for t in methods_csv.split(",")]
    if not names or any(n == "" for n in names):
        raise SystemExit("empty --methods token")
    if "all" in names:
        raise SystemExit("--methods all cannot mix with names")
    if len(names) != len(set(names)):
        raise SystemExit("duplicate --methods")
    unknown = [n for n in names if n not in _METHOD_NAMES]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")
    by_name = {m.name: m for m in all_m}
    return [by_name[n] for n in names]


def _result_path(hf_id: str, frame: str, outcomes_path: str, tag: str | None = None) -> Path:
    safe = hf_id.replace("/", "__")
    stem = Path(str(outcomes_path)).stem
    extra = f"_{tag}" if tag else ""
    return _ROOT / "results" / f"{safe}_{frame}_{stem}{extra}.json"


def _run_slot(
    slot: str,
    frame: str,
    outcomes_path: str,
    backend: str,
    seed: int | None = None,
    methods_csv: str | None = None,
    tag: str | None = None,
    system_override: str | None = None,
) -> tuple:
    decode = _load_yaml("configs/decode.yaml")
    if seed is not None:
        decode = {**decode, "seed": int(seed)}
    models = _load_yaml("configs/models.yaml")
    if slot not in models:
        raise SystemExit(f"unknown slot {slot}")
    spec = models[slot]
    hf_id = spec["hf"]
    ollama_tag = spec.get("ollama")
    outcomes = load_outcomes(outcomes_path)
    frame, system = _resolve_system(frame, system_override)
    inner = make_generate_fn(backend, hf_id, ollama_tag, decode)
    methods = _parse_methods(methods_csv)
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
            "tag": tag,
            "methods": [m.name for m in methods],
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
            # keep == len(methods) so a crash after M2 does not look like finished CMS
            if len(methods) < 2:
                out["cms_error"] = "skip headline CMS (need at least 2 methods)"
            elif len(scores) == len(methods) and all(v == 0 for v in miss.values()):
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
            _write_result(_blob(), hf_id)
        return _blob(), hf_id
    finally:
        pbar.close()


def _write_result(blob: dict, hf_id: str) -> Path:
    # frame + tag from blob so cmd_run cannot clobber headline after --system
    out = _result_path(hf_id, blob["frame"], blob.get("outcomes", "outcomes"), blob.get("tag"))
    out.parent.mkdir(exist_ok=True)
    text = json.dumps(blob, indent=2)
    out.write_text(text, encoding="utf-8")
    if blob.get("backend") == "ollama":
        stem = Path(str(blob.get("outcomes", "outcomes"))).stem
        debug = _ROOT / "results" / f"debug_ollama_{blob.get('slot', 'S')}_{stem}.json"
        debug.write_text(text, encoding="utf-8")
    return out


def cmd_smoke(args) -> int:
    backend = backend_from_env()
    path = args.outcomes or str(_ROOT / "data" / "outcomes.smoke.json")
    blob, hf_id = _run_slot("S", "default", path, backend)
    out = _write_result(blob, hf_id)
    print(out)
    return 0


def cmd_run(args) -> int:
    backend = backend_from_env()
    blob, hf_id = _run_slot(
        args.slot,
        args.frame,
        args.outcomes,
        backend,
        seed=getattr(args, "seed", None),
        methods_csv=getattr(args, "methods", None),
        tag=getattr(args, "tag", None),
        system_override=getattr(args, "system", None),
    )
    out = _write_result(blob, hf_id)
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
    rn.add_argument("--frame", default="default", choices=_FRAMES)
    rn.add_argument("--system", default=None, help="custom system string; records frame=custom")
    rn.add_argument("--seed", type=int, default=None, help="override configs/decode.yaml seed")
    rn.add_argument("--methods", default=None, help="comma-separated M1,M2,M3,M4 or all")
    rn.add_argument("--tag", default=None, help="filename suffix, e.g. seed1 / m4")
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
