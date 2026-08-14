"""E0 smoke: check-axioms then N=4 run (backend from PREFKIT_BACKEND)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefkit.cli import main

if __name__ == "__main__":
    rc = main(["check-axioms"])
    if rc != 0:
        raise SystemExit(rc)
    raise SystemExit(main(["smoke"]))
