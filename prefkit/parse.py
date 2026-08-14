import re

_M1 = re.compile(r"\b([AB])\b")
_M3 = re.compile(r"\b([ABCD])\b")
_INT = re.compile(r"-?\d+")


def parse_m1(text: str) -> str | None:
    m = _M1.search(text.strip().upper())
    return m.group(1) if m else None


def parse_m2(text: str) -> int | None:
    m = _INT.search(text.strip())
    if not m:
        return None
    n = int(m.group(0))
    return n if 1 <= n <= 7 else None


def parse_m3(text: str) -> str | None:
    m = _M3.search(text.strip().upper())
    return m.group(1) if m else None
