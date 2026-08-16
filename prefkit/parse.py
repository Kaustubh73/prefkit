import re

_M1 = re.compile(r"\b([AB])\b")
_INT = re.compile(r"-?\d+")


def _letter(text: str, pat: re.Pattern[str]) -> str | None:
    t = text.strip().upper()
    if not t:
        return None
    for chunk in (t, *(ln.strip() for ln in reversed(t.splitlines()))):
        m = pat.fullmatch(chunk)
        if m:
            return m.group(1)
    m = pat.search(t)
    return m.group(1) if m else None


def parse_m1(text: str) -> str | None:
    return _letter(text, _M1)


def parse_m2(text: str) -> int | None:
    def likert(s: str) -> int | None:
        m = _INT.fullmatch(s.strip())
        if not m:
            return None
        n = int(m.group(0))
        return n if 1 <= n <= 7 else None

    t = text.strip()
    if not t:
        return None
    n = likert(t)
    if n is not None:
        return n
    for line in reversed(t.splitlines()):
        n = likert(line)
        if n is not None:
            return n
    m = _INT.search(t)
    if not m:
        return None
    n = int(m.group(0))
    return n if 1 <= n <= 7 else None


parse_m3 = parse_m1

# Exact "BEST WORST" letters only. No prose search — chatty refusals stay None.
_M4 = re.compile(r"^([A-D]) ([A-D])$")


def parse_m4(text: str) -> tuple[str, str] | None:
    t = text.strip().upper()
    m = _M4.fullmatch(t)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if a == b:
        return None
    return a, b
