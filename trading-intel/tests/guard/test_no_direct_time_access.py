"""Nobody may read the wall clock except ``core/clock.py``.

``ruff``'s banned-api rule covers imports, but a call reached through an alias
(``import datetime as dt; dt.datetime.now()``) slips past it. This walks the AST
of every source file instead, so the rule holds no matter how the call is
spelled.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

#: The only file allowed to touch the real clock.
ALLOWED = {SRC_ROOT / "trading_intel" / "core" / "clock.py"}

BANNED_ATTRIBUTES = {"now", "utcnow", "today"}
BANNED_DOTTED = {
    "datetime.now",
    "datetime.utcnow",
    "datetime.today",
    "time.time",
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "datetime.datetime.today",
}


def _dotted_name(node: ast.expr) -> str:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if not dotted:
            continue
        tail = dotted.rsplit(".", 2)[-2:]
        is_banned = dotted in BANNED_DOTTED or (
            len(tail) == 2 and tail[0] == "datetime" and tail[1] in BANNED_ATTRIBUTES
        )
        if dotted.endswith("time.time") and "time" in dotted.split("."):
            is_banned = True
        if is_banned:
            found.append(f"{path}:{node.lineno}: {dotted}()")
    return found


def test_no_direct_time_access_outside_clock() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path in ALLOWED:
            continue
        offenders.extend(_violations(path))
    assert not offenders, (
        "direct wall-clock access is banned outside core/clock.py; "
        "use trading_intel.core.clock.utc_now instead:\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_planted_violation(tmp_path: Path) -> None:
    """The guard must actually fire — a green test that cannot fail is worthless."""
    planted = tmp_path / "offender.py"
    planted.write_text(
        "import datetime as dt\nimport time\n\n\n"
        "def f():\n"
        "    a = dt.datetime.now()\n"
        "    b = datetime.utcnow()\n"
        "    c = time.time()\n"
        "    return a, b, c\n",
        encoding="utf-8",
    )
    found = _violations(planted)
    assert len(found) == 3, found


def test_clock_module_is_the_only_exemption() -> None:
    clock_module = SRC_ROOT / "trading_intel" / "core" / "clock.py"
    assert list(ALLOWED) == [clock_module]
    # clock.py really does contain what everyone else is forbidden from doing,
    # so the exemption is load-bearing rather than decorative.
    assert _violations(clock_module)
