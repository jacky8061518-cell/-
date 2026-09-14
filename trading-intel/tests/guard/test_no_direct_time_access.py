"""除了 ``core/clock.py`` 以外，任何人都不得讀取牆上時鐘。

``ruff`` 的 banned-api 規則走 import graph，抓得到直接寫法，
但抓不到透過別名繞過去的呼叫（``import datetime as dt; dt.datetime.now()``）。
本測試改走每個原始檔的 AST，因此不論怎麼拼寫，規則都成立。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

#: 唯一允許碰真實時鐘的檔案。
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
        "core/clock.py 以外禁止直接存取牆上時鐘，"
        "請改用 trading_intel.core.clock.utc_now：\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_planted_violation(tmp_path: Path) -> None:
    """守門測試本身必須真的會失敗；永遠綠燈的測試沒有價值。"""
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
    # clock.py 內確實有其他人被禁止做的事，因此這項豁免是有作用的，不是裝飾。
    assert _violations(clock_module)
