"""The plane separation, enforced by test rather than by good intentions.

A research agent that can reach the execution path is not isolated, however the
directory tree is drawn. These tests fail the build if the import graph drifts.

Allowed direction of dependency:

    data     ← research ← trading
    data     ← control

``research`` must never import ``trading``: research generates hypotheses, it
does not place orders. ``control`` must never import ``research`` or
``trading``: the rules governing deployment cannot depend on the code being
deployed, or a researcher could widen a gate by editing the thing that checks it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[1] / "src" / "quant_platform"

FORBIDDEN = {
    "research": {"trading", "control"},
    "control": {"research", "trading"},
    "data": {"research", "trading", "control"},
}


def imported_planes(path: Path) -> set[str]:
    """Every quant_platform plane referenced by this module's imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    planes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative import, e.g. from ..data.pit import X
                parts = module.split(".")
                if parts and parts[0] in {"data", "research", "trading", "control"}:
                    planes.add(parts[0])
            elif module.startswith("quant_platform."):
                planes.add(module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("quant_platform."):
                    planes.add(alias.name.split(".")[1])
    return planes


def plane_files(plane: str) -> list[Path]:
    return sorted((PLATFORM / plane).glob("*.py"))


@pytest.mark.parametrize("plane", sorted(FORBIDDEN))
def test_plane_does_not_import_forbidden_planes(plane):
    violations = []
    for path in plane_files(plane):
        for imported in imported_planes(path):
            if imported in FORBIDDEN[plane]:
                violations.append(f"{path.name} imports {imported}")
    assert not violations, (
        f"{plane} plane 違反隔離規則：{violations}。"
        "四平面的意義在於邊界，不在於資料夾名稱。"
    )


def test_every_plane_exists_and_is_documented():
    for plane in ("data", "research", "trading", "control"):
        init = PLATFORM / plane / "__init__.py"
        assert init.exists(), f"缺少 {plane} plane"
        assert ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))), (
            f"{plane} plane 沒有 docstring；邊界必須寫下來才有人遵守"
        )


def test_control_plane_takes_no_natural_language_input():
    """The risk and registry surfaces accept data, never instructions.

    A risk engine that can be talked out of a limit is not a risk engine. This
    checks the weaker, mechanical property: no control-plane function signature
    accepts something named like a prompt or an agent message.
    """
    suspicious = {"prompt", "instruction", "message", "llm_response", "agent_message"}
    offenders = []
    for path in plane_files("control"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
                hits = names & suspicious
                if hits:
                    offenders.append(f"{path.name}:{node.name} 接受 {sorted(hits)}")
    assert not offenders, f"control plane 不得接受自然語言指令：{offenders}"
