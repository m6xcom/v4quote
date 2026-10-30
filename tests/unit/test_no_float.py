"""Guard: no floating point anywhere in the swap-math path (v4quote.math, v4quote.amm, v4quote.routing)."""

import ast
from pathlib import Path

import v4quote

SRC = Path(v4quote.__file__).resolve().parent
SWAP_PATH_DIRS = [SRC / "math", SRC / "amm"]
SWAP_PATH_FILES = [SRC / "routing.py"]
FORBIDDEN_MODULES = {"math", "decimal", "fractions", "statistics", "numpy", "v4quote.display"}


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        where = f"{path.relative_to(SRC)}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            out.append(f"{where} true division '/'")
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            out.append(f"{where} true division '/='")
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            out.append(f"{where} float literal {node.value!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("float", "round"):
            out.append(f"{where} call to {node.func.id}()")
        if isinstance(node, ast.Import):
            out += [f"{where} import {a.name}" for a in node.names if a.name in FORBIDDEN_MODULES]
        if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_MODULES:
            out.append(f"{where} from {node.module} import ...")
    return out


def test_swap_path_is_integer_only():
    files = [p for d in SWAP_PATH_DIRS for p in d.rglob("*.py")] + SWAP_PATH_FILES
    assert len(files) >= 10
    violations = [v for f in files for v in _violations(f)]
    assert not violations, "\n".join(violations)
