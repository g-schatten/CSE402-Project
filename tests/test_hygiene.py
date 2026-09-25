"""Architectural hygiene (PLAN.md section 3 hard rule / section 7 table):
core/sirlab must never import scipy or matplotlib, and must never do file
I/O outside of io/. SciPy/matplotlib are dev-only cross-check dependencies.
"""
from __future__ import annotations

import pathlib
import re

CORE = pathlib.Path(__file__).resolve().parents[1] / "core" / "sirlab"
BANNED = ("scipy", "matplotlib")


def test_no_banned_imports_in_core():
    offenders = []
    for path in CORE.rglob("*.py"):
        text = path.read_text()
        for name in BANNED:
            if re.search(rf"^\s*(import {name}|from {name})\b", text, re.MULTILINE):
                offenders.append((str(path.relative_to(CORE.parent.parent)), name))
    assert not offenders, f"banned imports found: {offenders}"


def test_no_numpy_eigensolvers_outside_tests():
    """Eigenvalues, singular values, and condition numbers come from our
    power method (linalg/eigen.py); NumPy's versions are for tests only."""
    pattern = re.compile(r"np\.linalg\.(eig\w*|svd|cond)\b|from numpy\.linalg import")
    offenders = []
    for root in (CORE, CORE.parents[1] / "experiments"):
        for path in root.rglob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(CORE.parents[1])}:{lineno}")
    assert not offenders, f"NumPy eigensolver calls found: {offenders}"


def test_core_package_importable():
    import sirlab  # noqa: F401
    import sirlab.models  # noqa: F401
    import sirlab.solvers  # noqa: F401
    import sirlab.reference  # noqa: F401
