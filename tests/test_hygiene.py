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


def test_core_package_importable():
    import sirlab  # noqa: F401
    import sirlab.models  # noqa: F401
    import sirlab.solvers  # noqa: F401
    import sirlab.reference  # noqa: F401
