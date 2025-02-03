"""Pytest configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def dirnames() -> list[str]:
    """Return test directory names."""
    _dirnames = [
        "cubic",
        "hexagonal",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "triclinic",
        "trigonal",
    ]
    return _dirnames
