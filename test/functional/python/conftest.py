"""Pytest configuration."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml

if TYPE_CHECKING:
    from typing import Self, TypeAlias

    TestCell: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


TEST_DIR = Path(__file__).parent


def get_cell(fname: Path) -> TestCell:
    with fname.open() as f:
        yaml_data = yaml.safe_load(f)["unitcell"]
    lattice = np.array(yaml_data["lattice"])
    numbers = np.array([v["number"] for v in yaml_data["points"]], dtype=int)
    points = np.array([v["coordinates"] for v in yaml_data["points"]])
    return lattice, points, numbers


@dataclasses.dataclass()
class CrystalData:
    name: str
    cell: TestCell
    # TODO: Convert this to a TypeDict
    ref: dict | None

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        if not isinstance(path, Path):
            path = Path(path)
        if path.is_absolute():
            name = path.stem
            cell_file = path
            ref_file = None
        else:
            name = str(path)
            cell_file = TEST_DIR / f"data/{path}.yaml"
            ref_file = TEST_DIR / f"ref/{path}-ref.yaml"
        cell_data = get_cell(cell_file)
        ref_data = None
        if ref_file and ref_file.exists():
            with ref_file.open() as f:
                ref_data = yaml.safe_load(f)
        return cls(
            name=name,
            cell=cell_data,
            ref=ref_data,
        )


@pytest.fixture()
def get_crystal_data():
    """Get a single crystal data."""

    def _get_crystal_data(path: str) -> CrystalData:
        return CrystalData.from_path(path)

    return _get_crystal_data


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Generate all crystal data tests tests."""
    if "crystal_data" in metafunc.fixturenames:
        data_root = TEST_DIR / "data"
        all_crystal_data = []
        # Path.walk was introduced in 3.12. For now use os.walk
        for root, dirs, files in os.walk(data_root):
            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix not in (".yaml", ".yml"):
                    continue
                relative_path = file_path.relative_to(data_root)
                # Strip the suffix
                relative_path = relative_path.with_suffix("")
                crystal_data = CrystalData.from_path(relative_path)
                all_crystal_data.append(
                    pytest.param(
                        crystal_data,
                        id=str(relative_path),
                    )
                )
        metafunc.parametrize("crystal_data", all_crystal_data)
