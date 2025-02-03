from __future__ import annotations

import pathlib

import pytest
from load_yaml_cell import get_cell
from spglib import get_symmetry_dataset

cwd = pathlib.Path(__file__).parent


@pytest.mark.benchmark(group="space-group")
def test_get_symmetry_dataset(
    benchmark,
    dirnames: list[str],
):
    """Benchmarking get_symmetry_dataset on all structures under test/data."""
    cells = []
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cells.append(get_cell(fname))

    print(f"Benchmark get_symmetry_dataset on {len(cells)} structures")

    def _get_symmetry_dataset_for_cells():
        for cell in cells:
            _ = get_symmetry_dataset(cell, symprec=1e-5)

    benchmark.pedantic(_get_symmetry_dataset_for_cells, rounds=4)
