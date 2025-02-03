"""Test of get_hall_number_from_symmetry."""

from __future__ import annotations

import pathlib

import pytest
from load_yaml_cell import get_cell
from spglib import get_hall_number_from_symmetry, get_symmetry_dataset

cwd = pathlib.Path(__file__).parent


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_hall_number_from_symmetry(dirnames: list[str]):
    """Test get_hall_number_from_symmetry."""
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cell = get_cell(fname)
            dataset = get_symmetry_dataset(cell, symprec=1e-5)
            hall_number = get_hall_number_from_symmetry(
                dataset.rotations,
                dataset.translations,
                symprec=1e-5,
            )
            assert hall_number == dataset.hall_number, "%d != %d in %s" % (
                hall_number,
                dataset.hall_number,
                fname,
            )
