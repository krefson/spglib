"""Test of get_hall_number_from_symmetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from spglib import get_hall_number_from_symmetry, get_symmetry_dataset

if TYPE_CHECKING:
    from conftest import CrystalData


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_hall_number_from_symmetry(crystal_data: CrystalData):
    """Test get_hall_number_from_symmetry."""
    dataset = get_symmetry_dataset(crystal_data.cell, symprec=1e-5)
    hall_number = get_hall_number_from_symmetry(
        dataset.rotations,
        dataset.translations,
        symprec=1e-5,
    )
    assert hall_number == dataset.hall_number
