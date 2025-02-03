"""Test of spacegroup_type_from_symmetry."""

from __future__ import annotations

import pathlib

import pytest
from load_yaml_cell import get_cell
from spglib import (
    get_spacegroup_type,
    get_spacegroup_type_from_symmetry,
    get_symmetry_dataset,
)

cwd = pathlib.Path(__file__).parent


@pytest.mark.parametrize("lattice_None", [True, False])
def test_spacegroup_type_from_symmetry(
    dirnames: list[str],
    lattice_None: bool,
):
    """Test spacegroup_type_from_symmetry."""
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cell = get_cell(fname)

            if lattice_None:
                lattice = None
            else:
                lattice = cell[0]

            dataset = get_symmetry_dataset(cell, symprec=1e-5)
            spgtype = get_spacegroup_type_from_symmetry(
                dataset.rotations,
                dataset.translations,
                lattice=lattice,
                symprec=1e-5,
            )

            assert spgtype.number == dataset.number, "%d != %d in %s" % (
                spgtype.number,
                dataset.number,
                fname,
            )
            spgtype_ref = get_spacegroup_type(dataset.hall_number)
            assert spgtype == spgtype_ref
