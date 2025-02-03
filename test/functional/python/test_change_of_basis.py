from __future__ import annotations

import pathlib

import numpy as np
from load_yaml_cell import get_cell
from spglib import (
    get_symmetry_dataset,
    get_symmetry_from_database,
)

cwd = pathlib.Path(__file__).parent


def test_change_of_basis(dirnames: list[str]):
    symprec = 1e-5
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cell = get_cell(fname)
            dataset = get_symmetry_dataset(cell, symprec=symprec)
            std_pos = dataset.std_positions
            tmat = dataset.transformation_matrix
            orig_shift = dataset.origin_shift
            lat = np.dot(cell[0].T, np.linalg.inv(tmat))
            pos = np.dot(cell[1], tmat.T) + orig_shift
            for p in pos:
                diff = std_pos - p
                diff -= np.rint(diff)
                diff = np.dot(diff, lat.T)
                delta = np.sqrt((diff**2).sum(axis=1))
                indices = np.where(delta < symprec)[0]
                assert len(indices) == 1, "multi: %s %s" % (p, indices)


def test_std_symmetry(dirnames: list[str]):
    symprec = 1e-5
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cell = get_cell(fname)
            dataset = get_symmetry_dataset(cell, symprec=symprec)
            symmetry = get_symmetry_from_database(dataset.hall_number)
            std_pos = dataset.std_positions

            # for r, t in zip(symmetry['rotations'], symmetry['translations']):
            #     for rp in (np.dot(std_pos, r.T) + t):
            #         diff = std_pos - rp
            #         diff -= np.rint(diff)
            #         num_match = len(np.where(abs(diff).sum(axis=1) < 1e-3)[0])
            #         self.assertEqual(num_match, 1, msg="%s" % fname)

            # Equivalent above by numpy hack
            # 15 sec on macOS 2.3 GHz Intel Core i5 (4times faster than above)
            rot = symmetry["rotations"]
            trans = symmetry["translations"]
            # (n_sym, 3, n_atom)
            rot_pos = np.dot(rot, std_pos.T) + trans[:, :, None]
            for p in std_pos:
                diff = rot_pos - p[None, :, None]
                diff -= np.rint(diff)
                num_match = (abs(diff).sum(axis=1) < 1e-3).sum(axis=1)
                assert (num_match == 1).all(), "%s" % fname


def test_std_rotation(dirnames: list[str]):
    symprec = 1e-5
    for d in dirnames:
        dirname = cwd / "data" / d
        for fname in dirname.iterdir():
            cell = get_cell(fname)
            dataset = get_symmetry_dataset(cell, symprec=symprec)
            std_lat = dataset.std_lattice
            tmat = dataset.transformation_matrix
            lat = np.dot(cell[0].T, np.linalg.inv(tmat))
            lat_rot = np.dot(dataset.std_rotation_matrix, lat)
            np.testing.assert_allclose(
                std_lat,
                lat_rot.T,
                atol=symprec,
                err_msg="%s" % fname,
            )
