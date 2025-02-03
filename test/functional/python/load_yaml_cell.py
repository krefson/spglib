from __future__ import annotations

import io
import lzma
import os
from typing import Union

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type:ignore[assignment]


def load_yaml(fp: Union[str, bytes, os.PathLike, io.IOBase]):
    """Load yaml file.

    Parameters
    ----------
    fp : str, bytes, os.PathLike or io.IOBase
        Filename, file path, or file stream.

    lzma and gzip comppressed non-stream files can be loaded.

    """
    if isinstance(fp, io.IOBase):
        yaml_data = yaml.load(fp, Loader=Loader)
    else:
        with lzma.open(fp) as f:
            yaml_data = yaml.load(f, Loader=Loader)

    return yaml_data


def get_cell(
    fname: Union[str, bytes, os.PathLike, io.IOBase],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaml_data = load_yaml(fname)["unitcell"]
    lattice = np.array(yaml_data["lattice"])
    numbers = np.array([v["number"] for v in yaml_data["points"]], dtype=int)
    points = np.array([v["coordinates"] for v in yaml_data["points"]])
    cell = (lattice, points, numbers)
    return cell
