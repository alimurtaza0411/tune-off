"""The fixed table of 32 scrambled-Sobol points shared by every cell (README: "What gets tuned").

The table stores u-values in [0, 1)^6 only, so it does not depend on the search ranges.
Generate it once with ``python -m mvt sobol --write``; never edit it by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from mvt.core.config import CONFIG_DIR, SearchSpaceConfig


def generate(seed: int = 4776, m: int = 5, d: int = 6) -> np.ndarray:
    return qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m=m)


def table_path(cfg: SearchSpaceConfig) -> Path:
    return CONFIG_DIR / cfg.sobol.table


def write_table(cfg: SearchSpaceConfig) -> Path:
    table = generate(cfg.sobol.seed, cfg.sobol.m, len(cfg.params))
    path = table_path(cfg)
    header = ",".join(p.name for p in cfg.params)
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.17g")
    return path


def load_table(cfg: SearchSpaceConfig) -> np.ndarray:
    return np.loadtxt(table_path(cfg), delimiter=",", skiprows=1, ndmin=2)


def check_table(cfg: SearchSpaceConfig) -> bool:
    """True if the committed table equals a fresh regeneration."""
    expected = generate(cfg.sobol.seed, cfg.sobol.m, len(cfg.params))
    return np.array_equal(load_table(cfg), expected)
