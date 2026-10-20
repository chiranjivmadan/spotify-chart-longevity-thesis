#!/usr/bin/env python3
"""Version-2 notebook utilities for the supervisor-confirmed update.

All general helpers are imported unchanged from ``thesis_utils``. Only the
notebook output locations are versioned so executing v2 cannot modify v1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from thesis_utils import *  # noqa: F401,F403
from thesis_utils import SEED, find_project_root


def setup_notebook(notebook_name: str) -> tuple[Path, Path, Path]:
    """Initialise v2 paths and deterministic plotting without touching v1."""

    root = find_project_root()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_final_notebooks_v2"),
    )
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    np.random.seed(SEED)
    figure_dir = root / "outputs/final_thesis_notebooks_v2/figures"
    record_dir = root / "notebooks/final_thesis_v2/execution_records"
    figure_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    return root, figure_dir, record_dir
