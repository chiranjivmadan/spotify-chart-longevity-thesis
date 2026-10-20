#!/usr/bin/env python3
"""Shared utilities for the final Spotify chart-longevity notebooks.

The module contains presentation and integrity helpers only.  Substantive
estimators remain in their validated, versioned analysis scripts so notebook
reproduction cannot drift from the frozen working specification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import Image, display
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 20260825


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root from a notebook or script working directory."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root. Run the notebook from within the Spotify_clv repository."
    )


def sha256(path: str | Path) -> str:
    """Return a file SHA-256 without loading the full file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: str | Path, root: str | Path | None = None) -> dict[str, str]:
    """Hash every file below a directory for a non-destructive integrity check."""

    directory = Path(path)
    root_path = Path(root or directory.parent).resolve()
    return {
        str(file.resolve().relative_to(root_path)): sha256(file)
        for file in sorted(directory.rglob("*"))
        if file.is_file()
    }


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON artifact."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    """Write a compact, deterministic notebook execution record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment_table() -> pd.DataFrame:
    """Return the software versions relevant to the validated workflow."""

    return pd.DataFrame(
        [
            ("Python", platform.python_version()),
            ("pandas", pd.__version__),
            ("NumPy", np.__version__),
            ("SciPy", scipy.__version__),
            ("scikit-learn", sklearn.__version__),
            ("Matplotlib", matplotlib.__version__),
            ("Platform", platform.platform()),
        ],
        columns=["component", "version"],
    )


def read_chart_archive(archive_path: str | Path) -> pd.DataFrame:
    """Read the single CSV member of the preserved chart-source archive."""

    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise AssertionError(f"Expected one CSV in {archive_path}, found {csv_members}.")
        with archive.open(csv_members[0]) as handle:
            return pd.read_csv(handle)


SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")


def extract_spotify_id(value: Any) -> str | None:
    """Extract and validate a 22-character Spotify track ID from a URL or ID."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().split("?")[0].rstrip("/")
    candidate = text.rsplit("/", 1)[-1]
    return candidate if SPOTIFY_ID_PATTERN.fullmatch(candidate) else None


def regression_metrics(y: Iterable[float], prediction: Iterable[float], upper: int = 48) -> dict[str, float]:
    """Calculate the shared held-out count-prediction metrics."""

    observed = np.asarray(list(y), dtype=float)
    predicted = np.asarray(list(prediction), dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted arrays are not aligned.")
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "bias": float(np.mean(predicted - observed)),
        "observed_mean": float(np.mean(observed)),
        "predicted_mean": float(np.mean(predicted)),
        "r2_supplementary": float(r2_score(observed, predicted)),
        "outside_support_n": int(np.sum((predicted < 0) | (predicted > upper))),
    }


def assert_close(actual: float, expected: float, tolerance: float = 1e-8, label: str = "value") -> None:
    """Fail with an interpretable message when a reconciliation value drifts."""

    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label} mismatch: notebook={actual!r}, validated={expected!r}, tolerance={tolerance}."
        )


def assert_unique_tracks(frame: pd.DataFrame, label: str) -> None:
    """Assert one analysis row per Spotify track ID."""

    if "spotify_track_id" not in frame:
        raise KeyError(f"{label} has no spotify_track_id column.")
    duplicated = int(frame["spotify_track_id"].duplicated().sum())
    if duplicated:
        raise AssertionError(f"{label} contains {duplicated} duplicate track IDs.")


def save_and_show_figure(figure: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a notebook figure and display the exact saved artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    display(Image(filename=str(target)))


def concise_table(frame: pd.DataFrame, rows: int = 12, decimals: int = 3) -> pd.DataFrame:
    """Return a compact display copy without altering stored precision."""

    shown = frame.head(rows).copy()
    numeric = shown.select_dtypes(include=[np.number]).columns
    shown[numeric] = shown[numeric].round(decimals)
    return shown


def record_values(record_path: str | Path, notebook: str, values: dict[str, Any]) -> None:
    """Save headline values produced during notebook execution."""

    serialisable: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.integer,)):
            serialisable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            serialisable[key] = float(value)
        else:
            serialisable[key] = value
    write_json(record_path, {"notebook": notebook, "values": serialisable})


BLOCKED_PREDICTOR_PATTERNS = (
    "track_popularity",
    "artist_popularity",
    "artist_followers",
    "average_popularity",
    "future_rank",
    "future_stream",
    "eventual_peak",
    "total_chart_duration",
    "first_spell_duration",
    "later_spell",
    "later_reentry",
    "eligibility_flag",
)


def blocked_predictor_hits(variables: Iterable[str]) -> list[str]:
    """Identify frozen blocklist violations before any model is fitted."""

    hits = []
    for variable in variables:
        lowered = str(variable).lower()
        if lowered.startswith(("y_", "l_", "z_")) or any(
            pattern in lowered for pattern in BLOCKED_PREDICTOR_PATTERNS
        ):
            hits.append(str(variable))
    return sorted(set(hits))


def setup_notebook(notebook_name: str) -> tuple[Path, Path, Path]:
    """Initialise consistent paths and deterministic plotting for one notebook."""

    root = find_project_root()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_final_notebooks"),
    )
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    np.random.seed(SEED)
    figure_dir = root / "outputs/final_thesis_notebooks_v1/figures"
    record_dir = root / "notebooks/final_thesis_v1/execution_records"
    figure_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    return root, figure_dir, record_dir
