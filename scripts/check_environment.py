#!/usr/bin/env python3
"""Fast preflight checks for the portable thesis reproducibility release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_MANIFEST = ROOT / "provenance/processed_artifact_manifest.csv"

PACKAGES = {
    "IPython": "IPython",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "seaborn": "seaborn",
}

RUNTIME_FILES = [
    "scripts/thesis_utils.py",
    "scripts/thesis_utils_v2.py",
    "scripts/run_provisional_lyrics_hurdle_modelling.py",
    "scripts/run_provisional_robustness_uncertainty.py",
    "scripts/run_provisional_final_model_comparison.py",
    "scripts/run_supervisor_confirmed_update.py",
    "scripts/execute_final_thesis_notebooks_v2.py",
]

RUNTIME_MODULES = [
    "thesis_utils",
    "thesis_utils_v2",
    "run_provisional_lyrics_hurdle_modelling",
    "run_provisional_robustness_uncertainty",
    "run_provisional_final_model_comparison",
    "run_supervisor_confirmed_update",
]

REBUILD_SCRIPTS = [
    "scripts/audit_raw_datasets.py",
    "scripts/build_chart_survival_dataset.py",
    "scripts/build_fixed_window_professor_diagnostics.py",
    "scripts/build_pre_model_analysis_ready.py",
    "scripts/build_pre_model_remaining_eda.py",
    "scripts/build_final_pre_model_specification.py",
    "scripts/run_provisional_benchmark_modelling.py",
    "scripts/run_provisional_lyrics_hurdle_modelling.py",
    "scripts/run_provisional_robustness_uncertainty.py",
    "scripts/run_provisional_final_model_comparison.py",
    "scripts/run_supervisor_confirmed_update.py",
]

NOTEBOOKS = [
    "01_data_sources_audit_and_cohort.ipynb",
    "02_outcome_construction_and_eda.ipynb",
    "03_feature_selection_and_benchmarks.ipynb",
    "04_lyrics_tfidf_and_hurdle.ipynb",
    "05_nonlinear_ml_comparisons.ipynb",
    "06_robustness_inference_and_final_results.ipynb",
]

RAW_INPUTS = [
    {
        "path": "data/raw/charts/source_metadata/bartomiejczyewski_top200_global_2017_2021.json",
        "name": "project-created chart source metadata",
        "url": "included in this release",
        "external": False,
    },
    {
        "path": "data/raw/charts/original/spotify_top_200_weekly_global_2017_2021.zip",
        "name": "Spotify Top 200 Weekly Global 2017-2021",
        "url": "https://www.kaggle.com/datasets/bartomiejczyewski/spotify-top-200-weekly-global-20172021",
        "external": True,
    },
    {
        "path": "data/raw/spotify_550k/songs.csv",
        "name": "550K Spotify Songs: Audio, Lyrics & Genres",
        "url": "https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres",
        "external": True,
    },
    {
        "path": "data/raw/spotify_160k/data.csv",
        "name": "Spotify Dataset 1921-2020, 160k+ Tracks",
        "url": "https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks",
        "external": True,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the release without running notebooks or fitting models."
    )
    parser.add_argument(
        "--allow-missing-raw",
        action="store_true",
        help="Package-maintainer check: report external raw downloads as pending without failing.",
    )
    parser.add_argument(
        "--verify-original-artifact-hashes",
        action="store_true",
        help="Optionally verify byte-for-byte identity with the original saved artifacts.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    print("Spotify chart-longevity reproducibility preflight")
    print(f"Project root: {ROOT}")
    print(f"Python: {platform.python_version()} ({platform.platform()})")

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_preflight"),
    )

    if not (ROOT / "README.md").is_file() or not (ROOT / "scripts").is_dir():
        errors.append("Project-root resolution failed: README.md or scripts/ is missing.")

    print("\nPython packages:")
    for label, module_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"  OK {label}: {version}")
        except Exception as exc:
            errors.append(f"Python package {label} could not be imported: {exc}")

    print("\nRelease files:")
    for relative in dict.fromkeys([*RUNTIME_FILES, *REBUILD_SCRIPTS]):
        path = ROOT / relative
        if path.is_file():
            print(f"  OK {relative}")
        else:
            errors.append(f"Missing required script: {relative}")

    notebook_dir = ROOT / "notebooks/final_thesis_v2"
    for name in NOTEBOOKS:
        path = notebook_dir / name
        if not path.is_file():
            errors.append(f"Missing notebook: notebooks/final_thesis_v2/{name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("cells"), list):
                raise ValueError("missing notebook cells list")
            print(f"  OK notebooks/final_thesis_v2/{name} ({len(payload['cells'])} cells)")
        except Exception as exc:
            errors.append(f"Invalid notebook JSON: {name}: {exc}")

    print("\nLocal helper imports:")
    sys.path.insert(0, str(ROOT / "scripts"))
    for module_name in RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"  OK {module_name}")
        except Exception as exc:
            errors.append(f"Local helper import failed for {module_name}: {exc}")

    print("\nProcessed artifacts:")
    if not PROCESSED_MANIFEST.is_file():
        errors.append("Missing provenance/processed_artifact_manifest.csv")
    else:
        with PROCESSED_MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 61:
            errors.append(f"Processed manifest has {len(rows)} rows; expected 61.")
        present = 0
        for row in rows:
            path = ROOT / row["relative_path"]
            if not path.is_file():
                errors.append(f"Missing processed artifact: {row['relative_path']}")
                continue
            present += 1
            if args.verify_original_artifact_hashes:
                expected_size = int(row["size_bytes"])
                if path.stat().st_size != expected_size:
                    errors.append(
                        f"Size mismatch for {row['relative_path']}: "
                        f"expected {expected_size}, found {path.stat().st_size}"
                    )
                    continue
                if sha256(path) != row["sha256"]:
                    errors.append(f"SHA-256 mismatch for {row['relative_path']}")
        if args.verify_original_artifact_hashes:
            print(f"  {present}/{len(rows)} processed artifacts present; original-copy hashes checked")
        else:
            print(f"  {present}/{len(rows)} processed artifacts present (contents/results not validated)")

    print("\nRaw inputs:")
    for item in RAW_INPUTS:
        path = ROOT / item["path"]
        if path.is_file():
            print(f"  OK {item['path']}")
            continue
        message = (
            f"MISSING: {item['path']}\n"
            f"  Download/source: {item['name']}\n"
            f"  URL: {item['url']}\n"
            f"  Place the file at: {item['path']}"
        )
        if item["external"] and args.allow_missing_raw:
            warnings.append(message)
        else:
            errors.append(message)

    print("\nOutput-directory write checks:")
    for relative in [
        "outputs/final_thesis_notebooks_v2/figures",
        "notebooks/final_thesis_v2/execution_records",
    ]:
        directory = ROOT / relative
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".preflight_write_test"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            print(f"  OK {relative}")
        except Exception as exc:
            errors.append(f"Output directory is not writable: {relative}: {exc}")

    if warnings:
        print("\nACTION REQUIRED BEFORE NOTEBOOK EXECUTION:")
        for warning in warnings:
            print(warning)

    if errors:
        print("\nPREFLIGHT FAILED")
        for error in errors:
            print(error)
        return 1

    if warnings:
        print("\nPACKAGE CHECK PASSED; EXTERNAL RAW DOWNLOADS ARE STILL REQUIRED.")
    else:
        print("\nPREFLIGHT PASSED. Required files are present; analytical results are not validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
