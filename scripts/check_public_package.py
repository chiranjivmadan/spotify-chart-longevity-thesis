#!/usr/bin/env python3
"""Fast, read-only check of the public code candidate; never runs models."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks/final_thesis_v2"
EXPECTED_NOTEBOOKS = [
    "01_data_sources_audit_and_cohort.ipynb",
    "02_outcome_construction_and_eda.ipynb",
    "03_feature_selection_and_benchmarks.ipynb",
    "04_lyrics_tfidf_and_hurdle.ipynb",
    "05_nonlinear_ml_comparisons.ipynb",
    "06_robustness_inference_and_final_results.ipynb",
]
EXPECTED_SCRIPTS = [
    "audit_raw_datasets.py",
    "build_chart_survival_dataset.py",
    "build_final_pre_model_specification.py",
    "build_fixed_window_professor_diagnostics.py",
    "build_pre_model_analysis_ready.py",
    "build_pre_model_remaining_eda.py",
    "check_environment.py",
    "execute_final_thesis_notebooks_v2.py",
    "run_provisional_benchmark_modelling.py",
    "run_provisional_final_model_comparison.py",
    "run_provisional_lyrics_hurdle_modelling.py",
    "run_provisional_robustness_uncertainty.py",
    "run_supervisor_confirmed_update.py",
    "thesis_utils.py",
    "thesis_utils_v2.py",
]


def main() -> int:
    errors: list[str] = []
    print(f"Public package root: {ROOT}")
    print(f"Python: {platform.python_version()}")

    for name in EXPECTED_SCRIPTS:
        if not (ROOT / "scripts" / name).is_file():
            errors.append(f"Missing script: scripts/{name}")

    for name in EXPECTED_NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        if not path.is_file():
            errors.append(f"Missing notebook: notebooks/final_thesis_v2/{name}")
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                raise ValueError("missing cells list")
            for index, cell in enumerate(notebook["cells"], 1):
                if cell.get("cell_type") == "code" and (
                    cell.get("outputs") or cell.get("execution_count") is not None
                ):
                    errors.append(f"Notebook {name} cell {index} contains execution output")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid notebook {name}: {exc}")

    manifest = ROOT / "provenance/processed_artifact_manifest.csv"
    if not manifest.is_file():
        errors.append("Missing processed-artifact manifest")
    else:
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 61:
            errors.append(f"Expected 61 processed-artifact records, found {len(rows)}")
        present = sum((ROOT / row["relative_path"]).is_file() for row in rows)
        print(f"Processed artifacts present locally: {present}/{len(rows)}")
        if present:
            print("NOTICE: local processed files should remain excluded from any public upload.")

    if (ROOT / "thesis/final_thesis.pdf").exists():
        errors.append("Thesis PDF must remain excluded from the public candidate")
    if not (ROOT / "data/README.md").is_file():
        errors.append("Missing external-data guide")

    if errors:
        print("PUBLIC PACKAGE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PUBLIC PACKAGE CHECK PASSED (code/docs only; data-run readiness not tested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
