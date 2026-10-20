#!/usr/bin/env python3
"""Execute the six supervisor-confirmed v2 notebooks without new packages."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_final_notebooks_v2"),
)
os.environ.setdefault(
    "IPYTHONDIR",
    str(Path(tempfile.gettempdir()) / "ipython_spotify_thesis_v2"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


NOTEBOOK_NAMES = [
    "01_data_sources_audit_and_cohort.ipynb",
    "02_outcome_construction_and_eda.ipynb",
    "03_feature_selection_and_benchmarks.ipynb",
    "04_lyrics_tfidf_and_hurdle.ipynb",
    "05_nonlinear_ml_comparisons.ipynb",
    "06_robustness_inference_and_final_results.ipynb",
]


def json_safe(value):
    """Convert captured rich output to notebook-JSON-compatible values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def execute_notebook(path: Path, root: Path) -> dict:
    """Execute code cells in order and stop at the first cell error."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    shell = InteractiveShell.instance()
    shell.reset(new_session=True)
    shell.run_line_magic("cd", str(root))
    execution_count = 0
    started = time.perf_counter()
    warning_lines = 0
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        execution_count += 1
        with capture_output() as captured:
            result = shell.run_cell(source, store_history=False, silent=False)
        outputs = []
        if captured.stdout:
            outputs.append({"output_type": "stream", "name": "stdout", "text": captured.stdout})
        if captured.stderr:
            outputs.append({"output_type": "stream", "name": "stderr", "text": captured.stderr})
            warning_lines += len([line for line in captured.stderr.splitlines() if line.strip()])
        for rich in captured.outputs:
            outputs.append({
                "output_type": "display_data",
                "data": json_safe(rich.data),
                "metadata": json_safe(rich.metadata),
            })
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(
                f"{path.name} failed in code cell {cell_index + 1}: "
                f"{type(error).__name__}: {error}"
            )
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    duration = time.perf_counter() - started
    notebook.setdefault("metadata", {})["execution"] = {
        "engine": "IPython custom runner (no dependency installation)",
        "status": "success",
        "duration_seconds": round(duration, 6),
        "project_working_directory": ".",
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "notebook": path.name,
        "status": "success",
        "code_cells": execution_count,
        "duration_seconds": duration,
        "cell_errors": 0,
        "captured_stderr_lines": warning_lines,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks/final_thesis_v2"
    missing = [name for name in NOTEBOOK_NAMES if not (notebook_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing v2 notebooks: {missing}")
    summary = []
    for name in NOTEBOOK_NAMES:
        print(f"Executing {name} ...", flush=True)
        result = execute_notebook(notebook_dir / name, root)
        summary.append(result)
        print(f"  completed in {result['duration_seconds']:.2f} seconds", flush=True)
    (notebook_dir / "notebook_execution_summary_v2.json").write_text(
        json.dumps({
            "execution_method": "IPython custom runner; no packages installed",
            "python": sys.version,
            "notebooks": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
