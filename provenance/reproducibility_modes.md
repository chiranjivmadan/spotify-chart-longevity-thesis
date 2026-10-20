# Reproducibility modes and public-release boundary

## Source-only public candidate

The proposed public repository has the six final notebooks, runtime helpers, upstream generator scripts, package specifications, source metadata and a manifest of **61** processed/intermediate artifacts. It does **not** carry the raw datasets, processed artifacts, saved predictions or executed notebook outputs. This keeps lyric text and third-party-derived track-level data out of the proposed GitHub contents. `python scripts/check_public_package.py` checks this code-only state without running models.

The processed artifact manifest gives the original relative path, size, SHA-256, generator and notebook use for each file. The 61 files total **95,887,448 bytes** in the tested private package. Some are class E frozen/cached results; the manifest does not imply that they are newly recomputed in the six final notebooks.

## Setup and checks

The known successful environment used Python 3.9.6; package versions are in `requirements.txt` and `environment_info.txt`. From the repository root, create a Python 3.9 virtual environment and install the requirements:

```bash
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements.txt
```

On Windows, `py -3.9 -m venv .venv` selects Python 3.9, and PowerShell activation is `.\.venv\Scripts\Activate.ps1`. On macOS/Linux, activation is `source .venv/bin/activate`. These recorded versions are not a guarantee for every newer platform.

Run `python scripts/check_public_package.py` for a **code-only** inventory check; it does not run models or validate results. Run `python examples/synthetic_workflow.py` for the separately tested, invented-data illustration. Once the external sources in `data/README.md` and all 61 processed files are available locally, run `python scripts/check_environment.py` before `python scripts/execute_final_thesis_notebooks_v2.py`. The environment check verifies packages, helper imports, required paths and write access; it does not validate analytical outputs. The runner executes notebooks 01–06 and writes into the working copy. It has **not** been run from a clean copy of this data-excluding public candidate.

The manifest's hashes identify the **original saved artifacts**. By default, `check_environment.py` requires the processed paths but does not compare regenerated files with those historical hashes. Use `--verify-original-artifact-hashes` only when checking a byte-for-byte copy of the original artifacts. A regeneration check instead requires executing the analytical workflow and comparing its substantive results and reconciliation checks; matching historical file bytes is not the criterion.

## Mode A — final-result reproduction with local artifacts

The previously tested portable package ran all six notebooks on Windows using the three external raw downloads and all 61 original processed artifacts. The same runtime code and relative path structure are retained here, but this public candidate omits those artifacts. To use Mode A locally, obtain authorized copies or generate them, place them at the manifest paths, then run `python scripts/check_environment.py` and `python scripts/execute_final_thesis_notebooks_v2.py` from the repository root. The first command is expected to fail clearly when inputs are absent. This public clone by itself is **not** a ready-to-run final-results release.

The final suite mixes fresh computation and frozen-output reconciliation. The F3_YEAR–LEX_YEAR protected-component bootstrap, temporal validation, predictive NB2/bounded-count, TF-IDF, Random Forest and some Extra Trees/HGB uncertainty/tuning/temporal outputs are loaded from upstream artifacts. Extra Trees/HGB grouped predictions, related-version models and bootstrap, and a separate descriptive NB2 association are recomputed by the final suite. The distinction between predictive and descriptive NB2 analyses matters.

## Mode B — optional full raw-data rebuild

The upstream generator code is included to document provenance. The three raw files in the quick-mode table are **not sufficient** for this route: the raw audit and analysis-ready builders open the 550k `artists.csv`, and chart construction opens the extracted weekly-chart CSV. The intended sequence is:

1. `scripts/audit_raw_datasets.py`
2. `scripts/build_chart_survival_dataset.py`
3. `scripts/build_fixed_window_professor_diagnostics.py`
4. `scripts/build_pre_model_analysis_ready.py`
5. `scripts/build_pre_model_remaining_eda.py`
6. `scripts/build_final_pre_model_specification.py`
7. `scripts/run_provisional_benchmark_modelling.py`
8. `scripts/run_provisional_lyrics_hurdle_modelling.py`
9. `scripts/run_provisional_robustness_uncertainty.py`
10. `scripts/run_provisional_final_model_comparison.py`
11. `scripts/run_supervisor_confirmed_update.py`
12. `scripts/execute_final_thesis_notebooks_v2.py`

This route is computationally expensive. It was not run during public-repository preparation and has **not** been verified from a clean public clone. Generator integrity guards and additional source files may require separate setup. Do not mistake the existence of generator code for a tested one-command rebuild. Never overwrite the only validated artifact copy while trying this mode.

## Public hosting decision

Do not add `data/processed` merely because the private portable release contains it. At least the analysis-ready 550k cohort files have `lyrics_text` columns, while other files contain chart-history records, artist/track identifiers or saved predictions derived from the external sources. Redistribution of each candidate artifact requires a separate rights review. A concise account of aggregated thesis results appears in the main README; the thesis PDF is intentionally excluded.
