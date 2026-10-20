# Final analysis notebooks

These six notebooks retain the final analysis cell sources and run order. Their public copies have execution outputs/counts and internal execution metadata removed to avoid publishing track-level previews, generated images and local execution details. No model code or reported result was changed.

Run from the repository root after installing requirements and supplying the external raw data **and** all 61 processed artifacts in the manifest. The complete-data preflight must pass first. Notebook 01 verifies data provenance and cohort construction; subsequent notebooks depend on saved outputs from the established upstream workflow.

1. `01_data_sources_audit_and_cohort.ipynb` — source audits, exact-ID linkage, exclusions, grouping and cohort reconciliation.
2. `02_outcome_construction_and_eda.ipynb` — X, Y52, Y26 and descriptive L52/Z outcomes.
3. `03_feature_selection_and_benchmarks.ipynb` — leakage controls, grouped validation and F1_YEAR/F3_YEAR.
4. `04_lyrics_tfidf_and_hurdle.ipynb` — lyric checks, LEX_YEAR, H3_YEAR and text sensitivities.
5. `05_nonlinear_ml_comparisons.ipynb` — nonlinear sensitivity comparisons.
6. `06_robustness_inference_and_final_results.ipynb` — uncertainty, temporal transfer, related-version sensitivity and reconciliation.

The established computational workflow uses literal directories such as `pre_model_analysis_ready_v1`, `provisional_final_model_comparison_v1` and `supervisor_confirmed_update_v1`; these are dependency paths, not statements about the final thesis status. Do not rename them.

The canonical command is:

```bash
python scripts/execute_final_thesis_notebooks_v2.py
```

The runner updates the six local notebook files in place with new outputs/counts and writes figures/records to local output directories ignored by `.gitignore`. It does not install packages. Manual interactive execution is possible in order 01–06 with a separately installed Jupyter interface.
