# Omitted processed inputs by final notebook

This map expands the 61-file artifact manifest for a first-time reader. Every path below is relative to `data/processed/` and is **absent from this public candidate**. The counts overlap because several notebooks use the same file. The manifest gives each file's SHA-256 and exact generator script.

## Notebook 01 — 13 files

- `audit/`: `raw_dataset_audit.json`
- `chart_survival/`: `matched_550k_baseline_features.csv`
- `final_pre_model_specification_v1/`: `analysis_unit_version_summary.csv`, `collaboration_grouping_diagnostic.csv`, `provisional_grouped_cv_assignments.csv`
- `fixed_window_count_v1/`: `feature_coverage_by_chart_entry_year.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_160k_26w.csv`, `analysis_ready_160k_52w.csv`, `analysis_ready_550k_26w.csv`, `analysis_ready_550k_52w.csv`, `cohort_summary.csv`
- `supervisor_confirmed_update_v1/`: `survival1357_to_y52_1014_reconciliation.csv`, `survival1357_to_y52_1014_summary.csv`

## Notebook 02 — 12 files

- `chart_survival/`: `first_spell_survival.csv`, `reentry_summary.csv`, `song_week_panel.csv`
- `final_pre_model_specification_v1/`: `Y26_Y52_increment_distribution.csv`
- `fixed_window_count_v1/`: `fixed_window_track_outcomes.csv`, `rank_200_annual_stream_summary.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_550k_26w.csv`, `analysis_ready_550k_52w.csv`, `cohort_feature_comparison.csv`, `entry_year_composition.csv`
- `supervisor_confirmed_update_v1/`: `L_Z_descriptives.csv`, `Z_distribution.csv`

## Notebook 03 — 14 files

- `final_pre_model_specification_v1/`: `provisional_grouped_cv_assignments.csv`, `provisional_temporal_validation_26w.csv`, `provisional_temporal_validation_52w.csv`, `provisional_variable_roles.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_550k_52w.csv`, `vif_diagnostic.csv`
- `provisional_final_model_comparison_v1/`: `final_coefficient_stability_audit.csv`, `final_feature_selection_record.csv`, `final_predictor_pearson_matrix.csv`
- `provisional_lyrics_hurdle_modelling_v1/`: `lyric_lexical_features_by_track.csv`
- `supervisor_confirmed_update_v1/`: `entry_year_encoding_audit.csv`, `historical_vs_year_controlled_comparison.csv`, `year_controlled_grouped_cv_metrics.csv`, `year_controlled_grouped_cv_predictions.csv`

## Notebook 04 — 15 files

- `final_pre_model_specification_v1/`: `analysis_unit_version_summary.csv`, `provisional_grouped_cv_assignments.csv`, `provisional_temporal_validation_52w.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_550k_52w.csv`, `lyric_quality_summary.csv`
- `pre_model_remaining_eda_v2/`: `duplicate_lyric_cohort_coverage_summary.csv`
- `provisional_lyrics_hurdle_modelling_v1/`: `lyric_feature_definitions.csv`, `lyric_grouped_cv_predictions.csv`, `lyric_model_specifications.csv`, `tfidf_fold_vocabulary_summary.csv`, `tfidf_regularisation_summary.csv`
- `provisional_robustness_uncertainty_v1/`: `tfidf_lower_alpha_sensitivity.csv`, `x4_paired_uncertainty.csv`
- `supervisor_confirmed_update_v1/`: `year_controlled_grouped_cv_predictions.csv`, `year_controlled_lexical_uncertainty.csv`

## Notebook 05 — 14 files

- `final_pre_model_specification_v1/`: `provisional_grouped_cv_assignments.csv`, `provisional_temporal_validation_26w.csv`, `provisional_temporal_validation_52w.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_550k_26w.csv`, `analysis_ready_550k_52w.csv`
- `provisional_final_model_comparison_v1/`: `ml_feature_importance.csv`, `ml_grouped_cv_predictions.csv`, `ml_inner_grouped_tuning.csv`, `ml_model_specifications.csv`, `ml_paired_cluster_bootstrap.csv`, `ml_temporal_metrics_52w.csv`, `thesis_final_model_comparison.csv`
- `provisional_robustness_uncertainty_v1/`: `robustness_validation_manifest.json`
- `supervisor_confirmed_update_v1/`: `year_controlled_grouped_cv_metrics.csv`

## Notebook 06 — 19 files

- `final_pre_model_specification_v1/`: `analysis_unit_version_audit.csv`, `provisional_grouped_cv_assignments.csv`, `provisional_temporal_validation_26w.csv`, `provisional_temporal_validation_52w.csv`
- `pre_model_analysis_ready_v1/`: `analysis_ready_550k_26w.csv`, `analysis_ready_550k_52w.csv`, `high_correlation_pairs.csv`
- `provisional_final_model_comparison_v1/`: `clustered_ols_coefficients.csv`, `final_candidate_model_roles.csv`, `final_finding_hierarchy.csv`, `final_robustness_evidence_matrix.csv`, `thesis_final_model_comparison.csv`
- `provisional_robustness_uncertainty_v1/`: `robustness_validation_manifest.json`
- `supervisor_confirmed_update_v1/`: `L_Z_descriptives.csv`, `survival1357_to_y52_1014_summary.csv`, `year_controlled_grouped_cv_metrics.csv`, `year_controlled_lexical_uncertainty.csv`, `year_controlled_related_version_sensitivity.csv`, `year_controlled_temporal_metrics.csv`

## Regeneration status

All 61 manifest records name a generator file, and those named scripts are included under `scripts/`. This is **code provenance**, not proof that a clean public clone can regenerate the files. The documented three-file quick-mode download is insufficient for Mode B: `scripts/audit_raw_datasets.py` and `scripts/build_pre_model_analysis_ready.py` read the 550k `artists.csv`, while chart construction reads the extracted weekly-chart CSV. The complete upstream sequence also depends on earlier generated intermediates; its clean-machine execution has not been tested here. The 160k `data_w_genres.csv` appears in metadata/genre-provenance descriptions but is not opened by the scanned final generator code.

The tested private Mode A package already contains the 61 artifacts and was run successfully on Windows. That result does not transfer automatically to this data-excluding public candidate. Never describe `check_public_package.py` as an end-to-end notebook test.
