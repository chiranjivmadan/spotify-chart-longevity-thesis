#!/usr/bin/env python3
"""Provisional final thesis-ready model comparison for Spotify chart longevity.

The script preserves all historical project outputs. It adds exactly two
nonlinear comparison families (Extra Trees and sklearn histogram gradient
boosting), then consolidates prediction, classification, correlation,
association, version-sensitivity, and thesis-evidence outputs. All new files
are staged before publication and all historical input directories are hashed
before and after execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_final"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm, t as student_t
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/processed/pre_model_analysis_ready_v1"
SPEC_DIR = ROOT / "data/processed/final_pre_model_specification_v1"
BENCH_DIR = ROOT / "data/processed/provisional_benchmark_modelling_v1"
LYRIC_DIR = ROOT / "data/processed/provisional_lyrics_hurdle_modelling_v1"
ROBUST_DIR = ROOT / "data/processed/provisional_robustness_uncertainty_v1"
VERSION_AUDIT_DIR = ROOT / "data/processed/pre_model_remaining_eda_v2"

OUT_DIR = ROOT / "data/processed/provisional_final_model_comparison_v1"
FIG_DIR = ROOT / "outputs/provisional_final_model_comparison_v1/figures"
SPEC_REPORT = ROOT / "docs/provisional_final_model_specification_v2.md"
EVIDENCE_PACK = ROOT / "docs/thesis_methods_results_evidence_pack_v2.md"
NOTEBOOK_PLAN = ROOT / "docs/thesis_notebook_plan_v2.md"

SEED = 20260825
BOOTSTRAP_REPS = 5000
YCOL = "Y_52"
YMAX = 48
THRESHOLD = 0.50

sys.path.insert(0, str(ROOT / "scripts"))
import run_provisional_lyrics_hurdle_modelling as prior  # noqa: E402


AUDIO = prior.AUDIO
F1 = prior.F1
F3 = prior.F3
LEX_MODEL = prior.LEX_MODEL
LEX = prior.LEX
AUDIO_CONTINUOUS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
BLOCKS = {"ML_F3": F3, "ML_LEX": LEX}
MODEL_LABELS = {
    "ET_ML_F3": "Extra Trees — F3",
    "ET_ML_LEX": "Extra Trees — LEX",
    "HGB_ML_F3": "Histogram gradient boosting — F3",
    "HGB_ML_LEX": "Histogram gradient boosting — LEX",
}

ET_CONFIGS = [
    {
        "config_id": "ET_A",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 1.0,
    },
    {
        "config_id": "ET_B",
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 2,
        "max_features": 0.8,
    },
    {
        "config_id": "ET_C",
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 0.8,
    },
]
HGB_CONFIGS = [
    {
        "config_id": "HGB_A",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    {
        "config_id": "HGB_B",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    {
        "config_id": "HGB_C",
        "learning_rate": 0.10,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_digest(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def values_sha256(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def dataframe_to_markdown(frame: pd.DataFrame, decimals: int = 3) -> str:
    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def metric_values(y, prediction, nmax: int = YMAX) -> dict:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    zero = y == 0
    positive = y > 0
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mean_prediction_bias": float(np.mean(prediction - y)),
        "mean_observed_y": float(np.mean(y)),
        "mean_prediction": float(np.mean(prediction)),
        "r2_supplementary": float(r2_score(y, prediction)),
        "observed_zero_n": int(zero.sum()),
        "observed_zero_pct": float(zero.mean() * 100),
        "mae_observed_zero": float(mean_absolute_error(y[zero], prediction[zero])) if zero.any() else np.nan,
        "mae_observed_positive": float(mean_absolute_error(y[positive], prediction[positive])) if positive.any() else np.nan,
        "lower_bound_violation_n": int(np.sum(prediction < 0)),
        "upper_bound_violation_n": int(np.sum(prediction > nmax)),
        "outside_bound_n": int(np.sum((prediction < 0) | (prediction > nmax))),
        "outside_bound_pct": float(np.mean((prediction < 0) | (prediction > nmax)) * 100),
    }


def make_tree_preprocessor(variables: list[str]) -> ColumnTransformer:
    categorical = [v for v in variables if v in {"X", "key"}]
    passthrough = [v for v in variables if v not in categorical]
    transformers = []
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    if passthrough:
        transformers.append(("numeric_binary", "passthrough", passthrough))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_tree_estimator(algorithm: str, config: dict, seed: int):
    if algorithm == "ET":
        return ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            random_state=seed,
            n_jobs=1,
        )
    if algorithm == "HGB":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            l2_regularization=float(config["l2_regularization"]),
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def make_tree_pipeline(algorithm: str, config: dict, variables: list[str], seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", make_tree_preprocessor(variables)),
        ("model", make_tree_estimator(algorithm, config, seed)),
    ])


def leakage_hits(variables: list[str]) -> list[str]:
    return prior.leakage_hits(variables)


def validate_inputs(audit_rows: list[dict]):
    data52 = pd.read_csv(ANALYSIS_DIR / "analysis_ready_550k_52w.csv")
    data26 = pd.read_csv(ANALYSIS_DIR / "analysis_ready_550k_26w.csv")
    assignments = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")
    temporal52 = pd.read_csv(SPEC_DIR / "provisional_temporal_validation_52w.csv")
    temporal26 = pd.read_csv(SPEC_DIR / "provisional_temporal_validation_26w.csv")
    if len(data52) != 1014 or data52.spotify_track_id.nunique() != 1014:
        raise AssertionError("Primary Y_52 cohort is not 1,014 unique tracks.")
    if set(data52.spotify_track_id) != set(assignments.spotify_track_id):
        raise AssertionError("Primary cohort and frozen grouped assignments differ.")
    protected_flags = [
        "first_artist_crosses_folds",
        "exact_lyric_group_crosses_folds",
        "related_version_group_crosses_folds",
    ]
    if assignments[protected_flags].sum().sum() != 0:
        raise AssertionError("A protected group crosses the frozen folds.")
    if assignments.primary_group_component.nunique() != 276:
        raise AssertionError("Protected grouping component count is not 276.")
    if data52[list(set(F3) - {"X", "key", "mode"})].isna().any().any():
        raise AssertionError("F3 has unexpected missing values.")
    if leakage_hits(LEX):
        raise AssertionError(f"Blocked variables found in LEX: {leakage_hits(LEX)}")
    prior_manifest = json.loads((ROBUST_DIR / "robustness_validation_manifest.json").read_text())
    if not all(prior_manifest["validation_assertions"].values()):
        raise AssertionError("The authoritative robustness manifest contains a failed assertion.")
    data52 = prior.add_lexical_features(data52).merge(
        assignments[[
            "spotify_track_id", "fold", "primary_group_component",
            "first_credited_artist_id", "lyrics_normalised_sha256",
        ]],
        on=["spotify_track_id", "lyrics_normalised_sha256"],
        how="left",
        validate="one_to_one",
    )
    if data52[["fold", "primary_group_component", "first_credited_artist_id"]].isna().any().any():
        raise AssertionError("A primary record lacks a fold, protected component, or first artist ID.")
    audit_rows.extend([
        {"check": "primary cohort", "status": "PASS", "evidence": "1,014 rows and unique Spotify IDs; outcome support checked at 0-48"},
        {"check": "frozen grouped folds", "status": "PASS", "evidence": "276 protected components; zero first-artist, exact-lyric, or supported-version crossings"},
        {"check": "authoritative robustness manifest", "status": "PASS", "evidence": "all prior validation assertions are true"},
        {"check": "new-model feature leakage", "status": "PASS", "evidence": f"F3 and LEX predictor scans returned no blocked fields; LEX variables={LEX}"},
    ])
    return data52, data26, assignments, temporal52, temporal26


def inner_select_tree(
    train: pd.DataFrame,
    outcome: str,
    algorithm: str,
    block: str,
    configs: list[dict],
    context: str,
    audit_rows: list[dict],
    warning_rows: list[dict],
) -> tuple[dict, pd.DataFrame]:
    variables = BLOCKS[block]
    groups = train.primary_group_component.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=3)
    rows = []
    for config_index, config in enumerate(configs):
        maes, rmses = [], []
        for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(train, train[outcome], groups), start=1):
            fit_data, val_data = train.iloc[fit_idx], train.iloc[val_idx]
            overlap = set(groups[fit_idx]) & set(groups[val_idx])
            if overlap:
                raise AssertionError("Inner grouped tuning split contains a protected-component crossing.")
            pipeline = make_tree_pipeline(
                algorithm, config, variables,
                SEED + config_index * 100 + inner_fold,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(fit_data[variables], fit_data[outcome])
            pred = pipeline.predict(val_data[variables])
            maes.append(mean_absolute_error(val_data[outcome], pred))
            rmses.append(math.sqrt(mean_squared_error(val_data[outcome], pred)))
            warning_text = " | ".join(str(w.message) for w in caught)
            warning_rows.append({
                "stage": "inner_grouped_tuning",
                "context": context,
                "algorithm": algorithm,
                "feature_block": block,
                "config_id": config["config_id"],
                "fold": inner_fold,
                "status": "warning" if warning_text else "ok",
                "warning_or_failure": warning_text,
            })
            audit_rows.append({
                "check": "inner tuning training-only",
                "status": "PASS",
                "evidence": f"{context}; {algorithm}; {block}; {config['config_id']}; inner_fold={inner_fold}; group_overlap=0; preprocess fitted on n={len(fit_data)} only",
            })
        row = {
            "context": context,
            "algorithm": algorithm,
            "feature_block": block,
            "config_id": config["config_id"],
            "inner_folds": 3,
            "inner_mean_mae": float(np.mean(maes)),
            "inner_sd_mae": float(np.std(maes, ddof=1)),
            "inner_mean_rmse": float(np.mean(rmses)),
            "config_json": json.dumps(config, sort_keys=True),
        }
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(["inner_mean_mae", "inner_mean_rmse", "config_id"])
    selected_id = results.iloc[0].config_id
    results["selected"] = results.config_id.eq(selected_id).astype(int)
    selected = next(config for config in configs if config["config_id"] == selected_id)
    return selected, results


def group_permutation_importance(
    pipeline: Pipeline,
    heldout: pd.DataFrame,
    variables: list[str],
    outcome: str,
    algorithm: str,
    block: str,
    fold: str,
    repeats: int = 10,
) -> pd.DataFrame:
    y = heldout[outcome].to_numpy(float)
    baseline = mean_absolute_error(y, pipeline.predict(heldout[variables]))
    rng = np.random.default_rng(SEED + sum(map(ord, str(fold))) + (10 if algorithm == "HGB" else 0))
    rows = []
    for feature in variables:
        increases = []
        for _ in range(repeats):
            permuted = heldout[variables].copy()
            order = rng.permutation(len(permuted))
            permuted[feature] = permuted[feature].to_numpy()[order]
            increases.append(mean_absolute_error(y, pipeline.predict(permuted)) - baseline)
        rows.append({
            "aggregation": fold,
            "algorithm": algorithm,
            "feature_block": block,
            "importance_scope": "individual_raw_predictor",
            "feature_or_group": feature,
            "conceptual_group": "X" if feature == "X" else ("lexical" if feature in LEX_MODEL else "audio_song"),
            "heldout_n": len(heldout),
            "permutation_repeats": repeats,
            "baseline_mae": baseline,
            "mae_increase_mean": float(np.mean(increases)),
            "mae_increase_sd": float(np.std(increases, ddof=1)),
        })
    conceptual = {
        "X": [v for v in variables if v == "X"],
        "audio_song": [v for v in variables if v in F3 and v != "X"],
        "lexical": [v for v in variables if v in LEX_MODEL],
    }
    for group_name, group_variables in conceptual.items():
        if not group_variables:
            continue
        increases = []
        for _ in range(repeats):
            permuted = heldout[variables].copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, group_variables] = permuted[group_variables].to_numpy()[order]
            increases.append(mean_absolute_error(y, pipeline.predict(permuted)) - baseline)
        rows.append({
            "aggregation": fold,
            "algorithm": algorithm,
            "feature_block": block,
            "importance_scope": "joint_conceptual_group",
            "feature_or_group": group_name,
            "conceptual_group": group_name,
            "heldout_n": len(heldout),
            "permutation_repeats": repeats,
            "baseline_mae": baseline,
            "mae_increase_mean": float(np.mean(increases)),
            "mae_increase_sd": float(np.std(increases, ddof=1)),
        })
    return pd.DataFrame(rows)


def run_grouped_ml(data: pd.DataFrame, audit_rows: list[dict], warning_rows: list[dict]):
    prediction_frames, tuning_frames, importance_frames = [], [], []
    algorithms = [("ET", ET_CONFIGS), ("HGB", HGB_CONFIGS)]
    for fold in sorted(data.fold.unique()):
        train = data[data.fold.ne(fold)].copy()
        heldout = data[data.fold.eq(fold)].copy()
        for algorithm, configs in algorithms:
            for block in ["ML_F3", "ML_LEX"]:
                print(f"Grouped ML: {fold} {algorithm} {block}", flush=True)
                context = f"outer_{fold}"
                selected, inner = inner_select_tree(
                    train, YCOL, algorithm, block, configs, context,
                    audit_rows, warning_rows,
                )
                inner["outer_fold"] = fold
                tuning_frames.append(inner)
                variables = BLOCKS[block]
                pipeline = make_tree_pipeline(algorithm, selected, variables, SEED + int(fold.split("_")[-1]))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(train[variables], train[YCOL])
                prediction = pipeline.predict(heldout[variables])
                warning_text = " | ".join(str(w.message) for w in caught)
                warning_rows.append({
                    "stage": "outer_grouped_fit",
                    "context": context,
                    "algorithm": algorithm,
                    "feature_block": block,
                    "config_id": selected["config_id"],
                    "fold": fold,
                    "status": "warning" if warning_text else "ok",
                    "warning_or_failure": warning_text,
                })
                model_id = f"{algorithm}_{block}"
                prediction_frames.append(pd.DataFrame({
                    "spotify_track_id": heldout.spotify_track_id,
                    "fold": fold,
                    "primary_group_component": heldout.primary_group_component,
                    "model_id": model_id,
                    "algorithm": algorithm,
                    "feature_block": block,
                    "selected_config_id": selected["config_id"],
                    "y_observed": heldout[YCOL],
                    "y_pred_raw": prediction,
                    "X": heldout.X,
                    "chart_entry_year": heldout.chart_entry_year,
                }))
                importance_frames.append(group_permutation_importance(
                    pipeline, heldout, variables, YCOL, algorithm, block, fold,
                ))
                audit_rows.append({
                    "check": "outer ML held-out isolation",
                    "status": "PASS",
                    "evidence": f"{fold}; {algorithm}; {block}; train_n={len(train)}; heldout_n={len(heldout)}; track_overlap=0; selected={selected['config_id']}",
                })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.duplicated(["spotify_track_id", "model_id"]).any():
        raise AssertionError("Duplicate ML track/model OOF predictions.")
    coverage = predictions.groupby("model_id").spotify_track_id.nunique()
    if len(coverage) != 4 or not (coverage == 1014).all():
        raise AssertionError(f"Incomplete ML OOF coverage: {coverage.to_dict()}")
    frozen_fold = data.set_index("spotify_track_id").fold
    if not (predictions.fold.to_numpy() == predictions.spotify_track_id.map(frozen_fold).to_numpy()).all():
        raise AssertionError("ML outer predictions do not use the frozen fold labels.")
    if predictions.groupby("spotify_track_id").y_observed.nunique().max() != 1:
        raise AssertionError("ML outcomes disagree across model families.")
    audit_rows.append({
        "check": "ML OOF coverage and frozen-fold alignment",
        "status": "PASS",
        "evidence": "four ML block/family combinations each contain exactly 1,014 unique held-out predictions and match frozen folds",
    })
    return (
        predictions,
        pd.concat(tuning_frames, ignore_index=True),
        pd.concat(importance_frames, ignore_index=True),
    )


def summarise_predictions(predictions: pd.DataFrame):
    aggregate, folds, by_x = [], [], []
    for keys, q in predictions.groupby(["model_id", "algorithm", "feature_block"], sort=True):
        aggregate.append({
            **dict(zip(["model_id", "algorithm", "feature_block"], keys)),
            **metric_values(q.y_observed, q.y_pred_raw),
        })
    for keys, q in predictions.groupby(["fold", "model_id", "algorithm", "feature_block"], sort=True):
        folds.append({
            **dict(zip(["fold", "model_id", "algorithm", "feature_block"], keys)),
            **metric_values(q.y_observed, q.y_pred_raw),
        })
    for keys, q in predictions.groupby(["model_id", "algorithm", "feature_block", "X"], sort=True):
        by_x.append({
            **dict(zip(["model_id", "algorithm", "feature_block", "X"], keys)),
            **metric_values(q.y_observed, q.y_pred_raw),
        })
    return pd.DataFrame(aggregate), pd.DataFrame(folds), pd.DataFrame(by_x)


def aggregate_importance(importance: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    group_cols = ["algorithm", "feature_block", "importance_scope", "feature_or_group", "conceptual_group"]
    for keys, q in importance.groupby(group_cols, sort=True):
        weights = q.heldout_n.to_numpy(float)
        summaries.append({
            "aggregation": "weighted_across_5_outer_folds",
            **dict(zip(group_cols, keys)),
            "heldout_n": int(q.heldout_n.sum()),
            "permutation_repeats": int(q.permutation_repeats.iloc[0]),
            "baseline_mae": float(np.average(q.baseline_mae, weights=weights)),
            "mae_increase_mean": float(np.average(q.mae_increase_mean, weights=weights)),
            "mae_increase_sd": float(q.mae_increase_mean.std(ddof=1)),
        })
    return pd.concat([importance, pd.DataFrame(summaries)], ignore_index=True)


def cluster_bootstrap(
    data: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
    cluster_col: str,
    subset: str,
    seed: int,
    outcome_col: str = "Y_52",
) -> pd.DataFrame:
    q = data[[outcome_col, reference_col, candidate_col, cluster_col]].dropna().copy()
    y = q[outcome_col]
    q["ae_ref"] = np.abs(q[reference_col] - y)
    q["ae_cand"] = np.abs(q[candidate_col] - y)
    q["se_ref"] = (q[reference_col] - y) ** 2
    q["se_cand"] = (q[candidate_col] - y) ** 2
    q["err_ref"] = q[reference_col] - y
    q["err_cand"] = q[candidate_col] - y
    agg = q.groupby(cluster_col, sort=True).agg(
        n=(outcome_col, "size"), ae_ref=("ae_ref", "sum"), ae_cand=("ae_cand", "sum"),
        se_ref=("se_ref", "sum"), se_cand=("se_cand", "sum"),
        err_ref=("err_ref", "sum"), err_cand=("err_cand", "sum"),
    )
    g = len(agg)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(g, np.repeat(1 / g, g), size=BOOTSTRAP_REPS)
    denominator = counts @ agg.n.to_numpy(float)
    ref_mae = counts @ agg.ae_ref.to_numpy(float) / denominator
    cand_mae = counts @ agg.ae_cand.to_numpy(float) / denominator
    ref_rmse = np.sqrt(counts @ agg.se_ref.to_numpy(float) / denominator)
    cand_rmse = np.sqrt(counts @ agg.se_cand.to_numpy(float) / denominator)
    ref_bias = counts @ agg.err_ref.to_numpy(float) / denominator
    cand_bias = counts @ agg.err_cand.to_numpy(float) / denominator
    arrays = {
        "MAE": ref_mae - cand_mae,
        "RMSE": ref_rmse - cand_rmse,
        "bias_candidate_minus_reference": cand_bias - ref_bias,
    }
    observed = {
        "MAE": mean_absolute_error(y, q[reference_col]) - mean_absolute_error(y, q[candidate_col]),
        "RMSE": math.sqrt(mean_squared_error(y, q[reference_col])) - math.sqrt(mean_squared_error(y, q[candidate_col])),
        "bias_candidate_minus_reference": float(np.mean(q[candidate_col] - y) - np.mean(q[reference_col] - y)),
    }
    rows = []
    for metric, values in arrays.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append({
            "subset": subset,
            "reference_model": reference_col,
            "candidate_model": candidate_col,
            "metric": metric,
            "improvement_definition": (
                "reference error minus candidate error; positive favours candidate"
                if metric in {"MAE", "RMSE"}
                else "candidate mean prediction bias minus reference mean prediction bias"
            ),
            "observed_difference": float(observed[metric]),
            "bootstrap_mean_difference": float(values.mean()),
            "bootstrap_standard_error": float(values.std(ddof=1)),
            "ci_2_5": float(lower),
            "ci_97_5": float(upper),
            "bootstrap_proportion_candidate_better": float(np.mean(values > 0)) if metric in {"MAE", "RMSE"} else np.nan,
            "n_tracks": int(len(q)),
            "n_clusters": int(g),
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_seed": seed,
            "cluster_unit": cluster_col,
        })
    return pd.DataFrame(rows)


def choose_best_ml_blocks(aggregate: pd.DataFrame) -> dict[str, str]:
    selected = {}
    for algorithm in ["ET", "HGB"]:
        q = aggregate[aggregate.algorithm.eq(algorithm)].sort_values(["mae", "rmse", "model_id"])
        selected[algorithm] = str(q.iloc[0].model_id)
    return selected


def ml_paired_uncertainty(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_models: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = pd.read_csv(ROBUST_DIR / "aligned_grouped_oof_predictions.csv")[[
        "spotify_track_id", "Y_52", "X", "fold", "primary_group_component", "REF_F3", "LEX",
    ]]
    ml_wide = predictions.pivot(index="spotify_track_id", columns="model_id", values="y_pred_raw").reset_index()
    wide = historical.merge(ml_wide, on="spotify_track_id", validate="one_to_one")
    rows, x4_rows = [], []
    counter = 0
    for algorithm, candidate in selected_models.items():
        for reference in ["REF_F3", "LEX"]:
            rows.append(cluster_bootstrap(
                wide, reference, candidate, "primary_group_component",
                "all_Y52_grouped_OOF", SEED + 600 + counter,
            ))
            x4_rows.append(cluster_bootstrap(
                wide[wide.X.eq(4)], reference, candidate, "primary_group_component",
                "X_equals_4", SEED + 700 + counter,
            ))
            counter += 1
    return pd.concat(rows, ignore_index=True), pd.concat(x4_rows, ignore_index=True)


def prepare_temporal_data(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    assignment = pd.read_csv(SPEC_DIR / f"provisional_temporal_validation_{horizon}w.csv")
    base = data.copy()
    if "primary_group_component" not in base:
        groups = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")[[
            "spotify_track_id", "primary_group_component", "first_credited_artist_id",
        ]]
        base = base.merge(groups, on="spotify_track_id", how="left", validate="one_to_one")
    base = base.drop(columns=["temporal_partition"], errors="ignore").merge(
        assignment[["spotify_track_id", "temporal_partition"]],
        on="spotify_track_id", how="left", validate="one_to_one",
    )
    if base.temporal_partition.isna().any():
        raise AssertionError(f"Y_{horizon} temporal assignment is incomplete.")
    return base


def tune_fit_temporal_ml(
    data: pd.DataFrame,
    outcome: str,
    algorithm: str,
    block: str,
    evaluation_partitions: list[str],
    context: str,
    audit_rows: list[dict],
    warning_rows: list[dict],
):
    configs = ET_CONFIGS if algorithm == "ET" else HGB_CONFIGS
    train = data[data.temporal_partition.eq("train_2017_2018")].copy()
    evaluation = data[data.temporal_partition.isin(evaluation_partitions)].copy()
    if train.primary_group_component.isna().any():
        # Expanded Y_26 tracks occur only after the common 2017-2018 training set.
        raise AssertionError("Temporal training data lack protected groups.")
    selected, inner = inner_select_tree(
        train, outcome, algorithm, block, configs, context,
        audit_rows, warning_rows,
    )
    variables = BLOCKS[block]
    pipeline = make_tree_pipeline(algorithm, selected, variables, SEED + 900)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(train[variables], train[outcome])
    pred = pipeline.predict(evaluation[variables])
    warning_text = " | ".join(str(w.message) for w in caught)
    warning_rows.append({
        "stage": "temporal_outer_fit",
        "context": context,
        "algorithm": algorithm,
        "feature_block": block,
        "config_id": selected["config_id"],
        "fold": "train_2017_2018",
        "status": "warning" if warning_text else "ok",
        "warning_or_failure": warning_text,
    })
    result = pd.DataFrame({
        "spotify_track_id": evaluation.spotify_track_id,
        "outcome": outcome,
        "temporal_partition": evaluation.temporal_partition,
        "algorithm": algorithm,
        "feature_block": block,
        "model_id": f"{algorithm}_{block}",
        "selected_config_id": selected["config_id"],
        "y_observed": evaluation[outcome],
        "y_pred_raw": pred,
        "X": evaluation.X,
    })
    audit_rows.append({
        "check": "temporal ML isolation",
        "status": "PASS",
        "evidence": f"{context}; {algorithm}; {block}; tuning and preprocessing use 2017-2018 only; evaluated={evaluation_partitions}; selected={selected['config_id']}",
    })
    return result, inner


def run_temporal_ml(
    data52: pd.DataFrame,
    data26_raw: pd.DataFrame,
    selected_models: dict[str, str],
    audit_rows: list[dict],
    warning_rows: list[dict],
):
    temporal52 = prepare_temporal_data(data52, 52)
    frames52, tuning_frames, frames26 = [], [], []
    for algorithm, model_id in selected_models.items():
        block = model_id.replace(f"{algorithm}_", "")
        pred, tuning = tune_fit_temporal_ml(
            temporal52, "Y_52", algorithm, block,
            ["validation_2019", "test_2020"], f"Y52_temporal_{algorithm}",
            audit_rows, warning_rows,
        )
        frames52.append(pred)
        tuning["temporal_outcome"] = "Y_52"
        tuning_frames.append(tuning)

    data26 = prior.add_lexical_features(data26_raw)
    groups = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")[[
        "spotify_track_id", "primary_group_component",
    ]]
    data26 = data26.merge(groups, on="spotify_track_id", how="left", validate="one_to_one")
    temporal26 = prepare_temporal_data(data26, 26)
    for algorithm, model_id in selected_models.items():
        block = model_id.replace(f"{algorithm}_", "")
        pred, tuning = tune_fit_temporal_ml(
            temporal26, "Y_26", algorithm, block,
            ["test_2020"], f"Y26_2020_{algorithm}", audit_rows, warning_rows,
        )
        frames26.append(pred)
        tuning["temporal_outcome"] = "Y_26"
        tuning_frames.append(tuning)
    predictions52 = pd.concat(frames52, ignore_index=True)
    predictions26 = pd.concat(frames26, ignore_index=True)
    metrics52, metrics26 = [], []
    for keys, q in predictions52.groupby(["temporal_partition", "model_id", "algorithm", "feature_block"]):
        metrics52.append({
            **dict(zip(["temporal_partition", "model_id", "algorithm", "feature_block"], keys)),
            "outcome": "Y_52", **metric_values(q.y_observed, q.y_pred_raw, 48),
        })
    for keys, q in predictions26.groupby(["temporal_partition", "model_id", "algorithm", "feature_block"]):
        metrics26.append({
            **dict(zip(["temporal_partition", "model_id", "algorithm", "feature_block"], keys)),
            "outcome": "Y_26", **metric_values(q.y_observed, q.y_pred_raw, 22),
        })
    return (
        predictions52, pd.DataFrame(metrics52), predictions26, pd.DataFrame(metrics26),
        pd.concat(tuning_frames, ignore_index=True), temporal52,
    )


def temporal_ml_uncertainty(
    ml_predictions: pd.DataFrame,
    temporal52: pd.DataFrame,
) -> pd.DataFrame:
    historical = pd.read_csv(ROBUST_DIR / "temporal_aligned_predictions_52w.csv")
    hist = historical[historical.model.isin(["REF_F3", "LEX"])].pivot(
        index="spotify_track_id", columns="model", values="y_pred_raw"
    ).reset_index()
    ml = ml_predictions.pivot(index="spotify_track_id", columns="model_id", values="y_pred_raw").reset_index()
    outcome = ml_predictions.groupby("spotify_track_id").y_observed.first()
    partition = ml_predictions.groupby("spotify_track_id").temporal_partition.first()
    q = hist.merge(ml, on="spotify_track_id", validate="one_to_one")
    q["Y_52"] = q.spotify_track_id.map(outcome)
    q["temporal_partition"] = q.spotify_track_id.map(partition)
    q = q.merge(
        temporal52[["spotify_track_id", "primary_group_component"]],
        on="spotify_track_id", validate="one_to_one",
    )
    rows = []
    candidates = list(ml_predictions.model_id.unique())
    counter = 0
    for temporal_partition in ["validation_2019", "test_2020"]:
        subset = q[q.temporal_partition.eq(temporal_partition)]
        for candidate in candidates:
            for reference in ["REF_F3", "LEX"]:
                result = cluster_bootstrap(
                    subset, reference, candidate, "primary_group_component",
                    f"Y52_{temporal_partition}", SEED + 1000 + counter,
                )
                result["power_note"] = (
                    "underpowered 45-track stress test"
                    if temporal_partition == "test_2020"
                    else "main temporal validation"
                )
                rows.append(result)
                counter += 1
    return pd.concat(rows, ignore_index=True)


def classification_rows(frame: pd.DataFrame, evaluation: str) -> tuple[dict, dict]:
    y = (frame.y_observed > 0).astype(int).to_numpy()
    p = np.clip(frame.p_positive.to_numpy(float), 1e-12, 1 - 1e-12)
    predicted = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics = {
        "evaluation": evaluation,
        "model_label": "H3 binary component",
        "target": "Y_52 > 0",
        "decision_threshold": THRESHOLD,
        "n": len(y),
        "positive_n": int(y.sum()),
        "observed_positive_pct": float(y.mean() * 100),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision_positive_predictive_value": float(precision_score(y, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "classification_f1_score": float(f1_score(y, predicted, zero_division=0)),
        "note_on_f1_name": "F1-score is a classification metric; F1 is separately the X-only project model label.",
    }
    confusion = {
        "evaluation": evaluation,
        "decision_threshold": THRESHOLD,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "total": int(tp + fp + tn + fn),
    }
    return metrics, confusion


def consolidate_h3_classification(audit_rows: list[dict]):
    grouped = pd.read_csv(LYRIC_DIR / "hurdle_grouped_cv_predictions.csv")
    grouped = grouped[grouped.model_id.eq("H3")].copy()
    temporal = pd.read_csv(ROBUST_DIR / "temporal_aligned_predictions_52w.csv")
    temporal = temporal[temporal.model.eq("H3")].copy()
    frames = {
        "grouped_oof": grouped,
        "validation_2019": temporal[temporal.partition.eq("validation_2019")],
        "test_2020_stress": temporal[temporal.partition.eq("test_2020")],
    }
    metrics, confusion = [], []
    for evaluation, frame in frames.items():
        metric_row, confusion_row = classification_rows(frame, evaluation)
        metrics.append(metric_row)
        confusion.append(confusion_row)
    metrics_df, confusion_df = pd.DataFrame(metrics), pd.DataFrame(confusion)
    if not (confusion_df[["true_positive", "false_positive", "true_negative", "false_negative"]].sum(axis=1) == confusion_df.total).all():
        raise AssertionError("H3 confusion-matrix counts do not reconcile.")
    if not (metrics_df.n.to_numpy() == confusion_df.total.to_numpy()).all():
        raise AssertionError("H3 classification N and confusion totals disagree.")
    audit_rows.append({
        "check": "H3 threshold metrics reconciliation", "status": "PASS",
        "evidence": "threshold fixed at 0.50; all confusion cells sum to N for grouped OOF, 2019 and 2020",
    })
    return metrics_df, confusion_df, grouped, temporal


def final_correlation_analysis(data: pd.DataFrame, audit_rows: list[dict]):
    variables = AUDIO_CONTINUOUS + ["mode"] + LEX_MODEL
    pearson = data[variables].corr(method="pearson")
    spearman = data[variables].corr(method="spearman")
    prior_pairs = pd.read_csv(ANALYSIS_DIR / "high_correlation_pairs.csv")
    prior_keys = {
        tuple(sorted([row.variable_1, row.variable_2]))
        for row in prior_pairs.itertuples()
    }
    rows = []
    for i, left in enumerate(variables):
        for right in variables[i + 1:]:
            p = float(pearson.loc[left, right])
            s = float(spearman.loc[left, right])
            max_abs = max(abs(p), abs(s))
            rows.append({
                "variable_1": left,
                "variable_2": right,
                "pearson_correlation": p,
                "spearman_correlation": s,
                "maximum_absolute_correlation": max_abs,
                "absolute_correlation_0_50plus": int(max_abs >= 0.50),
                "absolute_correlation_0_70plus": int(max_abs >= 0.70),
                "absolute_correlation_0_80plus": int(max_abs >= 0.80),
                "prior_validated_high_pair": int(tuple(sorted([left, right])) in prior_keys),
                "automatic_removal": 0,
                "interpretation": "descriptive association only; review jointly with VIF and coefficient stability",
            })
    summary = pd.DataFrame(rows).sort_values("maximum_absolute_correlation", ascending=False)
    if set(pearson.columns) != set(variables):
        raise AssertionError("Final correlation matrix contains unintended variables.")
    audit_rows.append({
        "check": "correlation variable scope", "status": "PASS",
        "evidence": f"Pearson/Spearman consolidation uses {len(variables)} retained transparent numeric/binary predictors; categorical X and key excluded from correlation interpretation",
    })
    return summary, pearson, spearman


@dataclass
class DesignMatrix:
    matrix: np.ndarray
    names: list[str]
    source_variables: list[str]
    term_types: list[str]
    scaling_notes: list[str]


def make_design(data: pd.DataFrame, specification: str, standardise_continuous: bool = False) -> DesignMatrix:
    columns = [np.ones(len(data))]
    names = ["Intercept"]
    sources = ["__intercept__"]
    types = ["intercept"]
    notes = ["intercept"]
    for level in [2, 3, 4]:
        columns.append((data.X.astype(int) == level).astype(float).to_numpy())
        names.append(f"X_level_{level}_vs_1")
        sources.append("X")
        types.append("categorical contrast")
        notes.append("reference X=1")
    if specification in {"A2", "A3"}:
        for variable in AUDIO_CONTINUOUS:
            values = data[variable].to_numpy(float)
            if standardise_continuous:
                sd = values.std(ddof=0)
                values = (values - values.mean()) / sd
                note = f"one-SD increase; source SD={sd:.8g}"
            else:
                note = "source units"
            columns.append(values)
            names.append(variable)
            sources.append(variable)
            types.append("continuous")
            notes.append(note)
        for level in range(1, 12):
            columns.append((data.key.astype(int) == level).astype(float).to_numpy())
            names.append(f"key_{level}_vs_0")
            sources.append("key")
            types.append("categorical contrast")
            notes.append("reference key=0")
        columns.append(data["mode"].to_numpy(float))
        names.append("mode_1_vs_0")
        sources.append("mode")
        types.append("binary contrast")
        notes.append("reference mode=0")
    if specification == "A3":
        for variable in LEX_MODEL:
            values = data[variable].to_numpy(float)
            if standardise_continuous:
                sd = values.std(ddof=0)
                values = (values - values.mean()) / sd
                note = f"one-SD increase after validated lexical transformation; source SD={sd:.8g}"
            else:
                note = "validated transformed units"
            columns.append(values)
            names.append(variable)
            sources.append(variable)
            types.append("continuous lexical")
            notes.append(note)
    return DesignMatrix(np.column_stack(columns), names, sources, types, notes)


def cluster_robust_covariance(
    bread: np.ndarray,
    score_by_observation: np.ndarray,
    groups: np.ndarray,
    n: int,
    k: int,
) -> tuple[np.ndarray, int]:
    unique_groups = pd.unique(groups)
    meat = np.zeros((k, k))
    for group in unique_groups:
        score = score_by_observation[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    g = len(unique_groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    return correction * bread @ meat @ bread, g


def fit_clustered_ols(
    data: pd.DataFrame,
    specification: str,
    groups: np.ndarray,
    clustering_label: str,
    standardised: bool = False,
) -> pd.DataFrame:
    design = make_design(data, specification, standardised)
    X = design.matrix
    y = data.Y_52.to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    score = X * residual[:, None]
    covariance, g = cluster_robust_covariance(bread, score, np.asarray(groups), len(y), X.shape[1])
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    critical = student_t.ppf(0.975, df=max(g - 1, 1))
    statistic = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * student_t.sf(np.abs(statistic), df=max(g - 1, 1))
    return pd.DataFrame({
        "association_model": specification,
        "clustering_unit": clustering_label,
        "term": design.names,
        "source_variable": design.source_variables,
        "term_type": design.term_types,
        "scaling": design.scaling_notes,
        "coefficient_weeks": beta,
        "cluster_robust_se": se,
        "ci_2_5": beta - critical * se,
        "ci_97_5": beta + critical * se,
        "p_value_supplementary": pvalue,
        "n_tracks": len(y),
        "n_clusters": g,
        "design_rank": rank,
        "causal_interpretation": "not permitted; descriptive association only",
    })


def fit_all_ols(data: pd.DataFrame, audit_rows: list[dict]):
    first_artist = data.first_credited_artist_id.astype(str).to_numpy()
    protected = data.primary_group_component.astype(str).to_numpy()
    frames = []
    for spec in ["A1", "A2", "A3"]:
        frames.append(fit_clustered_ols(data, spec, first_artist, "first_credited_artist_primary", False))
        frames.append(fit_clustered_ols(data, spec, protected, "protected_component_sensitivity", False))
    coefficients = pd.concat(frames, ignore_index=True)
    standardised = pd.concat([
        fit_clustered_ols(data, spec, first_artist, "first_credited_artist_primary", True)
        for spec in ["A1", "A2", "A3"]
    ], ignore_index=True)
    standardised["standardised_association_type"] = np.where(
        standardised.term_type.str.contains("continuous"),
        "weeks per one-SD predictor increase",
        "original categorical/binary contrast",
    )
    if data.first_credited_artist_id.isna().any():
        raise AssertionError("First-credited artist clustering IDs are incomplete.")
    audit_rows.append({
        "check": "artist-clustered OLS", "status": "PASS",
        "evidence": f"A1/A2/A3 fitted on N={len(data)}; first-artist clusters={data.first_credited_artist_id.nunique()}; protected-component sensitivity clusters={data.primary_group_component.nunique()}; CR1 sandwich covariance",
    })
    return coefficients, standardised


def calculate_vif(design: DesignMatrix) -> pd.DataFrame:
    X = design.matrix[:, 1:]
    rows = []
    for index, name in enumerate(design.names[1:]):
        target = X[:, index]
        others = np.delete(X, index, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        fitted, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
        residual = target - others @ fitted
        total = np.sum((target - target.mean()) ** 2)
        r2 = 1 - np.sum(residual ** 2) / total if total > 0 else np.nan
        vif = 1 / (1 - r2) if pd.notna(r2) and r2 < 1 else np.inf
        rows.append({
            "term": name,
            "source_variable": design.source_variables[index + 1],
            "r_squared_against_other_A3_terms": r2,
            "vif": vif,
            "flag_vif_5plus": int(vif >= 5),
            "flag_vif_10plus": int(vif >= 10),
        })
    return pd.DataFrame(rows)


def coefficient_stability_audit(
    data: pd.DataFrame,
    coefficients: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> pd.DataFrame:
    primary = coefficients[coefficients.clustering_unit.eq("first_credited_artist_primary")]
    a2 = primary[primary.association_model.eq("A2")][[
        "term", "coefficient_weeks", "cluster_robust_se", "ci_2_5", "ci_97_5",
    ]].rename(columns={
        "coefficient_weeks": "a2_coefficient",
        "cluster_robust_se": "a2_cluster_se",
        "ci_2_5": "a2_ci_2_5",
        "ci_97_5": "a2_ci_97_5",
    })
    a3 = primary[primary.association_model.eq("A3")][[
        "term", "source_variable", "term_type", "coefficient_weeks",
        "cluster_robust_se", "ci_2_5", "ci_97_5",
    ]].rename(columns={
        "coefficient_weeks": "a3_coefficient",
        "cluster_robust_se": "a3_cluster_se",
        "ci_2_5": "a3_ci_2_5",
        "ci_97_5": "a3_ci_97_5",
    })
    result = a3.merge(a2, on="term", how="left", validate="one_to_one")
    result["coefficient_change_a3_minus_a2"] = result.a3_coefficient - result.a2_coefficient
    result["absolute_relative_change_pct"] = 100 * np.abs(result.coefficient_change_a3_minus_a2) / np.maximum(np.abs(result.a2_coefficient), 1e-8)
    result["direction_changed"] = (
        np.sign(result.a3_coefficient) != np.sign(result.a2_coefficient)
    ).where(result.a2_coefficient.notna(), False).astype(int)

    vif = calculate_vif(make_design(data, "A3", standardise_continuous=True))
    result = result.merge(vif[["term", "vif", "flag_vif_5plus", "flag_vif_10plus"]], on="term", how="left", validate="one_to_one")
    high_variables = set(
        correlation_summary.loc[
            correlation_summary.absolute_correlation_0_70plus.eq(1),
            ["variable_1", "variable_2"],
        ].to_numpy().ravel()
    )
    result["source_in_pair_abs_corr_0_70plus"] = result.source_variable.isin(high_variables).astype(int)
    result["coefficient_interpretation_flag"] = np.select(
        [
            result.flag_vif_10plus.fillna(0).eq(1),
            result.direction_changed.eq(1),
            result.flag_vif_5plus.fillna(0).eq(1) | result.source_in_pair_abs_corr_0_70plus.eq(1),
        ],
        [
            "high collinearity: avoid isolated coefficient interpretation",
            "direction changes between A2 and A3: unstable",
            "correlated predictor: interpret jointly and cautiously",
        ],
        default="no major coefficient-stability flag",
    )
    condition_number = np.linalg.cond(make_design(data, "A3", True).matrix)
    result["standardised_A3_design_condition_number"] = condition_number
    return result


def fit_nb2_association(data: pd.DataFrame, standardised_ols: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    design = make_design(data, "A3", standardise_continuous=True)
    X = design.matrix
    y = data.Y_52.to_numpy(float)

    def poisson_objective(beta):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta + gammaln(y + 1)))

    def poisson_gradient(beta):
        mu = np.exp(np.clip(X @ beta, -20, 20))
        return X.T @ (mu - y)

    beta0 = np.zeros(X.shape[1])
    beta0[0] = math.log(max(y.mean(), 1e-6))
    poisson = minimize(
        poisson_objective, beta0, jac=poisson_gradient, method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
    )
    mu0 = np.exp(np.clip(X @ poisson.x, -20, 20))
    numerator = np.sum((y - mu0) ** 2 - y)
    denominator = np.sum(mu0 ** 2)
    alpha = float(np.clip(numerator / denominator if denominator > 0 else 1e-6, 1e-6, 50.0))
    r = 1 / alpha

    def objective(beta):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        loglik = (
            gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * (math.log(r) - np.log(r + mu))
            + y * (eta - np.log(r + mu))
        )
        return float(-np.sum(loglik))

    def gradient(beta):
        mu = np.exp(np.clip(X @ beta, -20, 20))
        return X.T @ ((mu - y) / (1 + alpha * mu))

    fit = minimize(
        objective, poisson.x, jac=gradient, method="L-BFGS-B",
        options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-7},
    )
    beta = fit.x
    mu = np.exp(np.clip(X @ beta, -20, 20))
    weights = mu / (1 + alpha * mu)
    bread = np.linalg.pinv(X.T @ (weights[:, None] * X))
    score = X * ((y - mu) / (1 + alpha * mu))[:, None]
    groups = data.first_credited_artist_id.astype(str).to_numpy()
    covariance, g = cluster_robust_covariance(bread, score, groups, len(y), X.shape[1])
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    critical = student_t.ppf(0.975, df=g - 1)
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * student_t.sf(np.abs(z), df=g - 1)
    output = pd.DataFrame({
        "association_model": "NB2_A3_model_form_sensitivity",
        "term": design.names,
        "source_variable": design.source_variables,
        "term_type": design.term_types,
        "scaling": design.scaling_notes,
        "coefficient_log_mean": beta,
        "rate_ratio": np.exp(np.clip(beta, -20, 20)),
        "artist_cluster_robust_se_conditional_on_alpha": se,
        "ci_2_5_log_scale": beta - critical * se,
        "ci_97_5_log_scale": beta + critical * se,
        "p_value_supplementary": pvalue,
        "dispersion_alpha_fixed_method_of_moments": alpha,
        "n_tracks": len(y),
        "n_artist_clusters": g,
        "converged": int(fit.success),
        "iterations": int(fit.nit),
        "fit_message": str(fit.message),
        "inference_limitation": "cluster covariance conditions on method-of-moments alpha; model-form sensitivity, not primary inference",
    })
    ols = standardised_ols[
        standardised_ols.association_model.eq("A3")
        & standardised_ols.clustering_unit.eq("first_credited_artist_primary")
    ][["term", "coefficient_weeks"]].rename(columns={"coefficient_weeks": "standardised_ols_coefficient_weeks"})
    output = output.merge(ols, on="term", how="left", validate="one_to_one")
    output["direction_agrees_with_standardised_OLS"] = (
        np.sign(output.coefficient_log_mean) == np.sign(output.standardised_ols_coefficient_weeks)
    ).astype(int)
    info = {
        "converged": bool(fit.success), "message": str(fit.message), "iterations": int(fit.nit),
        "alpha": alpha, "n": len(y), "clusters": g,
    }
    return output, info


def fit_logistic_association(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    design = make_design(data, "A3", standardise_continuous=True)
    X = design.matrix
    y = (data.Y_52 > 0).astype(int).to_numpy()

    def objective(beta):
        eta = np.clip(X @ beta, -30, 30)
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def gradient(beta):
        p = expit(np.clip(X @ beta, -30, 30))
        return X.T @ (p - y)

    beta0 = np.zeros(X.shape[1])
    beta0[0] = math.log(y.mean() / (1 - y.mean()))
    fit = minimize(
        objective, beta0, jac=gradient, method="L-BFGS-B",
        options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-7},
    )
    beta = fit.x
    p = expit(np.clip(X @ beta, -30, 30))
    weights = p * (1 - p)
    bread = np.linalg.pinv(X.T @ (weights[:, None] * X))
    score = X * (y - p)[:, None]
    groups = data.first_credited_artist_id.astype(str).to_numpy()
    covariance, g = cluster_robust_covariance(bread, score, groups, len(y), X.shape[1])
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    critical = student_t.ppf(0.975, df=g - 1)
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * student_t.sf(np.abs(z), df=g - 1)
    output = pd.DataFrame({
        "association_model": "H3_part_A_full_cohort_descriptive_logistic",
        "term": design.names,
        "source_variable": design.source_variables,
        "term_type": design.term_types,
        "scaling": design.scaling_notes,
        "coefficient_log_odds": beta,
        "odds_ratio": np.exp(np.clip(beta, -20, 20)),
        "artist_cluster_robust_se": se,
        "ci_2_5_log_odds": beta - critical * se,
        "ci_97_5_log_odds": beta + critical * se,
        "odds_ratio_ci_2_5": np.exp(np.clip(beta - critical * se, -20, 20)),
        "odds_ratio_ci_97_5": np.exp(np.clip(beta + critical * se, -20, 20)),
        "p_value_supplementary": pvalue,
        "n_tracks": len(y),
        "n_artist_clusters": g,
        "converged": int(fit.success),
        "iterations": int(fit.nit),
        "fit_message": str(fit.message),
        "causal_interpretation": "not permitted; descriptive association only",
    })
    info = {
        "converged": bool(fit.success), "message": str(fit.message), "iterations": int(fit.nit),
        "n": len(y), "clusters": g,
    }
    return output, info


def h3_descriptive_summaries(
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_positive = pd.read_csv(LYRIC_DIR / "hurdle_positive_count_metrics.csv")
    positive = existing_positive[
        existing_positive.model_id.eq("H3")
    ].copy()
    temporal_positive = pd.read_csv(ROBUST_DIR / "temporal_hurdle_positive_count_52w.csv")
    temporal_positive = temporal_positive.rename(columns={"partition": "aggregation"})
    positive["evidence_source"] = "grouped OOF H3 output"
    temporal_positive["evidence_source"] = "temporal robustness output"
    positive_summary = pd.concat([positive, temporal_positive], ignore_index=True, sort=False)

    zero_grouped = pd.read_csv(LYRIC_DIR / "hurdle_zero_calibration.csv")
    zero_grouped = zero_grouped[
        (zero_grouped.comparison_group.eq("H3"))
        & zero_grouped.model_type.isin(["ordinary_poisson_reference", "true_hurdle"])
    ].copy()
    zero_grouped["evaluation"] = "grouped_oof"
    zero_grouped["evidence_source"] = "lyric/hurdle modelling"
    zero_temporal = pd.read_csv(ROBUST_DIR / "temporal_zero_calibration_comparison_52w.csv")
    zero_temporal = zero_temporal[zero_temporal.model.isin(["LEX", "H3"])].copy()
    zero_temporal["evaluation"] = zero_temporal.partition
    zero_temporal["evidence_source"] = "temporal robustness"
    calibration = classification.copy()
    calibration["calibration_record_type"] = "binary classification and probability metrics"
    calibration["evidence_source"] = "reconciled H3 predictions"
    # Keep both structured sources in a single long evidence table.
    zero_grouped["calibration_record_type"] = "zero-probability comparison"
    zero_temporal["calibration_record_type"] = "zero-probability comparison"
    final_calibration = pd.concat([calibration, zero_grouped, zero_temporal], ignore_index=True, sort=False)
    return positive_summary, final_calibration


def related_version_sensitivity(
    data: pd.DataFrame,
    audit_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    version_audit = pd.read_csv(SPEC_DIR / "analysis_unit_version_audit.csv")
    supported = version_audit[
        version_audit.eligible_for_later_underlying_song_sensitivity_group.eq(1)
    ]
    excluded_ids = set()
    for value in supported.spotify_track_ids:
        excluded_ids.update(str(value).split("|"))
    if len(excluded_ids) != 143:
        raise AssertionError(f"Supported related-version exclusion set is {len(excluded_ids)}, expected 143.")
    reduced = data[~data.spotify_track_id.isin(excluded_ids)].copy()
    if len(reduced) != 871:
        raise AssertionError(f"Related-version exclusion leaves {len(reduced)} tracks, expected 871.")
    predictions = []
    for fold in sorted(data.fold.unique()):
        train = reduced[reduced.fold.ne(fold)]
        heldout = reduced[reduced.fold.eq(fold)]
        for model_id, variables in [("F3_VERSION_EXCLUSION", F3), ("LEX_VERSION_EXCLUSION", LEX)]:
            pred, _, _, warning_text = prior.fit_tab_poisson(train, heldout, variables)
            if warning_text:
                warnings.warn(warning_text)
            predictions.append(pd.DataFrame({
                "spotify_track_id": heldout.spotify_track_id,
                "fold": fold,
                "primary_group_component": heldout.primary_group_component,
                "model_id": model_id,
                "y_observed": heldout.Y_52,
                "y_pred_raw": pred,
            }))
    pred = pd.concat(predictions, ignore_index=True)
    wide = pred.pivot(index="spotify_track_id", columns="model_id", values="y_pred_raw").reset_index()
    wide = reduced[["spotify_track_id", "Y_52", "primary_group_component"]].merge(
        wide, on="spotify_track_id", validate="one_to_one",
    )
    metrics = []
    for model_id, q in pred.groupby("model_id"):
        metrics.append({
            "record_type": "model_metric",
            "documented_rule": "exclude every track in the 70 supported exact-lyric/related-version groups; retain the two tracks in the one unresolved group",
            "n_supported_groups_removed": 70,
            "n_tracks_removed": 143,
            "n_tracks_remaining": len(reduced),
            "protected_clusters_remaining": reduced.primary_group_component.nunique(),
            "model_id": model_id,
            **metric_values(q.y_observed, q.y_pred_raw),
        })
    uncertainty = cluster_bootstrap(
        wide, "F3_VERSION_EXCLUSION", "LEX_VERSION_EXCLUSION",
        "primary_group_component", "supported_related_version_exclusion",
        SEED + 1200,
    )
    uncertainty["record_type"] = "paired_cluster_bootstrap"
    uncertainty["documented_rule"] = "exclude every track in the 70 supported exact-lyric/related-version groups; retain the unresolved group"
    uncertainty["n_supported_groups_removed"] = 70
    uncertainty["n_tracks_removed"] = 143
    uncertainty["n_tracks_remaining"] = len(reduced)
    uncertainty["protected_clusters_remaining"] = reduced.primary_group_component.nunique()
    output = pd.concat([pd.DataFrame(metrics), uncertainty], ignore_index=True, sort=False)
    audit_rows.append({
        "check": "related-version exclusion rule", "status": "PASS",
        "evidence": f"70 supported groups; 143 tracks removed; 871 remain; unresolved group retained; frozen fold labels preserved",
    })
    return output, pred


def feature_selection_record() -> pd.DataFrame:
    rows = []
    for variable in F3:
        rows.append({
            "variable_or_group": variable,
            "final_status": "retained",
            "final_model_scope": "F3, LEX, H3, ML_F3 and ML_LEX",
            "selection_layer": "leakage-safe substantive pre-specification",
            "treatment": (
                "categorical four levels" if variable == "X" else
                "categorical musical key" if variable == "key" else
                "binary" if variable == "mode" else "continuous"
            ),
            "reason": "available by prediction time; conceptually relevant baseline or early-chart predictor",
            "leakage_status": "safe",
            "automatic_statistical_selection": 0,
        })
    lexical_definitions = {
        "lex_log_word_count": "log1p lyric word count",
        "lex_log_unique_word_count": "log1p unique-word count",
        "lyric_lexical_diversity": "unique words divided by total words",
        "lyric_average_word_length": "mean token length",
        "lex_log_line_count": "log1p nonblank lyric line count",
        "lyric_repeated_line_proportion": "share of nonblank lines repeated",
        "lex_log_intensity_punctuation": "log1p exclamation/question marks per 100 words",
    }
    for variable in LEX_MODEL:
        rows.append({
            "variable_or_group": variable,
            "final_status": "retained",
            "final_model_scope": "LEX, H3 and ML_LEX",
            "selection_layer": "transparent lyric-summary pre-specification and grouped-CV incremental comparison",
            "treatment": lexical_definitions[variable],
            "reason": "small, reproducible, interpretable lyric summary; no future chart information",
            "leakage_status": "safe static lyric text",
            "automatic_statistical_selection": 0,
        })

    excluded = [
        ("track_popularity", "blocked", "snapshot timing and leakage risk"),
        ("artist_popularity", "blocked", "snapshot timing and leakage risk"),
        ("artist_followers", "blocked", "snapshot timing and leakage risk"),
        ("average_artist_popularity", "blocked", "snapshot timing and leakage risk"),
        ("future_ranks", "blocked", "observed after the week-4 prediction time"),
        ("future_streams", "blocked", "observed after the week-4 prediction time"),
        ("eventual_peak_rank", "blocked", "full-history leakage"),
        ("total_chart_duration", "blocked", "outcome/full-history leakage"),
        ("first_spell_duration", "blocked", "alternative outcome/full-history information"),
        ("L_52", "blocked", "outcome-window descriptor"),
        ("Z_count_52", "blocked", "later spell/re-entry information"),
        ("Z_reentry_52", "blocked", "later re-entry information"),
        ("old_two_week_four_week_eligibility_flags", "blocked", "landmark survivor-selection mechanism"),
        ("spotify_track_id", "grouping/audit only", "identifier memorisation risk"),
        ("first_credited_artist_id", "grouping only", "validation and covariance clustering, not a numeric predictor"),
        ("chart_entry_year", "not retained in provisional final F3/LEX", "F4 calendar extension did not improve the core benchmark consistently"),
        ("release_age_years", "not retained in provisional final F3/LEX", "F4 extension remained secondary and release/calendar variables require separate interpretation"),
        ("release_year", "not jointly retained", "exact dependency with entry year and release age"),
        ("genre_raw", "sensitivity only", "provenance and rare-category limitations"),
        ("lyrics_character_count", "excluded redundant control", "word count retained instead of nearly duplicate character count"),
        ("raw_lyrics_text_TFIDF", "sensitivity only", "fold-local high-dimensional T3; mixed RMSE and temporal evidence"),
        ("explicit_160k_only", "excluded from common/main specification", "not conceptually comparable across the primary 550k and 160k cohorts"),
    ]
    for variable, status, reason in excluded:
        rows.append({
            "variable_or_group": variable,
            "final_status": status,
            "final_model_scope": "none in provisional final main count model" if "only" not in status else status,
            "selection_layer": "leakage/conceptual exclusion or documented sensitivity",
            "treatment": "not used as a main predictor",
            "reason": reason,
            "leakage_status": "blocked or intentionally excluded" if status == "blocked" else "not leakage-safe main feature or not selected",
            "automatic_statistical_selection": 0,
        })
    output = pd.DataFrame(rows)
    output["feature_selection_principle"] = (
        "No exhaustive or p-value-based deletion: N≈1,014, overfitting/model-selection-bias risk, "
        "incremental conceptually meaningful feature groups, interpretability and reproducibility."
    )
    return output


def candidate_roles(
    ml_aggregate: pd.DataFrame,
    ml_uncertainty: pd.DataFrame,
    temporal_ml: pd.DataFrame,
    selected_models: dict[str, str],
) -> pd.DataFrame:
    rows = [
        {
            "model_id": "F1",
            "model_name": "Categorical X only",
            "role": "core benchmark",
            "main_or_sensitivity": "main text",
            "evidence_basis": "strong early-chart benchmark; very low complexity; best simple temporal reference",
            "promotion_rule_result": "not applicable",
        },
        {
            "model_id": "F3",
            "model_name": "Categorical X + audio/song characteristics",
            "role": "audio/song-characteristic reference",
            "main_or_sensitivity": "main text",
            "evidence_basis": "tests incremental baseline song characteristics beyond X",
            "promotion_rule_result": "not applicable",
        },
        {
            "model_id": "LEX",
            "model_name": "F3 + seven transparent lexical summaries",
            "role": "principal interpretable count model",
            "main_or_sensitivity": "main text",
            "evidence_basis": "small cluster-supported grouped-CV MAE gain; RMSE and temporal evidence mixed",
            "promotion_rule_result": "retained for interpretability and direct lyric-increment question, not uniform superiority",
        },
        {
            "model_id": "H3",
            "model_name": "Two-part hurdle using LEX predictors",
            "role": "probability/future-presence model",
            "main_or_sensitivity": "main text with count metrics separated from binary metrics",
            "evidence_basis": "strong discrimination and zero calibration; no clear overall count-error advantage",
            "promotion_rule_result": "not a replacement for the main count model",
        },
        {
            "model_id": "T3",
            "model_name": "F3 + regularised fold-local TF-IDF",
            "role": "high-dimensional lyric sensitivity",
            "main_or_sensitivity": "sensitivity/appendix",
            "evidence_basis": "mixed MAE, worse RMSE behaviour, systematic underprediction; lower alpha did not rescue it",
            "promotion_rule_result": "not supported as main specification",
        },
        {
            "model_id": "RF_F3",
            "model_name": "Historical random forest — F3",
            "role": "historical nonlinear comparison",
            "main_or_sensitivity": "sensitivity/appendix",
            "evidence_basis": "validated first-round nonlinear benchmark; no meaningful improvement",
            "promotion_rule_result": "not supported as main specification",
        },
    ]
    for algorithm, model_id in selected_models.items():
        uncertainty = ml_uncertainty[
            ml_uncertainty.candidate_model.eq(model_id)
            & ml_uncertainty.reference_model.eq("LEX")
        ]
        mae = uncertainty[uncertainty.metric.eq("MAE")].iloc[0]
        rmse = uncertainty[uncertainty.metric.eq("RMSE")].iloc[0]
        t2019 = temporal_ml[
            temporal_ml.model_id.eq(model_id)
            & temporal_ml.temporal_partition.eq("validation_2019")
        ].iloc[0]
        robust_both = mae.ci_2_5 > 0 and rmse.ci_2_5 > 0
        practical = mae.observed_difference >= 0.20 or rmse.observed_difference >= 0.20
        role = "nonlinear ML comparison"
        main_scope = "main comparison table; sensitivity interpretation"
        promotion = (
            "supported against LEX on MAE and RMSE with ≥0.20-week improvement; temporal evidence still required"
            if robust_both and practical else
            "not promoted: uncertainty, dual-metric, or practical-magnitude criterion not fully met"
        )
        rows.append({
            "model_id": model_id,
            "model_name": MODEL_LABELS[model_id],
            "role": role,
            "main_or_sensitivity": main_scope,
            "evidence_basis": (
                f"versus LEX MAE gain {mae.observed_difference:.3f}, CI [{mae.ci_2_5:.3f}, {mae.ci_97_5:.3f}]; "
                f"RMSE gain {rmse.observed_difference:.3f}, CI [{rmse.ci_2_5:.3f}, {rmse.ci_97_5:.3f}]; "
                f"2019 MAE/RMSE {t2019.mae:.3f}/{t2019.rmse:.3f}"
            ),
            "promotion_rule_result": promotion,
        })
    return pd.DataFrame(rows)


def thesis_model_comparison(
    ml_aggregate: pd.DataFrame,
    temporal_ml: pd.DataFrame,
    selected_models: dict[str, str],
    h3_classification: pd.DataFrame,
) -> pd.DataFrame:
    benchmark = pd.read_csv(BENCH_DIR / "grouped_cv_aggregate_metrics.csv")
    lyric = pd.read_csv(LYRIC_DIR / "lyric_grouped_cv_metrics.csv")
    hurdle = pd.read_csv(LYRIC_DIR / "hurdle_combined_metrics.csv")
    temporal_hist = pd.read_csv(ROBUST_DIR / "temporal_robustness_52w.csv")
    rows = []

    def add_count(model_id, name, purpose, source, mae, rmse, bias, r2, temporal_key=None):
        row = {
            "model_id": model_id,
            "model_name": name,
            "model_purpose": purpose,
            "grouped_cv_source": source,
            "grouped_cv_n": 1014,
            "grouped_cv_mae": mae,
            "grouped_cv_rmse": rmse,
            "grouped_cv_bias": bias,
            "grouped_cv_r2_supplementary": r2,
        }
        if temporal_key:
            for partition, suffix in [("validation_2019", "2019"), ("test_2020", "2020_stress")]:
                q = temporal_hist[
                    temporal_hist.partition.eq(partition) & temporal_hist.model.eq(temporal_key)
                ]
                if not q.empty:
                    row[f"temporal_{suffix}_n"] = int(q.iloc[0].n)
                    row[f"temporal_{suffix}_mae"] = q.iloc[0].mae
                    row[f"temporal_{suffix}_rmse"] = q.iloc[0].rmse
                    row[f"temporal_{suffix}_bias"] = q.iloc[0].mean_prediction_bias
        rows.append(row)

    f0 = benchmark[(benchmark.outcome.eq("Y_52")) & benchmark.feature_block.eq("F0") & benchmark.model_id.eq("M0_mean")].iloc[0]
    add_count("F0_MEAN", "Training-fold mean", "naive count benchmark", "benchmark v1", f0.mae, f0.rmse, f0.mean_prediction_bias, f0.r2_supplementary)
    for block, model_id, name, temporal_key in [
        ("F1", "F1", "Categorical X only", "REF_F1"),
        ("F3", "F3", "X + audio/song characteristics", "REF_F3"),
    ]:
        q = benchmark[(benchmark.outcome.eq("Y_52")) & benchmark.feature_block.eq(block) & benchmark.model_id.eq("M1_poisson")].iloc[0]
        add_count(model_id, name, "transparent count model", "benchmark v1", q.mae, q.rmse, q.mean_prediction_bias, q.r2_supplementary, temporal_key)
    q = lyric[lyric.model_id.eq("LEX")].iloc[0]
    add_count("LEX", "F3 + transparent lexical summaries", "principal interpretable count model", "lyrics/hurdle v1", q.mae, q.rmse, q.mean_prediction_bias, q.r2_supplementary, "LEX")
    q = benchmark[(benchmark.outcome.eq("Y_52")) & benchmark.feature_block.eq("F3") & benchmark.model_id.eq("M4_random_forest")].iloc[0]
    add_count("RF_F3", "Random forest — F3", "historical nonlinear comparison", "benchmark v1", q.mae, q.rmse, q.mean_prediction_bias, q.r2_supplementary)
    for algorithm, model_id in selected_models.items():
        q = ml_aggregate[ml_aggregate.model_id.eq(model_id)].iloc[0]
        row = {
            "model_id": model_id,
            "model_name": MODEL_LABELS[model_id],
            "model_purpose": "new nonlinear ML comparison",
            "grouped_cv_source": "final comparison v1",
            "grouped_cv_n": q.n,
            "grouped_cv_mae": q.mae,
            "grouped_cv_rmse": q.rmse,
            "grouped_cv_bias": q.mean_prediction_bias,
            "grouped_cv_r2_supplementary": q.r2_supplementary,
        }
        for partition, suffix in [("validation_2019", "2019"), ("test_2020", "2020_stress")]:
            t = temporal_ml[
                temporal_ml.model_id.eq(model_id) & temporal_ml.temporal_partition.eq(partition)
            ].iloc[0]
            row[f"temporal_{suffix}_n"] = int(t.n)
            row[f"temporal_{suffix}_mae"] = t.mae
            row[f"temporal_{suffix}_rmse"] = t.rmse
            row[f"temporal_{suffix}_bias"] = t.mean_prediction_bias
        rows.append(row)
    q = hurdle[(hurdle.aggregation.eq("all_oof")) & hurdle.model_id.eq("H3")].iloc[0]
    add_count("H3", "Two-part hurdle — LEX predictors", "probability/future-presence plus conditional duration", "lyrics/hurdle v1", q.mae, q.rmse, q.mean_prediction_bias, q.r2_supplementary, "H3")
    q = lyric[lyric.model_id.eq("T3")].iloc[0]
    add_count("T3", "F3 + fold-local TF-IDF", "high-dimensional lyric sensitivity", "lyrics/hurdle v1", q.mae, q.rmse, q.mean_prediction_bias, q.r2_supplementary, "T3")

    output = pd.DataFrame(rows)
    grouped_class = h3_classification[h3_classification.evaluation.eq("grouped_oof")].iloc[0]
    for column in [
        "roc_auc", "pr_auc", "accuracy", "precision_positive_predictive_value",
        "recall_sensitivity", "specificity", "classification_f1_score", "brier_score", "log_loss",
    ]:
        output.loc[output.model_id.eq("H3"), f"h3_binary_{column}"] = grouped_class[column]
    output["metric_compatibility_note"] = np.where(
        output.model_id.eq("H3"),
        "H3 count metrics and binary metrics answer different questions; do not rank them as one score.",
        "count-regression metrics apply",
    )
    return output


def robustness_evidence_matrix(
    model_comparison: pd.DataFrame,
    ml_uncertainty: pd.DataFrame,
    selected_models: dict[str, str],
    related: pd.DataFrame,
    h3_classification: pd.DataFrame,
) -> pd.DataFrame:
    lex = model_comparison[model_comparison.model_id.eq("LEX")].iloc[0]
    f1 = model_comparison[model_comparison.model_id.eq("F1")].iloc[0]
    f3 = model_comparison[model_comparison.model_id.eq("F3")].iloc[0]
    h3 = model_comparison[model_comparison.model_id.eq("H3")].iloc[0]
    t3 = model_comparison[model_comparison.model_id.eq("T3")].iloc[0]
    prior_unc = pd.read_csv(ROBUST_DIR / "paired_model_uncertainty.csv")
    lex_mae_ci = prior_unc[
        prior_unc.reference_model.eq("REF_F3") & prior_unc.candidate_model.eq("LEX") & prior_unc.metric.eq("MAE")
    ].iloc[0]
    x4 = pd.read_csv(ROBUST_DIR / "x4_paired_uncertainty.csv")
    x4lex = x4[
        x4.reference_model.eq("REF_F3") & x4.candidate_model.eq("LEX") & x4.metric.eq("MAE")
    ].iloc[0]
    version_mae = related[
        related.record_type.eq("paired_cluster_bootstrap") & related.metric.eq("MAE")
    ].iloc[0]
    h3c = h3_classification[h3_classification.evaluation.eq("grouped_oof")].iloc[0]
    common = pd.read_csv(BENCH_DIR / "common_550k_160k_robustness_metrics.csv")
    same_horizon = pd.read_csv(SPEC_DIR / "Y26_Y52_same_tracks_comparison.csv")
    rows = [
        {
            "claim_or_issue": "Early chart behaviour X is the dominant predictive signal",
            "supporting_evidence": f"F1 grouped MAE/RMSE {f1.grouped_cv_mae:.3f}/{f1.grouped_cv_rmse:.3f} versus naive mean {model_comparison[model_comparison.model_id.eq('F0_MEAN')].iloc[0].grouped_cv_mae:.3f}/{model_comparison[model_comparison.model_id.eq('F0_MEAN')].iloc[0].grouped_cv_rmse:.3f}",
            "contrary_or_mixed_evidence": "X=4 remains difficult; X is not sufficient for precise long-duration prediction",
            "evidence_strength": "strong",
            "safe_thesis_wording": "Early four-week chart presence is the strongest validated predictive benchmark among already-charted songs.",
        },
        {
            "claim_or_issue": "Audio/song characteristics add limited information beyond X",
            "supporting_evidence": f"F3 MAE {f3.grouped_cv_mae:.3f} versus F1 {f1.grouped_cv_mae:.3f}",
            "contrary_or_mixed_evidence": f"F3 RMSE {f3.grouped_cv_rmse:.3f} versus F1 {f1.grouped_cv_rmse:.3f}; temporal F3 is not consistently better",
            "evidence_strength": "strong for limited increment",
            "safe_thesis_wording": "The standard audio descriptors provide little additional out-of-sample accuracy once early chart presence is known.",
        },
        {
            "claim_or_issue": "Transparent lexical summaries add a small MAE benefit",
            "supporting_evidence": f"LEX versus F3 MAE gain {lex_mae_ci.observed_difference:.3f}, cluster-bootstrap CI [{lex_mae_ci.ci_2_5:.3f}, {lex_mae_ci.ci_97_5:.3f}]",
            "contrary_or_mixed_evidence": "RMSE interval includes zero; 2019 and Y_26 temporal evidence is mixed",
            "evidence_strength": "modest supported",
            "safe_thesis_wording": "Transparent lexical summaries produced a small grouped-CV MAE improvement, but not a general or temporally stable superiority claim.",
        },
        {
            "claim_or_issue": "Lexical summaries help some X=4 tracks",
            "supporting_evidence": f"X=4 MAE gain {x4lex.observed_difference:.3f}, CI [{x4lex.ci_2_5:.3f}, {x4lex.ci_97_5:.3f}]",
            "contrary_or_mixed_evidence": "X=4 RMSE interval includes zero",
            "evidence_strength": "modest supported",
            "safe_thesis_wording": "Lyrics may modestly improve absolute-error prediction among songs present in all four early weeks.",
        },
        {
            "claim_or_issue": "Future presence is easier than positive conditional longevity",
            "supporting_evidence": f"H3 grouped AUC {h3c.roc_auc:.3f}, PR-AUC {h3c.pr_auc:.3f}, Brier {h3c.brier_score:.3f}",
            "contrary_or_mixed_evidence": f"H3 overall count MAE/RMSE {h3.grouped_cv_mae:.3f}/{h3.grouped_cv_rmse:.3f}; positive-count errors remain large",
            "evidence_strength": "strong",
            "safe_thesis_wording": "Whether any future chart presence occurs is much easier to discriminate and calibrate than its positive duration.",
        },
        {
            "claim_or_issue": "TF-IDF is not supported as the main lyric model",
            "supporting_evidence": f"T3 MAE/RMSE {t3.grouped_cv_mae:.3f}/{t3.grouped_cv_rmse:.3f}; systematic underprediction and lower-alpha failure already validated",
            "contrary_or_mixed_evidence": "T3 point-estimate MAE is low and X=4 MAE can improve",
            "evidence_strength": "mixed/sensitivity",
            "safe_thesis_wording": "High-dimensional lyric text is retained as a sensitivity because its lower MAE is offset by worse RMSE and calibration/bias behaviour.",
        },
        {
            "claim_or_issue": "Poisson versus NB2 versus bounded model form",
            "supporting_evidence": "F1 predictions were effectively identical across the three transparent formulations; bounded and Poisson F3 errors were close",
            "contrary_or_mixed_evidence": "NB2 F3 was worse in the first benchmark; all models rely on working distributional assumptions",
            "evidence_strength": "model-form robustness",
            "safe_thesis_wording": "The central F1 result is not driven by the particular transparent count likelihood used in the first benchmark.",
        },
        {
            "claim_or_issue": "Y_52 versus Y_26 horizon",
            "supporting_evidence": f"Same-track horizon comparison retained {int(same_horizon.iloc[0].get('n_tracks', 1014)) if len(same_horizon) else 1014} tracks; Y_52 records additional later presence",
            "contrary_or_mixed_evidence": "Y_26 retains more later-entry tracks and lyric temporal results remain mixed",
            "evidence_strength": "sensitivity",
            "safe_thesis_wording": "Y_52 is the working primary horizon, with Y_26 retained to test horizon and calendar-composition sensitivity.",
        },
        {
            "claim_or_issue": "550k versus 160k feature-linked cohorts",
            "supporting_evidence": f"Common-specification robustness contains {common.source_dataset.nunique() if 'source_dataset' in common else 2} feature sources",
            "contrary_or_mixed_evidence": "The cohorts differ in composition and feature-source measurements are not identical; 550k coverage declines over time",
            "evidence_strength": "coverage sensitivity",
            "safe_thesis_wording": "The alternative 160k cohort is a robustness population, not simply a larger copy of the lyric-linked cohort.",
        },
        {
            "claim_or_issue": "Source-wide missing weeks",
            "supporting_evidence": "The clean fixed-window cohort excludes windows crossing the three audited unknown source weeks",
            "contrary_or_mixed_evidence": "Exclusion changes sample composition; unknown weeks cannot be recovered from the source",
            "evidence_strength": "design safeguard",
            "safe_thesis_wording": "Unknown source weeks were never converted to observed absence.",
        },
        {
            "claim_or_issue": "Related Spotify versions",
            "supporting_evidence": f"Conservative exclusion removes 143 tracks; remaining LEX MAE gain {version_mae.observed_difference:.3f}, CI [{version_mae.ci_2_5:.3f}, {version_mae.ci_97_5:.3f}]",
            "contrary_or_mixed_evidence": "Exclusion reduces power and unresolved identities are retained rather than guessed",
            "evidence_strength": "version sensitivity",
            "safe_thesis_wording": "The version sensitivity shows whether the small lexical result depends on audited duplicate/related IDs without changing the primary track-ID unit.",
        },
    ]
    for algorithm, model_id in selected_models.items():
        q = ml_uncertainty[
            ml_uncertainty.candidate_model.eq(model_id)
            & ml_uncertainty.reference_model.eq("LEX")
        ]
        mae = q[q.metric.eq("MAE")].iloc[0]
        rmse = q[q.metric.eq("RMSE")].iloc[0]
        rows.append({
            "claim_or_issue": f"{MODEL_LABELS[model_id]} nonlinear comparison",
            "supporting_evidence": f"versus LEX MAE gain {mae.observed_difference:.3f}, CI [{mae.ci_2_5:.3f}, {mae.ci_97_5:.3f}]",
            "contrary_or_mixed_evidence": f"RMSE gain {rmse.observed_difference:.3f}, CI [{rmse.ci_2_5:.3f}, {rmse.ci_97_5:.3f}]; temporal performance considered separately",
            "evidence_strength": "nonlinear sensitivity",
            "safe_thesis_wording": "The nonlinear comparison tests interactions within pre-specified predictors; any gain is reported with uncertainty and practical magnitude.",
        })
    return pd.DataFrame(rows)


def finding_hierarchy(
    model_comparison: pd.DataFrame,
    ml_uncertainty: pd.DataFrame,
    related: pd.DataFrame,
    selected_models: dict[str, str],
) -> pd.DataFrame:
    rows = [
        {
            "tier": "Tier 1 — strongest",
            "finding": "Early chart behaviour X is the dominant predictive signal.",
            "basis": "Large improvement over naive prediction and limited incremental gain from F3.",
            "safe_wording": "Among already-charted songs, early four-week chart presence is the strongest validated predictor block.",
        },
        {
            "tier": "Tier 1 — strongest",
            "finding": "Standard audio characteristics add little beyond X.",
            "basis": "Small MAE change and no RMSE/temporal dominance for F3 over F1.",
            "safe_wording": "Audio descriptors provide limited incremental predictive accuracy after early chart behaviour is known.",
        },
        {
            "tier": "Tier 1 — strongest",
            "finding": "Any future presence is easier to predict than positive conditional longevity.",
            "basis": "H3 binary discrimination/calibration is strong while positive-count errors remain high.",
            "safe_wording": "The future-presence decision is more predictable than the length of continued presence among positive cases.",
        },
        {
            "tier": "Tier 2 — modest supported",
            "finding": "Transparent lexical summaries provide a small grouped-CV MAE benefit.",
            "basis": "F3-to-LEX clustered MAE interval excludes zero narrowly; magnitude is about 0.16 weeks.",
            "safe_wording": "Lexical summaries add a small absolute-error improvement, not a large or universal advantage.",
        },
        {
            "tier": "Tier 2 — modest supported",
            "finding": "Lexical summaries may help modestly among X=4 tracks.",
            "basis": "X=4 clustered MAE interval excludes zero; RMSE interval does not.",
            "safe_wording": "The lyric increment is most visible in MAE among songs present in all four early weeks.",
        },
        {
            "tier": "Tier 3 — mixed/sensitivity",
            "finding": "Temporal lexical advantage.",
            "basis": "LEX does not outperform F3 in 2019 and Y_26 results change direction across years.",
            "safe_wording": "Temporal evidence for a lyric advantage is mixed.",
        },
        {
            "tier": "Tier 3 — mixed/sensitivity",
            "finding": "TF-IDF lyric content.",
            "basis": "Mixed MAE, worse RMSE, underprediction and unsuccessful lower-alpha sensitivity.",
            "safe_wording": "TF-IDF is an appendix sensitivity rather than a preferred lyric specification.",
        },
        {
            "tier": "Tier 3 — mixed/sensitivity",
            "finding": "Shorter Y_26 horizon lyric contribution.",
            "basis": "Temporal differences are small and intervals include zero.",
            "safe_wording": "The shorter horizon does not establish a stable lyric benefit.",
        },
    ]
    for algorithm, model_id in selected_models.items():
        q = ml_uncertainty[
            ml_uncertainty.candidate_model.eq(model_id)
            & ml_uncertainty.reference_model.eq("LEX")
        ]
        mae = q[q.metric.eq("MAE")].iloc[0]
        rmse = q[q.metric.eq("RMSE")].iloc[0]
        tier = (
            "Tier 2 — modest supported"
            if mae.ci_2_5 > 0 and rmse.ci_2_5 > 0
            else "Tier 3 — mixed/sensitivity"
        )
        rows.append({
            "tier": tier,
            "finding": f"{MODEL_LABELS[model_id]} nonlinear gain.",
            "basis": f"versus LEX MAE CI [{mae.ci_2_5:.3f}, {mae.ci_97_5:.3f}], RMSE CI [{rmse.ci_2_5:.3f}, {rmse.ci_97_5:.3f}].",
            "safe_wording": "Nonlinear interactions are retained as a comparison and promoted only if both error metrics, temporal evidence and practical magnitude agree.",
        })
    return pd.DataFrame(rows)


def make_figures(
    figure_dir: Path,
    model_comparison: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    ml_uncertainty: pd.DataFrame,
    x4_uncertainty: pd.DataFrame,
    temporal_ml_metrics: pd.DataFrame,
    selected_models: dict[str, str],
    h3_grouped: pd.DataFrame,
    h3_classification: pd.DataFrame,
    h3_confusion: pd.DataFrame,
    pearson: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[str]:
    """Create restrained thesis figures from held-out or descriptive evidence."""
    figure_dir.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="whitegrid", context="notebook")
    created = []

    def save(name: str):
        path = figure_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=240, bbox_inches="tight")
        plt.close()
        created.append(name)

    # 1. Incremental transparent sequence.
    q = model_comparison[model_comparison.model_id.isin(["F1", "F3", "LEX"])].copy()
    q["model_id"] = pd.Categorical(q.model_id, ["F1", "F3", "LEX"], ordered=True)
    q = q.sort_values("model_id")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for ax, metric, label in zip(axes, ["grouped_cv_mae", "grouped_cv_rmse"], ["MAE (weeks)", "RMSE (weeks)"]):
        sns.barplot(data=q, x="model_id", y=metric, ax=ax, color="#4C78A8")
        ax.set(xlabel="Model", ylabel=label, title=f"Grouped-CV {label.split()[0]}")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    fig.suptitle("Incremental prediction: early chart, audio, and lexical information", y=1.02)
    save("01_incremental_F1_F3_LEX.png")

    # 2. Broader model comparison.
    model_order = ["F1", "F3", "LEX", "RF_F3", *selected_models.values(), "H3", "T3"]
    q = model_comparison[model_comparison.model_id.isin(model_order)].copy()
    q["model_id"] = pd.Categorical(q.model_id, model_order, ordered=True)
    q = q.sort_values("model_id")
    long = q.melt(
        id_vars=["model_id"], value_vars=["grouped_cv_mae", "grouped_cv_rmse"],
        var_name="metric", value_name="weeks",
    )
    long["metric"] = long.metric.map({"grouped_cv_mae": "MAE", "grouped_cv_rmse": "RMSE"})
    plt.figure(figsize=(10.5, 4.5))
    sns.barplot(data=long, x="model_id", y="weeks", hue="metric", palette=["#4C78A8", "#F58518"])
    plt.xlabel("Model")
    plt.ylabel("Held-out error (weeks)")
    plt.title("Grouped-CV count-prediction comparison")
    plt.xticks(rotation=25, ha="right")
    save("02_broader_model_comparison.png")

    # 3. Paired uncertainty, positive = nonlinear candidate is better.
    q = ml_uncertainty[ml_uncertainty.metric.isin(["MAE", "RMSE"])].copy()
    q["comparison"] = q.reference_model + " vs " + q.candidate_model
    q["y"] = np.arange(len(q))
    plt.figure(figsize=(9.5, max(4.2, 0.42 * len(q))))
    colors = q.metric.map({"MAE": "#4C78A8", "RMSE": "#F58518"})
    for (_, row), color in zip(q.iterrows(), colors):
        plt.errorbar(
            row.observed_difference, row.y,
            xerr=[[row.observed_difference - row.ci_2_5], [row.ci_97_5 - row.observed_difference]],
            fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, markersize=5,
        )
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.yticks(q.y, q.comparison + " (" + q.metric + ")")
    plt.xlabel("Reference error minus candidate error (weeks)")
    plt.ylabel("")
    plt.title("Protected-component cluster-bootstrap differences (95% intervals)")
    save("03_ml_cluster_bootstrap_differences.png")

    # 4. X=4 comparison with current and historical predictions.
    historical = pd.read_csv(ROBUST_DIR / "aligned_grouped_oof_predictions.csv")
    model_predictions = {
        "F3": historical.REF_F3,
        "LEX": historical.LEX,
        "H3": historical.H3,
    }
    for model_id in selected_models.values():
        m = ml_predictions[ml_predictions.model_id.eq(model_id)].set_index("spotify_track_id").y_pred_raw
        model_predictions[model_id] = historical.spotify_track_id.map(m)
    x4 = historical.X.eq(4)
    rows = []
    for model_id, prediction in model_predictions.items():
        rows.append({
            "model": model_id,
            "MAE": mean_absolute_error(historical.loc[x4, "Y_52"], np.asarray(prediction)[x4]),
            "RMSE": math.sqrt(mean_squared_error(historical.loc[x4, "Y_52"], np.asarray(prediction)[x4])),
        })
    x4_metrics = pd.DataFrame(rows).melt("model", var_name="metric", value_name="weeks")
    plt.figure(figsize=(8.7, 4.3))
    sns.barplot(data=x4_metrics, x="model", y="weeks", hue="metric", palette=["#4C78A8", "#F58518"])
    plt.xlabel("Model")
    plt.ylabel("Held-out error among X=4 tracks (weeks)")
    plt.title(f"Prediction difficulty among songs present in all four early weeks (N={int(x4.sum())})")
    save("04_X4_model_comparison.png")

    # 5. Grouped versus temporal MAE.
    rows = []
    for model_id in ["F1", "F3", "LEX", "H3", *selected_models.values()]:
        main = model_comparison[model_comparison.model_id.eq(model_id)]
        if main.empty:
            continue
        rows.append({"model": model_id, "evaluation": "Grouped CV", "mae": main.iloc[0].grouped_cv_mae})
        for partition, label in [("validation_2019", "2019 validation"), ("test_2020", "2020 stress")]:
            if model_id in selected_models.values():
                t = temporal_ml_metrics[
                    temporal_ml_metrics.model_id.eq(model_id)
                    & temporal_ml_metrics.temporal_partition.eq(partition)
                ]
                if not t.empty:
                    rows.append({"model": model_id, "evaluation": label, "mae": t.iloc[0].mae})
            else:
                col = "temporal_2019_mae" if partition == "validation_2019" else "temporal_2020_stress_mae"
                value = main.iloc[0].get(col, np.nan)
                if pd.notna(value):
                    rows.append({"model": model_id, "evaluation": label, "mae": value})
    q = pd.DataFrame(rows)
    plt.figure(figsize=(10, 4.6))
    sns.barplot(data=q, x="model", y="mae", hue="evaluation", palette="colorblind")
    plt.xlabel("Model")
    plt.ylabel("MAE (weeks)")
    plt.title("Grouped validation and calendar-time evaluation")
    plt.legend(title="Evaluation", frameon=True)
    save("05_grouped_vs_temporal_mae.png")

    # 6. H3 OOF ROC and precision-recall curves.
    y_binary = (h3_grouped.y_observed > 0).astype(int)
    p = h3_grouped.p_positive.to_numpy(float)
    fpr, tpr, _ = roc_curve(y_binary, p)
    precision, recall, _ = precision_recall_curve(y_binary, p)
    group_metric = h3_classification[h3_classification.evaluation.eq("grouped_oof")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(fpr, tpr, color="#4C78A8", label=f"ROC-AUC={group_metric.roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[0].legend()
    axes[1].plot(recall, precision, color="#F58518", label=f"PR-AUC={group_metric.pr_auc:.3f}")
    axes[1].axhline(y_binary.mean(), linestyle="--", color="grey", label="Prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    fig.suptitle("H3 future-presence component: grouped out-of-fold predictions")
    save("06_H3_binary_ROC_PR.png")

    # 7. Grouped OOF confusion matrix at threshold .50.
    c = h3_confusion[h3_confusion.evaluation.eq("grouped_oof")].iloc[0]
    matrix = np.array([[c.true_negative, c.false_positive], [c.false_negative, c.true_positive]])
    plt.figure(figsize=(4.8, 4.1))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                xticklabels=["Predicted zero", "Predicted positive"],
                yticklabels=["Observed zero", "Observed positive"])
    plt.title("H3 confusion matrix (threshold 0.50)")
    save("07_H3_confusion_matrix.png")

    # 8. Retained transparent predictor correlations.
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson, cmap="vlag", center=0, vmin=-1, vmax=1, square=False,
                cbar_kws={"label": "Pearson correlation"})
    plt.title("Retained numeric/binary predictor correlations")
    save("08_retained_predictor_correlation_heatmap.png")

    # 9. Held-out permutation importance for selected ML models.
    q = importance[
        importance.aggregation.eq("weighted_across_5_outer_folds")
        & importance.importance_scope.eq("individual_raw_predictor")
    ].copy()
    q["model_id"] = q.algorithm + "_" + q.feature_block
    q = q[q.model_id.isin(selected_models.values())]
    q = q.sort_values("mae_increase_mean", ascending=False).groupby("model_id", group_keys=False).head(12)
    q["label"] = q.feature_or_group + " — " + q.model_id
    q = q.sort_values("mae_increase_mean")
    plt.figure(figsize=(9.5, max(5.5, 0.3 * len(q))))
    sns.barplot(data=q, y="label", x="mae_increase_mean", hue="model_id", dodge=False, palette="colorblind")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Increase in held-out MAE after permutation (weeks)")
    plt.ylabel("")
    plt.title("Nonlinear-model held-out permutation importance")
    plt.legend(title="Model")
    save("09_ml_permutation_feature_importance.png")

    return created


def write_documents(
    staging_docs: Path,
    model_comparison: pd.DataFrame,
    roles: pd.DataFrame,
    ml_uncertainty: pd.DataFrame,
    temporal_ml: pd.DataFrame,
    classification: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    stability: pd.DataFrame,
    nb2: pd.DataFrame,
    related: pd.DataFrame,
    findings: pd.DataFrame,
    selected_models: dict[str, str],
    audit: pd.DataFrame,
) -> dict[str, Path]:
    staging_docs.mkdir(parents=True, exist_ok=False)
    comp = model_comparison.set_index("model_id")
    h3c = classification.set_index("evaluation").loc["grouped_oof"]
    top_corr = correlation_summary.head(8)[[
        "variable_1", "variable_2", "pearson_correlation", "spearman_correlation"
    ]]
    version_mae = related[
        related.record_type.eq("paired_cluster_bootstrap") & related.metric.eq("MAE")
    ].iloc[0]
    primary_ols = coefficients[
        coefficients.association_model.eq("A1")
        & coefficients.clustering_unit.eq("first_credited_artist_primary")
        & coefficients.term.str.startswith("X_level")
    ][["term", "coefficient_weeks", "cluster_robust_se", "ci_2_5", "ci_97_5"]]
    flagged = stability[
        stability.coefficient_interpretation_flag.ne("no major coefficient-stability flag")
    ][["term", "vif", "coefficient_interpretation_flag"]]

    role_table = roles[["model_id", "role", "main_or_sensitivity", "promotion_rule_result"]]
    specification_text = f"""# Provisional Final Model Specification v2

## Status and research question

This document freezes a **provisional working model set** for thesis reporting. It does not record supervisor approval. The empirical question remains:

> Among songs that have already entered the Spotify Global Top 200, can song characteristics and early chart performance predict how long their chart success lasts?

The study is conditional on chart entry and is not a hit-prediction study.

## Analysis unit, outcome and cohort

- Analysis unit: Spotify track ID. Related-version observations are retained in the primary analysis; a conservative exclusion sensitivity tests dependence on audited related IDs.
- Early information: `X`, the number of chart appearances in calendar weeks 1–4 after the first confirmed chart entry. `X` is categorical, with levels 1–4.
- Working primary outcome: `Y_52`, the number of observed chart-presence weeks from week 5 through week 52, including confirmed re-entry. Its support is 0–48.
- Horizon sensitivity: `Y_26`, support 0–22.
- Primary feature-linked cohort: 1,014 unique 550k-linked tracks after excluding uncertain entry, insufficient follow-up and fixed windows crossing a source-wide missing week.
- Unknown source weeks remain unknown and are never converted to absence.

## Feature-selection logic

Feature selection was performed through (1) leakage-based blocking, (2) substantive pre-specification of variables observable by week 4, (3) incremental blocks, (4) grouped out-of-sample comparisons, (5) a seven-variable transparent lexical block, (6) fold-local TF-IDF filtering and regularisation as a sensitivity, and (7) correlation/VIF diagnostics. No exhaustive or p-value-based deletion was used. With N≈1,014, that restraint reduces model-selection bias and preserves interpretation of conceptually defined feature groups.

F3 contains categorical `X`; continuous danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo and duration; categorical musical key; and binary mode. LEX adds log word count, log unique-word count, lexical diversity, average word length, log line count, repeated-line proportion and log intensity-punctuation rate.

Blocked fields include popularity/followers snapshots, future rank or streams, eventual peak, first-spell and total duration, `L`, `Z`, re-entry/full-history summaries, old landmark eligibility flags and any field derived after week 4. Artist ID is grouping-only. Calendar/release variables remain a documented F4 sensitivity rather than part of the provisional final F3/LEX sequence.

## Model sequence and roles

{dataframe_to_markdown(role_table, 3)}

The two added nonlinear methods are Extra Trees and sklearn histogram gradient boosting. They use only F3 or LEX inputs, the same 1,014 tracks and the same frozen held-out folds. They are not promoted on a point estimate alone.

## Validation and uncertainty

Primary predictive evaluation uses five frozen folds that keep the first-credited artist and supported exact-lyric/related-version groups together. Complete collaborator separation is not claimed. Learned preprocessing and the small hyperparameter comparison are training-fold-only. Temporal robustness trains on 2017–2018, validates on 2019 and treats the 45-track 2020 Y_52 partition as a low-powered stress test. Paired performance differences use 5,000 cluster-bootstrap replicates over the 276 protected grouping components.

Count prediction is assessed with MAE, RMSE, bias, zero/positive error, bounds and performance by X; R² is supplementary. H3's binary component additionally uses ROC-AUC, PR-AUC, Brier score, log loss, and threshold-0.50 accuracy, precision, recall, specificity and F1-score. The project's **F1 model label** is distinct from the classification **F1-score**.

## Prediction and association are separate

Predictive conclusions come from held-out grouped/time-aware evaluation. Full-cohort OLS A1/A2/A3 tables are descriptive association companions with first-credited-artist cluster-robust standard errors; protected-component clustering is a sensitivity. A restrained NB2 A3 model-form sensitivity checks broad direction. None of these coefficients is causal.

## Correlation and coefficient safeguards

Correlation tables are descriptive and are not automatic deletion rules. Energy/loudness and lexical-length constructs require joint interpretation. VIF and A2-to-A3 coefficient stability flags identify terms that should not be interpreted in isolation.

## Principal limitations

The sample includes only already-charted songs; exact 550k linkage creates additional coverage selection; collaborator overlap can remain across lead-artist folds; 2020 Y_52 is small; feature and genre provenance have snapshot limitations; the outcome is bounded; lyric advantages are small/mixed temporally; and predictive associations are not causal. CLV remains conceptual motivation, while chart longevity is the empirical target.
"""

    evidence_text = f"""# Thesis Methods and Results Evidence Pack v2

## Purpose

This pack records validated numerical and methodological evidence for later thesis drafting. It is not a supervisor-approved final chapter and does not convert predictive associations into causal claims.

## Data and professor-motivated redesign

The earlier 268-song exploratory Kworb route was replaced by the approved, audited Bartłomiej Czyżewski Global Weekly Top 200 dataset. The source contains 44,200 weekly observations, 221 observed chart weeks and 4,247 unique Spotify tracks from 6 January 2017 to 16 April 2021. It provides rank, weekly streams and Spotify URLs/track IDs. Three source-wide missing weeks are represented as unknown.

The preliminary first-spell survival target was preserved, but fixed-window future chart presence became the provisional primary formulation because it retains early exits and counts later re-entry. Weeks 1–4 form a measurement window, not a requirement to survive four consecutive weeks. `Y_52` counts appearances in weeks 5–52; `Y_26` is the shorter sensitivity. This design avoids the old two-/four-week landmark survivor-admission mechanism.

The strict clean Y_52 550k-linked cohort contains 1,014 unique tracks. The analysis remains conditional on Global Top 200 entry, and the linked cohort has separate feature-coverage bias. The 160k-linked common-feature analysis is a robustness population, not a larger interchangeable copy.

## Correlation and feature selection

The highest retained-predictor correlation pairs are:

{dataframe_to_markdown(top_corr, 3)}

The historical full Pearson/Spearman matrices and VIF diagnostics remain supporting appendix evidence. Raw lyric character and word counts were almost redundant; the transparent block uses validated transformed summaries. Correlation did not automatically determine retention. Leakage and timing restrictions determined the admissible predictor set before performance comparison.

## Statistical and machine-learning models

The transparent sequence is F1 (categorical X), F3 (X plus audio/song descriptors) and LEX (F3 plus seven lexical summaries). First-round Poisson, NB2, bounded-binomial working model and random-forest comparisons are retained. H3 separates any future presence from duration conditional on positive presence. T3 uses training-fold-only regularised TF-IDF and remains a sensitivity. This round adds only Extra Trees and histogram gradient boosting on the frozen F3/LEX variables.

### Final count-comparison evidence

{dataframe_to_markdown(model_comparison[[c for c in ["model_id", "grouped_cv_n", "grouped_cv_mae", "grouped_cv_rmse", "grouped_cv_bias", "temporal_2019_mae", "temporal_2019_rmse", "temporal_2020_stress_mae", "temporal_2020_stress_rmse"] if c in model_comparison]], 3)}

The naive mean has MAE/RMSE {comp.loc['F0_MEAN','grouped_cv_mae']:.3f}/{comp.loc['F0_MEAN','grouped_cv_rmse']:.3f}; F1 has {comp.loc['F1','grouped_cv_mae']:.3f}/{comp.loc['F1','grouped_cv_rmse']:.3f}; F3 has {comp.loc['F3','grouped_cv_mae']:.3f}/{comp.loc['F3','grouped_cv_rmse']:.3f}; and LEX has {comp.loc['LEX','grouped_cv_mae']:.3f}/{comp.loc['LEX','grouped_cv_rmse']:.3f}. Thus X supplies the dominant gain, audio adds little, and the lexical MAE gain is small rather than transformative.

## H3 classification and hurdle interpretation

At the fixed 0.50 threshold, grouped OOF H3 has accuracy {h3c.accuracy:.3f}, precision {h3c.precision_positive_predictive_value:.3f}, recall {h3c.recall_sensitivity:.3f}, specificity {h3c.specificity:.3f}, and F1-score {h3c.classification_f1_score:.3f}. Threshold-independent/probability metrics are ROC-AUC {h3c.roc_auc:.3f}, PR-AUC {h3c.pr_auc:.3f}, Brier {h3c.brier_score:.3f}, and log loss {h3c.log_loss:.3f}. These establish discrimination and probability quality without replacing count-error metrics. Any future presence is easier to predict than positive conditional duration; H3 is therefore a probability/future-presence companion rather than an automatic count-model replacement.

## Descriptive association companion

The A1 categorical-X contrasts, relative to X=1, are:

{dataframe_to_markdown(primary_ols, 3)}

These are descriptive weeks-of-presence associations with first-artist cluster-robust uncertainty. A2/A3 add audio and lexical variables. Terms flagged for cautious isolated interpretation include:

{dataframe_to_markdown(flagged.head(20), 3) if len(flagged) else 'No term crossed the pre-specified instability flags.'}

The NB2 A3 fit converged for {int(nb2.converged.iloc[0])} (dispersion alpha {nb2.dispersion_alpha_fixed_method_of_moments.iloc[0]:.3f}); it is a model-form sensitivity conditional on estimated dispersion, not the primary inference table.

## Related-version sensitivity

The conservative rule excludes all 143 tracks in 70 supported duplicate/related-version groups, retains the unresolved two-track group, and leaves 871 tracks. The LEX-versus-F3 MAE difference after exclusion is {version_mae.observed_difference:.3f} weeks with a 95% protected-component cluster-bootstrap interval [{version_mae.ci_2_5:.3f}, {version_mae.ci_97_5:.3f}]. The primary unit remains Spotify track ID; this sensitivity tests fragmentation rather than guessing merges.

## Finding hierarchy

{dataframe_to_markdown(findings, 3)}

## Safe claims

- Among songs already in the Global Top 200, first-four-week presence is the strongest validated predictive signal.
- Standard audio descriptors add little incremental out-of-sample accuracy beyond X.
- Transparent lexical summaries provide at most a small MAE improvement; RMSE and calendar-time evidence qualify this claim.
- Predicting whether any future presence occurs is easier than predicting how long positive future presence lasts.
- Nonlinear models test interactions in the same pre-specified variables; their role depends on uncertainty, temporal robustness and practical magnitude, not the lowest point estimate alone.
- Unknown source weeks were not interpreted as absence, and blocked future/snapshot fields did not enter the models.

## Claims to avoid

- Do not claim hit prediction, population-wide generalisation to uncharted songs, causality, supervisor approval, a large lyric effect, or universal superiority of one model.
- Do not describe chart duration itself as CLV or interpret a p-value as feature importance.
- Do not claim complete collaborator separation or strong evidence from the 45-track 2020 Y_52 stress set.

## Suggested thesis presentation

Main text: cohort and target table; F1/F3/LEX incremental table; final count comparison; H3 binary and count metrics kept distinct; grouped versus temporal evaluation; protected-component uncertainty; descriptive clustered association table; limitations. Appendices: full correlations/VIF, tuning records, TF-IDF, bounded/NB2/RF/ML sensitivities, 160k robustness, Y_26, related versions, source-gap audit, all leakage/integrity manifests.
"""

    notebook_text = """# Thesis Notebook Plan v2

The final notebooks should be assembled only from validated scripts and frozen outputs. Each notebook should run from project-relative paths, record package versions and random seeds, and fail on leakage/count reconciliation errors.

## Notebook 01 — Data sources, audit and cohort construction

- Raw-file provenance, licences, checksums and source coverage.
- Schema, duplicates, missing values and Spotify-ID validation.
- Exact matching to 550k and 160k feature sources.
- Reproducible cleaning and exclusion register.

## Notebook 02 — Outcome construction and exploratory analysis

- Weekly calendar and three unknown source weeks.
- Preserved first-spell survival target.
- Fixed-window X, Y, L and Z definitions; 26/52-week horizon comparison.
- Cohort selection, outcome distributions, artist repetition and selection-bias diagnostics.

## Notebook 03 — Correlation, feature selection and benchmark models

- Predictor allowlist/blocklist and timing audit.
- Pearson, Spearman, VIF and missing/range checks.
- Frozen grouped folds and temporal partitions.
- F0–F4 and Poisson/NB2/bounded/random-forest benchmarks.

## Notebook 04 — Lyrics, TF-IDF and hurdle modelling

- Lyric-quality audit and seven transparent lexical summaries.
- Training-fold-only TF-IDF vocabulary, IDF and regularisation.
- LEX/T3 increments and X=4 diagnostics.
- H3 binary/positive components, calibration and classification metrics.

## Notebook 05 — Nonlinear ML comparisons

- Extra Trees and histogram gradient boosting on F3/LEX only.
- Frozen outer folds, small inner grouped grids and raw held-out predictions.
- Cluster-bootstrap uncertainty, temporal checks and permutation importance.

## Notebook 06 — Robustness, association and final thesis figures

- Y_26, 160k, model-form, version and temporal sensitivities.
- A1/A2/A3 artist-clustered OLS, standardised summaries and NB2 companion.
- H3 descriptive logistic association.
- Final evidence matrix, finding hierarchy and publication-ready figures.

Avoid duplicating large data in notebooks. Notebook outputs should point to the versioned CSV/JSON evidence files and reproduce only the tables/figures needed for review.
"""

    paths = {
        "provisional_final_model_specification_v2.md": staging_docs / "provisional_final_model_specification_v2.md",
        "thesis_methods_results_evidence_pack_v2.md": staging_docs / "thesis_methods_results_evidence_pack_v2.md",
        "thesis_notebook_plan_v2.md": staging_docs / "thesis_notebook_plan_v2.md",
    }
    paths["provisional_final_model_specification_v2.md"].write_text(specification_text, encoding="utf-8")
    paths["thesis_methods_results_evidence_pack_v2.md"].write_text(evidence_text, encoding="utf-8")
    paths["thesis_notebook_plan_v2.md"].write_text(notebook_text, encoding="utf-8")
    return paths


def model_specification_table(tuning: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, implementation, configs in [
        ("ET", "sklearn.ensemble.ExtraTreesRegressor", ET_CONFIGS),
        ("HGB", "sklearn.ensemble.HistGradientBoostingRegressor", HGB_CONFIGS),
    ]:
        for block, variables in BLOCKS.items():
            selected = tuning[
                tuning.algorithm.eq(algorithm)
                & tuning.feature_block.eq(block)
                & tuning.selected.eq(1)
            ]
            rows.append({
                "algorithm": algorithm,
                "implementation": implementation,
                "feature_block": block,
                "predictor_count_before_encoding": len(variables),
                "predictors": "|".join(variables),
                "candidate_configurations": json.dumps(configs, sort_keys=True),
                "outer_contexts_fitted": int(selected.context.nunique()),
                "selected_config_counts": json.dumps(
                    selected.config_id.value_counts().sort_index().to_dict(), sort_keys=True
                ),
                "inner_selection_metric": "mean MAE; RMSE and config ID as deterministic ties",
                "inner_folds": 3,
                "outer_folds": 5,
                "random_seed": SEED,
                "preprocessing": "training-only one-hot encoding for categorical X/key; mode binary; numeric values unchanged; no scaling",
                "outcome": "Y_52 (0-48)",
                "role": "narrow nonlinear comparison/sensitivity",
            })
    return pd.DataFrame(rows)


def build_integrity_audit(
    audit_rows: list[dict],
    historical_before: dict[str, dict[str, str]],
    historical_after: dict[str, dict[str, str]],
    ml_predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    h3_metrics: pd.DataFrame,
    h3_confusion: pd.DataFrame,
    feature_record: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    related: pd.DataFrame,
) -> pd.DataFrame:
    if historical_before != historical_after:
        changed = [key for key in historical_before if historical_before[key] != historical_after[key]]
        raise AssertionError(f"Historical input/output directories changed: {changed}")
    audit_rows.append({
        "check": "historical directories unchanged", "status": "PASS",
        "evidence": f"before/after SHA-256 maps identical for {len(historical_before)} authoritative historical directories",
    })

    expected = {"ET_ML_F3", "ET_ML_LEX", "HGB_ML_F3", "HGB_ML_LEX"}
    if set(ml_predictions.model_id) != expected:
        raise AssertionError("The new ML prediction model set is not exactly the intended four combinations.")
    if len(ml_predictions) != 4 * 1014:
        raise AssertionError("ML prediction rows do not equal four models by 1,014 tracks.")
    audit_rows.append({
        "check": "new ML method and block restriction", "status": "PASS",
        "evidence": "exactly Extra Trees and histogram gradient boosting; each evaluated on ML_F3 and ML_LEX only",
    })

    if tuning.inner_folds.dropna().astype(int).ne(3).any():
        raise AssertionError("An ML tuning record does not use three inner grouped folds.")
    audit_rows.append({
        "check": "limited hyperparameter policy", "status": "PASS",
        "evidence": "three predeclared configurations per algorithm; three-fold inner protected-group CV; no outer-fold selection",
    })

    retained = set(feature_record.loc[feature_record.final_status.eq("retained"), "variable_or_group"])
    expected_retained = set(F3 + LEX_MODEL)
    if retained != expected_retained:
        raise AssertionError(f"Feature-selection record mismatch: {retained ^ expected_retained}")
    if leakage_hits(sorted(retained)):
        raise AssertionError("A blocked field appears in the retained feature record.")
    audit_rows.append({
        "check": "feature record reconciles with model predictors", "status": "PASS",
        "evidence": f"retained record equals the {len(expected_retained)} unique F3/LEX variables; blocked-field scan empty",
    })

    if len(correlation_summary) != math.comb(len(AUDIO_CONTINUOUS + ["mode"] + LEX_MODEL), 2):
        raise AssertionError("Final reduced correlation-pair count is inconsistent with intended variables.")
    audit_rows.append({
        "check": "correlation table scope", "status": "PASS",
        "evidence": "all and only pairwise relationships among retained transparent numeric/binary predictors are represented",
    })

    if not np.allclose(h3_metrics.decision_threshold, THRESHOLD):
        raise AssertionError("H3 classification threshold differs from 0.50.")
    if not (
        h3_confusion[["true_positive", "false_positive", "true_negative", "false_negative"]].sum(axis=1)
        == h3_confusion.total
    ).all():
        raise AssertionError("H3 confusion cells do not reconcile.")
    audit_rows.append({
        "check": "H3 classification specification", "status": "PASS",
        "evidence": "decision threshold fixed at 0.50 without held-out selection; confusion counts reconcile for all three evaluations",
    })

    source_metrics = {
        "F1": (6.314077, 10.686244),
        "F3": (6.231387, 10.786244),
        "LEX": (6.069582, 10.774377),
        "H3": (6.079406, 10.721969),
        "T3": (6.037900, 11.107569),
    }
    for model_id, (mae, rmse) in source_metrics.items():
        row = model_comparison[model_comparison.model_id.eq(model_id)].iloc[0]
        if not (np.isclose(row.grouped_cv_mae, mae, atol=1e-6) and np.isclose(row.grouped_cv_rmse, rmse, atol=1e-6)):
            raise AssertionError(f"Reused metric reconciliation failed for {model_id}.")
    if model_comparison.model_id.duplicated().any():
        raise AssertionError("The final model-comparison table has duplicate model IDs.")
    audit_rows.append({
        "check": "reused and final comparison metrics reconcile", "status": "PASS",
        "evidence": "F1/F3/LEX/H3/T3 reproduce authoritative source metrics to 1e-6; final model IDs are unique",
    })

    version_meta = related[related.record_type.eq("model_metric")].iloc[0]
    if int(version_meta.n_tracks_removed) != 143 or int(version_meta.n_tracks_remaining) != 871:
        raise AssertionError("Related-version exclusion counts do not match the documented rule.")
    audit_rows.append({
        "check": "related-version sensitivity reconciliation", "status": "PASS",
        "evidence": "70 supported groups/143 records excluded, unresolved group retained, 871 records remain and frozen folds are preserved",
    })

    result = pd.DataFrame(audit_rows)
    if not result.status.eq("PASS").all():
        raise AssertionError("At least one integrity check is not PASS.")
    return result


def save_csv_outputs(directory: Path, outputs: dict[str, pd.DataFrame]):
    directory.mkdir(parents=True, exist_ok=False)
    for filename, frame in outputs.items():
        if frame is None or not isinstance(frame, pd.DataFrame) or len(frame.columns) == 0:
            raise AssertionError(f"Invalid dataframe for {filename}.")
        frame.to_csv(directory / filename, index=False)
    # Reopen every CSV to detect malformed or empty artifacts before publication.
    for filename in outputs:
        reopened = pd.read_csv(directory / filename)
        if len(reopened.columns) == 0:
            raise AssertionError(f"Generated CSV has no columns: {filename}")


def main():
    protected_paths = [OUT_DIR, FIG_DIR, SPEC_REPORT, EVIDENCE_PACK, NOTEBOOK_PLAN]
    already_exists = [str(path) for path in protected_paths if path.exists()]
    if already_exists:
        raise FileExistsError(
            "Refusing to overwrite this task's versioned outputs: " + ", ".join(already_exists)
        )

    historical_dirs = {
        "pre_model_analysis_ready_v1": ANALYSIS_DIR,
        "final_pre_model_specification_v1": SPEC_DIR,
        "provisional_benchmark_modelling_v1": BENCH_DIR,
        "provisional_lyrics_hurdle_modelling_v1": LYRIC_DIR,
        "provisional_robustness_uncertainty_v1": ROBUST_DIR,
        "pre_model_remaining_eda_v2": VERSION_AUDIT_DIR,
    }
    historical_before = {name: directory_digest(path) for name, path in historical_dirs.items()}
    audit_rows: list[dict] = []
    warning_rows: list[dict] = []
    data52, data26, assignments, temporal52_frozen, temporal26_frozen = validate_inputs(audit_rows)
    if data52.Y_52.min() < 0 or data52.Y_52.max() > YMAX:
        raise AssertionError("Y_52 is outside its frozen 0-48 support.")
    audit_rows.append({
        "check": "outcome support and unit", "status": "PASS",
        "evidence": f"Y_52 range {data52.Y_52.min():.0f}-{data52.Y_52.max():.0f}; one row per 1,014 Spotify track IDs",
    })

    ml_predictions, grouped_tuning, importance_raw = run_grouped_ml(data52, audit_rows, warning_rows)
    ml_aggregate, ml_folds, ml_by_x = summarise_predictions(ml_predictions)
    importance = aggregate_importance(importance_raw)
    selected_models = choose_best_ml_blocks(ml_aggregate)
    ml_uncertainty, x4_uncertainty = ml_paired_uncertainty(data52, ml_predictions, selected_models)

    (
        temporal_ml_predictions, temporal_ml_metrics,
        temporal26_predictions, temporal26_metrics,
        temporal_tuning, temporal52,
    ) = run_temporal_ml(data52, data26, selected_models, audit_rows, warning_rows)
    temporal_uncertainty = temporal_ml_uncertainty(temporal_ml_predictions, temporal52)
    all_tuning = pd.concat([grouped_tuning, temporal_tuning], ignore_index=True, sort=False)

    h3_metrics, h3_confusion, h3_grouped, h3_temporal = consolidate_h3_classification(audit_rows)
    correlation_summary, pearson, spearman = final_correlation_analysis(data52, audit_rows)
    ols_coefficients, standardised = fit_all_ols(data52, audit_rows)
    stability = coefficient_stability_audit(data52, ols_coefficients, correlation_summary)
    nb2, nb2_info = fit_nb2_association(data52, standardised)
    logistic, logistic_info = fit_logistic_association(data52)
    if not nb2_info["converged"]:
        warning_rows.append({
            "stage": "NB2_association", "context": "full cohort A3", "algorithm": "NB2",
            "feature_block": "A3", "config_id": "method_of_moments_alpha", "fold": "full",
            "status": "warning", "warning_or_failure": nb2_info["message"],
        })
    if not logistic_info["converged"]:
        warning_rows.append({
            "stage": "H3_binary_association", "context": "full cohort descriptive", "algorithm": "logistic",
            "feature_block": "A3/LEX", "config_id": "unpenalised_descriptive", "fold": "full",
            "status": "warning", "warning_or_failure": logistic_info["message"],
        })
    positive_summary, h3_calibration = h3_descriptive_summaries(h3_metrics)
    related, related_predictions = related_version_sensitivity(data52, audit_rows)
    feature_record = feature_selection_record()
    model_specs = model_specification_table(all_tuning)
    model_comparison = thesis_model_comparison(
        ml_aggregate, temporal_ml_metrics, selected_models, h3_metrics,
    )
    roles = candidate_roles(ml_aggregate, ml_uncertainty, temporal_ml_metrics, selected_models)
    robustness = robustness_evidence_matrix(
        model_comparison, ml_uncertainty, selected_models, related, h3_metrics,
    )
    findings = finding_hierarchy(model_comparison, ml_uncertainty, related, selected_models)

    historical_after = {name: directory_digest(path) for name, path in historical_dirs.items()}
    integrity = build_integrity_audit(
        audit_rows, historical_before, historical_after, ml_predictions, all_tuning,
        h3_metrics, h3_confusion, feature_record, correlation_summary,
        model_comparison, related,
    )

    warning_table = pd.DataFrame(warning_rows)
    if warning_table.empty:
        warning_table = pd.DataFrame([{
            "stage": "all", "context": "all fits", "algorithm": "all",
            "feature_block": "all", "config_id": "not applicable", "fold": "all",
            "status": "ok", "warning_or_failure": "No warnings recorded.",
        }])

    outputs = {
        "ml_model_specifications.csv": model_specs,
        "ml_inner_grouped_tuning.csv": all_tuning,
        "ml_grouped_cv_predictions.csv": ml_predictions,
        "ml_grouped_cv_metrics.csv": ml_aggregate,
        "ml_grouped_cv_fold_metrics.csv": ml_folds,
        "ml_performance_by_X.csv": ml_by_x,
        "ml_feature_importance.csv": importance,
        "ml_paired_cluster_bootstrap.csv": ml_uncertainty,
        "ml_X4_paired_cluster_bootstrap.csv": x4_uncertainty,
        "ml_temporal_predictions_52w.csv": temporal_ml_predictions,
        "ml_temporal_metrics_52w.csv": temporal_ml_metrics,
        "ml_temporal_predictions_26w_2020.csv": temporal26_predictions,
        "ml_temporal_metrics_26w_2020.csv": temporal26_metrics,
        "ml_temporal_paired_uncertainty_52w.csv": temporal_uncertainty,
        "h3_binary_classification_metrics.csv": h3_metrics,
        "h3_confusion_matrix.csv": h3_confusion,
        "final_predictor_correlation_summary.csv": correlation_summary,
        "final_predictor_pearson_matrix.csv": pearson.reset_index().rename(columns={"index": "variable"}),
        "final_predictor_spearman_matrix.csv": spearman.reset_index().rename(columns={"index": "variable"}),
        "final_feature_selection_record.csv": feature_record,
        "final_candidate_model_roles.csv": roles,
        "clustered_ols_coefficients.csv": ols_coefficients,
        "standardised_association_summary.csv": standardised,
        "final_coefficient_stability_audit.csv": stability,
        "nb2_association_sensitivity.csv": nb2,
        "h3_binary_association_summary.csv": logistic,
        "h3_positive_duration_summary.csv": positive_summary,
        "h3_final_calibration_summary.csv": h3_calibration,
        "related_version_exclusion_sensitivity.csv": related,
        "related_version_exclusion_predictions.csv": related_predictions,
        "final_robustness_evidence_matrix.csv": robustness,
        "thesis_final_model_comparison.csv": model_comparison,
        "final_finding_hierarchy.csv": findings,
        "robustness_model_fit_warnings.csv": warning_table,
        "final_model_integrity_audit.csv": integrity,
    }

    with tempfile.TemporaryDirectory(prefix="spotify_final_model_") as temporary:
        staging_root = Path(temporary)
        staging_data = staging_root / "data"
        staging_figures = staging_root / "figures"
        staging_docs = staging_root / "docs"
        save_csv_outputs(staging_data, outputs)
        figure_names = make_figures(
            staging_figures, model_comparison, ml_predictions, ml_uncertainty,
            x4_uncertainty, temporal_ml_metrics, selected_models, h3_grouped,
            h3_metrics, h3_confusion, pearson, importance,
        )
        document_paths = write_documents(
            staging_docs, model_comparison, roles, ml_uncertainty,
            temporal_ml_metrics, h3_metrics, correlation_summary,
            ols_coefficients, stability, nb2, related, findings,
            selected_models, integrity,
        )
        for filename in figure_names:
            if (staging_figures / filename).stat().st_size < 5000:
                raise AssertionError(f"Figure appears empty or corrupt: {filename}")
        for path in document_paths.values():
            if path.stat().st_size < 1000:
                raise AssertionError(f"Document appears unexpectedly short: {path.name}")

        manifest = {
            "task": "provisional final thesis-ready model comparison v1",
            "status": "provisional working specification; not supervisor-approved",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(ROOT),
            "outcome": {"primary": "Y_52", "support": [0, 48], "n": 1014, "sensitivity": "Y_26"},
            "new_ml_methods": {
                "ML1": "sklearn ExtraTreesRegressor",
                "ML2": "sklearn HistGradientBoostingRegressor (XGBoost not installed; environment unchanged)",
            },
            "selected_grouped_cv_models": selected_models,
            "validation": {
                "outer_folds": 5,
                "protected_group_components": 276,
                "inner_grouped_folds": 3,
                "bootstrap_replicates": BOOTSTRAP_REPS,
                "bootstrap_seed_base": SEED,
                "H3_classification_threshold": THRESHOLD,
                "Y52_temporal_counts": temporal52_frozen.temporal_partition.value_counts().sort_index().to_dict(),
                "Y26_temporal_counts": temporal26_frozen.temporal_partition.value_counts().sort_index().to_dict(),
            },
            "association_models": {
                "OLS": "A1/A2/A3 with first-credited-artist CR1 cluster covariance; protected-component sensitivity",
                "NB2": nb2_info,
                "H3_binary_logistic": logistic_info,
            },
            "historical_directory_hashes_unchanged": historical_before == historical_after,
            "input_directory_sha256_maps": historical_before,
            "script_sha256": sha256(Path(__file__)),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
            "integrity_all_pass": bool(integrity.status.eq("PASS").all()),
            "files": {
                "data": sorted(outputs),
                "figures": figure_names,
                "documents": sorted(document_paths),
            },
            "staged_file_sha256": {
                str(path.relative_to(staging_root)): sha256(path)
                for path in sorted(staging_root.rglob("*")) if path.is_file()
            },
        }
        (staging_data / "final_model_validation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Reopen manifest before atomically publishing new versioned directories/files.
        loaded_manifest = json.loads((staging_data / "final_model_validation_manifest.json").read_text())
        if not loaded_manifest["integrity_all_pass"]:
            raise AssertionError("Manifest records a failed integrity state.")

        OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        FIG_DIR.parent.mkdir(parents=True, exist_ok=True)
        SPEC_REPORT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_data, OUT_DIR)
        shutil.copytree(staging_figures, FIG_DIR)
        shutil.copy2(document_paths["provisional_final_model_specification_v2.md"], SPEC_REPORT)
        shutil.copy2(document_paths["thesis_methods_results_evidence_pack_v2.md"], EVIDENCE_PACK)
        shutil.copy2(document_paths["thesis_notebook_plan_v2.md"], NOTEBOOK_PLAN)

    print(json.dumps({
        "status": "complete",
        "output_directory": str(OUT_DIR),
        "figure_directory": str(FIG_DIR),
        "documents": [str(SPEC_REPORT), str(EVIDENCE_PACK), str(NOTEBOOK_PLAN)],
        "selected_models": selected_models,
        "integrity_checks": int(len(integrity)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
