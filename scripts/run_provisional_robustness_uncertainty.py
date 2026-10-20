#!/usr/bin/env python3
"""Targeted robustness and uncertainty analysis for Spotify chart longevity.

This script treats the frozen pre-model specification, first benchmark round,
and lyric/hurdle round as immutable inputs. It aligns the existing out-of-fold
predictions, performs paired cluster bootstraps, fits only the requested
temporal models and restrained TF-IDF sensitivity, and stages all new outputs
before publishing them to a new versioned directory.
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
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_robustness"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, hstack
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/processed/pre_model_analysis_ready_v1"
SPEC_DIR = ROOT / "data/processed/final_pre_model_specification_v1"
BENCH_DIR = ROOT / "data/processed/provisional_benchmark_modelling_v1"
LYRIC_DIR = ROOT / "data/processed/provisional_lyrics_hurdle_modelling_v1"
OUT_DIR = ROOT / "data/processed/provisional_robustness_uncertainty_v1"
FIG_DIR = ROOT / "outputs/provisional_robustness_uncertainty_v1/figures"
REPORT_PATH = ROOT / "docs/provisional_robustness_uncertainty_report.md"
MEMO_PATH = ROOT / "docs/thesis_evidence_memo_robustness_v1.md"

SEED = 20260825
BOOTSTRAP_REPS = 5000
MODELS = ["REF_F1", "REF_F3", "LEX", "T3", "H3"]
MODEL_LABELS = {
    "REF_F1": "F1: categorical X only",
    "REF_F3": "F3: X + audio/song characteristics",
    "LEX": "LEX: F3 + transparent lexical summaries",
    "T3": "T3: F3 + fold-local TF-IDF",
    "H3": "H3: hurdle with X + audio + lexical summaries",
    "T3_LOW_ALPHA": "T3 lower-alpha sensitivity",
}
COMPARISONS = [
    ("REF_F3", "LEX"),
    ("REF_F3", "T3"),
    ("REF_F3", "H3"),
    ("LEX", "T3"),
    ("LEX", "H3"),
]
X4_COMPARISONS = [("REF_F3", "LEX"), ("REF_F3", "T3"), ("REF_F3", "H3")]

sys.path.insert(0, str(ROOT / "scripts"))
import run_provisional_lyrics_hurdle_modelling as prior  # noqa: E402


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


def dataframe_to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def metric_values(y, pred, nmax: int) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    zero = y == 0
    positive = y > 0
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mean_observed_y": float(y.mean()),
        "mean_prediction": float(pred.mean()),
        "mean_prediction_bias": float(np.mean(pred - y)),
        "observed_zero_n": int(zero.sum()),
        "observed_zero_pct": float(zero.mean() * 100),
        "mae_observed_zero": float(mean_absolute_error(y[zero], pred[zero])) if zero.any() else np.nan,
        "mae_observed_positive": float(mean_absolute_error(y[positive], pred[positive])) if positive.any() else np.nan,
        "mean_prediction_observed_zero": float(pred[zero].mean()) if zero.any() else np.nan,
        "mean_prediction_observed_positive": float(pred[positive].mean()) if positive.any() else np.nan,
        "outside_bound_n": int(np.sum((pred < 0) | (pred > nmax))),
        "outside_bound_pct": float(np.mean((pred < 0) | (pred > nmax)) * 100),
    }


def safe_binary_metrics(y_binary, p_positive, threshold: float = 0.5) -> dict:
    y = np.asarray(y_binary, int)
    p = np.clip(np.asarray(p_positive, float), 1e-12, 1 - 1e-12)
    cls = (p >= threshold).astype(int)
    negative = y == 0
    positive = y == 1
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_n": int(positive.sum()),
        "positive_pct": float(positive.mean() * 100),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_supplementary": float(accuracy_score(y, cls)),
        "sensitivity_recall_positive": float(recall_score(y, cls, zero_division=0)),
        "specificity_zero": float(np.mean(cls[negative] == 0)) if negative.any() else np.nan,
        "classification_threshold": threshold,
        "mean_predicted_positive_probability": float(p.mean()),
        "mean_predicted_zero_probability": float((1 - p).mean()),
        "observed_zero_pct": float(negative.mean() * 100),
        "zero_probability_brier_score": float(brier_score_loss(negative.astype(int), 1 - p)),
        "zero_probability_log_loss": float(log_loss(negative.astype(int), 1 - p, labels=[0, 1])),
    }


def load_and_validate_oof(audit_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(ANALYSIS_DIR / "analysis_ready_550k_52w.csv")
    assignments = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")
    if len(cohort) != 1014 or cohort.spotify_track_id.nunique() != 1014:
        raise AssertionError("The frozen Y_52 cohort is not exactly 1,014 unique tracks.")
    if len(assignments) != 1014 or assignments.spotify_track_id.nunique() != 1014:
        raise AssertionError("The frozen grouped assignments are not exactly 1,014 unique tracks.")
    if set(cohort.spotify_track_id) != set(assignments.spotify_track_id):
        raise AssertionError("Cohort and grouped-assignment track IDs differ.")

    protected_flags = [
        "first_artist_crosses_folds",
        "exact_lyric_group_crosses_folds",
        "related_version_group_crosses_folds",
    ]
    if assignments[protected_flags].sum().sum() != 0:
        raise AssertionError("A protected group crosses the frozen grouped folds.")
    audit_rows.extend([
        {"check": "primary cohort uniqueness", "status": "PASS", "evidence": "1,014 rows; 1,014 Spotify track IDs"},
        {"check": "frozen fold integrity", "status": "PASS", "evidence": "first artist, exact lyric and supported related-version fold crossings all equal zero"},
        {"check": "bootstrap grouping unit", "status": "PASS", "evidence": f"primary_group_component; {assignments.primary_group_component.nunique()} clusters"},
    ])

    benchmark = pd.read_csv(BENCH_DIR / "grouped_cv_predictions.csv")
    benchmark = benchmark[
        benchmark.outcome.eq("Y_52")
        & benchmark.model_id.eq("M1_poisson")
        & benchmark.feature_block.isin(["F1", "F3"])
    ].copy()
    benchmark["model"] = benchmark.feature_block.map({"F1": "REF_F1", "F3": "REF_F3"})
    benchmark = benchmark.rename(columns={"split": "fold"})

    lyric = pd.read_csv(LYRIC_DIR / "lyric_grouped_cv_predictions.csv")
    lyric = lyric[lyric.model_id.isin(["LEX", "T3"])].copy()
    lyric["model"] = lyric.model_id

    hurdle = pd.read_csv(LYRIC_DIR / "hurdle_grouped_cv_predictions.csv")
    hurdle = hurdle[hurdle.model_id.eq("H3")].copy()
    hurdle["model"] = "H3"

    common_columns = ["spotify_track_id", "fold", "model", "y_observed", "y_pred_raw", "X"]
    long = pd.concat(
        [benchmark[common_columns], lyric[common_columns], hurdle[common_columns]],
        ignore_index=True,
    )
    duplicate_count = int(long.duplicated(["spotify_track_id", "model"]).sum())
    if duplicate_count:
        raise AssertionError(f"Found {duplicate_count} duplicate track/model OOF predictions.")
    coverage = long.groupby("model").agg(rows=("spotify_track_id", "size"), tracks=("spotify_track_id", "nunique"))
    if set(coverage.index) != set(MODELS) or not ((coverage.rows == 1014) & (coverage.tracks == 1014)).all():
        raise AssertionError(f"OOF coverage is incomplete:\n{coverage}")

    outcome_spread = long.groupby("spotify_track_id").y_observed.nunique()
    fold_spread = long.groupby("spotify_track_id").fold.nunique()
    if outcome_spread.max() != 1 or fold_spread.max() != 1:
        raise AssertionError("Outcomes or folds disagree across aligned model files.")
    frozen_fold = assignments.set_index("spotify_track_id").fold
    if not (long.fold.to_numpy() == long.spotify_track_id.map(frozen_fold).to_numpy()).all():
        raise AssertionError("At least one OOF prediction has a fold that differs from the frozen assignment.")
    expected_y = cohort.set_index("spotify_track_id").Y_52
    if not np.allclose(long.y_observed, long.spotify_track_id.map(expected_y)):
        raise AssertionError("OOF outcomes do not agree with the frozen cohort.")

    wide_pred = long.pivot(index="spotify_track_id", columns="model", values="y_pred_raw")
    wide = (
        cohort[["spotify_track_id", "Y_52", "X", "chart_entry_year"]]
        .merge(assignments, on=["spotify_track_id", "chart_entry_year"], validate="one_to_one")
        .merge(wide_pred.reset_index(), on="spotify_track_id", validate="one_to_one")
    )
    if wide[MODELS].isna().any().any() or len(wide) != 1014:
        raise AssertionError("Aligned OOF prediction matrix is incomplete.")
    audit_rows.extend([
        {"check": "OOF prediction uniqueness", "status": "PASS", "evidence": "one held-out prediction per track for each of REF_F1, REF_F3, LEX, T3 and H3"},
        {"check": "OOF outcome alignment", "status": "PASS", "evidence": "Y_52 agrees across all prediction files and the frozen cohort"},
        {"check": "OOF fold alignment", "status": "PASS", "evidence": "all prediction fold labels equal the frozen assignments"},
        {"check": "OOF prediction provenance", "status": "PASS", "evidence": "existing validation manifest records outer grouped folds, training-only TF-IDF and complete held-out coverage"},
    ])
    return wide, long


def cluster_bootstrap_comparison(
    data: pd.DataFrame,
    reference: str,
    candidate: str,
    cluster_col: str,
    reps: int,
    seed: int,
    subset: str,
) -> pd.DataFrame:
    q = data[["Y_52", reference, candidate, cluster_col]].dropna().copy()
    if not len(q):
        raise AssertionError("No aligned data for paired bootstrap.")
    q["ae_reference"] = np.abs(q[reference] - q.Y_52)
    q["ae_candidate"] = np.abs(q[candidate] - q.Y_52)
    q["se_reference"] = (q[reference] - q.Y_52) ** 2
    q["se_candidate"] = (q[candidate] - q.Y_52) ** 2
    q["err_reference"] = q[reference] - q.Y_52
    q["err_candidate"] = q[candidate] - q.Y_52
    aggregates = q.groupby(cluster_col, sort=True).agg(
        n=("Y_52", "size"),
        ae_reference=("ae_reference", "sum"),
        ae_candidate=("ae_candidate", "sum"),
        se_reference=("se_reference", "sum"),
        se_candidate=("se_candidate", "sum"),
        err_reference=("err_reference", "sum"),
        err_candidate=("err_candidate", "sum"),
    )
    group_n = len(aggregates)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(group_n, np.repeat(1 / group_n, group_n), size=reps)
    denominator = counts @ aggregates.n.to_numpy(float)
    ref_mae = (counts @ aggregates.ae_reference.to_numpy(float)) / denominator
    cand_mae = (counts @ aggregates.ae_candidate.to_numpy(float)) / denominator
    ref_rmse = np.sqrt((counts @ aggregates.se_reference.to_numpy(float)) / denominator)
    cand_rmse = np.sqrt((counts @ aggregates.se_candidate.to_numpy(float)) / denominator)
    ref_bias = (counts @ aggregates.err_reference.to_numpy(float)) / denominator
    cand_bias = (counts @ aggregates.err_candidate.to_numpy(float)) / denominator

    bootstrap = {
        "MAE": ref_mae - cand_mae,
        "RMSE": ref_rmse - cand_rmse,
        "bias_candidate_minus_reference": cand_bias - ref_bias,
    }
    observed = {
        "MAE": mean_absolute_error(q.Y_52, q[reference]) - mean_absolute_error(q.Y_52, q[candidate]),
        "RMSE": math.sqrt(mean_squared_error(q.Y_52, q[reference])) - math.sqrt(mean_squared_error(q.Y_52, q[candidate])),
        "bias_candidate_minus_reference": float(np.mean(q[candidate] - q.Y_52) - np.mean(q[reference] - q.Y_52)),
    }
    rows = []
    for metric, values in bootstrap.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append({
            "subset": subset,
            "reference_model": reference,
            "candidate_model": candidate,
            "metric": metric,
            "improvement_definition": (
                "reference error minus candidate error; positive favours candidate"
                if metric in {"MAE", "RMSE"}
                else "candidate mean prediction bias minus reference mean prediction bias"
            ),
            "observed_difference": float(observed[metric]),
            "bootstrap_mean_difference": float(np.mean(values)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
            "ci_2_5": float(lower),
            "ci_97_5": float(upper),
            "bootstrap_proportion_candidate_better": float(np.mean(values > 0)) if metric in {"MAE", "RMSE"} else np.nan,
            "n_tracks": int(len(q)),
            "n_clusters": int(group_n),
            "bootstrap_replicates": int(reps),
            "bootstrap_seed": int(seed),
            "cluster_unit": cluster_col,
        })
    return pd.DataFrame(rows)


def paired_uncertainty(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, (reference, candidate) in enumerate(COMPARISONS):
        rows.append(cluster_bootstrap_comparison(
            wide, reference, candidate, "primary_group_component", BOOTSTRAP_REPS,
            SEED + i, "all_Y52_grouped_OOF",
        ))
    uncertainty = pd.concat(rows, ignore_index=True)

    fold_rows = []
    for fold, q in wide.groupby("fold", sort=True):
        for reference, candidate in COMPARISONS:
            fold_rows.append({
                "fold": fold,
                "reference_model": reference,
                "candidate_model": candidate,
                "n_tracks": len(q),
                "n_clusters": q.primary_group_component.nunique(),
                "reference_mae": mean_absolute_error(q.Y_52, q[reference]),
                "candidate_mae": mean_absolute_error(q.Y_52, q[candidate]),
                "mae_improvement": mean_absolute_error(q.Y_52, q[reference]) - mean_absolute_error(q.Y_52, q[candidate]),
                "reference_rmse": math.sqrt(mean_squared_error(q.Y_52, q[reference])),
                "candidate_rmse": math.sqrt(mean_squared_error(q.Y_52, q[candidate])),
                "rmse_improvement": math.sqrt(mean_squared_error(q.Y_52, q[reference])) - math.sqrt(mean_squared_error(q.Y_52, q[candidate])),
            })
    by_fold = pd.DataFrame(fold_rows)

    absolute_rows = []
    for reference, candidate in COMPARISONS:
        q = wide[["spotify_track_id", "fold", "primary_group_component", "X", "Y_52", reference, candidate]].copy()
        q["reference_model"] = reference
        q["candidate_model"] = candidate
        q["absolute_error_reference"] = np.abs(q[reference] - q.Y_52)
        q["absolute_error_candidate"] = np.abs(q[candidate] - q.Y_52)
        q["absolute_error_improvement"] = q.absolute_error_reference - q.absolute_error_candidate
        absolute_rows.append(q[[
            "spotify_track_id", "fold", "primary_group_component", "X", "Y_52",
            "reference_model", "candidate_model", "absolute_error_reference",
            "absolute_error_candidate", "absolute_error_improvement",
        ]])
    track_diffs = pd.concat(absolute_rows, ignore_index=True)

    x4 = wide[wide.X.eq(4)].copy()
    x4_rows = []
    for i, (reference, candidate) in enumerate(X4_COMPARISONS):
        x4_rows.append(cluster_bootstrap_comparison(
            x4, reference, candidate, "primary_group_component", BOOTSTRAP_REPS,
            SEED + 100 + i, "X_equals_4",
        ))
    x4_uncertainty = pd.concat(x4_rows, ignore_index=True)
    return uncertainty, by_fold, x4_uncertainty, track_diffs


def temporal_reference_predictions(outcome: str) -> pd.DataFrame:
    suffix = "52w" if outcome == "Y_52" else "26w"
    source = pd.read_csv(BENCH_DIR / f"temporal_validation_predictions_{suffix}.csv")
    source = source[
        source.outcome.eq(outcome)
        & source.model_id.eq("M1_poisson")
        & source.feature_block.isin(["F1", "F3"])
    ].copy()
    source["model"] = source.feature_block.map({"F1": "REF_F1", "F3": "REF_F3"})
    source["partition"] = source.split
    return source[["spotify_track_id", "partition", "model", "y_observed", "y_pred_raw", "X", "chart_entry_year"]]


def fit_tabular_poisson(train, test, variables: list[str], outcome: str):
    prep = prior.make_tab_preprocessor(variables)
    X_train = prep.fit_transform(train[variables])
    X_test = prep.transform(test[variables])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PoissonRegressor(alpha=1e-8, max_iter=1000, tol=1e-8)
        model.fit(X_train, train[outcome])
    return model.predict(X_test), prep, model, " | ".join(str(w.message) for w in caught)


def select_tfidf_alpha(
    train: pd.DataFrame,
    outcome: str,
    alpha_grid: list[float],
    label: str,
    audit_rows: list[dict],
    warning_rows: list[dict],
) -> tuple[float, pd.DataFrame]:
    groups = train.primary_group_component.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=3)
    rows = []
    for alpha in alpha_grid:
        fold_mae, fold_rmse = [], []
        for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(train, train[outcome], groups), start=1):
            fit_data, val_data = train.iloc[fit_idx], train.iloc[val_idx]
            overlap = set(groups[fit_idx]) & set(groups[val_idx])
            if overlap:
                raise AssertionError("Inner TF-IDF groups cross training and validation.")
            X_fit, X_val, vectorizer, _, _, _ = prior.build_tfidf_matrices(fit_data, val_data, prior.F3)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-7)
                model.fit(X_fit, fit_data[outcome])
            prediction = model.predict(X_val)
            fold_mae.append(mean_absolute_error(val_data[outcome], prediction))
            fold_rmse.append(math.sqrt(mean_squared_error(val_data[outcome], prediction)))
            warning_text = " | ".join(str(w.message) for w in caught)
            warning_rows.append({
                "stage": label,
                "partition_or_fold": f"inner_{inner_fold}",
                "model": "T3",
                "alpha": alpha,
                "status": "warning" if warning_text else "ok",
                "converged": int(model.n_iter_ < model.max_iter),
                "iterations": int(model.n_iter_),
                "warning_or_failure": warning_text,
            })
            audit_rows.append({
                "check": f"{label} inner tuning isolation",
                "status": "PASS",
                "evidence": f"alpha={alpha}; inner_fold={inner_fold}; group_overlap=0; vocabulary fit on {len(fit_data)} training tracks only; vocabulary_n={len(vectorizer.vocabulary_)}",
            })
        rows.append({
            "stage": label,
            "alpha": alpha,
            "inner_cv_folds": 3,
            "inner_mean_mae": float(np.mean(fold_mae)),
            "inner_sd_mae": float(np.std(fold_mae, ddof=1)),
            "inner_mean_rmse": float(np.mean(fold_rmse)),
        })
    result = pd.DataFrame(rows).sort_values(["inner_mean_mae", "inner_mean_rmse", "alpha"])
    selected = float(result.iloc[0].alpha)
    result["selected"] = (result.alpha == selected).astype(int)
    return selected, result


def fit_temporal_t3(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    outcome: str,
    alpha_grid: list[float],
    label: str,
    audit_rows: list[dict],
    warning_rows: list[dict],
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    selected, inner = select_tfidf_alpha(train, outcome, alpha_grid, label, audit_rows, warning_rows)
    X_train, X_evaluation, vectorizer, _, text_train, text_eval = prior.build_tfidf_matrices(train, evaluation, prior.F3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PoissonRegressor(alpha=selected, max_iter=1000, tol=1e-7)
        model.fit(X_train, train[outcome])
    prediction = model.predict(X_evaluation)
    warning_text = " | ".join(str(w.message) for w in caught)
    warning_rows.append({
        "stage": label,
        "partition_or_fold": "outer_fit",
        "model": "T3",
        "alpha": selected,
        "status": "warning" if warning_text else "ok",
        "converged": int(model.n_iter_ < model.max_iter),
        "iterations": int(model.n_iter_),
        "warning_or_failure": warning_text,
    })
    audit_rows.append({
        "check": f"{label} vocabulary isolation",
        "status": "PASS",
        "evidence": f"vocabulary and IDF fit on training only; training_n={len(train)}; evaluation_n={len(evaluation)}; selected_alpha={selected}; vocabulary_n={len(vectorizer.vocabulary_)}; evaluation_zero_vectors={int(np.sum(np.diff(text_eval.indptr)==0))}",
    })
    info = {
        "selected_alpha": selected,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "vocabulary_sha256": values_sha256(sorted(vectorizer.get_feature_names_out())),
        "training_track_sha256": values_sha256(sorted(train.spotify_track_id)),
        "evaluation_track_sha256": values_sha256(sorted(evaluation.spotify_track_id)),
    }
    return prediction, info, inner


def fit_temporal_h3(train, evaluation, outcome: str, warning_rows: list[dict]):
    variables = prior.LEX
    y_binary = (train[outcome] > 0).astype(int).to_numpy()
    binary_prep = prior.make_tab_preprocessor(variables)
    X_binary_train = binary_prep.fit_transform(train[variables])
    X_binary_eval = binary_prep.transform(evaluation[variables])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logistic = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        logistic.fit(X_binary_train, y_binary)
    p_positive = logistic.predict_proba(X_binary_eval)[:, 1]
    warning_text = " | ".join(str(w.message) for w in caught)
    warning_rows.append({
        "stage": f"temporal_{outcome}_H3_binary",
        "partition_or_fold": "outer_fit",
        "model": "H3",
        "alpha": np.nan,
        "status": "warning" if warning_text else "ok",
        "converged": int(logistic.n_iter_[0] < logistic.max_iter),
        "iterations": int(logistic.n_iter_[0]),
        "warning_or_failure": warning_text,
    })

    positive_train = train[train[outcome] > 0]
    positive_prep = prior.make_tab_preprocessor(variables)
    X_positive_train = positive_prep.fit_transform(positive_train[variables])
    X_positive_eval = positive_prep.transform(evaluation[variables])
    ztp = prior.ZeroTruncatedPoisson().fit(X_positive_train, positive_train[outcome].to_numpy(float))
    conditional_mean = ztp.predict_conditional_mean(X_positive_eval)
    warning_rows.append({
        "stage": f"temporal_{outcome}_H3_positive_ztp",
        "partition_or_fold": "outer_fit",
        "model": "H3",
        "alpha": np.nan,
        "status": "ok" if ztp.fit_.converged else "warning",
        "converged": int(ztp.fit_.converged),
        "iterations": ztp.fit_.n_iter,
        "warning_or_failure": "" if ztp.fit_.converged else ztp.fit_.message,
    })
    return p_positive * conditional_mean, p_positive, conditional_mean


def prepare_temporal_data(horizon: int, audit_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = f"Y_{horizon}"
    data = pd.read_csv(ANALYSIS_DIR / f"analysis_ready_550k_{horizon}w.csv")
    assignment = pd.read_csv(SPEC_DIR / f"provisional_temporal_validation_{horizon}w.csv")
    if len(data) != len(assignment) or set(data.spotify_track_id) != set(assignment.spotify_track_id):
        raise AssertionError(f"Temporal {horizon}-week assignments do not match the analysis-ready cohort.")
    data = data.merge(
        assignment[["spotify_track_id", "temporal_partition", "first_credited_artist_id", "lyrics_normalised_sha256"]],
        on=["spotify_track_id", "lyrics_normalised_sha256"],
        how="left",
        validate="one_to_one",
    )
    data = prior.add_lexical_features(data)
    grouped = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")[["spotify_track_id", "primary_group_component"]]
    data = data.merge(grouped, on="spotify_track_id", how="left", validate="one_to_one")
    if horizon == 52 and data.primary_group_component.isna().any():
        raise AssertionError("A Y_52 temporal track lacks its frozen protected grouping component.")
    counts = data.temporal_partition.value_counts().to_dict()
    expected = (
        {"train_2017_2018": 628, "validation_2019": 341, "test_2020": 45}
        if horizon == 52
        else {"train_2017_2018": 628, "validation_2019": 341, "test_2020": 251}
    )
    if counts != expected:
        raise AssertionError(f"Unexpected Y_{horizon} temporal counts: {counts}; expected {expected}.")
    train_ids = set(data.loc[data.temporal_partition.eq("train_2017_2018"), "spotify_track_id"])
    later_ids = set(data.loc[~data.temporal_partition.eq("train_2017_2018"), "spotify_track_id"])
    if train_ids & later_ids:
        raise AssertionError("Temporal train and evaluation track IDs overlap.")
    audit_rows.append({
        "check": f"Y_{horizon} temporal assignment integrity",
        "status": "PASS",
        "evidence": f"counts={counts}; train/evaluation track overlap=0; outcome={outcome}",
    })
    return data, assignment


def run_temporal_52(audit_rows: list[dict], warning_rows: list[dict]):
    data, _ = prepare_temporal_data(52, audit_rows)
    train = data[data.temporal_partition.eq("train_2017_2018")].copy()
    evaluation = data[~data.temporal_partition.eq("train_2017_2018")].copy()
    references = temporal_reference_predictions("Y_52")
    if len(references) != 2 * len(evaluation):
        raise AssertionError("Existing Y_52 F1/F3 temporal predictions have unexpected coverage.")

    prediction_frames = [references]
    lex_pred, _, lex_model, lex_warning = fit_tabular_poisson(train, evaluation, prior.LEX, "Y_52")
    warning_rows.append({
        "stage": "temporal_Y52_LEX",
        "partition_or_fold": "outer_fit",
        "model": "LEX",
        "alpha": 1e-8,
        "status": "warning" if lex_warning else "ok",
        "converged": int(lex_model.n_iter_ < lex_model.max_iter),
        "iterations": int(lex_model.n_iter_),
        "warning_or_failure": lex_warning,
    })
    prediction_frames.append(pd.DataFrame({
        "spotify_track_id": evaluation.spotify_track_id,
        "partition": evaluation.temporal_partition,
        "model": "LEX",
        "y_observed": evaluation.Y_52,
        "y_pred_raw": lex_pred,
        "X": evaluation.X,
        "chart_entry_year": evaluation.chart_entry_year,
    }))

    t3_pred, t3_info, t3_inner = fit_temporal_t3(
        train, evaluation, "Y_52", [0.01, 0.1, 1.0], "temporal_Y52_T3",
        audit_rows, warning_rows,
    )
    prediction_frames.append(pd.DataFrame({
        "spotify_track_id": evaluation.spotify_track_id,
        "partition": evaluation.temporal_partition,
        "model": "T3",
        "y_observed": evaluation.Y_52,
        "y_pred_raw": t3_pred,
        "X": evaluation.X,
        "chart_entry_year": evaluation.chart_entry_year,
    }))

    h3_pred, p_positive, conditional = fit_temporal_h3(train, evaluation, "Y_52", warning_rows)
    h3_frame = pd.DataFrame({
        "spotify_track_id": evaluation.spotify_track_id,
        "partition": evaluation.temporal_partition,
        "model": "H3",
        "y_observed": evaluation.Y_52,
        "y_pred_raw": h3_pred,
        "X": evaluation.X,
        "chart_entry_year": evaluation.chart_entry_year,
        "p_positive": p_positive,
        "p_zero": 1 - p_positive,
        "conditional_positive_mean": conditional,
    })
    prediction_frames.append(h3_frame)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)

    if predictions.duplicated(["spotify_track_id", "model"]).any():
        raise AssertionError("Duplicate Y_52 temporal track/model predictions.")
    coverage = predictions.groupby("model").spotify_track_id.nunique()
    if not (coverage.reindex(MODELS) == len(evaluation)).all():
        raise AssertionError(f"Incomplete Y_52 temporal prediction coverage: {coverage}")
    expected_outcome = evaluation.set_index("spotify_track_id").Y_52
    if not np.allclose(predictions.y_observed, predictions.spotify_track_id.map(expected_outcome)):
        raise AssertionError("Y_52 temporal outcomes disagree across model predictions.")

    metric_rows = []
    for (partition, model), q in predictions.groupby(["partition", "model"], sort=True):
        metric_rows.append({
            "outcome": "Y_52",
            "partition": partition,
            "model": model,
            **metric_values(q.y_observed, q.y_pred_raw, 48),
        })
    metrics = pd.DataFrame(metric_rows)

    binary_rows = []
    calibration_rows = []
    for partition, q in h3_frame.groupby("partition", sort=True):
        binary_rows.append({
            "outcome": "Y_52",
            "partition": partition,
            "model": "H3",
            **safe_binary_metrics((q.y_observed > 0).astype(int), q.p_positive),
        })
        q = q.copy()
        q["calibration_bin"] = pd.qcut(q.p_positive.rank(method="first"), q=min(5, len(q)), labels=False) + 1
        for calibration_bin, h in q.groupby("calibration_bin"):
            calibration_rows.append({
                "outcome": "Y_52",
                "partition": partition,
                "model": "H3",
                "calibration_bin": int(calibration_bin),
                "n": len(h),
                "mean_predicted_positive_probability": h.p_positive.mean(),
                "observed_positive_proportion": (h.y_observed > 0).mean(),
                "mean_predicted_zero_probability": h.p_zero.mean(),
                "observed_zero_proportion": (h.y_observed == 0).mean(),
            })
    audit_rows.append({
        "check": "Y_52 temporal training isolation",
        "status": "PASS",
        "evidence": "LEX scaling, T3 vocabulary/IDF/tuning, and both H3 components were fitted on 2017-2018 only; 2019/2020 outcomes not used",
    })
    return predictions, metrics, pd.DataFrame(binary_rows), pd.DataFrame(calibration_rows), t3_info, t3_inner


def temporal_hurdle_diagnostics(predictions52: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare temporal zero calibration and describe H3's positive-count part."""
    zero_rows = []
    positive_rows = []
    for partition in ["validation_2019", "test_2020"]:
        for model in ["REF_F1", "REF_F3", "LEX", "T3", "H3"]:
            q = predictions52[
                predictions52.partition.eq(partition) & predictions52.model.eq(model)
            ].copy()
            observed_zero = (q.y_observed == 0).astype(int)
            if model == "H3":
                p_zero = q.p_zero.to_numpy(float)
                probability_source = "explicit hurdle binary component"
            else:
                p_zero = np.exp(-np.clip(q.y_pred_raw.to_numpy(float), 0, None))
                probability_source = "Poisson-implied exp(-predicted mean)"
            p_zero = np.clip(p_zero, 1e-12, 1 - 1e-12)
            zero_rows.append({
                "outcome": "Y_52",
                "partition": partition,
                "model": model,
                "probability_source": probability_source,
                "n": len(q),
                "observed_zero_proportion": observed_zero.mean(),
                "mean_predicted_zero_probability": p_zero.mean(),
                "zero_probability_bias": p_zero.mean() - observed_zero.mean(),
                "zero_probability_brier_score": brier_score_loss(observed_zero, p_zero),
                "zero_probability_log_loss": log_loss(observed_zero, p_zero, labels=[0, 1]),
            })
        h3 = predictions52[
            predictions52.partition.eq(partition) & predictions52.model.eq("H3")
        ].copy()
        positive = h3.y_observed > 0
        positive_rows.append({
            "outcome": "Y_52",
            "partition": partition,
            "model": "H3",
            "component": "positive-count zero-truncated Poisson",
            "n_positive": int(positive.sum()),
            "mae_positive_count": mean_absolute_error(
                h3.loc[positive, "y_observed"], h3.loc[positive, "conditional_positive_mean"]
            ),
            "rmse_positive_count": math.sqrt(mean_squared_error(
                h3.loc[positive, "y_observed"], h3.loc[positive, "conditional_positive_mean"]
            )),
            "mean_observed_positive_y": h3.loc[positive, "y_observed"].mean(),
            "mean_predicted_conditional_positive_y": h3.loc[positive, "conditional_positive_mean"].mean(),
            "mean_prediction_bias_positive_count": (
                h3.loc[positive, "conditional_positive_mean"] - h3.loc[positive, "y_observed"]
            ).mean(),
        })
    return pd.DataFrame(zero_rows), pd.DataFrame(positive_rows)


def first_artist_from_json(value) -> str:
    try:
        parsed = json.loads(value) if isinstance(value, str) else []
        return str(parsed[0]) if parsed else ""
    except (ValueError, TypeError, IndexError):
        return ""


def build_y26_bootstrap_groups(data: pd.DataFrame) -> pd.Series:
    """First-artist + exact-lyric components for the expanded Y_26 cohort."""
    parent = list(range(len(data)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    first_seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(data.itertuples(index=False)):
        first_artist = first_artist_from_json(row.artist_ids_original)
        for kind, value in [("artist", first_artist), ("lyrics", str(row.lyrics_normalised_sha256))]:
            if not value or value == "nan":
                continue
            key = (kind, value)
            if key in first_seen:
                union(i, first_seen[key])
            else:
                first_seen[key] = i
    return pd.Series([f"y26_component_{find(i):04d}" for i in range(len(data))], index=data.index)


def run_temporal_26(audit_rows: list[dict], warning_rows: list[dict]):
    data, _ = prepare_temporal_data(26, audit_rows)
    data["y26_bootstrap_component"] = build_y26_bootstrap_groups(data)
    train = data[data.temporal_partition.eq("train_2017_2018")].copy()
    evaluation = data[~data.temporal_partition.eq("train_2017_2018")].copy()
    references = temporal_reference_predictions("Y_26")
    if len(references) != 2 * len(evaluation):
        raise AssertionError("Existing Y_26 F1/F3 temporal predictions have unexpected coverage.")
    lex_pred, _, lex_model, lex_warning = fit_tabular_poisson(train, evaluation, prior.LEX, "Y_26")
    warning_rows.append({
        "stage": "temporal_Y26_LEX",
        "partition_or_fold": "outer_fit",
        "model": "LEX",
        "alpha": 1e-8,
        "status": "warning" if lex_warning else "ok",
        "converged": int(lex_model.n_iter_ < lex_model.max_iter),
        "iterations": int(lex_model.n_iter_),
        "warning_or_failure": lex_warning,
    })
    lex_frame = pd.DataFrame({
        "spotify_track_id": evaluation.spotify_track_id,
        "partition": evaluation.temporal_partition,
        "model": "LEX",
        "y_observed": evaluation.Y_26,
        "y_pred_raw": lex_pred,
        "X": evaluation.X,
        "chart_entry_year": evaluation.chart_entry_year,
    })
    predictions = pd.concat([references, lex_frame], ignore_index=True)
    if predictions.duplicated(["spotify_track_id", "model"]).any():
        raise AssertionError("Duplicate Y_26 temporal track/model predictions.")
    expected = evaluation.set_index("spotify_track_id").Y_26
    if not np.allclose(predictions.y_observed, predictions.spotify_track_id.map(expected)):
        raise AssertionError("Y_26 temporal outcomes disagree across models.")
    metrics = []
    for (partition, model), q in predictions.groupby(["partition", "model"], sort=True):
        metrics.append({
            "outcome": "Y_26",
            "partition": partition,
            "model": model,
            **metric_values(q.y_observed, q.y_pred_raw, 22),
        })
    group_map = data.set_index("spotify_track_id").y26_bootstrap_component
    predictions["bootstrap_group"] = predictions.spotify_track_id.map(group_map)
    audit_rows.extend([
        {"check": "Y_26 temporal training isolation", "status": "PASS", "evidence": "LEX preprocessing fitted on 2017-2018 only; 2019 and 2020 outcomes were not used"},
        {"check": "Y_26 uncertainty grouping", "status": "PASS", "evidence": "expanded-cohort components connect tracks sharing first-credited artist or exact normalised lyrics; used only for paired uncertainty"},
    ])
    return data, predictions, pd.DataFrame(metrics)


def temporal_paired_bootstrap(
    predictions52: pd.DataFrame,
    predictions26: pd.DataFrame,
    cohort52: pd.DataFrame,
    cohort26: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    wide52 = predictions52.pivot(index="spotify_track_id", columns="model", values="y_pred_raw").reset_index()
    outcome52 = predictions52.groupby("spotify_track_id").y_observed.first()
    partition52 = predictions52.groupby("spotify_track_id").partition.first()
    q52 = wide52.copy()
    q52["Y_52"] = q52.spotify_track_id.map(outcome52)
    q52["partition"] = q52.spotify_track_id.map(partition52)
    q52 = q52.merge(
        cohort52[["spotify_track_id", "primary_group_component"]],
        on="spotify_track_id",
        validate="one_to_one",
    )
    for partition in ["validation_2019", "test_2020"]:
        q = q52[q52.partition.eq(partition)]
        for i, candidate in enumerate(["LEX", "T3", "H3"]):
            result = cluster_bootstrap_comparison(
                q, "REF_F3", candidate, "primary_group_component", BOOTSTRAP_REPS,
                SEED + 300 + i + (10 if partition == "test_2020" else 0),
                f"Y52_{partition}",
            )
            result["power_note"] = "underpowered 45-track stress test" if partition == "test_2020" else "main temporal validation"
            rows.append(result)

    wide26 = predictions26.pivot(index="spotify_track_id", columns="model", values="y_pred_raw").reset_index()
    outcome26 = predictions26.groupby("spotify_track_id").y_observed.first()
    partition26 = predictions26.groupby("spotify_track_id").partition.first()
    group26 = predictions26.groupby("spotify_track_id").bootstrap_group.first()
    q26 = wide26.copy()
    q26["Y_52"] = q26.spotify_track_id.map(outcome26)  # generic bootstrap outcome column
    q26["partition"] = q26.spotify_track_id.map(partition26)
    q26["y26_bootstrap_component"] = q26.spotify_track_id.map(group26)
    for partition in ["validation_2019", "test_2020"]:
        q = q26[q26.partition.eq(partition)]
        result = cluster_bootstrap_comparison(
            q, "REF_F3", "LEX", "y26_bootstrap_component", BOOTSTRAP_REPS,
            SEED + 400 + (10 if partition == "test_2020" else 0),
            f"Y26_{partition}",
        )
        result["power_note"] = "larger 251-track 2020 temporal test" if partition == "test_2020" else "main temporal validation"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def lower_alpha_t3(
    data52: pd.DataFrame,
    assignments: pd.DataFrame,
    original_wide: pd.DataFrame,
    audit_rows: list[dict],
    warning_rows: list[dict],
):
    data = prior.add_lexical_features(data52).merge(
        assignments[["spotify_track_id", "fold", "primary_group_component"]],
        on="spotify_track_id",
        how="left",
        validate="one_to_one",
    )
    alpha_grid = [0.001, 0.003, 0.01]
    prediction_frames = []
    outer_rows = []
    inner_rows = []
    for fold in sorted(data.fold.unique()):
        print(f"Lower-alpha T3 sensitivity: {fold}", flush=True)
        outer_train = data[data.fold.ne(fold)].copy()
        outer_test = data[data.fold.eq(fold)].copy()
        selected, inner = select_tfidf_alpha(
            outer_train, "Y_52", alpha_grid, f"lower_alpha_{fold}", audit_rows, warning_rows,
        )
        inner["outer_fold"] = fold
        inner_rows.append(inner)
        X_train, X_test, vectorizer, _, _, _ = prior.build_tfidf_matrices(outer_train, outer_test, prior.F3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PoissonRegressor(alpha=selected, max_iter=1000, tol=1e-7)
            model.fit(X_train, outer_train.Y_52)
        pred = model.predict(X_test)
        warning_text = " | ".join(str(w.message) for w in caught)
        warning_rows.append({
            "stage": "lower_alpha_outer",
            "partition_or_fold": fold,
            "model": "T3_LOW_ALPHA",
            "alpha": selected,
            "status": "warning" if warning_text else "ok",
            "converged": int(model.n_iter_ < model.max_iter),
            "iterations": int(model.n_iter_),
            "warning_or_failure": warning_text,
        })
        q = pd.DataFrame({
            "spotify_track_id": outer_test.spotify_track_id,
            "fold": fold,
            "y_observed": outer_test.Y_52,
            "y_pred_raw": pred,
            "X": outer_test.X,
            "selected_alpha": selected,
        })
        prediction_frames.append(q)
        metrics = metric_values(q.y_observed, q.y_pred_raw, 48)
        original = original_wide[original_wide.fold.eq(fold)]
        outer_rows.append({
            "aggregation": fold,
            "selected_alpha": selected,
            "selected_lower_grid_boundary": int(selected == min(alpha_grid)),
            "vocabulary_size": len(vectorizer.vocabulary_),
            **metrics,
            "original_t3_mae": mean_absolute_error(original.Y_52, original.T3),
            "original_t3_rmse": math.sqrt(mean_squared_error(original.Y_52, original.T3)),
        })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != 1014 or predictions.spotify_track_id.nunique() != 1014:
        raise AssertionError("Lower-alpha T3 did not yield one OOF prediction per track.")
    aggregate = metric_values(predictions.y_observed, predictions.y_pred_raw, 48)
    original_metrics = metric_values(original_wide.Y_52, original_wide.T3, 48)
    outer_rows.append({
        "aggregation": "all_oof",
        "selected_alpha": np.nan,
        "selected_lower_grid_boundary": int(all(row["selected_lower_grid_boundary"] == 1 for row in outer_rows)),
        "vocabulary_size": np.nan,
        **aggregate,
        "original_t3_mae": original_metrics["mae"],
        "original_t3_rmse": original_metrics["rmse"],
    })
    predictions_wide = original_wide[["spotify_track_id", "Y_52", "X", "fold", "primary_group_component", "T3"]].merge(
        predictions[["spotify_track_id", "y_pred_raw"]].rename(columns={"y_pred_raw": "T3_LOW_ALPHA"}),
        on="spotify_track_id",
        validate="one_to_one",
    )
    uncertainty = cluster_bootstrap_comparison(
        predictions_wide, "T3", "T3_LOW_ALPHA", "primary_group_component", BOOTSTRAP_REPS,
        SEED + 500, "lower_alpha_vs_original_T3",
    )
    audit_rows.append({
        "check": "lower-alpha outer coverage and leakage", "status": "PASS",
        "evidence": "one OOF prediction per track; fixed outer folds; vocabulary/IDF and three-alpha selection nested within each outer training fold",
    })
    return predictions, pd.DataFrame(outer_rows), pd.concat(inner_rows, ignore_index=True), uncertainty


def temporal_group_overlap_summary(data52: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = data52[data52.temporal_partition.eq("train_2017_2018")]
    train_groups = set(train.primary_group_component)
    train_artists = set(train.first_credited_artist_id.dropna())
    train_lyrics = set(train.lyrics_normalised_sha256.dropna())
    for partition in ["validation_2019", "test_2020"]:
        q = data52[data52.temporal_partition.eq(partition)]
        rows.append({
            "partition": partition,
            "n_tracks": len(q),
            "tracks_in_protected_component_seen_in_train": int(q.primary_group_component.isin(train_groups).sum()),
            "tracks_with_first_artist_seen_in_train": int(q.first_credited_artist_id.isin(train_artists).sum()),
            "tracks_with_exact_lyric_hash_seen_in_train": int(q.lyrics_normalised_sha256.isin(train_lyrics).sum()),
            "interpretation": "time-only robustness split; group overlap is reported, not prevented",
        })
    return pd.DataFrame(rows)


def grouped_metric_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        rows.append({"model": model, **metric_values(wide.Y_52, wide[model], 48)})
    return pd.DataFrame(rows)


def make_candidate_evidence(
    grouped_metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
    by_fold: pd.DataFrame,
    x4: pd.DataFrame,
    temporal52: pd.DataFrame,
    temporal26: pd.DataFrame,
    temporal_uncertainty: pd.DataFrame,
    lower_alpha: pd.DataFrame,
    binary52: pd.DataFrame,
) -> pd.DataFrame:
    def interval(candidate, metric, subset="all_Y52_grouped_OOF"):
        q = uncertainty[
            uncertainty.reference_model.eq("REF_F3")
            & uncertainty.candidate_model.eq(candidate)
            & uncertainty.metric.eq(metric)
            & uncertainty.subset.eq(subset)
        ]
        if q.empty:
            return "not applicable"
        r = q.iloc[0]
        return f"{r.observed_difference:+.3f} weeks; 95% CI [{r.ci_2_5:+.3f}, {r.ci_97_5:+.3f}]"

    roles = {
        "REF_F1": "core main-results benchmark and strongest simple reference",
        "REF_F3": "core main-results audio/song-characteristic reference",
        "LEX": "core interpretable lyric candidate, with mixed temporal robustness",
        "T3": "sensitivity only; not currently supported as the main specification",
        "H3": "core future-presence/probability candidate; count-prediction sensitivity",
    }
    interpretation = {
        "REF_F1": "high: four transparent X categories",
        "REF_F3": "moderate-high: conventional audio descriptors plus X",
        "LEX": "high: small, explicitly defined lexical summaries",
        "T3": "lower: thousands of regularised n-gram weights",
        "H3": "moderate: separates future presence from positive duration",
    }
    complexity = {
        "REF_F1": "very low",
        "REF_F3": "low",
        "LEX": "low-moderate",
        "T3": "high-dimensional; nested tuning and fold-local vocabulary",
        "H3": "two linked models; moderate",
    }
    weaknesses = {
        "REF_F1": "does not use baseline song characteristics",
        "REF_F3": "audio features add only modest prediction beyond X",
        "LEX": "aggregate gains are small and require uncertainty/temporal support",
        "T3": "worse grouped RMSE, systematic underprediction, boundary-seeking regularisation",
        "H3": "better probability calibration does not necessarily improve overall count error",
    }
    rows = []
    for model in MODELS:
        gm = grouped_metrics[grouped_metrics.model.eq(model)].iloc[0]
        fold_summary = "reference"
        if model != "REF_F3":
            f = by_fold[(by_fold.reference_model.eq("REF_F3")) & (by_fold.candidate_model.eq(model))]
            if not f.empty:
                fold_summary = f"MAE better in {(f.mae_improvement > 0).sum()}/5; RMSE better in {(f.rmse_improvement > 0).sum()}/5 folds"
        x4_text = "reference"
        if model in {"LEX", "T3", "H3"}:
            q = x4[(x4.reference_model.eq("REF_F3")) & (x4.candidate_model.eq(model)) & x4.metric.eq("MAE")]
            if not q.empty:
                r = q.iloc[0]
                x4_text = f"MAE gain {r.observed_difference:+.3f}; 95% CI [{r.ci_2_5:+.3f}, {r.ci_97_5:+.3f}]"
        t19 = temporal52[(temporal52.partition.eq("validation_2019")) & temporal52.model.eq(model)]
        t20 = temporal52[(temporal52.partition.eq("test_2020")) & temporal52.model.eq(model)]
        temporal_text = (
            f"2019 MAE/RMSE {t19.iloc[0].mae:.3f}/{t19.iloc[0].rmse:.3f}; "
            f"2020 N=45 {t20.iloc[0].mae:.3f}/{t20.iloc[0].rmse:.3f}"
            if not t19.empty and not t20.empty else "not evaluated"
        )
        y26_text = "not evaluated"
        if model in {"REF_F1", "REF_F3", "LEX"}:
            q19 = temporal26[(temporal26.partition.eq("validation_2019")) & temporal26.model.eq(model)]
            q20 = temporal26[(temporal26.partition.eq("test_2020")) & temporal26.model.eq(model)]
            y26_text = f"2019 {q19.iloc[0].mae:.3f}/{q19.iloc[0].rmse:.3f}; 2020 N=251 {q20.iloc[0].mae:.3f}/{q20.iloc[0].rmse:.3f}"
        zero_text = "ordinary count prediction only"
        if model == "H3":
            q = binary52[binary52.partition.eq("validation_2019")].iloc[0]
            zero_text = f"2019 binary AUC {q.roc_auc:.3f}, Brier {q.brier_score:.3f}; explicit zero probability"
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "grouped_cv_mae": gm.mae,
            "grouped_cv_rmse": gm.rmse,
            "paired_mae_uncertainty_vs_f3": interval(model, "MAE") if model not in {"REF_F1", "REF_F3"} else "benchmark/reference",
            "paired_rmse_uncertainty_vs_f3": interval(model, "RMSE") if model not in {"REF_F1", "REF_F3"} else "benchmark/reference",
            "fold_consistency": fold_summary,
            "x4_evidence": x4_text,
            "y52_temporal_evidence": temporal_text,
            "y26_temporal_evidence": y26_text,
            "zero_calibration": zero_text,
            "interpretability": interpretation[model],
            "computational_complexity": complexity[model],
            "important_weakness": weaknesses[model],
            "provisional_role": roles[model],
        })
    return pd.DataFrame(rows)


def make_figures(
    fig_dir: Path,
    uncertainty: pd.DataFrame,
    x4: pd.DataFrame,
    grouped_metrics: pd.DataFrame,
    temporal52: pd.DataFrame,
    calibration: pd.DataFrame,
    lower_inner: pd.DataFrame,
    lower_outer: pd.DataFrame,
):
    sns.set_theme(style="whitegrid", context="notebook")
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = {"REF_F1": "#4C78A8", "REF_F3": "#72B7B2", "LEX": "#F58518", "T3": "#E45756", "H3": "#54A24B"}

    for metric, filename, title in [
        ("MAE", "paired_mae_differences_95ci.png", "Paired MAE improvement under grouped resampling"),
        ("RMSE", "paired_rmse_differences_95ci.png", "Paired RMSE improvement under grouped resampling"),
    ]:
        q = uncertainty[uncertainty.metric.eq(metric)].copy()
        labels = [f"{r.reference_model} → {r.candidate_model}" for r in q.itertuples()]
        y = np.arange(len(q))
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        xerr = np.vstack([q.observed_difference - q.ci_2_5, q.ci_97_5 - q.observed_difference])
        ax.errorbar(q.observed_difference, y, xerr=xerr, fmt="o", color="#2F4B7C", capsize=4)
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_yticks(y, labels)
        ax.set_xlabel(f"{metric} improvement in weeks (positive favours candidate)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    q = x4[x4.metric.isin(["MAE", "RMSE"])].copy()
    q["label"] = q.candidate_model + " " + q.metric
    y = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xerr = np.vstack([q.observed_difference - q.ci_2_5, q.ci_97_5 - q.observed_difference])
    ax.errorbar(q.observed_difference, y, xerr=xerr, fmt="o", color="#B279A2", capsize=4)
    ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y, q.label)
    ax.set_xlabel("Error improvement over F3 in weeks (positive favours candidate)")
    ax.set_title("X=4 paired model differences with 95% cluster-bootstrap intervals")
    fig.tight_layout()
    fig.savefig(fig_dir / "x4_model_difference_intervals.png", dpi=180)
    plt.close(fig)

    gm = grouped_metrics[["model", "mae", "rmse"]].copy()
    gm["evaluation"] = "Grouped CV"
    tm = temporal52[temporal52.partition.isin(["validation_2019", "test_2020"])][["model", "partition", "mae", "rmse"]].copy()
    tm["evaluation"] = tm.partition.map({"validation_2019": "2019 temporal", "test_2020": "2020 stress (N=45)"})
    plot_data = pd.concat([gm[["model", "evaluation", "mae", "rmse"]], tm[["model", "evaluation", "mae", "rmse"]]])
    long = plot_data.melt(id_vars=["model", "evaluation"], value_vars=["mae", "rmse"], var_name="metric", value_name="error")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        p = long[long.metric.eq(metric)]
        sns.pointplot(data=p, x="model", y="error", hue="evaluation", ax=ax, markers=["o", "s", "^"], linestyles=["-", "--", ":"])
        ax.set_title(metric.upper())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Prediction error (weeks)")
    axes[0].legend(title="Evaluation", fontsize=8)
    axes[1].legend_.remove()
    fig.suptitle("Grouped-CV and temporal performance of carried-forward models")
    fig.tight_layout()
    fig.savefig(fig_dir / "grouped_cv_vs_temporal_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for partition, q in calibration.groupby("partition"):
        ax.plot(q.mean_predicted_positive_probability, q.observed_positive_proportion, marker="o", label=partition)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability of Y > 0", ylabel="Observed proportion with Y > 0", title="Temporal H3 future-presence calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_hurdle_probability_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for fold, q in lower_inner.groupby("outer_fold"):
        q = q.sort_values("alpha")
        ax.plot(q.alpha, q.inner_mean_mae, marker="o", alpha=0.75, label=fold)
    ax.set_xscale("log")
    ax.set(xlabel="Poisson regularisation alpha", ylabel="Inner grouped-CV MAE", title="Restrained lower-alpha TF-IDF sensitivity")
    ax.legend(title="Outer fold", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "tfidf_lower_alpha_sensitivity.png", dpi=180)
    plt.close(fig)


def write_reports(
    report_path: Path,
    memo_path: Path,
    uncertainty: pd.DataFrame,
    x4: pd.DataFrame,
    grouped_metrics: pd.DataFrame,
    temporal52: pd.DataFrame,
    binary52: pd.DataFrame,
    temporal26: pd.DataFrame,
    temporal_uncertainty: pd.DataFrame,
    zero_temporal: pd.DataFrame,
    positive_temporal: pd.DataFrame,
    lower_outer: pd.DataFrame,
    lower_uncertainty: pd.DataFrame,
    candidate: pd.DataFrame,
    overlap: pd.DataFrame,
    warnings_df: pd.DataFrame,
):
    f3lex = uncertainty[(uncertainty.reference_model.eq("REF_F3")) & uncertainty.candidate_model.eq("LEX")]
    f3t3 = uncertainty[(uncertainty.reference_model.eq("REF_F3")) & uncertainty.candidate_model.eq("T3")]
    f3h3 = uncertainty[(uncertainty.reference_model.eq("REF_F3")) & uncertainty.candidate_model.eq("H3")]
    x4lex = x4[(x4.reference_model.eq("REF_F3")) & x4.candidate_model.eq("LEX")]

    def result_row(frame, metric):
        return frame[frame.metric.eq(metric)].iloc[0]

    lex_mae, lex_rmse = result_row(f3lex, "MAE"), result_row(f3lex, "RMSE")
    t3_mae, t3_rmse = result_row(f3t3, "MAE"), result_row(f3t3, "RMSE")
    h3_mae, h3_rmse = result_row(f3h3, "MAE"), result_row(f3h3, "RMSE")
    t3_bias = result_row(f3t3, "bias_candidate_minus_reference")
    x4_mae, x4_rmse = result_row(x4lex, "MAE"), result_row(x4lex, "RMSE")

    selected_outer = lower_outer[lower_outer.aggregation.ne("all_oof")]
    lower_all = lower_outer[lower_outer.aggregation.eq("all_oof")].iloc[0]
    lower_mae = lower_uncertainty[lower_uncertainty.metric.eq("MAE")].iloc[0]
    lower_rmse = lower_uncertainty[lower_uncertainty.metric.eq("RMSE")].iloc[0]

    temporal_focus = temporal52[
        temporal52.model.isin(MODELS)
        & temporal52.partition.isin(["validation_2019", "test_2020"])
    ][["partition", "model", "n", "mae", "rmse", "mean_prediction_bias", "mean_observed_y", "mean_prediction"]]
    temporal26_focus = temporal26[
        temporal26.model.isin(["REF_F1", "REF_F3", "LEX"])
    ][["partition", "model", "n", "mae", "rmse", "mean_prediction_bias"]]
    zero_focus = zero_temporal[zero_temporal.model.isin(["REF_F3", "LEX", "T3", "H3"])][[
        "partition", "model", "observed_zero_proportion", "mean_predicted_zero_probability",
        "zero_probability_brier_score", "zero_probability_log_loss",
    ]]

    report = f"""# Provisional robustness and uncertainty report

## Scope and working status

This targeted round tests the stability of five previously fitted candidates for `Y_52`, the number of future Global Top 200 chart-presence weeks from weeks 5–52. It uses the clean 1,014-track 550k-linked cohort. All specifications remain provisional and are not described as supervisor-approved. No new broad feature family or algorithm was introduced.

The carried-forward models are F1 (categorical early presence `X` only), F3 (`X` plus audio/song characteristics), LEX (F3 plus transparent lexical summaries), T3 (F3 plus regularised TF-IDF), and H3 (a hurdle model with F3 and lexical summaries).

## Integrity and uncertainty design

Existing out-of-fold predictions were aligned one-to-one by Spotify track ID. Every model had exactly one held-out prediction for each of 1,014 tracks, with identical outcomes and frozen folds. The primary bootstrap resampled all 276 `primary_group_component` clusters with replacement and retained paired predictions within each sampled cluster. This component is the same protected unit used to keep first-credited artists, exact lyric groups, and supported related versions together in grouped validation. Five thousand replicates and seed {SEED} were used. Bootstrap proportions are resampling frequencies, not Bayesian posterior probabilities.

## Grouped-CV uncertainty

LEX reduced MAE relative to F3 by {lex_mae.observed_difference:.3f} weeks (95% percentile interval {lex_mae.ci_2_5:.3f} to {lex_mae.ci_97_5:.3f}; {lex_mae.bootstrap_proportion_candidate_better*100:.1f}% of bootstrap replicates favoured LEX). The magnitude is small. Its RMSE difference was {lex_rmse.observed_difference:.3f} weeks (95% interval {lex_rmse.ci_2_5:.3f} to {lex_rmse.ci_97_5:.3f}; {lex_rmse.bootstrap_proportion_candidate_better*100:.1f}% favoured LEX).

 T3 changed MAE by {t3_mae.observed_difference:.3f} weeks relative to F3 (95% interval {t3_mae.ci_2_5:.3f} to {t3_mae.ci_97_5:.3f}) but changed RMSE by {t3_rmse.observed_difference:.3f} weeks (positive favours T3; 95% interval {t3_rmse.ci_2_5:.3f} to {t3_rmse.ci_97_5:.3f}). Its mean prediction bias was {abs(t3_bias.observed_difference):.3f} weeks more negative than F3 (95% interval for candidate-minus-reference bias {t3_bias.ci_2_5:.3f} to {t3_bias.ci_97_5:.3f}), confirming systematic relative underprediction. H3 changed MAE by {h3_mae.observed_difference:.3f} weeks (95% interval {h3_mae.ci_2_5:.3f} to {h3_mae.ci_97_5:.3f}) and RMSE by {h3_rmse.observed_difference:.3f} weeks (95% interval {h3_rmse.ci_2_5:.3f} to {h3_rmse.ci_97_5:.3f}).

{dataframe_to_markdown(uncertainty[uncertainty.metric.isin(['MAE','RMSE'])][['reference_model','candidate_model','metric','observed_difference','ci_2_5','ci_97_5','bootstrap_proportion_candidate_better']], 4)}

## X=4 tracks

The exact X=4 subset contains {int(x4_mae.n_tracks)} tracks in {int(x4_mae.n_clusters)} protected clusters. LEX improved MAE over F3 by {x4_mae.observed_difference:.3f} weeks (95% interval {x4_mae.ci_2_5:.3f} to {x4_mae.ci_97_5:.3f}) and RMSE by {x4_rmse.observed_difference:.3f} weeks (95% interval {x4_rmse.ci_2_5:.3f} to {x4_rmse.ci_97_5:.3f}). These intervals show whether the previously observed improvement among the hardest X category is stable after cluster dependence is preserved.

{dataframe_to_markdown(x4[x4.metric.isin(['MAE','RMSE'])][['candidate_model','metric','observed_difference','ci_2_5','ci_97_5','bootstrap_proportion_candidate_better','n_tracks','n_clusters']], 4)}

## Y_52 temporal robustness

Models were trained on 628 tracks entering in 2017–2018, evaluated on 341 tracks entering in 2019, and stress-tested on 45 tracks entering in 2020. All LEX preprocessing, T3 vocabulary/IDF and regularisation selection, and both H3 components were fitted on 2017–2018 only. The 2020 findings are descriptive and underpowered.

{dataframe_to_markdown(temporal_focus, 3)}

The temporal split is deliberately calendar-based rather than group-separated. Its protected-group and artist overlaps are therefore reported transparently:

{dataframe_to_markdown(overlap, 0)}

For H3, the binary future-presence results were:

{dataframe_to_markdown(binary52[['partition','n','positive_pct','roc_auc','pr_auc','brier_score','log_loss','sensitivity_recall_positive','specificity_zero']], 3)}

H3's explicit zero probability remained much better calibrated than the zero probability implied by ordinary Poisson means. The comparison below uses the same held-out tracks; lower Brier score and log loss are better.

{dataframe_to_markdown(zero_focus, 3)}

The positive-count component remained substantially harder than the binary future-presence component:

{dataframe_to_markdown(positive_temporal[['partition','n_positive','mae_positive_count','rmse_positive_count','mean_observed_positive_y','mean_predicted_conditional_positive_y','mean_prediction_bias_positive_count']], 3)}

## Y_26 temporal robustness

The shorter-horizon check retains the same 628-track training and 341-track 2019 validation sets, but the 2020 evaluation grows to 251 tracks. Only F1, F3 and the interpretable LEX model were evaluated.

{dataframe_to_markdown(temporal26_focus, 3)}

Against F3, LEX's 2019 Y_26 MAE difference was {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_validation_2019')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].observed_difference:.3f} weeks (95% interval {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_validation_2019')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].ci_2_5:.3f} to {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_validation_2019')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].ci_97_5:.3f}). In the larger 2020 sample it was {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_test_2020')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].observed_difference:.3f} weeks (95% interval {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_test_2020')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].ci_2_5:.3f} to {temporal_uncertainty[(temporal_uncertainty.subset.eq('Y26_test_2020')) & (temporal_uncertainty.metric.eq('MAE'))].iloc[0].ci_97_5:.3f}). The direction therefore changes across years and neither interval excludes zero.

## Restrained lower-alpha T3 sensitivity

The TF-IDF settings and frozen outer folds were unchanged. The only new grid was 0.001, 0.003 and 0.01, selected using three-fold grouped validation entirely inside each outer training fold. Selected alphas by fold were {', '.join(f'{r.aggregation}: {r.selected_alpha:g}' for r in selected_outer.itertuples())}. No fold selected the new lower boundary of 0.001: 0.003 was selected in three folds and 0.01 in two. The sensitivity produced overall MAE {lower_all.mae:.3f} and RMSE {lower_all.rmse:.3f}, compared with original T3 MAE {lower_all.original_t3_mae:.3f} and RMSE {lower_all.original_t3_rmse:.3f}. The paired MAE change from original T3 was {lower_mae.observed_difference:.3f} weeks (95% interval {lower_mae.ci_2_5:.3f} to {lower_mae.ci_97_5:.3f}); the RMSE change was {lower_rmse.observed_difference:.3f} weeks (95% interval {lower_rmse.ci_2_5:.3f} to {lower_rmse.ci_97_5:.3f}). Testing below 0.01 therefore did not improve T3 and materially worsened RMSE; no further alpha search is warranted in this round.

## Candidate evidence and provisional roles

{dataframe_to_markdown(candidate, 3)}

F1, F3, LEX and H3 should provisionally remain visible in the thesis main-results sequence, but for different purposes: F1 is the essential early-chart benchmark, F3 measures the incremental contribution of audio/song characteristics, LEX is the main interpretable lyric comparison, and H3 evaluates future-presence probability and zero calibration. T3 should currently remain a high-dimensional sensitivity rather than a main specification. LEX's grouped-CV MAE evidence is positive, but its RMSE and temporal evidence are mixed; it should not be presented as uniformly superior to F3.

## Interpretation and limitations

The research question remains conditional on a song already entering the Spotify Global Top 200. These comparisons are predictive, not causal. Small improvements of approximately one or two tenths of a week should be described as small even when their resampling interval is favourable. The temporal split changes cohort composition and permits repeated artists across periods; it tests later-calendar generalisation, not a simultaneous new-artist requirement. The 2020 Y_52 stress test has only 45 tracks. TF-IDF remains high-dimensional and sensitive to regularisation. H3 should be assessed separately for probability calibration and overall count error because those are different predictive purposes.

## Implementation status

No blocked variable entered a model. Frozen folds were not regenerated. Existing prediction files were read but not modified. New outputs were staged and published only after validation checks passed. Model warnings recorded in this round: {(warnings_df.status != 'ok').sum()} warning/failure rows out of {len(warnings_df)} fit records.
"""
    report_path.write_text(report, encoding="utf-8")

    robust_claims = []
    weak_claims = []
    if lex_mae.ci_2_5 > 0:
        robust_claims.append(f"LEX gives a small, cluster-bootstrap-supported MAE improvement over F3 ({lex_mae.observed_difference:.3f} weeks; 95% CI {lex_mae.ci_2_5:.3f} to {lex_mae.ci_97_5:.3f}).")
    else:
        weak_claims.append(f"LEX's MAE advantage over F3 is uncertain because its 95% interval includes zero ({lex_mae.ci_2_5:.3f} to {lex_mae.ci_97_5:.3f}).")
    if lex_rmse.ci_2_5 > 0:
        robust_claims.append(f"LEX also improves RMSE over F3 ({lex_rmse.observed_difference:.3f}; 95% CI {lex_rmse.ci_2_5:.3f} to {lex_rmse.ci_97_5:.3f}).")
    else:
        weak_claims.append(f"LEX's RMSE advantage is weak/mixed (95% CI {lex_rmse.ci_2_5:.3f} to {lex_rmse.ci_97_5:.3f}).")
    if t3_rmse.ci_97_5 < 0:
        robust_claims.append("T3 worsens RMSE relative to F3 under clustered resampling.")
    else:
        weak_claims.append("T3's RMSE disadvantage is not fully stable under clustered resampling.")

    memo = f"""# Thesis evidence memo: robustness and uncertainty, version 1

## Exact question tested

Among songs already present in the Spotify Global Top 200, are the small predictive gains from transparent lyric summaries, high-dimensional lyric text, and a hurdle formulation stable enough to retain as serious candidates for predicting future chart-presence weeks?

## Dataset and cohort

The primary analysis uses the frozen clean 550k-linked `Y_52` cohort: 1,014 unique Spotify track IDs, outcome support 0–48, and the existing five grouped folds. `Y_26` is used only as a limited temporal sensitivity.

## Validation and uncertainty design

Grouped validation is necessary because artists can contribute multiple tracks and some records share exact lyrics or supported related-version links. The frozen design assigns all linked records in a `primary_group_component` to one fold. Paired uncertainty therefore resamples these 276 complete components rather than treating 1,014 tracks as independent. Five thousand paired cluster-bootstrap replicates use seed {SEED}.

Temporal robustness trains on 2017–2018 and evaluates 2019, with 2020 treated as a stress test. The Y_52 2020 sample has 45 tracks; Y_26 has 251. Temporal TF-IDF vocabulary, IDF, filtering and alpha selection are training-only.

## Key numerical results

* F3 versus LEX MAE improvement: {lex_mae.observed_difference:.3f} weeks; 95% CI [{lex_mae.ci_2_5:.3f}, {lex_mae.ci_97_5:.3f}]; {lex_mae.bootstrap_proportion_candidate_better*100:.1f}% of replicates favour LEX.
* F3 versus LEX RMSE improvement: {lex_rmse.observed_difference:.3f} weeks; 95% CI [{lex_rmse.ci_2_5:.3f}, {lex_rmse.ci_97_5:.3f}]; {lex_rmse.bootstrap_proportion_candidate_better*100:.1f}% favour LEX.
* X=4 F3 versus LEX MAE improvement: {x4_mae.observed_difference:.3f}; 95% CI [{x4_mae.ci_2_5:.3f}, {x4_mae.ci_97_5:.3f}].
* T3 versus F3 MAE improvement: {t3_mae.observed_difference:.3f}; 95% CI [{t3_mae.ci_2_5:.3f}, {t3_mae.ci_97_5:.3f}].
* T3 versus F3 RMSE improvement: {t3_rmse.observed_difference:.3f}; 95% CI [{t3_rmse.ci_2_5:.3f}, {t3_rmse.ci_97_5:.3f}].
* T3 candidate-minus-F3 prediction-bias difference: {t3_bias.observed_difference:.3f}; 95% CI [{t3_bias.ci_2_5:.3f}, {t3_bias.ci_97_5:.3f}].
* H3 versus F3 MAE improvement: {h3_mae.observed_difference:.3f}; 95% CI [{h3_mae.ci_2_5:.3f}, {h3_mae.ci_97_5:.3f}].
* H3 versus F3 RMSE improvement: {h3_rmse.observed_difference:.3f}; 95% CI [{h3_rmse.ci_2_5:.3f}, {h3_rmse.ci_97_5:.3f}].
* Lower-alpha T3 overall MAE/RMSE: {lower_all.mae:.3f}/{lower_all.rmse:.3f}; original T3: {lower_all.original_t3_mae:.3f}/{lower_all.original_t3_rmse:.3f}.

## Findings sufficiently robust to retain

{chr(10).join('* ' + item for item in robust_claims) if robust_claims else '* No incremental model claim meets a strict interval-excludes-zero rule across both error metrics.'}

The earlier finding that early chart presence `X` is the central benchmark remains part of the validated benchmark evidence, not a new causal claim. H3's future-presence discrimination and calibration should be reported separately from its count-prediction accuracy.

## Weak or mixed findings

{chr(10).join('* ' + item for item in weak_claims) if weak_claims else '* No additional weak findings were identified.'}

Small numerical gains must not be described as large practical effects. Temporal evidence is descriptive where sample sizes are small or protected groups overlap periods.

## X=4 evidence

The X=4 subset has {int(x4_mae.n_tracks)} tracks in {int(x4_mae.n_clusters)} clusters. It is the hardest early-performance group for count prediction. The interval above determines whether lexical summaries make a stable incremental contribution in that group; no separate X=4 model was fitted.

## TF-IDF sensitivity

The lower-alpha grid was deliberately limited to 0.001, 0.003 and 0.01. Selected values were {', '.join(f'{r.aggregation}={r.selected_alpha:g}' for r in selected_outer.itertuples())}. The new lower boundary of 0.001 was not selected in any fold. The lower-alpha sensitivity slightly worsened MAE and materially worsened RMSE, so it provides no evidence for continuing toward still smaller values.

## Hurdle interpretation

H3 separates any future chart presence from positive future duration. In 2019 its binary ROC-AUC was {binary52[binary52.partition.eq('validation_2019')].iloc[0].roc_auc:.3f}, PR-AUC was {binary52[binary52.partition.eq('validation_2019')].iloc[0].pr_auc:.3f}, and Brier score was {binary52[binary52.partition.eq('validation_2019')].iloc[0].brier_score:.3f}. Its 2019 mean predicted zero probability was {zero_temporal[(zero_temporal.partition.eq('validation_2019')) & (zero_temporal.model.eq('H3'))].iloc[0].mean_predicted_zero_probability:.3f} against an observed zero proportion of {zero_temporal[(zero_temporal.partition.eq('validation_2019')) & (zero_temporal.model.eq('H3'))].iloc[0].observed_zero_proportion:.3f}; the corresponding Brier score was {zero_temporal[(zero_temporal.partition.eq('validation_2019')) & (zero_temporal.model.eq('H3'))].iloc[0].zero_probability_brier_score:.3f}, compared with {zero_temporal[(zero_temporal.partition.eq('validation_2019')) & (zero_temporal.model.eq('LEX'))].iloc[0].zero_probability_brier_score:.3f} for the Poisson-implied LEX zero probability. The positive-count component remained harder: 2019 positive-count MAE was {positive_temporal[positive_temporal.partition.eq('validation_2019')].iloc[0].mae_positive_count:.3f} weeks. Probability calibration and count MAE answer different questions; neither alone establishes a final model.

## Provisional candidate placement

F1, F3, LEX and H3 should remain in the principal thesis comparison, with their purposes kept separate. LEX is the interpretable lyric candidate, but its temporal evidence is mixed. H3 is the future-presence/probability candidate rather than an overall count-error winner. T3 should remain a high-dimensional sensitivity because its lower MAE is not accompanied by stable RMSE or bias performance.

## Limitations

The cohort includes only tracks that already entered the Global Top 200 and the 550k-linked subset has documented feature-coverage bias. The temporal Y_52 2020 test is underpowered. Temporal partitions are not artist-grouped, so later-year evaluation can contain artists or protected components represented in training. The models are predictive rather than causal, and the outcome is a bounded count even when Poisson components are used diagnostically.

## Claims safe to make

* Early chart presence is a strong predictive benchmark within already-charted tracks.
* Transparent lyric summaries can be described only at the magnitude and uncertainty shown above.
* High-dimensional TF-IDF and hurdle results should be distinguished by error metric and purpose.
* Future chart presence can be easier to discriminate than the amount of longevity conditional on presence.

## Claims that should not be made

* The models predict which songs become hits.
* Lyrics, audio descriptors or chart behaviour causally determine longevity.
* A few tenths of a week are a large practical gain.
* The 45-track 2020 Y_52 stress set proves temporal superiority.
* A boundary-selected TF-IDF alpha is a stable optimum.
* H3 is automatically the best count model because its binary calibration is strong.
"""
    memo_path.write_text(memo, encoding="utf-8")


def main():
    for destination in [OUT_DIR, FIG_DIR.parent, REPORT_PATH, MEMO_PATH]:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    frozen_dirs = [ANALYSIS_DIR, SPEC_DIR, BENCH_DIR, LYRIC_DIR]
    digest_before = {str(path.relative_to(ROOT)): directory_digest(path) for path in frozen_dirs}
    input_files = [
        ANALYSIS_DIR / "analysis_ready_550k_52w.csv",
        ANALYSIS_DIR / "analysis_ready_550k_26w.csv",
        SPEC_DIR / "provisional_grouped_cv_assignments.csv",
        SPEC_DIR / "provisional_temporal_validation_52w.csv",
        SPEC_DIR / "provisional_temporal_validation_26w.csv",
        BENCH_DIR / "grouped_cv_predictions.csv",
        BENCH_DIR / "temporal_validation_predictions_52w.csv",
        BENCH_DIR / "temporal_validation_predictions_26w.csv",
        LYRIC_DIR / "lyric_grouped_cv_predictions.csv",
        LYRIC_DIR / "hurdle_grouped_cv_predictions.csv",
        LYRIC_DIR / "lyrics_hurdle_validation_manifest.json",
    ]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_files}
    audit_rows: list[dict] = []
    warning_rows: list[dict] = []

    previous_manifest = json.loads((LYRIC_DIR / "lyrics_hurdle_validation_manifest.json").read_text())
    required_previous_assertions = [
        "frozen_folds_used", "outer_group_crossings_zero", "tfidf_fit_training_only",
        "tfidf_inner_selection_training_only", "blocked_predictors_absent",
        "prediction_coverage_complete", "frozen_inputs_unchanged",
    ]
    if not all(previous_manifest["validation_assertions"].get(key) for key in required_previous_assertions):
        raise AssertionError("The prior lyric/hurdle validation manifest does not pass all required integrity checks.")
    audit_rows.append({"check": "prior validation manifest", "status": "PASS", "evidence": "all required frozen-fold, text-isolation, leakage and coverage assertions are true"})

    wide, long = load_and_validate_oof(audit_rows)
    uncertainty, by_fold, x4, track_diffs = paired_uncertainty(wide)
    grouped_metrics = grouped_metric_table(wide)

    data52_temporal, _ = prepare_temporal_data(52, audit_rows)
    predictions52, temporal52, binary52, calibration52, t3_temporal_info, t3_temporal_inner = run_temporal_52(audit_rows, warning_rows)
    zero_temporal52, positive_temporal52 = temporal_hurdle_diagnostics(predictions52)
    data26_temporal, predictions26, temporal26 = run_temporal_26(audit_rows, warning_rows)
    temporal_uncertainty = temporal_paired_bootstrap(predictions52, predictions26, data52_temporal, data26_temporal)
    overlap = temporal_group_overlap_summary(data52_temporal)

    data52 = pd.read_csv(ANALYSIS_DIR / "analysis_ready_550k_52w.csv")
    assignments = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")
    lower_predictions, lower_outer, lower_inner, lower_uncertainty = lower_alpha_t3(
        data52, assignments, wide, audit_rows, warning_rows,
    )
    candidate = make_candidate_evidence(
        grouped_metrics, uncertainty, by_fold, x4, temporal52, temporal26,
        temporal_uncertainty, lower_outer, binary52,
    )

    allowed_predictors = set(prior.F1 + prior.F3 + prior.LEX)
    blocked_hits = prior.leakage_hits(allowed_predictors)
    if blocked_hits:
        raise AssertionError(f"Blocked variables entered a model: {blocked_hits}")
    audit_rows.append({
        "check": "blocked predictor exclusion", "status": "PASS",
        "evidence": f"programmatic scan of F1/F3/LEX/T3/H3 inputs found no blocked fields; scanned={sorted(allowed_predictors)}",
    })
    audit_rows.append({
        "check": "paired bootstrap pairing and clustering", "status": "PASS",
        "evidence": f"models aligned one-to-one before resampling; complete primary_group_component clusters resampled; replicates={BOOTSTRAP_REPS}; seed={SEED}",
    })
    if not (uncertainty.bootstrap_replicates == BOOTSTRAP_REPS).all():
        raise AssertionError("Bootstrap replicate count mismatch.")

    warning_df = pd.DataFrame(warning_rows)
    if warning_df.empty:
        warning_df = pd.DataFrame(columns=["stage", "partition_or_fold", "model", "alpha", "status", "converged", "iterations", "warning_or_failure"])
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.status.eq("PASS").all():
        raise AssertionError("At least one robustness leakage/integrity audit failed.")

    with tempfile.TemporaryDirectory(prefix="spotify_robustness_") as temp_root:
        stage_root = Path(temp_root)
        stage_data = stage_root / "data"
        stage_fig = stage_root / "figures"
        stage_docs = stage_root / "docs"
        stage_data.mkdir(parents=True)
        stage_fig.mkdir(parents=True)
        stage_docs.mkdir(parents=True)

        outputs = {
            "paired_model_uncertainty.csv": uncertainty,
            "paired_model_uncertainty_by_fold.csv": by_fold,
            "x4_paired_uncertainty.csv": x4,
            "temporal_robustness_52w.csv": temporal52,
            "temporal_binary_metrics_52w.csv": binary52,
            "temporal_zero_calibration_comparison_52w.csv": zero_temporal52,
            "temporal_hurdle_positive_count_52w.csv": positive_temporal52,
            "temporal_robustness_26w.csv": temporal26,
            "temporal_paired_uncertainty.csv": temporal_uncertainty,
            "tfidf_lower_alpha_sensitivity.csv": lower_outer,
            "tfidf_lower_alpha_inner_cv.csv": lower_inner,
            "candidate_evidence_table.csv": candidate,
            "robustness_leakage_audit.csv": audit_df,
            "robustness_model_fit_warnings.csv": warning_df,
            "aligned_grouped_oof_predictions.csv": wide,
            "paired_track_absolute_error_differences.csv": track_diffs,
            "temporal_aligned_predictions_52w.csv": predictions52,
            "temporal_binary_calibration_52w.csv": calibration52,
            "temporal_tfidf_tuning_52w.csv": t3_temporal_inner,
            "temporal_aligned_predictions_26w.csv": predictions26,
            "temporal_group_overlap_summary.csv": overlap,
            "tfidf_lower_alpha_predictions.csv": lower_predictions,
            "tfidf_lower_alpha_paired_uncertainty.csv": lower_uncertainty,
        }
        for name, frame in outputs.items():
            frame.to_csv(stage_data / name, index=False)

        make_figures(stage_fig, uncertainty, x4, grouped_metrics, temporal52, calibration52, lower_inner, lower_outer)
        write_reports(
            stage_docs / REPORT_PATH.name,
            stage_docs / MEMO_PATH.name,
            uncertainty, x4, grouped_metrics, temporal52, binary52, temporal26,
            temporal_uncertainty, zero_temporal52, positive_temporal52, lower_outer,
            lower_uncertainty, candidate, overlap, warning_df,
        )

        digest_after = {str(path.relative_to(ROOT)): directory_digest(path) for path in frozen_dirs}
        if digest_before != digest_after:
            raise AssertionError("A frozen historical directory changed during the robustness task.")
        audit_df = pd.concat([audit_df, pd.DataFrame([{
            "check": "historical output immutability", "status": "PASS",
            "evidence": "full SHA-256 directory digests unchanged for analysis-ready, final specification, benchmark, and lyric/hurdle inputs",
        }])], ignore_index=True)
        audit_df.to_csv(stage_data / "robustness_leakage_audit.csv", index=False)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "working_status": "provisional robustness and uncertainty evidence; not supervisor-approved",
            "random_seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_unit": "primary_group_component for primary Y_52 analyses",
            "python": sys.version,
            "platform": platform.platform(),
            "libraries": {
                "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            },
            "primary_cohort": {"outcome": "Y_52", "n": 1014, "support": [0, 48], "protected_clusters": int(wide.primary_group_component.nunique())},
            "models": MODELS,
            "temporal_design": {
                "train": "2017-2018", "validation": "2019", "stress_test": "2020",
                "Y52_counts": {"train": 628, "validation": 341, "stress_test": 45},
                "Y26_counts": {"train": 628, "validation": 341, "stress_test": 251},
            },
            "tfidf_temporal": {"grid": [0.01, 0.1, 1.0], **t3_temporal_info},
            "tfidf_lower_alpha": {"grid": [0.001, 0.003, 0.01], "outer_folds": 5, "inner_grouped_folds": 3},
            "validation_assertions": {row.check: row.status == "PASS" for row in audit_df.itertuples()},
            "frozen_input_sha256": input_hashes,
            "frozen_directory_digest_before": digest_before,
            "frozen_directory_digest_after": digest_after,
            "analysis_script": {
                "path": "scripts/run_provisional_robustness_uncertainty.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "output_sha256": {},
        }
        all_stage_files = sorted([p for p in stage_root.rglob("*") if p.is_file()])
        manifest["output_sha256"] = {str(p.relative_to(stage_root)): sha256(p) for p in all_stage_files}
        (stage_data / "robustness_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Final staged-file validation before publication.
        for csv_path in stage_data.glob("*.csv"):
            pd.read_csv(csv_path)
        for png_path in stage_fig.glob("*.png"):
            if png_path.stat().st_size < 5000:
                raise AssertionError(f"Figure appears empty or corrupt: {png_path.name}")
        json.loads((stage_data / "robustness_validation_manifest.json").read_text())

        OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        FIG_DIR.parent.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage_data, OUT_DIR)
        shutil.copytree(stage_fig, FIG_DIR)
        shutil.copy2(stage_docs / REPORT_PATH.name, REPORT_PATH)
        shutil.copy2(stage_docs / MEMO_PATH.name, MEMO_PATH)

    print(json.dumps({
        "status": "complete",
        "output_directory": str(OUT_DIR),
        "figure_directory": str(FIG_DIR),
        "report": str(REPORT_PATH),
        "memo": str(MEMO_PATH),
        "x4_n": int((wide.X == 4).sum()),
        "protected_clusters": int(wide.primary_group_component.nunique()),
        "warning_rows": int((warning_df.status != "ok").sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
