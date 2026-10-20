#!/usr/bin/env python3
"""Implement the supervisor-confirmed, entry-year-controlled thesis update.

The script reuses the frozen Y52 cohort, protected folds, feature definitions,
and estimator implementations. It adds only the requested entry-year control,
L/Z descriptives, and the exact survival-to-fixed-window cohort bridge. New
outputs are staged before publication; historical artefacts are checksum-guarded.
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

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv_supervisor_update"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/processed/pre_model_analysis_ready_v1"
SPEC_DIR = ROOT / "data/processed/final_pre_model_specification_v1"
SURVIVAL_DIR = ROOT / "data/processed/chart_survival"
FIXED_DIR = ROOT / "data/processed/fixed_window_count_v1"
HIST_FINAL_DIR = ROOT / "data/processed/provisional_final_model_comparison_v1"
OUT_DIR = ROOT / "data/processed/supervisor_confirmed_update_v1"
PRESENTATION_DIR = ROOT / "outputs/supervisor_confirmed_update_v1"
SPEC_DOC = ROOT / "docs/supervisor_confirmed_final_model_specification_v1.md"
MEMO_DOC = ROOT / "docs/supervisor_feedback_implementation_2026-09-05_v1.md"

SEED = 20260825
VERSION_SENSITIVITY_SEED = 20262025
BOOTSTRAP_REPS = 5000
NMAX = 48

sys.path.insert(0, str(ROOT / "scripts"))
import run_provisional_lyrics_hurdle_modelling as lyric_prior  # noqa: E402
import run_provisional_robustness_uncertainty as robust_prior  # noqa: E402


AUDIO = list(lyric_prior.AUDIO)
LEXICAL = list(lyric_prior.LEX_MODEL)
MODEL_VARIABLES = {
    "F1_YEAR": ["X", "chart_entry_year"],
    "F3_YEAR": ["X", *AUDIO, "chart_entry_year"],
    "LEX_YEAR": ["X", *AUDIO, *LEXICAL, "chart_entry_year"],
    "H3_YEAR": ["X", *AUDIO, *LEXICAL, "chart_entry_year"],
}

HISTORICAL_DIRS = [
    ROOT / "data/processed/pre_model_analysis_ready_v1",
    ROOT / "data/processed/pre_model_remaining_eda_v2",
    ROOT / "data/processed/final_pre_model_specification_v1",
    ROOT / "data/processed/provisional_benchmark_modelling_v1",
    ROOT / "data/processed/provisional_lyrics_hurdle_modelling_v1",
    ROOT / "data/processed/provisional_robustness_uncertainty_v1",
    ROOT / "data/processed/provisional_final_model_comparison_v1",
    ROOT / "notebooks/final_thesis_v1",
]


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an integrity check."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash every file in the frozen directories without changing them."""

    hashes: dict[str, str] = {}
    for directory in paths:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def verify_existing_manifest() -> dict:
    """Recheck files covered by the historical final-model manifest."""

    path = HIST_FINAL_DIR / "final_model_validation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool]] = []
    for mapping in manifest["input_directory_sha256_maps"].values():
        for relative, expected in mapping.items():
            target = ROOT / relative
            checks.append((relative, target.exists() and sha256(target) == expected))
    for staged_relative, expected in manifest["staged_file_sha256"].items():
        if staged_relative.startswith("data/"):
            target = HIST_FINAL_DIR / Path(staged_relative).name
        elif staged_relative.startswith("docs/"):
            target = ROOT / staged_relative
        elif staged_relative.startswith("figures/"):
            target = ROOT / "outputs/provisional_final_model_comparison_v1" / staged_relative
        else:
            continue
        checks.append((str(target.relative_to(ROOT)), target.exists() and sha256(target) == expected))
    failed = [name for name, passed in checks if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "files_checked": len(checks),
        "failures": failed,
        "manifest": str(path.relative_to(ROOT)),
    }


def ensure_new_targets() -> None:
    """Refuse to overwrite an earlier run or any requested new document."""

    targets = [OUT_DIR, PRESENTATION_DIR, SPEC_DOC, MEMO_DOC]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Supervisor update targets already exist: {existing}")


def regression_metrics(y: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Calculate the frozen count-prediction metrics on raw predictions."""

    observed = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mean_observed_y": float(observed.mean()),
        "mean_prediction": float(predicted.mean()),
        "mean_prediction_bias": float(np.mean(predicted - observed)),
        "lower_bound_violation_n": int(np.sum(predicted < 0)),
        "upper_bound_violation_n": int(np.sum(predicted > NMAX)),
        "outside_bound_n": int(np.sum((predicted < 0) | (predicted > NMAX))),
        "outside_bound_pct": float(100 * np.mean((predicted < 0) | (predicted > NMAX))),
    }


def binary_metrics(y: pd.Series | np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    """Evaluate the H3 future-presence probability at the frozen threshold."""

    observed = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(observed, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(observed)),
        "positive_n": int(observed.sum()),
        "observed_positive_pct": float(100 * observed.mean()),
        "decision_threshold": float(threshold),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "pr_auc": float(average_precision_score(observed, probability)),
        "brier_score": float(brier_score_loss(observed, probability)),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(observed, predicted)),
        "precision_positive_predictive_value": float(precision_score(observed, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(observed, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "classification_f1_score": float(f1_score(observed, predicted, zero_division=0)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def load_model_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the frozen cohort, lexical definitions, folds, and temporal labels."""

    base = pd.read_csv(ANALYSIS_DIR / "analysis_ready_550k_52w.csv")
    assignments = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")
    temporal = pd.read_csv(SPEC_DIR / "provisional_temporal_validation_52w.csv")
    data = lyric_prior.add_lexical_features(base).merge(
        assignments[[
            "spotify_track_id", "fold", "primary_group_component",
            "first_credited_artist_id", "supported_related_version_group",
        ]],
        on="spotify_track_id",
        validate="one_to_one",
    )
    data = data.merge(
        temporal[["spotify_track_id", "temporal_partition"]],
        on="spotify_track_id",
        validate="one_to_one",
    ).sort_values("spotify_track_id").reset_index(drop=True)

    if len(data) != 1014 or data.spotify_track_id.nunique() != 1014:
        raise AssertionError("The primary cohort must contain 1,014 unique Spotify track IDs.")
    if data.primary_group_component.nunique() != 276:
        raise AssertionError("The frozen protected grouping must contain 276 components.")
    if sorted(data.fold.value_counts().tolist()) != [202, 203, 203, 203, 203]:
        raise AssertionError("The frozen five-fold assignments changed.")
    crossing = assignments[[
        "first_artist_crosses_folds", "exact_lyric_group_crosses_folds",
        "related_version_group_crosses_folds",
    ]].fillna(0).to_numpy().sum()
    if crossing:
        raise AssertionError("A protected artist/lyric/version group crosses folds.")
    if data.Y_52.min() != 0 or data.Y_52.max() != 48:
        raise AssertionError("Y52 support changed from 0–48.")
    if any(lyric_prior.leakage_hits(variables) for variables in MODEL_VARIABLES.values()):
        raise AssertionError("A supervisor-confirmed feature block contains a blocked predictor.")
    for variables in MODEL_VARIABLES.values():
        if data[variables].isna().any().any():
            raise AssertionError(f"Unexpected missing value in model variables: {variables}")
    if any(name.startswith(("L_", "Z_", "Y_")) for values in MODEL_VARIABLES.values() for name in values):
        raise AssertionError("Outcome-window L/Y/Z descriptors entered a predictor block.")
    return data, assignments, temporal


def fit_count(train: pd.DataFrame, heldout: pd.DataFrame, variables: list[str]):
    """Fit the established Poisson pipeline with training-only preprocessing."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction, preprocessor, model, warning_text = lyric_prior.fit_tab_poisson(
            train, heldout, variables
        )
    external_warnings = " | ".join(str(item.message) for item in caught)
    warning_text = " | ".join(filter(None, [warning_text, external_warnings]))
    return np.asarray(prediction, dtype=float), preprocessor, model, warning_text


def fit_hurdle(train: pd.DataFrame, heldout: pd.DataFrame, variables: list[str]):
    """Fit the unchanged logistic plus zero-truncated-Poisson H3 structure."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_binary = train.Y_52.gt(0).astype(int).to_numpy()
        binary_preprocessor = lyric_prior.make_tab_preprocessor(variables)
        x_binary_train = binary_preprocessor.fit_transform(train[variables])
        x_binary_test = binary_preprocessor.transform(heldout[variables])
        logistic = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        logistic.fit(x_binary_train, y_binary)
        probability_positive = logistic.predict_proba(x_binary_test)[:, 1]

        positive_train = train[train.Y_52.gt(0)]
        positive_preprocessor = lyric_prior.make_tab_preprocessor(variables)
        x_positive_train = positive_preprocessor.fit_transform(positive_train[variables])
        x_positive_test = positive_preprocessor.transform(heldout[variables])
        positive_model = lyric_prior.ZeroTruncatedPoisson().fit(
            x_positive_train, positive_train.Y_52.to_numpy(float)
        )
        conditional_positive_mean = positive_model.predict_conditional_mean(x_positive_test)
    warning_text = " | ".join(str(item.message) for item in caught)
    if not positive_model.fit_.converged:
        warning_text = " | ".join(filter(None, [warning_text, positive_model.fit_.message]))
    return {
        "prediction": probability_positive * conditional_positive_mean,
        "probability_positive": probability_positive,
        "conditional_positive_mean": conditional_positive_mean,
        "binary_iterations": int(logistic.n_iter_[0]),
        "positive_iterations": int(positive_model.fit_.n_iter),
        "positive_converged": bool(positive_model.fit_.converged),
        "warning_text": warning_text,
    }


def grouped_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one held-out prediction per track using the frozen five folds."""

    rows: list[pd.DataFrame] = []
    warnings_rows: list[dict] = []
    for fold in sorted(data.fold.unique()):
        train = data[data.fold.ne(fold)].copy()
        heldout = data[data.fold.eq(fold)].copy()
        if set(train.primary_group_component) & set(heldout.primary_group_component):
            raise AssertionError(f"Protected group crossing detected in {fold}.")
        train_years = sorted(train.chart_entry_year.unique().tolist())
        heldout_years = sorted(heldout.chart_entry_year.unique().tolist())
        if set(heldout_years) - set(train_years):
            raise AssertionError(f"Grouped fold {fold} contains an unseen entry-year level.")

        for model_id in ["F1_YEAR", "F3_YEAR", "LEX_YEAR"]:
            prediction, _, model, warning_text = fit_count(
                train, heldout, MODEL_VARIABLES[model_id]
            )
            rows.append(pd.DataFrame({
                "spotify_track_id": heldout.spotify_track_id.to_numpy(),
                "fold": fold,
                "primary_group_component": heldout.primary_group_component.to_numpy(),
                "model_id": model_id,
                "y_observed": heldout.Y_52.to_numpy(float),
                "y_pred_raw": prediction,
                "X": heldout.X.to_numpy(int),
                "chart_entry_year": heldout.chart_entry_year.to_numpy(int),
            }))
            warnings_rows.append({
                "stage": "grouped_oof_count",
                "partition": fold,
                "model_id": model_id,
                "status": "warning" if warning_text else "ok",
                "converged": int(model.n_iter_ < model.max_iter),
                "iterations": int(model.n_iter_),
                "warning_or_failure": warning_text,
            })

        hurdle = fit_hurdle(train, heldout, MODEL_VARIABLES["H3_YEAR"])
        rows.append(pd.DataFrame({
            "spotify_track_id": heldout.spotify_track_id.to_numpy(),
            "fold": fold,
            "primary_group_component": heldout.primary_group_component.to_numpy(),
            "model_id": "H3_YEAR",
            "y_observed": heldout.Y_52.to_numpy(float),
            "y_pred_raw": hurdle["prediction"],
            "p_positive": hurdle["probability_positive"],
            "conditional_positive_mean": hurdle["conditional_positive_mean"],
            "X": heldout.X.to_numpy(int),
            "chart_entry_year": heldout.chart_entry_year.to_numpy(int),
        }))
        warnings_rows.append({
            "stage": "grouped_oof_hurdle",
            "partition": fold,
            "model_id": "H3_YEAR",
            "status": "warning" if hurdle["warning_text"] else "ok",
            "converged": int(hurdle["positive_converged"]),
            "iterations": hurdle["positive_iterations"],
            "warning_or_failure": hurdle["warning_text"],
        })

    predictions = pd.concat(rows, ignore_index=True)
    if predictions.duplicated(["spotify_track_id", "model_id"]).any():
        raise AssertionError("Grouped OOF predictions are not unique by track and model.")
    counts = predictions.groupby("model_id").spotify_track_id.nunique().to_dict()
    if set(counts.values()) != {1014}:
        raise AssertionError(f"Incomplete grouped OOF coverage: {counts}")
    return predictions, pd.DataFrame(warnings_rows)


def summarise_grouped(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarise aggregate, fold-level, and H3 component performance."""

    aggregate = []
    folds = []
    for model_id, part in predictions.groupby("model_id", sort=True):
        aggregate.append({"model_id": model_id, **regression_metrics(part.y_observed, part.y_pred_raw)})
    for (fold, model_id), part in predictions.groupby(["fold", "model_id"], sort=True):
        folds.append({"fold": fold, "model_id": model_id, **regression_metrics(part.y_observed, part.y_pred_raw)})

    h3 = predictions[predictions.model_id.eq("H3_YEAR")].copy()
    h3_binary_rows = [{
        "aggregation": "all_oof",
        "model_id": "H3_YEAR",
        **binary_metrics(h3.y_observed.gt(0), h3.p_positive),
    }]
    h3_positive_rows = []
    positive = h3.y_observed.gt(0)
    h3_positive_rows.append({
        "aggregation": "all_oof",
        "model_id": "H3_YEAR",
        "n_positive": int(positive.sum()),
        "mae_positive_count": float(mean_absolute_error(
            h3.loc[positive, "y_observed"], h3.loc[positive, "conditional_positive_mean"]
        )),
        "rmse_positive_count": float(math.sqrt(mean_squared_error(
            h3.loc[positive, "y_observed"], h3.loc[positive, "conditional_positive_mean"]
        ))),
        "mean_observed_positive_y": float(h3.loc[positive, "y_observed"].mean()),
        "mean_predicted_conditional_positive_y": float(h3.loc[positive, "conditional_positive_mean"].mean()),
    })
    for fold, part in h3.groupby("fold", sort=True):
        h3_binary_rows.append({
            "aggregation": fold,
            "model_id": "H3_YEAR",
            **binary_metrics(part.y_observed.gt(0), part.p_positive),
        })
        pos = part.y_observed.gt(0)
        h3_positive_rows.append({
            "aggregation": fold,
            "model_id": "H3_YEAR",
            "n_positive": int(pos.sum()),
            "mae_positive_count": float(mean_absolute_error(
                part.loc[pos, "y_observed"], part.loc[pos, "conditional_positive_mean"]
            )),
            "rmse_positive_count": float(math.sqrt(mean_squared_error(
                part.loc[pos, "y_observed"], part.loc[pos, "conditional_positive_mean"]
            ))),
            "mean_observed_positive_y": float(part.loc[pos, "y_observed"].mean()),
            "mean_predicted_conditional_positive_y": float(part.loc[pos, "conditional_positive_mean"].mean()),
        })
    component = pd.concat(
        [
            pd.DataFrame(h3_binary_rows).assign(component="binary_future_presence"),
            pd.DataFrame(h3_positive_rows).assign(component="positive_duration"),
        ],
        ignore_index=True,
        sort=False,
    )
    return pd.DataFrame(aggregate), pd.DataFrame(folds), component


def temporal_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the fixed 2017–2018 training design on 2019 and 2020."""

    train = data[data.temporal_partition.eq("train_2017_2018")].copy()
    if len(train) != 628 or sorted(train.chart_entry_year.unique()) != [2017, 2018]:
        raise AssertionError("The frozen Y52 temporal training partition changed.")
    rows: list[pd.DataFrame] = []
    warning_rows: list[dict] = []
    encoding_rows: list[dict] = []
    for partition, expected_n in [("validation_2019", 341), ("test_2020", 45)]:
        heldout = data[data.temporal_partition.eq(partition)].copy()
        if len(heldout) != expected_n:
            raise AssertionError(f"{partition} contains {len(heldout)} tracks, expected {expected_n}.")
        unseen_years = sorted(set(heldout.chart_entry_year) - set(train.chart_entry_year))
        encoding_rows.append({
            "validation_context": partition,
            "training_entry_year_levels": "|".join(map(str, sorted(train.chart_entry_year.unique()))),
            "heldout_entry_year_levels": "|".join(map(str, sorted(heldout.chart_entry_year.unique()))),
            "unseen_entry_year_levels": "|".join(map(str, unseen_years)),
            "unseen_track_count": int(heldout.chart_entry_year.isin(unseen_years).sum()),
            "handling": "OneHotEncoder(handle_unknown='ignore'): unseen levels map to the all-zero fitted reference vector",
            "outcome_used_for_encoding_or_tuning": 0,
        })
        for model_id in ["F1_YEAR", "F3_YEAR", "LEX_YEAR"]:
            prediction, _, model, warning_text = fit_count(
                train, heldout, MODEL_VARIABLES[model_id]
            )
            rows.append(pd.DataFrame({
                "spotify_track_id": heldout.spotify_track_id.to_numpy(),
                "temporal_partition": partition,
                "model_id": model_id,
                "y_observed": heldout.Y_52.to_numpy(float),
                "y_pred_raw": prediction,
                "X": heldout.X.to_numpy(int),
                "chart_entry_year": heldout.chart_entry_year.to_numpy(int),
            }))
            warning_rows.append({
                "stage": "temporal_count",
                "partition": partition,
                "model_id": model_id,
                "status": "warning" if warning_text else "ok",
                "converged": int(model.n_iter_ < model.max_iter),
                "iterations": int(model.n_iter_),
                "warning_or_failure": warning_text,
            })

        hurdle = fit_hurdle(train, heldout, MODEL_VARIABLES["H3_YEAR"])
        rows.append(pd.DataFrame({
            "spotify_track_id": heldout.spotify_track_id.to_numpy(),
            "temporal_partition": partition,
            "model_id": "H3_YEAR",
            "y_observed": heldout.Y_52.to_numpy(float),
            "y_pred_raw": hurdle["prediction"],
            "p_positive": hurdle["probability_positive"],
            "conditional_positive_mean": hurdle["conditional_positive_mean"],
            "X": heldout.X.to_numpy(int),
            "chart_entry_year": heldout.chart_entry_year.to_numpy(int),
        }))
        warning_rows.append({
            "stage": "temporal_hurdle",
            "partition": partition,
            "model_id": "H3_YEAR",
            "status": "warning" if hurdle["warning_text"] else "ok",
            "converged": int(hurdle["positive_converged"]),
            "iterations": hurdle["positive_iterations"],
            "warning_or_failure": hurdle["warning_text"],
        })

    predictions = pd.concat(rows, ignore_index=True)
    metrics = []
    for (partition, model_id), part in predictions.groupby(["temporal_partition", "model_id"], sort=True):
        metrics.append({
            "temporal_partition": partition,
            "model_id": model_id,
            **regression_metrics(part.y_observed, part.y_pred_raw),
        })
        if model_id == "H3_YEAR":
            metrics[-1].update({
                f"binary_{key}": value
                for key, value in binary_metrics(part.y_observed.gt(0), part.p_positive).items()
                if key not in {"n"}
            })
    return predictions, pd.DataFrame(metrics), pd.concat(
        [pd.DataFrame(encoding_rows), pd.DataFrame(warning_rows)],
        ignore_index=True,
        sort=False,
    )


def historical_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Compare the new pre-specified year control with preserved historical results."""

    old = pd.read_csv(HIST_FINAL_DIR / "thesis_final_model_comparison.csv").set_index("model_id")
    mapping = {
        "F1_YEAR": "F1",
        "F3_YEAR": "F3",
        "LEX_YEAR": "LEX",
        "H3_YEAR": "H3",
    }
    rows = []
    indexed = aggregate.set_index("model_id")
    for new_id, old_id in mapping.items():
        new = indexed.loc[new_id]
        historic = old.loc[old_id]
        rows.append({
            "historical_model_id": old_id,
            "year_controlled_model_id": new_id,
            "historical_mae": historic.grouped_cv_mae,
            "year_controlled_mae": new.mae,
            "mae_change_year_minus_historical": new.mae - historic.grouped_cv_mae,
            "historical_rmse": historic.grouped_cv_rmse,
            "year_controlled_rmse": new.rmse,
            "rmse_change_year_minus_historical": new.rmse - historic.grouped_cv_rmse,
            "historical_bias": historic.grouped_cv_bias,
            "year_controlled_bias": new.mean_prediction_bias,
            "interpretation_rule": "negative error change favours the year-controlled specification; entry year was included by supervisor request, not selected by performance",
        })
    return pd.DataFrame(rows)


def lexical_uncertainty(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply the established paired protected-component bootstrap to F3_YEAR–LEX_YEAR."""

    selected = predictions[predictions.model_id.isin(["F3_YEAR", "LEX_YEAR"])]
    wide = selected.pivot(index="spotify_track_id", columns="model_id", values="y_pred_raw").reset_index()
    wide = data[["spotify_track_id", "Y_52", "primary_group_component"]].merge(
        wide, on="spotify_track_id", validate="one_to_one"
    )
    output = robust_prior.cluster_bootstrap_comparison(
        wide,
        "F3_YEAR",
        "LEX_YEAR",
        "primary_group_component",
        BOOTSTRAP_REPS,
        SEED,
        "supervisor_confirmed_Y52_grouped_OOF",
    )
    output["uncertainty_scope_note"] = (
        "Paired bootstrap conditional on the fixed OOF predictions; it does not include model-selection uncertainty."
    )
    return output


def related_version_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit F3_YEAR and LEX_YEAR after the unchanged supported-version exclusion."""

    version_audit = pd.read_csv(SPEC_DIR / "analysis_unit_version_audit.csv")
    supported = version_audit[
        version_audit.eligible_for_later_underlying_song_sensitivity_group.eq(1)
    ]
    excluded_ids: set[str] = set()
    for value in supported.spotify_track_ids:
        excluded_ids.update(str(value).split("|"))
    if len(excluded_ids) != 143:
        raise AssertionError(f"Related-version set contains {len(excluded_ids)} IDs, expected 143.")
    reduced = data[~data.spotify_track_id.isin(excluded_ids)].copy()
    if len(reduced) != 871:
        raise AssertionError(f"Related-version exclusion leaves {len(reduced)} tracks, expected 871.")

    frames = []
    for fold in sorted(reduced.fold.unique()):
        train = reduced[reduced.fold.ne(fold)]
        heldout = reduced[reduced.fold.eq(fold)]
        for model_id in ["F3_YEAR", "LEX_YEAR"]:
            prediction, _, _, warning_text = fit_count(train, heldout, MODEL_VARIABLES[model_id])
            if warning_text:
                warnings.warn(warning_text)
            frames.append(pd.DataFrame({
                "spotify_track_id": heldout.spotify_track_id.to_numpy(),
                "fold": fold,
                "primary_group_component": heldout.primary_group_component.to_numpy(),
                "model_id": f"{model_id}_VERSION_EXCLUSION",
                "y_observed": heldout.Y_52.to_numpy(float),
                "y_pred_raw": prediction,
            }))
    predictions = pd.concat(frames, ignore_index=True)
    metrics = []
    for model_id, part in predictions.groupby("model_id", sort=True):
        metrics.append({
            "record_type": "model_metric",
            "model_id": model_id,
            "n_supported_groups_removed": int(len(supported)),
            "n_tracks_removed": 143,
            "n_tracks_remaining": 871,
            "protected_clusters_remaining": int(reduced.primary_group_component.nunique()),
            **regression_metrics(part.y_observed, part.y_pred_raw),
        })
    wide = predictions.pivot(index="spotify_track_id", columns="model_id", values="y_pred_raw").reset_index()
    wide = reduced[["spotify_track_id", "Y_52", "primary_group_component"]].merge(
        wide, on="spotify_track_id", validate="one_to_one"
    )
    uncertainty = robust_prior.cluster_bootstrap_comparison(
        wide,
        "F3_YEAR_VERSION_EXCLUSION",
        "LEX_YEAR_VERSION_EXCLUSION",
        "primary_group_component",
        BOOTSTRAP_REPS,
        VERSION_SENSITIVITY_SEED,
        "supported_related_version_exclusion_year_controlled",
    )
    uncertainty["record_type"] = "paired_cluster_bootstrap"
    uncertainty["n_supported_groups_removed"] = int(len(supported))
    uncertainty["n_tracks_removed"] = 143
    uncertainty["n_tracks_remaining"] = 871
    uncertainty["protected_clusters_remaining"] = int(reduced.primary_group_component.nunique())
    uncertainty["documented_rule"] = (
        "Remove every track in the 70 supported exact-lyric/related-version groups; retain the unresolved group."
    )
    return pd.concat([pd.DataFrame(metrics), uncertainty], ignore_index=True, sort=False), predictions


def l_z_descriptives(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise L and Z as outcomes/descriptors, never as predictors."""

    pearson = pearsonr(data.Y_52, data.L_52)
    spearman = spearmanr(data.Y_52, data.L_52)
    historical = pd.read_csv(ANALYSIS_DIR / "exploratory_outcome_correlations.csv")
    historical_row = historical[
        historical.variable_1.eq("Y_52") & historical.variable_2.eq("L_52")
    ].iloc[0]
    if not np.isclose(pearson.statistic, historical_row.pearson, atol=1e-12):
        raise AssertionError("Recomputed Y52–L Pearson correlation differs from frozen evidence.")
    if not np.isclose(spearman.statistic, historical_row.spearman, atol=1e-12):
        raise AssertionError("Recomputed Y52–L Spearman correlation differs from frozen evidence.")

    z = data.Z_count_52.astype(int)
    if not np.array_equal(data.Z_reentry_52.astype(int), z.gt(1).astype(int)):
        raise AssertionError("Z_reentry_52 does not reconcile with Z_count_52 > 1.")
    z_distribution = (
        z.value_counts().sort_index().rename_axis("Z_count_52").rename("tracks").reset_index()
    )
    z_distribution["percent"] = 100 * z_distribution.tracks / len(data)
    z_distribution["interpretation"] = np.select(
        [z_distribution.Z_count_52.eq(0), z_distribution.Z_count_52.eq(1)],
        ["no chart presence in weeks 5–52", "one continuous presence block in weeks 5–52"],
        default="multiple future presence blocks (fragmentation/re-entry)",
    )
    rows = [
        {"measure": "Y52_L52_Pearson", "value": float(pearson.statistic), "n": len(data), "unit": "correlation", "role": "descriptive outcome relationship"},
        {"measure": "Y52_L52_Spearman", "value": float(spearman.statistic), "n": len(data), "unit": "rank correlation", "role": "descriptive outcome relationship"},
        {"measure": "Z_mean", "value": float(z.mean()), "n": len(data), "unit": "presence blocks", "role": "descriptive"},
        {"measure": "Z_median", "value": float(z.median()), "n": len(data), "unit": "presence blocks", "role": "descriptive"},
        {"measure": "Z_minimum", "value": int(z.min()), "n": len(data), "unit": "presence blocks", "role": "descriptive"},
        {"measure": "Z_maximum", "value": int(z.max()), "n": len(data), "unit": "presence blocks", "role": "descriptive"},
        {"measure": "Z_equals_1_count", "value": int(z.eq(1).sum()), "n": len(data), "unit": "tracks", "role": "descriptive"},
        {"measure": "Z_equals_1_percent", "value": float(100 * z.eq(1).mean()), "n": len(data), "unit": "percent", "role": "descriptive"},
        {"measure": "Z_greater_than_1_count", "value": int(z.gt(1).sum()), "n": len(data), "unit": "tracks", "role": "fragmentation/re-entry descriptor"},
        {"measure": "Z_greater_than_1_percent", "value": float(100 * z.gt(1).mean()), "n": len(data), "unit": "percent", "role": "fragmentation/re-entry descriptor"},
    ]
    return pd.DataFrame(rows), z_distribution


def cohort_reconciliation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact two-way bridge from strict survival to clean Y52."""

    survival = pd.read_csv(
        SURVIVAL_DIR / "first_spell_survival_matched_550k_eligible.csv",
        dtype={"spotify_track_id": "string"},
    )
    all_survival = pd.read_csv(
        SURVIVAL_DIR / "first_spell_survival.csv",
        dtype={"spotify_track_id": "string"},
    )
    final = pd.read_csv(
        ANALYSIS_DIR / "analysis_ready_550k_52w.csv",
        dtype={"spotify_track_id": "string"},
    )
    register = pd.read_csv(
        ANALYSIS_DIR / "cohort_exclusion_register.csv",
        dtype={"spotify_track_id": "string"},
    )
    register = register[register.horizon_weeks.eq(52)].copy()
    fixed = pd.read_csv(
        FIXED_DIR / "fixed_window_track_outcomes.csv",
        dtype={"spotify_track_id": "string"},
    )
    fixed = fixed[fixed.horizon_weeks.eq(52)].copy()
    fixed_exclusions = pd.read_csv(
        FIXED_DIR / "fixed_window_exclusions.csv",
        dtype={"spotify_track_id": "string"},
    )
    fixed_exclusions = fixed_exclusions[fixed_exclusions.horizon_weeks.eq(52)][
        ["spotify_track_id", "source_gap_dates"]
    ]

    if len(survival) != 1357 or survival.spotify_track_id.nunique() != 1357:
        raise AssertionError("Strict 550k-linked survival cohort is not 1,357 unique tracks.")
    if len(final) != 1014 or final.spotify_track_id.nunique() != 1014:
        raise AssertionError("Final Y52 cohort is not 1,014 unique tracks.")
    survival_ids = set(survival.spotify_track_id)
    final_ids = set(final.spotify_track_id)
    retained_ids = survival_ids & final_ids
    removed_ids = survival_ids - final_ids
    added_ids = final_ids - survival_ids
    if (len(retained_ids), len(removed_ids), len(added_ids)) != (1004, 353, 10):
        raise AssertionError(
            f"Unexpected cohort overlap: retained={len(retained_ids)}, removed={len(removed_ids)}, added={len(added_ids)}"
        )

    universe = pd.DataFrame({"spotify_track_id": sorted(survival_ids | final_ids)})
    survival_meta = survival[[
        "spotify_track_id", "track", "artist", "first_observed_chart_entry",
        "outcome_status", "first_spell_source_gap_affected",
    ]].rename(columns={
        "track": "survival_track_title",
        "artist": "survival_artist",
        "first_observed_chart_entry": "survival_first_observed_entry",
    })
    final_meta = final[[
        "spotify_track_id", "chart_track", "chart_artist", "chart_entry_date",
        "X", "Y_52", "L_52", "Z_count_52",
    ]]
    full_status = all_survival[[
        "spotify_track_id", "outcome_status", "survival_outcome_eligible",
        "first_spell_source_gap_affected",
    ]].rename(columns={
        "outcome_status": "all_chart_first_spell_outcome_status",
        "survival_outcome_eligible": "all_chart_survival_outcome_eligible",
        "first_spell_source_gap_affected": "all_chart_first_spell_source_gap_affected",
    })
    frame = (
        universe.merge(survival_meta, on="spotify_track_id", how="left", validate="one_to_one")
        .merge(final_meta, on="spotify_track_id", how="left", validate="one_to_one")
        .merge(full_status, on="spotify_track_id", how="left", validate="one_to_one")
        .merge(register, on="spotify_track_id", how="left", validate="one_to_one", suffixes=("", "_register"))
        .merge(fixed[[
            "spotify_track_id", "enough_follow_up", "source_gap_in_horizon",
            "fixed_window_eligible", "exclusion_reason",
        ]], on="spotify_track_id", how="left", validate="one_to_one")
        .merge(fixed_exclusions, on="spotify_track_id", how="left", validate="one_to_one")
    )
    frame["in_survival_1357"] = frame.spotify_track_id.isin(survival_ids).astype(int)
    frame["in_final_y52_1014"] = frame.spotify_track_id.isin(final_ids).astype(int)

    def transition(row: pd.Series) -> tuple[str, str, int]:
        if row.in_survival_1357 and row.in_final_y52_1014:
            return "retained_from_survival_in_final_y52", "", 0
        if row.in_survival_1357 and not row.in_final_y52_1014:
            if row.analysis_550k_status_exclusive == "insufficient_follow_up":
                return "removed_from_survival", "insufficient_52_week_follow_up", -1
            if row.analysis_550k_status_exclusive == "missing_source_week_contamination":
                if row.source_gap_dates == "2017-06-02;2017-06-09":
                    return "removed_from_survival", "required_52_week_window_crosses_2017_06_source_gaps", -1
                if row.source_gap_dates == "2021-02-19":
                    return "removed_from_survival", "required_52_week_window_crosses_2021_02_19_source_gap", -1
            raise AssertionError(f"Unresolved survival exclusion for {row.spotify_track_id}")
        if not row.in_survival_1357 and row.in_final_y52_1014:
            if (
                row.all_chart_first_spell_outcome_status == "source_gap_unknown"
                and row.all_chart_survival_outcome_eligible == 0
                and row.fixed_window_eligible == 1
            ):
                return (
                    "added_to_fixed_window_from_outside_strict_survival",
                    "first_spell_crosses_later_source_gap_but_required_52_week_window_is_complete",
                    1,
                )
            raise AssertionError(f"Unresolved fixed-window addition for {row.spotify_track_id}")
        raise AssertionError(f"Unexpected cohort state for {row.spotify_track_id}")

    classified = frame.apply(transition, axis=1, result_type="expand")
    classified.columns = ["transition_status", "exact_exclusion_or_addition_reason", "net_count_effect"]
    frame = pd.concat([frame, classified], axis=1)
    counts = frame.groupby("exact_exclusion_or_addition_reason", dropna=False).size().to_dict()
    expected = {
        "": 1004,
        "insufficient_52_week_follow_up": 198,
        "required_52_week_window_crosses_2017_06_source_gaps": 75,
        "required_52_week_window_crosses_2021_02_19_source_gap": 80,
        "first_spell_crosses_later_source_gap_but_required_52_week_window_is_complete": 10,
    }
    if counts != expected:
        raise AssertionError(f"Cohort transition counts do not reconcile: {counts}")

    final_columns = [
        "spotify_track_id", "in_survival_1357", "in_final_y52_1014",
        "transition_status", "exact_exclusion_or_addition_reason", "net_count_effect",
        "survival_track_title", "survival_artist", "chart_track", "chart_artist",
        "survival_first_observed_entry", "chart_entry_date", "source_gap_dates",
        "enough_follow_up", "source_gap_in_horizon", "fixed_window_eligible",
        "all_chart_first_spell_outcome_status", "all_chart_survival_outcome_eligible",
        "all_chart_first_spell_source_gap_affected", "X", "Y_52", "L_52", "Z_count_52",
    ]
    detailed = frame[final_columns].rename(columns={"spotify_track_id": "track_id"})

    # The two-way bridge is necessary because ten complete fixed windows were
    # excluded from strict survival after their later first spell hit a source gap.
    summary_rows = [
        ("survival_1357_to_final_y52", 1, "starting strict 550k-linked first-spell survival cohort", 1357, 1357),
        ("survival_1357_to_final_y52", 2, "remove: insufficient 52-week follow-up", -198, 1159),
        ("survival_1357_to_final_y52", 3, "remove: 52-week window crosses 2017-06-02 and 2017-06-09 gaps", -75, 1084),
        ("survival_1357_to_final_y52", 4, "remove: 52-week window crosses 2021-02-19 gap", -80, 1004),
        ("survival_1357_to_final_y52", 5, "add: fixed window valid although strict first-spell survival outcome is source-gap unknown", 10, 1014),
        ("survival_1357_to_final_y52", 6, "final clean Y52 550k-linked cohort", 0, 1014),
        ("preliminary_1025_to_final_y52", 1, "preliminary eligible 52-week 550k-linked count", 1025, 1025),
        ("preliminary_1025_to_final_y52", 2, "remove: uncertain entry on 2017-06-16 immediately after June source gaps", -11, 1014),
        ("preliminary_1025_to_final_y52", 3, "final clean Y52 550k-linked cohort", 0, 1014),
    ]
    summary = pd.DataFrame(summary_rows, columns=[
        "reconciliation_scope", "sequence", "step", "count_change", "running_total"
    ])
    if 1357 - 198 - 75 - 80 + 10 != 1014:
        raise AssertionError("The 1,357-to-1,014 bridge does not reconcile.")
    if 1025 - 11 != 1014:
        raise AssertionError("The 1,025-to-1,014 bridge does not reconcile.")
    return detailed, summary


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create compact probability calibration bins for H3_YEAR."""

    h3 = predictions[predictions.model_id.eq("H3_YEAR")].copy()
    h3["calibration_bin"] = pd.qcut(
        h3.p_positive.rank(method="first"), q=5, labels=False
    ) + 1
    return h3.groupby("calibration_bin", as_index=False).agg(
        tracks=("spotify_track_id", "size"),
        mean_predicted_positive_probability=("p_positive", "mean"),
        observed_positive_proportion=("y_observed", lambda values: values.gt(0).mean()),
        mean_predicted_zero_probability=("p_positive", lambda values: 1 - values.mean()),
        observed_zero_proportion=("y_observed", lambda values: values.eq(0).mean()),
    )


def model_specifications() -> pd.DataFrame:
    """Record the final supervisor-confirmed model roles without feature search."""

    rows = []
    descriptions = {
        "F1_YEAR": "categorical X + categorical chart-entry year",
        "F3_YEAR": "F1_YEAR + frozen audio/song-characteristic block",
        "LEX_YEAR": "F3_YEAR + seven frozen transparent lexical summaries",
        "H3_YEAR": "two-part H3 using the LEX_YEAR predictor block",
    }
    for model_id, variables in MODEL_VARIABLES.items():
        rows.append({
            "model_id": model_id,
            "model_role": descriptions[model_id],
            "predictors": "|".join(variables),
            "entry_year_encoding": "categorical OneHotEncoder; sorted levels; drop first; handle_unknown=ignore",
            "continuous_preprocessing": "StandardScaler fitted within each training fold",
            "categorical_preprocessing": "OneHotEncoder fitted within each training fold",
            "count_estimator": (
                "PoissonRegressor alpha=1e-8, max_iter=1000, tol=1e-8"
                if model_id != "H3_YEAR"
                else "P(Y>0): L2 logistic C=1; E(Y|Y>0): zero-truncated Poisson; expected count is their product"
            ),
            "performance_selected": 0,
            "supervisor_requested_control": "chart_entry_year",
            "post_week_4_predictor_count": 0,
        })
    return pd.DataFrame(rows)


def leakage_audit() -> pd.DataFrame:
    """Verify that entry year is safe and L/Z remain descriptive only."""

    rows = []
    for model_id, variables in MODEL_VARIABLES.items():
        hits = lyric_prior.leakage_hits(variables)
        lz_hits = [name for name in variables if name.startswith(("L_", "Z_", "Y_"))]
        rows.append({
            "check": f"{model_id} predictor block",
            "status": "PASS" if not hits and not lz_hits and "chart_entry_year" in variables else "FAIL",
            "evidence": (
                f"entry year present; blocked hits={hits or 'none'}; L/Y/Z hits={lz_hits or 'none'}; "
                f"variables={'|'.join(variables)}"
            ),
        })
    rows.extend([
        {
            "check": "entry year pre-specified rather than searched",
            "status": "PASS",
            "evidence": "single canonical categorical encoding reused; no alternative encoding or performance selection",
        },
        {
            "check": "L and Z roles",
            "status": "PASS",
            "evidence": "L_52, Z_count_52 and Z_reentry_52 appear only in descriptive outputs",
        },
        {
            "check": "unknown source weeks",
            "status": "PASS",
            "evidence": "frozen clean cohort and conservative source-gap flags reused; no unknown week converted to absence",
        },
        {
            "check": "model and feature search",
            "status": "PASS",
            "evidence": "no new model family, feature family, tuning grid or hyperparameter selection introduced",
        },
    ])
    audit = pd.DataFrame(rows)
    if not audit.status.eq("PASS").all():
        raise AssertionError("Leakage or scope audit failed.")
    return audit


def make_thesis_tables(
    aggregate: pd.DataFrame,
    historical: pd.DataFrame,
    uncertainty: pd.DataFrame,
    temporal: pd.DataFrame,
    lz_summary: pd.DataFrame,
    z_distribution: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    related: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create concise, machine-readable tables for later LaTeX drafting."""

    new_metrics = aggregate[[
        "model_id", "n", "mae", "rmse", "mean_prediction_bias",
        "lower_bound_violation_n", "upper_bound_violation_n",
    ]].copy()
    principal = historical.merge(
        new_metrics,
        left_on="year_controlled_model_id",
        right_on="model_id",
        validate="one_to_one",
    )
    lz_long = pd.concat([
        lz_summary.assign(record_type="summary_statistic"),
        z_distribution.rename(columns={
            "Z_count_52": "value", "tracks": "n", "percent": "unit_value"
        }).assign(
            record_type="Z_distribution",
            measure=lambda frame: "Z_count_52=" + frame.value.astype(str),
            unit="tracks; unit_value is percent",
            role="descriptive",
        )[["record_type", "measure", "value", "n", "unit", "role", "unit_value", "interpretation"]],
    ], ignore_index=True, sort=False)
    version_mae = related[
        related.record_type.eq("paired_cluster_bootstrap") & related.metric.eq("MAE")
    ].copy()
    lexical_mae = uncertainty[uncertainty.metric.eq("MAE")].iloc[0]
    robustness = pd.DataFrame([
        {
            "check": "historical versus entry-year-controlled sequence",
            "result": "see final_principal_model_comparison.csv",
            "interpretation": "entry year is included by supervisor request; error changes are evaluated as consequences, not a selection criterion",
        },
        {
            "check": "protected-component lexical uncertainty",
            "result": f"MAE difference={lexical_mae.observed_difference:.6f}",
            "interpretation": "positive means LEX_YEAR improves on F3_YEAR; conditional on fixed OOF predictions",
        },
        {
            "check": "related-version exclusion",
            "result": (
                f"143 removed; 871 retained; MAE difference={version_mae.iloc[0].observed_difference:.6f}; "
                f"95% CI [{version_mae.iloc[0].ci_2_5:.6f}, {version_mae.iloc[0].ci_97_5:.6f}]"
            ),
            "interpretation": "same 70 supported-group exclusion rule as the historical sensitivity",
        },
        {
            "check": "temporal validation",
            "result": "2017–2018 train; 2019 validation; 2020 stress test N=45",
            "interpretation": "unseen entry-year categories map to the fitted reference vector; the 2020 comparison remains descriptive",
        },
        {
            "check": "unchanged historical sensitivities",
            "result": "Y26, 160k, TF-IDF, nonlinear ML, bounded/NB2 and source-gap evidence preserved",
            "interpretation": "not rerun because the professor requested no new model or feature search",
        },
    ])
    return {
        "final_cohort_reconciliation.csv": cohort_summary,
        "final_principal_model_comparison.csv": principal,
        "final_lexical_uncertainty.csv": uncertainty,
        "final_temporal_validation.csv": temporal,
        "final_L_Z_descriptives.csv": lz_long,
        "final_robustness_summary.csv": robustness,
    }


def write_latex_fragments(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Export compact booktabs fragments without drafting thesis prose."""

    output_dir.mkdir(parents=True, exist_ok=True)

    def escape(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    for filename, frame in tables.items():
        alignment = "l" * len(frame.columns)
        lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
        lines.append(" & ".join(escape(column) for column in frame.columns) + r" \\")
        lines.append("\\midrule")
        for row in frame.itertuples(index=False, name=None):
            lines.append(" & ".join(escape(value) for value in row) + r" \\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        fragment = "\n".join(lines) + "\n"
        target = output_dir / (Path(filename).stem + ".tex")
        target.write_text(
            "% Machine-generated table fragment from supervisor_confirmed_update_v1.\n"
            "% Review column selection and caption in the later thesis-drafting task.\n"
            + fragment,
            encoding="utf-8",
        )


def build_specification_document(
    aggregate: pd.DataFrame,
    historical: pd.DataFrame,
    uncertainty: pd.DataFrame,
    temporal: pd.DataFrame,
    lz_summary: pd.DataFrame,
    related: pd.DataFrame,
    cohort_summary: pd.DataFrame,
) -> str:
    """Write the supervisor-confirmed computational specification for drafting."""

    metrics = aggregate.set_index("model_id")
    lex_mae = uncertainty[uncertainty.metric.eq("MAE")].iloc[0]
    lex_rmse = uncertainty[uncertainty.metric.eq("RMSE")].iloc[0]
    version = related[
        related.record_type.eq("paired_cluster_bootstrap") & related.metric.eq("MAE")
    ].iloc[0]
    lz = lz_summary.set_index("measure").value
    transition = cohort_summary[cohort_summary.reconciliation_scope.eq("survival_1357_to_final_y52")]
    transition_text = "; ".join(
        f"{row.step} ({row.count_change:+d})" for row in transition.itertuples() if row.sequence in {2, 3, 4, 5}
    )
    temporal_2019 = temporal[temporal.temporal_partition.eq("validation_2019")]

    return f"""# Supervisor-Confirmed Final Model Specification v1

**Status date:** 5 September 2026  
**Scope:** computational specification for subsequent thesis drafting

## Confirmed design

Professor Nicola Ciaramella confirmed the main design after reviewing the August progress update. The thesis uses Y52 as the primary outcome, Y26 as a sensitivity, the 550k-linked cohort as the primary rich-feature/lyrics population, and the 160k-linked population as a separate robustness sample. Source-wide missing weeks remain unknown. Protected grouped validation and temporal validation remain the principal generalisation checks. The seven transparent lexical summaries are the principal lyric representation, while TF-IDF remains a sensitivity. F1, F3 and LEX form the principal count sequence; H3 remains the companion future-presence analysis.

The supervisor requested three additions implemented here: chart-entry year in every principal model, descriptive treatment of L and Z, and a complete bridge from the 1,357-track strict survival cohort to the 1,014-track Y52 cohort. No broad model, feature, hyperparameter, target or dataset search was reopened.

## Analysis unit, timing and outcomes

The analysis unit remains the Spotify track ID. X is the number of Global Top-200 appearances during calendar weeks 1–4 after the first confirmed entry and is represented categorically. Y52 counts chart appearances in weeks 5–52, including later re-entry, and ranges from 0 to 48. Y26 counts weeks 5–26 and ranges from 0 to 22. The primary analysis contains 1,014 clean 550k-linked tracks.

L52 is the last numbered week with chart presence within the fixed 52-week window. Z counts separate presence blocks in weeks 5–52. L and Z are outcome-window descriptors and are never predictors.

## Entry-year control

`chart_entry_year` is included in F1_YEAR, F3_YEAR, LEX_YEAR and both H3_YEAR components. It uses the project's canonical categorical implementation: sorted year levels are encoded with `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`. The encoder is fitted within each training fold. This is a pre-specified control introduced at the supervisor's request, not a performance-selected feature.

All grouped-fold training sets contain the entry-year levels occurring in their held-out fold. In temporal validation, the model is trained on 2017–2018; 2019 and 2020 are unseen categorical levels and therefore map to the fitted all-zero reference vector. This avoids using future-year data, but it means the temporal year term cannot estimate a separate 2019 or 2020 adjustment. Temporal performance is consequently interpreted as generalisation under calendar shift, not as an estimate of the later-year coefficients.

Release age is not added to the final sequence. The supervisor requested entry year specifically, and the prior release/calendar block remains historical sensitivity evidence.

## Final principal sequence

| Model | Predictors | Grouped-CV MAE | Grouped-CV RMSE |
|---|---|---:|---:|
| F1_YEAR | categorical X + categorical entry year | {metrics.loc['F1_YEAR', 'mae']:.3f} | {metrics.loc['F1_YEAR', 'rmse']:.3f} |
| F3_YEAR | F1_YEAR + frozen audio/song descriptors | {metrics.loc['F3_YEAR', 'mae']:.3f} | {metrics.loc['F3_YEAR', 'rmse']:.3f} |
| LEX_YEAR | F3_YEAR + seven transparent lexical summaries | {metrics.loc['LEX_YEAR', 'mae']:.3f} | {metrics.loc['LEX_YEAR', 'rmse']:.3f} |
| H3_YEAR | binary continuation + conditional positive duration using LEX_YEAR predictors | {metrics.loc['H3_YEAR', 'mae']:.3f} | {metrics.loc['H3_YEAR', 'rmse']:.3f} |

The seven lexical summaries remain log-transformed word count, log-transformed unique-token count, lexical diversity, average token length, log-transformed line count, repeated-line proportion and log-transformed punctuation/intensity rate. TF-IDF and nonlinear machine learning remain sensitivity/appendix analyses.

## Validation and uncertainty

Primary evaluation uses the unchanged five folds over 276 protected components. First-credited artist, exact-lyric groups and supported related-version groups do not cross folds. Complete collaborator separation is not claimed. Learned preprocessing is fitted only on training data.

Temporal validation retains the fixed 2017–2018 training, 2019 validation and 2020 stress-test design. The Y52 2020 set contains 45 tracks. The updated 2019 rows are preserved in `year_controlled_temporal_metrics.csv` ({len(temporal_2019)} model rows).

The paired protected-component bootstrap for F3_YEAR to LEX_YEAR uses 5,000 replicates and seed {SEED}. The MAE difference is {lex_mae.observed_difference:.4f} weeks (95% percentile interval [{lex_mae.ci_2_5:.4f}, {lex_mae.ci_97_5:.4f}]); the RMSE difference is {lex_rmse.observed_difference:.4f} weeks (95% interval [{lex_rmse.ci_2_5:.4f}, {lex_rmse.ci_97_5:.4f}]). Positive values favour LEX_YEAR. This interval is conditional on fixed OOF predictions and does not include full model-selection uncertainty.

## H3 companion analysis

H3_YEAR retains the established two-part structure. Part A estimates the probability of any chart presence after week 4 using L2 logistic regression at the unchanged 0.50 classification threshold. Part B estimates positive future duration using the established zero-truncated Poisson model. Entry year is included in both components using the same training-only categorical encoding. H3 answers whether continuation occurs separately from how long positive continuation lasts; it does not replace the principal count sequence.

## L and Z descriptive role

For the 1,014-track cohort, the Pearson correlation between Y52 and L52 is {lz['Y52_L52_Pearson']:.4f} and the Spearman correlation is {lz['Y52_L52_Spearman']:.4f}. L records how far chart activity extends within the horizon, while Y records its total volume of weekly presence. Z has mean {lz['Z_mean']:.3f}, median {lz['Z_median']:.1f}, and range {int(lz['Z_minimum'])}–{int(lz['Z_maximum'])}. Multiple future blocks occur for {int(lz['Z_greater_than_1_count'])} tracks ({lz['Z_greater_than_1_percent']:.2f}%). These values characterise fragmentation/re-entry; they are not independent predictors.

## Cohort reconciliation

The numerical difference 1,357 − 1,014 = 343 is a net change rather than 343 one-way exclusions. The exact transition is: {transition_text}. Thus 1,357 − 198 − 75 − 80 + 10 = 1,014. The 198 insufficient-follow-up removals confirm that late entry is the largest single reason, but it is not the complete explanation.

The separate August bridge remains 1,025 − 11 = 1,014: the 11 tracks first observed on 16 June 2017 immediately after the two June source gaps were excluded under the final confirmed-entry rule.

## Leakage blocklist

The final models exclude popularity and follower snapshots; future ranks and streams; eventual peak; total chart duration; first-spell duration; L, Z and re-entry summaries; old survivor-landmark eligibility flags; and all other information derived after week 4. Artist IDs remain grouping variables rather than ordinary predictors.

## Robustness retained

Y26, the 160k common-feature population, source-gap treatment, related-version exclusion, bounded and NB2 formulations, TF-IDF, Random Forest, Extra Trees and histogram gradient boosting remain documented sensitivity evidence. Under the unchanged supported-version rule, 143 tracks are removed and 871 remain. The updated F3_YEAR-to-LEX_YEAR MAE difference is {version.observed_difference:.4f} weeks, with 95% interval [{version.ci_2_5:.4f}, {version.ci_97_5:.4f}].

## Claims permitted

- Among songs already in the Global Top 200, early four-week chart persistence is the dominant predictive signal in the examined feature blocks.
- Standard audio characteristics contribute comparatively little incremental held-out accuracy after X.
- Transparent lexical summaries provide, at most, a small and qualified incremental signal.
- Predicting any future chart presence is easier than predicting exact positive duration.

## Claims to avoid

- The study does not predict which songs become hits.
- The results do not generalise automatically to uncharted songs or the complete Spotify catalogue.
- Predictive and descriptive associations are not causal effects.
- Chart presence is an empirical longevity outcome; it is not itself a monetary CLV measure.
- The small lyric increment should not be described as strong or uniform when temporal and related-version evidence is mixed.
"""


def build_feedback_memo(
    metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
    lz_summary: pd.DataFrame,
    related: pd.DataFrame,
) -> str:
    """Create a compact implementation record for the supervisor feedback."""

    indexed = metrics.set_index("model_id")
    lz = lz_summary.set_index("measure").value
    lex = uncertainty[uncertainty.metric.eq("MAE")].iloc[0]
    version = related[
        related.record_type.eq("paired_cluster_bootstrap") & related.metric.eq("MAE")
    ].iloc[0]
    return f"""# Supervisor Feedback Implementation — 5 September 2026

This memo records the targeted changes made after Professor Nicola Ciaramella confirmed the main design. It is a traceability document, not thesis prose. Historical outputs remain unchanged.

| Professor request or confirmed choice | Implementation | Affected new files | Result | Thesis consequence |
|---|---|---|---|---|
| Include entry year in the principal model | Added categorical, training-fold-only `chart_entry_year` to F1_YEAR, F3_YEAR and LEX_YEAR; no release-age addition or encoding search | `year_controlled_*`; Notebooks 03, 04 and 06 v2 | MAE/RMSE: F1_YEAR {indexed.loc['F1_YEAR','mae']:.3f}/{indexed.loc['F1_YEAR','rmse']:.3f}; F3_YEAR {indexed.loc['F3_YEAR','mae']:.3f}/{indexed.loc['F3_YEAR','rmse']:.3f}; LEX_YEAR {indexed.loc['LEX_YEAR','mae']:.3f}/{indexed.loc['LEX_YEAR','rmse']:.3f} | Year-controlled sequence becomes the thesis-facing principal count specification |
| Include L descriptively | Recomputed Pearson and Spearman Y52–L52 correlations on N=1,014 and reconciled them to frozen EDA | `L_Z_descriptives.csv`; Notebook 02 v2 | Pearson {lz['Y52_L52_Pearson']:.4f}; Spearman {lz['Y52_L52_Spearman']:.4f} | L explains how far activity extends but remains excluded from predictors |
| Include Z/fragmentation descriptively | Summarised Z count and reconciled `Z_reentry_52` with Z>1 | `Z_distribution.csv`; Notebook 02 v2 | {int(lz['Z_greater_than_1_count'])} tracks ({lz['Z_greater_than_1_percent']:.2f}%) have multiple future blocks | Re-entry/fragmentation is reported without leaking it into prediction |
| Explain 1,357 to 1,014 | Built an exact ID bridge over the union of both cohorts | `survival1357_to_y52_1014_reconciliation.csv`; Notebook 01 v2 | 198 insufficient follow-up, 75 June-gap windows, 80 February-2021-gap windows, and 10 fixed-window-valid additions; net −343 | Methods must describe a two-way cohort transition, not 343 one-way exclusions |
| Y52 primary; Y26 sensitivity | Preserved definitions and existing clean cohorts | final specification and reconciliation manifest | Y52 N=1,014; Y26 N=1,220 | Primary and sensitivity horizons are now fixed for drafting |
| 550k primary; 160k robustness | Preserved separate roles; no pooling | final specification; Notebook 01 v2 | 550k supplies lyrics; 160k remains common-feature robustness | Coverage limitations remain explicit |
| Grouped plus temporal validation | Reused the exact five folds, 276 protected components and frozen temporal partitions | all year-controlled prediction files | No fold or split redesign | Main results use grouped OOF evidence; temporal results test later-period transfer |
| Transparent lexical summaries; TF-IDF sensitivity | Added year only to the seven-summary LEX model; TF-IDF was not reopened | `year_controlled_lexical_uncertainty.csv`; Notebook 04 v2 | F3_YEAR→LEX_YEAR MAE difference {lex.observed_difference:.4f}, 95% CI [{lex.ci_2_5:.4f}, {lex.ci_97_5:.4f}] | Lexical claim remains modest and uncertainty-qualified |
| F1/F3/LEX principal; H3 companion | Updated all four with the same entry-year control; preserved estimator families and threshold | `year_controlled_grouped_cv_*`; H3 component files | H3_YEAR MAE/RMSE {indexed.loc['H3_YEAR','mae']:.3f}/{indexed.loc['H3_YEAR','rmse']:.3f} | Count prediction and continuation probability remain separate results |
| Related-version qualification | Reused the unchanged 70-group/143-track exclusion rule | `year_controlled_related_version_sensitivity.csv` | 871 remain; MAE difference {version.observed_difference:.4f}, 95% CI [{version.ci_2_5:.4f}, {version.ci_97_5:.4f}] | Version structure remains an explicit qualification of the lyric increment |

No new model family, feature family, target, data source, hyperparameter grid or validation split was searched.
"""


def main() -> None:
    """Run the targeted update, validate it, and publish new versioned outputs."""

    ensure_new_targets()
    existing_manifest_check = verify_existing_manifest()
    if existing_manifest_check["status"] != "PASS":
        raise AssertionError(f"Historical manifest check failed: {existing_manifest_check}")
    historical_before = directory_hashes(HISTORICAL_DIRS)

    data, assignments, temporal_assignments = load_model_data()
    predictions, fit_warnings = grouped_predictions(data)
    aggregate, folds, h3_components = summarise_grouped(predictions)
    historical = historical_comparison(aggregate)
    uncertainty = lexical_uncertainty(data, predictions)
    temporal_prediction, temporal_metrics, temporal_audit = temporal_predictions(data)
    related, related_predictions = related_version_sensitivity(data)
    lz_summary, z_distribution = l_z_descriptives(data)
    reconciliation, reconciliation_summary = cohort_reconciliation()
    specs = model_specifications()
    leak = leakage_audit()
    calibration = calibration_table(predictions)

    if not set(predictions.model_id) == {"F1_YEAR", "F3_YEAR", "LEX_YEAR", "H3_YEAR"}:
        raise AssertionError("Unexpected year-controlled model set.")
    if assignments.primary_group_component.nunique() != 276:
        raise AssertionError("Protected group count changed during fitting.")
    if int((temporal_assignments.temporal_partition == "test_2020").sum()) != 45:
        raise AssertionError("The Y52 2020 temporal partition is no longer N=45.")
    if not (fit_warnings.status.isin(["ok", "warning"])).all():
        raise AssertionError("An unexpected model fit status occurred.")
    if predictions.y_observed.min() != 0 or predictions.y_observed.max() != 48:
        raise AssertionError("Prediction outcomes no longer have Y52 support 0–48.")

    stage_root = Path(tempfile.mkdtemp(prefix="spotify_supervisor_confirmed_update_"))
    stage_data = stage_root / "data"
    stage_tables = stage_root / "outputs/thesis_tables"
    stage_latex = stage_root / "outputs/latex_tables"
    stage_figures = stage_root / "outputs/figures"
    stage_docs = stage_root / "docs"
    for directory in [stage_data, stage_tables, stage_latex, stage_figures, stage_docs]:
        directory.mkdir(parents=True, exist_ok=True)

    binary_components = h3_components[h3_components.component.eq("binary_future_presence")].copy()
    positive_components = h3_components[h3_components.component.eq("positive_duration")].copy()
    grouped_bound = aggregate[[
        "model_id", "n", "lower_bound_violation_n", "upper_bound_violation_n",
        "outside_bound_n", "outside_bound_pct",
    ]].copy()
    grouped_files: dict[str, pd.DataFrame] = {
        "supervisor_confirmed_model_specifications.csv": specs,
        "entry_year_encoding_audit.csv": temporal_audit,
        "year_controlled_grouped_cv_predictions.csv": predictions,
        "year_controlled_grouped_cv_metrics.csv": aggregate,
        "year_controlled_grouped_cv_fold_metrics.csv": folds,
        "year_controlled_bound_diagnostics.csv": grouped_bound,
        "historical_vs_year_controlled_comparison.csv": historical,
        "year_controlled_lexical_uncertainty.csv": uncertainty,
        "year_controlled_temporal_predictions.csv": temporal_prediction,
        "year_controlled_temporal_metrics.csv": temporal_metrics,
        "year_controlled_h3_binary_metrics.csv": binary_components,
        "year_controlled_h3_positive_count_metrics.csv": positive_components,
        "year_controlled_h3_calibration.csv": calibration,
        "year_controlled_related_version_predictions.csv": related_predictions,
        "year_controlled_related_version_sensitivity.csv": related,
        "L_Z_descriptives.csv": lz_summary,
        "Z_distribution.csv": z_distribution,
        "survival1357_to_y52_1014_reconciliation.csv": reconciliation,
        "survival1357_to_y52_1014_summary.csv": reconciliation_summary,
        "supervisor_confirmed_leakage_audit.csv": leak,
        "supervisor_confirmed_model_fit_warnings.csv": fit_warnings,
    }
    for name, frame in grouped_files.items():
        frame.to_csv(stage_data / name, index=False)

    thesis_tables = make_thesis_tables(
        aggregate,
        historical,
        uncertainty,
        temporal_metrics,
        lz_summary,
        z_distribution,
        reconciliation_summary,
        related,
    )
    for name, frame in thesis_tables.items():
        frame.to_csv(stage_tables / name, index=False)
    write_latex_fragments(thesis_tables, stage_latex)

    # One compact descriptive figure is retained because the mass at Z=0 and
    # the small multiple-block tail are clearer visually than in prose alone.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(z_distribution.Z_count_52.astype(str), z_distribution.tracks, color="#4C78A8")
    ax.set(
        title="Separate chart-presence blocks in weeks 5–52",
        xlabel="Z: number of presence blocks",
        ylabel="Tracks",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(stage_figures / "z_presence_blocks_y52.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    display_order = ["F1_YEAR", "F3_YEAR", "LEX_YEAR", "H3_YEAR"]
    plot_data = aggregate.set_index("model_id").loc[display_order].reset_index()
    x = np.arange(len(plot_data))
    width = 0.36
    ax.bar(x - width / 2, plot_data.mae, width, label="MAE", color="#4C78A8")
    ax.bar(x + width / 2, plot_data.rmse, width, label="RMSE", color="#F58518")
    ax.set_xticks(x, display_order)
    ax.set(ylabel="Weeks", title="Supervisor-confirmed year-controlled model sequence")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(stage_figures / "year_controlled_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    spec_text = build_specification_document(
        aggregate, historical, uncertainty, temporal_metrics, lz_summary, related, reconciliation_summary
    )
    memo_text = build_feedback_memo(aggregate, uncertainty, lz_summary, related)
    (stage_docs / SPEC_DOC.name).write_text(spec_text, encoding="utf-8")
    (stage_docs / MEMO_DOC.name).write_text(memo_text, encoding="utf-8")

    # Validate every saved table by reading it back. This catches malformed CSV
    # output before the staged directory is published.
    for path in sorted([*stage_data.glob("*.csv"), *stage_tables.glob("*.csv")]):
        pd.read_csv(path)

    historical_after = directory_hashes(HISTORICAL_DIRS)
    changed = sorted(
        name for name in set(historical_before) | set(historical_after)
        if historical_before.get(name) != historical_after.get(name)
    )
    if changed:
        raise AssertionError(f"Historical files changed during the update: {changed}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "supervisor-confirmed design; targeted additions only",
        "supervisor_feedback_date": "2026-09-05",
        "random_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "existing_historical_manifest_check": existing_manifest_check,
        "historical_files_hash_guarded": len(historical_before),
        "historical_files_changed": changed,
        "cohort": {"Y52_550k_n": 1014, "Y26_550k_n": 1220, "protected_components": 276},
        "entry_year": {
            "encoding": "categorical OneHotEncoder(drop=first, handle_unknown=ignore)",
            "selected_by_performance": False,
            "temporal_unseen_handling": "all-zero fitted reference vector",
        },
        "model_variables": MODEL_VARIABLES,
        "grouped_metrics": aggregate.set_index("model_id").to_dict(orient="index"),
        "L_Z": lz_summary.set_index("measure").value.to_dict(),
        "cohort_reconciliation_equation": "1357 - 198 - 75 - 80 + 10 = 1014",
        "scope_assertions": {
            "no_new_model_search": True,
            "no_new_feature_search": True,
            "no_hyperparameter_search": True,
            "L_Z_excluded_from_predictors": True,
            "frozen_grouped_folds_reused": True,
            "frozen_temporal_partitions_reused": True,
            "related_version_rule_unchanged": True,
        },
    }
    staged_paths = [path for path in stage_root.rglob("*") if path.is_file()]
    manifest["staged_sha256"] = {
        str(path.relative_to(stage_root)): sha256(path) for path in sorted(staged_paths)
    }
    (stage_data / "supervisor_confirmed_update_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.parent.mkdir(parents=True, exist_ok=True)
    SPEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stage_data, OUT_DIR)
    shutil.copytree(stage_root / "outputs", PRESENTATION_DIR)
    shutil.copy2(stage_docs / SPEC_DOC.name, SPEC_DOC)
    shutil.copy2(stage_docs / MEMO_DOC.name, MEMO_DOC)
    shutil.rmtree(stage_root)

    print(json.dumps({
        "status": "complete",
        "data_output": str(OUT_DIR.relative_to(ROOT)),
        "presentation_output": str(PRESENTATION_DIR.relative_to(ROOT)),
        "specification": str(SPEC_DOC.relative_to(ROOT)),
        "feedback_memo": str(MEMO_DOC.relative_to(ROOT)),
        "metrics": aggregate.set_index("model_id")[["mae", "rmse"]].to_dict(orient="index"),
        "historical_files_checked": len(historical_before),
    }, indent=2))


if __name__ == "__main__":
    main()
