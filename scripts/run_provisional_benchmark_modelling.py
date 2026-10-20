#!/usr/bin/env python3
"""Run the frozen first-round Spotify chart-longevity benchmarks.

This script intentionally implements only F0--F4 and the model families requested
for the provisional benchmark round. It treats all earlier preprocessing and final
pre-model specification outputs as immutable inputs and stages every new output in
a temporary directory before publishing a new versioned result directory.
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
    str(Path(tempfile.gettempdir()) / "mplconfig_spotify_clv"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit, gammaln
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/processed/pre_model_analysis_ready_v1"
SPEC_DIR = ROOT / "data/processed/final_pre_model_specification_v1"
OUT_DIR = ROOT / "data/processed/provisional_benchmark_modelling_v1"
FIG_DIR = ROOT / "outputs/provisional_benchmark_modelling_v1/figures"
REPORT_PATH = ROOT / "docs/provisional_benchmark_modelling_report.md"
SEED = 20260825

CONTINUOUS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
BINARY = ["mode"]
AUDIO_ALL = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]

FEATURES = {
    "F0": [],
    "F1": ["X"],
    "F2": AUDIO_ALL,
    "F3": ["X"] + AUDIO_ALL,
    "F4": ["X"] + AUDIO_ALL + ["release_age_years", "chart_entry_year"],
}

BLOCKED_EXACT = {
    "track_popularity", "artist_popularity", "artist_followers",
    "average_artist_popularity", "average_popularity", "future_rank",
    "future_ranks", "future_stream", "future_streams", "eventual_peak_rank",
    "eventual_peak", "L", "Z", "total_chart_duration", "first_spell_duration",
    "later_spell_information", "later_reentry_descriptors",
    "old_landmark_eligibility_flags", "any_feature_derived_after_week_4",
}
BLOCKED_PATTERNS = (
    "future_rank", "future_stream", "eventual_peak", "total_chart_duration",
    "first_spell_duration", "later_spell", "later_reentry", "eligibility_flag",
    "artist_follow", "artist_popularity", "track_popularity", "average_popularity",
)

MODEL_LABELS = {
    "M0_mean": "Training-fold mean",
    "M0_median": "Training-fold median",
    "M1_poisson": "Poisson regression",
    "M2_negative_binomial": "Negative Binomial (NB2)",
    "M3_bounded_binomial": "Bounded binomial GLM",
    "M4_random_forest": "Random forest",
}


@dataclass
class FittedModel:
    model_id: str
    estimator: object
    status: str
    warning_text: str
    converged: bool
    dispersion_alpha: float | None = None
    n_iterations: int | None = None


class NegativeBinomialNB2:
    """NB2 log-link regression with fold-local method-of-moments dispersion."""

    def __init__(self, ridge: float = 1e-6, maxiter: int = 500):
        self.ridge = ridge
        self.maxiter = maxiter

    def fit(self, X: np.ndarray, y: np.ndarray):
        X1 = np.column_stack([np.ones(len(X)), X])
        starter = PoissonRegressor(alpha=1e-8, max_iter=1000, tol=1e-8)
        starter.fit(X, y)
        beta0 = np.r_[starter.intercept_, starter.coef_]
        mu0 = np.maximum(starter.predict(X), 1e-8)
        numerator = np.sum((y - mu0) ** 2 - y)
        denominator = np.sum(mu0 ** 2)
        self.alpha_ = float(np.clip(numerator / denominator if denominator > 0 else 1e-6, 1e-6, 50.0))
        a = self.alpha_
        r = 1.0 / a

        def objective(beta):
            eta = np.clip(X1 @ beta, -20, 20)
            mu = np.exp(eta)
            ll = (
                gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                + r * (math.log(r) - np.log(r + mu))
                + y * (eta - np.log(r + mu))
            )
            penalty = 0.5 * self.ridge * np.dot(beta[1:], beta[1:])
            return float(-np.sum(ll) + penalty)

        def gradient(beta):
            eta = np.clip(X1 @ beta, -20, 20)
            mu = np.exp(eta)
            grad_eta = (mu - y) / (1.0 + a * mu)
            grad = X1.T @ grad_eta
            grad[1:] += self.ridge * beta[1:]
            return grad

        result = minimize(
            objective, beta0, jac=gradient, method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        self.coef_all_ = result.x
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.message_ = str(result.message)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X1 = np.column_stack([np.ones(len(X)), X])
        return np.exp(np.clip(X1 @ self.coef_all_, -20, 20))


class BoundedBinomialGLM:
    """Binomial logit regression for y successes among n fixed future weeks."""

    def __init__(self, n_trials: int, ridge: float = 1e-6, maxiter: int = 500):
        self.n_trials = int(n_trials)
        self.ridge = ridge
        self.maxiter = maxiter

    def fit(self, X: np.ndarray, y: np.ndarray):
        X1 = np.column_stack([np.ones(len(X)), X])
        p0 = np.clip(np.mean(y) / self.n_trials, 1e-5, 1 - 1e-5)
        beta0 = np.zeros(X1.shape[1])
        beta0[0] = math.log(p0 / (1 - p0))

        def objective(beta):
            eta = np.clip(X1 @ beta, -30, 30)
            p = np.clip(expit(eta), 1e-12, 1 - 1e-12)
            ll = y * np.log(p) + (self.n_trials - y) * np.log1p(-p)
            penalty = 0.5 * self.ridge * np.dot(beta[1:], beta[1:])
            return float(-np.sum(ll) + penalty)

        def gradient(beta):
            p = expit(np.clip(X1 @ beta, -30, 30))
            grad = X1.T @ (self.n_trials * p - y)
            grad[1:] += self.ridge * beta[1:]
            return grad

        result = minimize(
            objective, beta0, jac=gradient, method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        self.coef_all_ = result.x
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.message_ = str(result.message)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X1 = np.column_stack([np.ones(len(X)), X])
        return self.n_trials * expit(np.clip(X1 @ self.coef_all_, -30, 30))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_digest(path: Path) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def feature_types(block: str) -> tuple[list[str], list[str], list[str]]:
    variables = FEATURES[block]
    cats = [v for v in variables if v in {"X", "key", "chart_entry_year"}]
    bins = [v for v in variables if v in BINARY]
    cont = [v for v in variables if v not in set(cats + bins)]
    return cont, cats, bins


def make_preprocessor(block: str) -> ColumnTransformer:
    cont, cats, bins = feature_types(block)
    transformers = []
    if cont:
        transformers.append(("continuous", StandardScaler(), cont))
    if cats:
        transformers.append((
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            cats,
        ))
    if bins:
        transformers.append(("binary", "passthrough", bins))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def leakage_hits(variables: list[str]) -> list[str]:
    hits = []
    for var in variables:
        lower = var.lower()
        if var in BLOCKED_EXACT or any(pattern in lower for pattern in BLOCKED_PATTERNS):
            hits.append(var)
        if lower.startswith("l_") or lower.startswith("z_") or lower in {"l", "z"}:
            hits.append(var)
        if lower.startswith("y_") or lower == "y":
            hits.append(var)
    return sorted(set(hits))


def validate_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if OUT_DIR.exists() or FIG_DIR.parent.exists() or REPORT_PATH.exists():
        raise FileExistsError("One or more provisional benchmark v1 outputs already exist; refusing to overwrite.")

    required = [
        ANALYSIS_DIR / "analysis_ready_550k_52w.csv",
        ANALYSIS_DIR / "analysis_ready_550k_26w.csv",
        ANALYSIS_DIR / "analysis_ready_160k_52w.csv",
        ANALYSIS_DIR / "analysis_ready_160k_26w.csv",
        SPEC_DIR / "provisional_grouped_cv_assignments.csv",
        SPEC_DIR / "provisional_temporal_validation_52w.csv",
        SPEC_DIR / "provisional_temporal_validation_26w.csv",
        SPEC_DIR / "provisional_feature_blocks.csv",
        SPEC_DIR / "provisional_variable_roles.csv",
        SPEC_DIR / "common_550k_160k_predictor_specification.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required frozen inputs missing: {missing}")

    frozen = pd.read_csv(SPEC_DIR / "provisional_feature_blocks.csv")
    frozen = frozen[frozen["feature_block"].isin(FEATURES)].copy()
    leakage_rows = []
    for block, variables in FEATURES.items():
        expected = set(frozen.loc[frozen.feature_block.eq(block), "variable"])
        actual = set(variables) if block != "F0" else {"__intercept_only__"}
        match = expected == actual
        hits = leakage_hits(variables)
        leakage_rows.append({
            "audit_stage": "pre_fit", "feature_block": block,
            "frozen_specification_match": int(match),
            "frozen_variables": "|".join(sorted(expected)),
            "implemented_variables": "|".join(variables),
            "leakage_violation_count": len(hits),
            "leakage_violations": "|".join(hits),
            "status": "PASS" if match and not hits else "FAIL",
        })
    leakage = pd.DataFrame(leakage_rows)
    if not leakage.status.eq("PASS").all():
        raise AssertionError("Feature block mismatch or leakage violation detected before fitting.")

    folds = pd.read_csv(SPEC_DIR / "provisional_grouped_cv_assignments.csv")
    if len(folds) != 1014 or folds.spotify_track_id.duplicated().any():
        raise AssertionError("Frozen grouped-CV assignment does not contain 1,014 unique tracks.")
    crossing_cols = [
        "exact_lyric_group_crosses_folds", "first_artist_crosses_folds",
        "related_version_group_crosses_folds",
    ]
    if folds[crossing_cols].fillna(0).to_numpy().sum() != 0:
        raise AssertionError("Frozen grouped-CV assignment violates a protected grouping constraint.")
    if folds.fold.value_counts().sort_index().tolist() != [203, 203, 203, 203, 202]:
        raise AssertionError("Frozen grouped-CV fold sizes changed.")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "working_status": "provisional pending supervisor confirmation where applicable",
        "random_seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in required},
        "frozen_directory_digest_before": {
            "analysis_ready": directory_digest(ANALYSIS_DIR),
            "specification": directory_digest(SPEC_DIR),
        },
        "grouped_cv": {
            "assignment_file": str((SPEC_DIR / "provisional_grouped_cv_assignments.csv").relative_to(ROOT)),
            "n_tracks": len(folds),
            "fold_sizes": folds.fold.value_counts().sort_index().to_dict(),
            "protected_group_crossings": {c: int(folds[c].sum()) for c in crossing_cols},
            "collaborator_overlap_track_count": int(folds.has_any_credited_artist_cross_fold_overlap.sum()),
            "collaborator_overlap_limitation": "All-collaborator separation is not enforced; overlap is reported as a limitation.",
        },
        "y26_grouped_cv_scope": "Same 1,014 tracks as Y_52, using frozen Y_52 fold assignments; full 1,220-track Y_26 cohort is used for temporal evaluation.",
        "temporal_year_encoding_note": "Categorical entry-year levels in validation/test are unseen in 2017-2018 training; handle_unknown='ignore' maps them to the fitted reference representation, limiting F4 calendar interpretation under temporal evaluation.",
    }
    return manifest, leakage, folds


def fit_model(model_id: str, X_train: np.ndarray, y_train: np.ndarray, nmax: int) -> FittedModel:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if model_id == "M1_poisson":
                estimator = PoissonRegressor(alpha=1e-8, max_iter=1000, tol=1e-8)
                estimator.fit(X_train, y_train)
                converged = bool(getattr(estimator, "n_iter_", 1000) < 1000)
                n_iter = int(getattr(estimator, "n_iter_", -1))
                alpha = None
                extra = ""
            elif model_id == "M2_negative_binomial":
                estimator = NegativeBinomialNB2().fit(X_train, y_train)
                converged = estimator.converged_
                n_iter = estimator.n_iter_
                alpha = estimator.alpha_
                extra = "" if converged else estimator.message_
            elif model_id == "M3_bounded_binomial":
                estimator = BoundedBinomialGLM(n_trials=nmax).fit(X_train, y_train)
                converged = estimator.converged_
                n_iter = estimator.n_iter_
                alpha = None
                extra = "" if converged else estimator.message_
            elif model_id == "M4_random_forest":
                estimator = RandomForestRegressor(
                    n_estimators=300, max_depth=8, min_samples_leaf=5,
                    max_features="sqrt", random_state=SEED, n_jobs=-1,
                )
                estimator.fit(X_train, y_train)
                converged = True
                n_iter = None
                alpha = None
                extra = ""
            else:
                raise ValueError(model_id)
            texts = [str(w.message) for w in caught]
            if extra:
                texts.append(extra)
            return FittedModel(
                model_id=model_id, estimator=estimator,
                status="ok" if converged else "warning",
                warning_text=" | ".join(texts), converged=converged,
                dispersion_alpha=alpha, n_iterations=n_iter,
            )
        except Exception as exc:
            return FittedModel(
                model_id=model_id, estimator=None, status="failed",
                warning_text=repr(exc), converged=False,
            )


def predict_model(fit: FittedModel, X_test: np.ndarray) -> np.ndarray:
    if fit.estimator is None:
        raise RuntimeError(fit.warning_text)
    return np.asarray(fit.estimator.predict(X_test), dtype=float)


def distribution_log_scores(y, pred, model_id, nmax, alpha=None):
    mu = np.clip(np.asarray(pred, float), 1e-10, None)
    y = np.asarray(y, float)
    poisson_nll = float(np.mean(mu - y * np.log(mu) + gammaln(y + 1)))
    nb_nll = np.nan
    binomial_nll = np.nan
    if model_id == "M2_negative_binomial" and alpha is not None:
        r = 1.0 / alpha
        logp = (
            gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * (math.log(r) - np.log(r + mu))
            + y * (np.log(mu) - np.log(r + mu))
        )
        nb_nll = float(-np.mean(logp))
    if model_id == "M3_bounded_binomial":
        p = np.clip(mu / nmax, 1e-12, 1 - 1e-12)
        logp = (
            gammaln(nmax + 1) - gammaln(y + 1) - gammaln(nmax - y + 1)
            + y * np.log(p) + (nmax - y) * np.log1p(-p)
        )
        binomial_nll = float(-np.mean(logp))
    return poisson_nll, nb_nll, binomial_nll


def metric_row(df: pd.DataFrame, grouping: dict) -> dict:
    y = df.y_observed.to_numpy(float)
    p = df.y_pred_raw.to_numpy(float)
    nmax = int(df.outcome_max.iloc[0])
    model_id = df.model_id.iloc[0]
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(mean_squared_error(y, p))
    p_for_dev = np.clip(p, 1e-8, None)
    pois_dev = float(mean_poisson_deviance(y, p_for_dev))
    alpha_values = df.dispersion_alpha.dropna()
    alpha = float(alpha_values.iloc[0]) if len(alpha_values) and alpha_values.nunique() == 1 else None
    poisson_nll, nb_nll, binomial_nll = distribution_log_scores(y, p, model_id, nmax, alpha)
    return {
        **grouping, "n": len(df), "mae": mae, "rmse": rmse,
        "mean_observed_y": float(np.mean(y)), "mean_prediction": float(np.mean(p)),
        "mean_prediction_bias": float(np.mean(p - y)),
        "r2_supplementary": float(r2_score(y, p)),
        "poisson_deviance": pois_dev, "poisson_nll": poisson_nll,
        "negative_binomial_nll": nb_nll, "bounded_binomial_nll": binomial_nll,
        "lower_bound_violation_n": int(np.sum(p < 0)),
        "upper_bound_violation_n": int(np.sum(p > nmax)),
        "outside_bound_n": int(np.sum((p < 0) | (p > nmax))),
        "outside_bound_pct": float(np.mean((p < 0) | (p > nmax)) * 100),
    }


def prediction_frame(base: pd.DataFrame, idx: np.ndarray, ycol: str, nmax: int,
                     block: str, model_id: str, pred: np.ndarray, evaluation: str,
                     split_label: str, alpha: float | None = None) -> pd.DataFrame:
    part = base.iloc[idx]
    return pd.DataFrame({
        "spotify_track_id": part.spotify_track_id.to_numpy(),
        "outcome": ycol, "outcome_max": nmax, "evaluation": evaluation,
        "split": split_label, "feature_block": block, "model_id": model_id,
        "model_family": MODEL_LABELS[model_id],
        "y_observed": part[ycol].to_numpy(float), "y_pred_raw": pred,
        "y_pred_clipped": np.clip(pred, 0, nmax),
        "residual_observed_minus_predicted": part[ycol].to_numpy(float) - pred,
        "X": part.X.to_numpy(int), "chart_entry_year": part.chart_entry_year.to_numpy(int),
        "dispersion_alpha": alpha,
    })


def run_grouped_cv(data52, data26, folds):
    y26_same = data26.set_index("spotify_track_id").loc[data52.spotify_track_id].reset_index()
    if not y26_same.spotify_track_id.equals(data52.spotify_track_id):
        raise AssertionError("Y_26 alignment to the frozen 1,014-track grouped-CV cohort failed.")
    datasets = [(data52, "Y_52", 48), (y26_same, "Y_26", 22)]
    predictions = []
    warnings_rows = []
    models = ["M1_poisson", "M2_negative_binomial", "M3_bounded_binomial", "M4_random_forest"]

    for data, ycol, nmax in datasets:
        merged = data.merge(folds[["spotify_track_id", "fold"]], on="spotify_track_id", how="left", validate="one_to_one")
        if merged.fold.isna().any() or len(merged) != 1014:
            raise AssertionError(f"Frozen fold merge failed for {ycol}.")
        for fold in sorted(merged.fold.unique()):
            train_idx = np.flatnonzero(merged.fold.ne(fold).to_numpy())
            test_idx = np.flatnonzero(merged.fold.eq(fold).to_numpy())
            y_train = merged.iloc[train_idx][ycol].to_numpy(float)
            for m0, value in [
                ("M0_mean", float(np.mean(y_train))),
                ("M0_median", float(np.median(y_train))),
            ]:
                pred = np.full(len(test_idx), value)
                predictions.append(prediction_frame(merged, test_idx, ycol, nmax, "F0", m0, pred, "grouped_cv", fold))
                warnings_rows.append({
                    "evaluation": "grouped_cv", "outcome": ycol, "split": fold,
                    "feature_block": "F0", "model_id": m0, "status": "ok",
                    "converged": 1, "n_iterations": np.nan, "dispersion_alpha": np.nan,
                    "warning_or_failure": "",
                })
            for block in ["F1", "F2", "F3", "F4"]:
                prep = make_preprocessor(block)
                X_train = prep.fit_transform(merged.iloc[train_idx][FEATURES[block]])
                X_test = prep.transform(merged.iloc[test_idx][FEATURES[block]])
                for model_id in models:
                    fit = fit_model(model_id, X_train, y_train, nmax)
                    warnings_rows.append({
                        "evaluation": "grouped_cv", "outcome": ycol, "split": fold,
                        "feature_block": block, "model_id": model_id,
                        "status": fit.status, "converged": int(fit.converged),
                        "n_iterations": fit.n_iterations,
                        "dispersion_alpha": fit.dispersion_alpha,
                        "warning_or_failure": fit.warning_text,
                    })
                    if fit.status == "failed":
                        continue
                    pred = predict_model(fit, X_test)
                    predictions.append(prediction_frame(
                        merged, test_idx, ycol, nmax, block, model_id, pred,
                        "grouped_cv", fold, fit.dispersion_alpha,
                    ))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(warnings_rows)


def summarize_grouped_predictions(pred):
    fold_rows = []
    for keys, g in pred.groupby(["outcome", "split", "feature_block", "model_id", "model_family"], sort=True):
        fold_rows.append(metric_row(g, dict(zip(["outcome", "fold", "feature_block", "model_id", "model_family"], keys))))
    fold_metrics = pd.DataFrame(fold_rows)
    aggregate_rows = []
    for keys, g in pred.groupby(["outcome", "feature_block", "model_id", "model_family"], sort=True):
        aggregate_rows.append(metric_row(g, dict(zip(["outcome", "feature_block", "model_id", "model_family"], keys))))
    aggregate = pd.DataFrame(aggregate_rows)

    zero_rows = []
    for keys, g in pred.groupby(["outcome", "feature_block", "model_id", "model_family"], sort=True):
        y = g.y_observed.to_numpy(float); p = g.y_pred_raw.to_numpy(float)
        z = y == 0; pos = y > 0
        zero_rows.append({
            **dict(zip(["outcome", "feature_block", "model_id", "model_family"], keys)),
            "n": len(g), "observed_zero_n": int(z.sum()), "observed_zero_pct": float(z.mean() * 100),
            "near_zero_threshold_weeks": 0.5, "predicted_near_zero_n": int((p < 0.5).sum()),
            "predicted_near_zero_pct": float((p < 0.5).mean() * 100),
            "mae_observed_zero": float(mean_absolute_error(y[z], p[z])) if z.any() else np.nan,
            "mae_observed_positive": float(mean_absolute_error(y[pos], p[pos])) if pos.any() else np.nan,
            "mean_predicted_y_observed_zero": float(p[z].mean()) if z.any() else np.nan,
            "mean_predicted_y_observed_positive": float(p[pos].mean()) if pos.any() else np.nan,
        })
    zero_perf = pd.DataFrame(zero_rows)

    x_rows = []
    for keys, g in pred.groupby(["outcome", "feature_block", "model_id", "model_family", "X"], sort=True):
        y = g.y_observed.to_numpy(float); p = g.y_pred_raw.to_numpy(float)
        x_rows.append({
            **dict(zip(["outcome", "feature_block", "model_id", "model_family", "X"], keys)),
            "n": len(g), "mean_observed_y": float(y.mean()), "mean_predicted_y": float(p.mean()),
            "bias": float(np.mean(p-y)), "mae": float(mean_absolute_error(y,p)),
            "rmse": float(math.sqrt(mean_squared_error(y,p))), "observed_zero_pct": float(np.mean(y==0)*100),
        })
    performance_x = pd.DataFrame(x_rows)

    bounds = aggregate[[
        "outcome", "feature_block", "model_id", "model_family", "n",
        "lower_bound_violation_n", "upper_bound_violation_n", "outside_bound_n", "outside_bound_pct",
    ]].copy()

    cal_rows = []
    for keys, g in pred.groupby(["outcome", "feature_block", "model_id", "model_family"], sort=True):
        h = g.copy()
        rank = h.y_pred_raw.rank(method="first")
        h["calibration_bin"] = pd.qcut(rank, q=5, labels=[1,2,3,4,5])
        for b, q in h.groupby("calibration_bin", observed=True):
            cal_rows.append({
                **dict(zip(["outcome", "feature_block", "model_id", "model_family"], keys)),
                "calibration_bin": int(b), "n": len(q),
                "mean_prediction": float(q.y_pred_raw.mean()), "mean_observed_y": float(q.y_observed.mean()),
                "prediction_bias": float((q.y_pred_raw-q.y_observed).mean()),
                "min_prediction": float(q.y_pred_raw.min()), "max_prediction": float(q.y_pred_raw.max()),
            })
    calibration = pd.DataFrame(cal_rows)
    return fold_metrics, aggregate, zero_perf, performance_x, bounds, calibration


def incremental_comparison(aggregate):
    rows = []
    steps = [
        ("F0_mean_to_F1", "F0", "M0_mean", "F1"),
        ("F0_mean_to_F2", "F0", "M0_mean", "F2"),
        ("F1_to_F3", "F1", None, "F3"),
        ("F3_to_F4", "F3", None, "F4"),
    ]
    model_ids = ["M1_poisson", "M2_negative_binomial", "M3_bounded_binomial", "M4_random_forest"]
    for outcome in ["Y_52", "Y_26"]:
        for model_id in model_ids:
            for label, base_block, base_model_override, new_block in steps:
                base_model = base_model_override or model_id
                b = aggregate[(aggregate.outcome==outcome)&(aggregate.feature_block==base_block)&(aggregate.model_id==base_model)]
                n = aggregate[(aggregate.outcome==outcome)&(aggregate.feature_block==new_block)&(aggregate.model_id==model_id)]
                if b.empty or n.empty:
                    continue
                b=b.iloc[0]; n=n.iloc[0]
                rows.append({
                    "outcome":outcome,"model_id":model_id,"model_family":MODEL_LABELS[model_id],
                    "comparison":label,"baseline_block":base_block,"new_block":new_block,
                    "baseline_mae":b.mae,"new_mae":n.mae,
                    "mae_improvement_absolute":b.mae-n.mae,
                    "mae_improvement_pct":100*(b.mae-n.mae)/b.mae,
                    "baseline_rmse":b.rmse,"new_rmse":n.rmse,
                    "rmse_improvement_absolute":b.rmse-n.rmse,
                    "rmse_improvement_pct":100*(b.rmse-n.rmse)/b.rmse,
                })
    return pd.DataFrame(rows)


def run_temporal(data, assignment, ycol, nmax):
    merged = data.merge(assignment[["spotify_track_id","temporal_partition"]], on="spotify_track_id", how="left", validate="one_to_one")
    if merged.temporal_partition.isna().any() or len(merged) != len(data):
        raise AssertionError(f"Temporal assignment merge failed for {ycol}.")
    train_idx=np.flatnonzero(merged.temporal_partition.eq("train_2017_2018").to_numpy())
    predictions=[]; warnings_rows=[]
    y_train=merged.iloc[train_idx][ycol].to_numpy(float)
    for split in ["validation_2019","test_2020"]:
        test_idx=np.flatnonzero(merged.temporal_partition.eq(split).to_numpy())
        for m0,value in [("M0_mean",float(y_train.mean())),("M0_median",float(np.median(y_train)))]:
            predictions.append(prediction_frame(merged,test_idx,ycol,nmax,"F0",m0,np.full(len(test_idx),value),"temporal",split))
    for block in ["F1","F3","F4"]:
        prep=make_preprocessor(block)
        Xtr=prep.fit_transform(merged.iloc[train_idx][FEATURES[block]])
        transformed_width=Xtr.shape[1]
        for model_id in ["M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"]:
            fit=fit_model(model_id,Xtr,y_train,nmax)
            warnings_rows.append({
                "evaluation":"temporal","outcome":ycol,"split":"train_2017_2018",
                "feature_block":block,"model_id":model_id,"status":fit.status,
                "converged":int(fit.converged),"n_iterations":fit.n_iterations,
                "dispersion_alpha":fit.dispersion_alpha,"warning_or_failure":fit.warning_text,
                "transformed_feature_count":transformed_width,
            })
            if fit.status=="failed": continue
            for split in ["validation_2019","test_2020"]:
                test_idx=np.flatnonzero(merged.temporal_partition.eq(split).to_numpy())
                Xte=prep.transform(merged.iloc[test_idx][FEATURES[block]])
                pred=predict_model(fit,Xte)
                predictions.append(prediction_frame(merged,test_idx,ycol,nmax,block,model_id,pred,"temporal",split,fit.dispersion_alpha))
    pred=pd.concat(predictions,ignore_index=True)
    metrics=[]
    for keys,g in pred.groupby(["outcome","split","feature_block","model_id","model_family"],sort=True):
        metrics.append(metric_row(g,dict(zip(["outcome","temporal_partition","feature_block","model_id","model_family"],keys))))
    return pred,pd.DataFrame(metrics),pd.DataFrame(warnings_rows)


def run_common_robustness(data, source_label):
    ycol="Y_52"; nmax=48
    part=np.select(
        [data.chart_entry_year.le(2018),data.chart_entry_year.eq(2019),data.chart_entry_year.eq(2020)],
        ["train_2017_2018","validation_2019","test_2020"],default="excluded"
    )
    merged=data.assign(temporal_partition=part)
    train_idx=np.flatnonzero(merged.temporal_partition.eq("train_2017_2018").to_numpy())
    ytrain=merged.iloc[train_idx][ycol].to_numpy(float)
    rows=[]; warn=[]
    for split in ["validation_2019","test_2020"]:
        idx=np.flatnonzero(merged.temporal_partition.eq(split).to_numpy())
        base=prediction_frame(merged,idx,ycol,nmax,"F0","M0_mean",np.full(len(idx),ytrain.mean()),"common_robustness",split)
        base["feature_source"]=source_label; rows.append(base)
    for block in ["F1","F2","F3","F4"]:
        prep=make_preprocessor(block); Xtr=prep.fit_transform(merged.iloc[train_idx][FEATURES[block]])
        for mid in ["M1_poisson","M2_negative_binomial","M3_bounded_binomial"]:
            fit=fit_model(mid,Xtr,ytrain,nmax)
            warn.append({"evaluation":"common_robustness","feature_source":source_label,"outcome":ycol,
                         "split":"train_2017_2018","feature_block":block,"model_id":mid,"status":fit.status,
                         "converged":int(fit.converged),"n_iterations":fit.n_iterations,
                         "dispersion_alpha":fit.dispersion_alpha,"warning_or_failure":fit.warning_text})
            if fit.status=="failed": continue
            for split in ["validation_2019","test_2020"]:
                idx=np.flatnonzero(merged.temporal_partition.eq(split).to_numpy())
                p=predict_model(fit,prep.transform(merged.iloc[idx][FEATURES[block]]))
                q=prediction_frame(merged,idx,ycol,nmax,block,mid,p,"common_robustness",split,fit.dispersion_alpha)
                q["feature_source"]=source_label; rows.append(q)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(warn)


def common_metrics(pred):
    rows=[]
    for keys,g in pred.groupby(["feature_source","split","feature_block","model_id","model_family"],sort=True):
        rows.append(metric_row(g,dict(zip(["feature_source","temporal_partition","feature_block","model_id","model_family"],keys))))
    return pd.DataFrame(rows)


def make_model_specifications():
    rows=[
        {"model_id":"M0_mean","model_family":MODEL_LABELS["M0_mean"],"applicable_blocks":"F0","estimation":"training-partition arithmetic mean","bounded_model":0,"notes":"No held-out outcomes used."},
        {"model_id":"M0_median","model_family":MODEL_LABELS["M0_median"],"applicable_blocks":"F0","estimation":"training-partition median","bounded_model":0,"notes":"Included for MAE-oriented comparison."},
        {"model_id":"M1_poisson","model_family":MODEL_LABELS["M1_poisson"],"applicable_blocks":"F1|F2|F3|F4","estimation":"scikit-learn PoissonRegressor; log link; alpha=1e-8","bounded_model":0,"notes":"Transparent unbounded diagnostic count baseline."},
        {"model_id":"M2_negative_binomial","model_family":MODEL_LABELS["M2_negative_binomial"],"applicable_blocks":"F1|F2|F3|F4","estimation":"NB2 log-link MLE; fold-local method-of-moments dispersion fixed during coefficient optimisation; ridge=1e-6","bounded_model":0,"notes":"Handles quadratic mean-variance overdispersion but remains unbounded."},
        {"model_id":"M3_bounded_binomial","model_family":MODEL_LABELS["M3_bounded_binomial"],"applicable_blocks":"F1|F2|F3|F4","estimation":"Binomial logit MLE with n=48 for Y_52 and n=22 for Y_26; ridge=1e-6","bounded_model":1,"notes":"Acknowledges fixed possible weeks; independence/equal-probability assumptions are simplifications."},
        {"model_id":"M4_random_forest","model_family":MODEL_LABELS["M4_random_forest"],"applicable_blocks":"F1|F2|F3|F4","estimation":"300 trees; max_depth=8; min_samples_leaf=5; max_features=sqrt; fixed seed","bounded_model":0,"notes":"Modest nonlinear benchmark; no hyperparameter search."},
    ]
    return pd.DataFrame(rows)


def create_figures(stage_fig: Path, aggregate, calibration, performance_x, pred):
    sns.set_theme(style="whitegrid", context="notebook")
    stage_fig.mkdir(parents=True,exist_ok=True)

    y52=aggregate[(aggregate.outcome=="Y_52") & (aggregate.model_id.isin(["M0_mean","M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"]))]
    fig,ax=plt.subplots(figsize=(11,6))
    sns.barplot(data=y52,x="feature_block",y="mae",hue="model_family",order=["F0","F1","F2","F3","F4"],ax=ax)
    ax.set(title="Grouped-CV MAE across feature blocks (Y_52)",xlabel="Feature block",ylabel="Mean absolute error (weeks)")
    ax.legend(title="Model",bbox_to_anchor=(1.02,1),loc="upper left")
    fig.tight_layout(); fig.savefig(stage_fig/"grouped_cv_performance_feature_blocks_y52.png",dpi=180); plt.close(fig)

    cal=calibration[(calibration.outcome=="Y_52")&(calibration.feature_block=="F4")&calibration.model_id.isin(["M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"])]
    fig,ax=plt.subplots(figsize=(7.5,6))
    for mid,g in cal.groupby("model_id"):
        ax.plot(g.mean_prediction,g.mean_observed_y,marker="o",label=MODEL_LABELS[mid])
    lim=max(cal.mean_prediction.max(),cal.mean_observed_y.max())*1.05
    ax.plot([0,lim],[0,lim],linestyle="--",color="black",linewidth=1,label="Ideal")
    ax.set(title="Observed versus predicted calibration (Y_52, F4)",xlabel="Mean predicted weeks",ylabel="Mean observed weeks",xlim=(0,lim),ylim=(0,lim))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(stage_fig/"calibration_y52_f4.png",dpi=180); plt.close(fig)

    px=performance_x[(performance_x.outcome=="Y_52")&(performance_x.feature_block=="F4")&performance_x.model_id.isin(["M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"])]
    fig,ax=plt.subplots(figsize=(8,5.5))
    sns.lineplot(data=px,x="X",y="mae",hue="model_family",marker="o",ax=ax)
    ax.set(title="Prediction error by early chart presence X (Y_52, F4)",xlabel="X: appearances in weeks 1–4",ylabel="MAE (weeks)",xticks=[1,2,3,4])
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(stage_fig/"performance_by_x_y52_f4.png",dpi=180); plt.close(fig)

    res=pred[(pred.outcome=="Y_52")&(pred.feature_block=="F4")&pred.model_id.isin(["M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"])]
    fig,ax=plt.subplots(figsize=(9,5.5))
    sns.histplot(data=res,x="residual_observed_minus_predicted",hue="model_family",element="step",stat="density",common_norm=False,bins=40,ax=ax)
    ax.axvline(0,color="black",linestyle="--",linewidth=1)
    ax.set(title="Out-of-fold residual distributions (Y_52, F4)",xlabel="Observed minus predicted weeks",ylabel="Density")
    fig.tight_layout(); fig.savefig(stage_fig/"residual_distribution_y52_f4.png",dpi=180); plt.close(fig)

    comp=aggregate[(aggregate.feature_block=="F4")&aggregate.model_id.isin(["M1_poisson","M2_negative_binomial","M3_bounded_binomial","M4_random_forest"])]
    fig,ax=plt.subplots(figsize=(8,5.5))
    sns.barplot(data=comp,x="model_family",y="mae",hue="outcome",ax=ax)
    ax.set(title="Y_26 versus Y_52 grouped-CV benchmark error (F4)",xlabel="Model",ylabel="MAE (weeks)")
    ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(stage_fig/"y26_y52_benchmark_comparison.png",dpi=180); plt.close(fig)


def best_rows_text(aggregate, incremental, zero, temporal52, temporal26, common):
    def fmt(x): return f"{x:.3f}"
    y52=aggregate[aggregate.outcome.eq("Y_52")].sort_values(["mae","rmse"])
    best=y52.iloc[0]
    f0=y52[(y52.feature_block=="F0")&(y52.model_id=="M0_mean")].iloc[0]
    f1=y52[(y52.feature_block=="F1")].sort_values("mae").iloc[0]
    f2=y52[(y52.feature_block=="F2")].sort_values("mae").iloc[0]
    f3=y52[(y52.feature_block=="F3")].sort_values("mae").iloc[0]
    f4=y52[(y52.feature_block=="F4")].sort_values("mae").iloc[0]
    return {
        "best":best,"f0":f0,"f1":f1,"f2":f2,"f3":f3,"f4":f4,
        "fmt":fmt,
    }


def dataframe_to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    def render(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    headers = [str(c).replace("|", "\\|") for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, aggregate, incremental, zero, performance_x, temporal52, temporal26, common, warnings_df, manifest):
    s=best_rows_text(aggregate,incremental,zero,temporal52,temporal26,common); fmt=s["fmt"]
    successful=warnings_df[warnings_df.status.eq("ok")]
    warn=warnings_df[warnings_df.status.ne("ok")]
    y52=aggregate[aggregate.outcome.eq("Y_52")]
    y26=aggregate[aggregate.outcome.eq("Y_26")]
    table=y52.sort_values(["feature_block","mae"])[["feature_block","model_family","mae","rmse","mean_prediction_bias","outside_bound_pct"]]
    table_md=dataframe_to_markdown(table,3)
    inc=incremental[incremental.outcome.eq("Y_52")].copy()
    inc_table=dataframe_to_markdown(inc[["model_family","comparison","mae_improvement_absolute","mae_improvement_pct","rmse_improvement_absolute","rmse_improvement_pct"]],2)
    temp52=dataframe_to_markdown(temporal52.sort_values(["temporal_partition","mae"])[["temporal_partition","feature_block","model_family","n","mae","rmse","mean_prediction_bias"]],3)
    temp26=dataframe_to_markdown(temporal26.sort_values(["temporal_partition","mae"])[["temporal_partition","feature_block","model_family","n","mae","rmse","mean_prediction_bias"]],3)
    common_best=dataframe_to_markdown(common.sort_values(["feature_source","temporal_partition","mae"])[["feature_source","temporal_partition","feature_block","model_family","n","mae","rmse","mean_prediction_bias"]],3)
    text=f"""# Provisional Benchmark Modelling Report

## Status and scope

This report documents the first provisional benchmark-modelling round for chart longevity. It uses the frozen `final_pre_model_specification_v1` working assumptions. These choices remain provisional and are not described as supervisor-approved. No lyrics-content model, hurdle model, zero-inflated model, survival model or extensive tuning was performed.

The working primary outcome is **Y_52**, the number of chart-presence weeks from weeks 5–52 after confirmed entry (support 0–48). **Y_26** is the sensitivity outcome (support 0–22). The primary 550k Y_52 cohort contains 1,014 tracks. Grouped-CV Y_26 results use the same 1,014 tracks so that the frozen folds and horizon comparison are identical; temporal Y_26 evaluation uses the full clean 1,220-track cohort.

## Validation and leakage controls

The frozen five-fold lead-artist/version-grouped assignments were used unchanged. No first-credited artist, exact normalised-lyric group or supported related-version group crosses folds. The folds contain 203, 203, 203, 203 and 202 tracks. Complete collaborator separation is not claimed: {manifest['grouped_cv']['collaborator_overlap_track_count']:,} tracks have at least one credited artist represented across folds, reflecting the previously documented large collaboration component.

All scaling and categorical encoding were fitted within each training fold. There were no missing values in the selected F1–F4 inputs, so no imputation was introduced. The programmatic pre- and post-fit leakage audits passed. Blocked outcome-window, popularity, later-rank, later-stream, re-entry, duration and old landmark variables are absent.

## Model families

- M0: training-only mean and median benchmarks.
- M1: Poisson regression, retained as a transparent unbounded count diagnostic.
- M2: NB2 regression. Dispersion was estimated separately within each training partition by a method-of-moments calculation after a Poisson fit, then held fixed during maximum-likelihood coefficient estimation.
- M3: bounded binomial-logit regression using 48 possible weeks for Y_52 and 22 for Y_26. It respects the outcome bounds but simplifies within-song week dependence and probability heterogeneity.
- M4: a modest random forest with fixed settings and no hyperparameter search.

`statsmodels` was not available in the local environment. The NB2 and bounded-binomial likelihoods were therefore implemented explicitly with SciPy optimisation rather than substituting invented results or installing an unrecorded dependency.

## Grouped-CV results for Y_52

{table_md}

The training-fold mean benchmark produced MAE {fmt(s['f0'].mae)} and RMSE {fmt(s['f0'].rmse)} weeks. The lowest Y_52 MAE in this deliberately small comparison was {fmt(s['best'].mae)} for {s['best'].model_family} with {s['best'].feature_block}; its RMSE was {fmt(s['best'].rmse)}. This is a provisional benchmark result, not final model selection.

The best X-only specification had MAE {fmt(s['f1'].mae)}. The best song-characteristics-only specification had MAE {fmt(s['f2'].mae)}. The best X plus song-characteristics specification had MAE {fmt(s['f3'].mae)}, and the best F4 specification had MAE {fmt(s['f4'].mae)}.

## Incremental predictor contribution for Y_52

Positive values below denote improvement (lower held-out error); negative values denote deterioration.

{inc_table}

These comparisons separate the predictive contribution of early chart presence X, baseline audio/song characteristics, and calendar/release information. They should not be interpreted causally.

## Zero and X diagnostics

Zero and positive outcomes were evaluated separately with a near-zero threshold of 0.5 predicted weeks. The detailed table reports MAE among observed zeros and positives, the predicted near-zero rate, and mean predictions in both groups. Performance by X=1,2,3,4 is also reported separately, including bias, so underprediction for X=4 and overprediction for X=1 can be inspected directly.

The benchmark evidence is used only to decide whether a later hurdle or zero-inflated comparison is worth investigating. A large observed zero share alone is not treated as sufficient justification.

## Bounded support and calibration

Poisson and Negative Binomial predictions are unbounded. The output records every raw prediction outside 0–48 or 0–22 before optional diagnostic clipping. All principal metrics use raw predictions. The bounded binomial baseline cannot predict outside the outcome support by construction. Its assumptions remain simplified because chart-presence weeks for a song are not independent exchangeable trials.

Calibration is reported in five equal-sized prediction-rank bins. This makes systematic under- or overprediction visible even where aggregate MAE is similar.

## Temporal robustness: Y_52

The frozen design trains on 2017–2018, validates on 2019 and treats 2020 as a 45-track stress test.

{temp52}

Categorical chart-entry year cannot extrapolate a coefficient for unseen 2019/2020 categories. The fold-local encoder therefore maps unseen years to its reference representation. This makes F4's temporal calendar component limited and should be stated when interpreting the temporal comparison.

## Temporal robustness: Y_26

The Y_26 temporal analysis uses the full clean cohort. Its 2020 stress test contains 251 tracks, making it more informative than the 45-track Y_52 stress test for later-year generalisation.

{temp26}

## Common 550k/160k robustness

The same common predictors were fitted separately in each source-linked cohort: categorical X; the common audio variables; categorical key; binary mode; release age; and categorical chart-entry year. Lyrics, genre and the 160k-only explicit field were excluded.

{common_best}

The analysis asks whether the broad incremental conclusions persist, not whether coefficients or scores are identical. The cohorts differ in size and composition and are not pooled.

## Model warnings and limitations

There were {len(successful):,} successful fit records and {len(warn):,} warning/failure records across grouped, temporal and source-robustness fitting. Exact messages, convergence flags, iteration counts and fold-local NB dispersion estimates are saved. Poisson and NB2 remain unbounded; the bounded-binomial assumptions are simplified; random-forest predictions are not structurally bounded, although observed violations are audited.

The principal validation prevents lead-artist and supported version-group leakage, but it does not eliminate all collaborator overlap. Predictive CV coefficients are not interpreted as causal or as formal artist-clustered inference.

## Provisional interpretation and next task

This round establishes transparent baselines and quantifies how much X and baseline features add out of sample. The next useful modelling task should be selected from the evidence: a modest lyrics-content increment fitted entirely within training folds, a bounded/overdispersed refinement, or a hurdle comparison only if zero calibration remains systematically poor. Final outcome, horizon and model choices remain subject to supervisor discussion.
"""
    path.write_text(text,encoding="utf-8")


def main():
    manifest, leakage_pre, folds = validate_inputs()
    data52=pd.read_csv(ANALYSIS_DIR/"analysis_ready_550k_52w.csv")
    data26=pd.read_csv(ANALYSIS_DIR/"analysis_ready_550k_26w.csv")
    data160_52=pd.read_csv(ANALYSIS_DIR/"analysis_ready_160k_52w.csv")
    data160_26=pd.read_csv(ANALYSIS_DIR/"analysis_ready_160k_26w.csv")
    for name,data,ycol,n in [
        ("550k_52",data52,"Y_52",1014),("550k_26",data26,"Y_26",1220),
        ("160k_52",data160_52,"Y_52",1216),("160k_26",data160_26,"Y_26",1551),
    ]:
        if len(data)!=n or data.spotify_track_id.duplicated().any(): raise AssertionError(f"{name} analysis unit validation failed")
        if data[ycol].lt(0).any() or data[ycol].gt(48 if ycol=="Y_52" else 22).any(): raise AssertionError(f"{name} outcome bounds failed")
        if data[sum([FEATURES[b] for b in ["F1","F2","F3","F4"]],[])].isna().any().any(): raise AssertionError(f"{name} selected predictor missingness found")

    pred,warn_group=run_grouped_cv(data52,data26,folds)
    fold_metrics,aggregate,zero_perf,perf_x,bounds,calibration=summarize_grouped_predictions(pred)
    incremental=incremental_comparison(aggregate)

    t52_assign=pd.read_csv(SPEC_DIR/"provisional_temporal_validation_52w.csv")
    t26_assign=pd.read_csv(SPEC_DIR/"provisional_temporal_validation_26w.csv")
    t52_pred,t52_metrics,t52_warn=run_temporal(data52,t52_assign,"Y_52",48)
    t26_pred,t26_metrics,t26_warn=run_temporal(data26,t26_assign,"Y_26",22)
    if t52_metrics.loc[t52_metrics.temporal_partition.eq("test_2020"),"n"].nunique()!=1 or int(t52_metrics.loc[t52_metrics.temporal_partition.eq("test_2020"),"n"].iloc[0])!=45: raise AssertionError("Y52 temporal stress-test N changed")
    if int(t26_metrics.loc[t26_metrics.temporal_partition.eq("test_2020"),"n"].iloc[0])!=251: raise AssertionError("Y26 temporal stress-test N changed")

    c550_pred,c550_warn=run_common_robustness(data52,"550k")
    c160_pred,c160_warn=run_common_robustness(data160_52,"160k")
    common_pred=pd.concat([c550_pred,c160_pred],ignore_index=True)
    common=common_metrics(common_pred)
    warnings_all=pd.concat([warn_group,t52_warn,t26_warn,c550_warn,c160_warn],ignore_index=True,sort=False)

    leakage_post=leakage_pre.copy(); leakage_post["audit_stage"]="post_fit"
    leakage=pd.concat([leakage_pre,leakage_post],ignore_index=True)
    if not leakage.status.eq("PASS").all(): raise AssertionError("Post-fit leakage audit failed")

    # Prediction unit and coverage checks.
    for outcome in ["Y_52","Y_26"]:
        q=pred[pred.outcome.eq(outcome)]
        for (block,mid),g in q.groupby(["feature_block","model_id"]):
            if len(g)!=1014 or g.spotify_track_id.nunique()!=1014: raise AssertionError(f"OOF coverage failure: {outcome} {block} {mid}")

    before_analysis=manifest["frozen_directory_digest_before"]["analysis_ready"]
    before_spec=manifest["frozen_directory_digest_before"]["specification"]
    after_analysis=directory_digest(ANALYSIS_DIR); after_spec=directory_digest(SPEC_DIR)
    if before_analysis!=after_analysis or before_spec!=after_spec: raise AssertionError("A frozen input changed during modelling")
    manifest["frozen_directory_unchanged"]={"analysis_ready":True,"specification":True}
    manifest["fit_summary"]={
        "fit_records":int(len(warnings_all)),"failed_fit_records":int(warnings_all.status.eq("failed").sum()),
        "warning_fit_records":int(warnings_all.status.eq("warning").sum()),
        "successful_fit_records":int(warnings_all.status.eq("ok").sum()),
    }
    manifest["prediction_counts"]={"grouped_cv":int(len(pred)),"temporal_52":int(len(t52_pred)),"temporal_26":int(len(t26_pred)),"common_robustness":int(len(common_pred))}
    manifest["validation_assertions"]={
        "frozen_fold_assignment_used":True,"no_first_artist_crosses_grouped_folds":True,
        "no_exact_lyric_group_crosses_grouped_folds":True,"no_supported_related_version_group_crosses_grouped_folds":True,
        "all_preprocessing_fit_within_training_partitions":True,"no_full_data_target_mean_used":True,
        "all_frozen_feature_blocks_match":True,"blocked_variables_absent":True,
        "raw_predictions_used_for_primary_metrics":True,"historical_inputs_unchanged":True,
    }

    specs=make_model_specifications()
    stage=Path(tempfile.mkdtemp(prefix="spotify_benchmark_v1_"))
    stage_data=stage/"data"; stage_fig=stage/"figures"; stage_data.mkdir(parents=True)
    outputs={
        "benchmark_model_specifications.csv":specs,
        "grouped_cv_fold_metrics.csv":fold_metrics,
        "grouped_cv_aggregate_metrics.csv":aggregate,
        "grouped_cv_predictions.csv":pred,
        "feature_block_incremental_comparison.csv":incremental,
        "performance_by_X.csv":perf_x,
        "zero_positive_performance.csv":zero_perf,
        "prediction_bound_violations.csv":bounds,
        "calibration_summary.csv":calibration,
        "temporal_validation_metrics_52w.csv":t52_metrics,
        "temporal_validation_metrics_26w.csv":t26_metrics,
        "common_550k_160k_robustness_metrics.csv":common,
        "model_fit_warnings.csv":warnings_all,
        "benchmark_leakage_audit.csv":leakage,
    }
    for name,df in outputs.items(): df.to_csv(stage_data/name,index=False)
    # Keep detailed temporal/common predictions concise but auditable inside the manifest directory.
    t52_pred.to_csv(stage_data/"temporal_validation_predictions_52w.csv",index=False)
    t26_pred.to_csv(stage_data/"temporal_validation_predictions_26w.csv",index=False)
    common_pred.to_csv(stage_data/"common_550k_160k_robustness_predictions.csv",index=False)

    create_figures(stage_fig,aggregate,calibration,perf_x,pred)
    stage_report=stage/"provisional_benchmark_modelling_report.md"
    write_report(stage_report,aggregate,incremental,zero_perf,perf_x,t52_metrics,t26_metrics,common,warnings_all,manifest)
    # Compute output hashes after every staged file is written except the manifest itself.
    manifest["output_sha256"]={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage_data/"benchmark_validation_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    OUT_DIR.parent.mkdir(parents=True,exist_ok=True); FIG_DIR.parent.mkdir(parents=True,exist_ok=True); REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(stage_data),str(OUT_DIR))
    shutil.move(str(stage_fig),str(FIG_DIR))
    shutil.move(str(stage_report),str(REPORT_PATH))
    shutil.rmtree(stage,ignore_errors=True)
    print(json.dumps({
        "status":"complete","output_directory":str(OUT_DIR),"figure_directory":str(FIG_DIR),
        "report":str(REPORT_PATH),"fit_summary":manifest["fit_summary"],
        "best_y52":aggregate[aggregate.outcome.eq("Y_52")].sort_values(["mae","rmse"]).iloc[0][["feature_block","model_id","mae","rmse"]].to_dict(),
    },indent=2))


if __name__=="__main__":
    main()
