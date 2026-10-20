#!/usr/bin/env python3
"""Focused second-round lyric and hurdle benchmarks for Spotify chart longevity.

The script uses the immutable 1,014-track Y_52 cohort and frozen grouped folds.
Every text vocabulary, IDF, tabular scaler, encoder, and regularisation decision is
fit inside the relevant training data. New outputs are staged in the system temporary directory and
published only after all leakage and reproducibility assertions pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
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
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, gammaln
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/processed/pre_model_analysis_ready_v1"
SPEC_DIR = ROOT / "data/processed/final_pre_model_specification_v1"
BENCH_DIR = ROOT / "data/processed/provisional_benchmark_modelling_v1"
OUT_DIR = ROOT / "data/processed/provisional_lyrics_hurdle_modelling_v1"
FIG_DIR = ROOT / "outputs/provisional_lyrics_hurdle_modelling_v1/figures"
REPORT_PATH = ROOT / "docs/provisional_lyrics_hurdle_modelling_report.md"
SEED = 20260825

YCOL = "Y_52"
NMAX = 48
ALPHA_GRID = [0.01, 0.1, 1.0]

AUDIO = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]
F1 = ["X"]
F3 = ["X"] + AUDIO
F4 = F3 + ["release_age_years", "chart_entry_year"]

LEX_RAW = [
    "lyric_word_count", "lyric_unique_word_count", "lyric_lexical_diversity",
    "lyric_average_word_length", "lyric_line_count",
    "lyric_repeated_line_proportion", "lyric_intensity_punctuation_per_100_words",
]
LEX_MODEL = [
    "lex_log_word_count", "lex_log_unique_word_count", "lyric_lexical_diversity",
    "lyric_average_word_length", "lex_log_line_count",
    "lyric_repeated_line_proportion", "lex_log_intensity_punctuation",
]
LWC = F3 + ["lex_log_word_count"]
LEX = F3 + LEX_MODEL
LEX_CAL = F4 + LEX_MODEL

TAB_BLOCKS = {
    "LWC": LWC,
    "LEX": LEX,
    "LEX+CAL": LEX_CAL,
    "T1": [],
    "T2": F1,
    "T3": F3,
    "T4": F4,
    "H1": F1,
    "H2": F3,
    "H3": LEX,
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

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
TFIDF_PARAMS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.95,
    "max_features": 3000,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float64,
    "token_pattern": r"(?u)\b[\w'’]{2,}\b",
}


@dataclass
class ZTPFit:
    coef_all: np.ndarray
    converged: bool
    n_iter: int
    message: str


class ZeroTruncatedPoisson:
    """Zero-truncated Poisson log-link regression with light L2 stabilisation."""

    def __init__(self, ridge: float = 1e-6, maxiter: int = 750):
        self.ridge = ridge
        self.maxiter = maxiter

    def fit(self, X, y):
        y = np.asarray(y, float)
        if np.any(y <= 0):
            raise ValueError("ZeroTruncatedPoisson requires strictly positive outcomes.")
        X = np.asarray(X, float)
        X1 = np.column_stack([np.ones(len(X)), X])
        starter = PoissonRegressor(alpha=1e-8, max_iter=1000, tol=1e-8)
        starter.fit(X, y)
        beta0 = np.r_[starter.intercept_, starter.coef_]

        def objective(beta):
            eta = np.clip(X1 @ beta, -20, 20)
            mu = np.exp(eta)
            log_truncation = np.log(np.maximum(-np.expm1(-mu), 1e-15))
            ll = y * eta - mu - gammaln(y + 1) - log_truncation
            penalty = 0.5 * self.ridge * np.dot(beta[1:], beta[1:])
            return float(-np.sum(ll) + penalty)

        def gradient(beta):
            eta = np.clip(X1 @ beta, -20, 20)
            mu = np.exp(eta)
            ratio = np.zeros_like(mu)
            mask = mu < 50
            ratio[mask] = mu[mask] / np.maximum(np.expm1(mu[mask]), 1e-15)
            grad_eta = mu + ratio - y
            grad = X1.T @ grad_eta
            grad[1:] += self.ridge * beta[1:]
            return grad

        result = minimize(
            objective, beta0, jac=gradient, method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        self.fit_ = ZTPFit(result.x, bool(result.success), int(result.nit), str(result.message))
        return self

    def predict_rate(self, X):
        X = np.asarray(X, float)
        X1 = np.column_stack([np.ones(len(X)), X])
        return np.exp(np.clip(X1 @ self.fit_.coef_all, -20, 20))

    def predict_conditional_mean(self, X):
        mu = self.predict_rate(X)
        return mu / np.maximum(-np.expm1(-mu), 1e-15)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def string_sha256(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def directory_digest(path: Path) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def leakage_hits(variables) -> list[str]:
    hits = []
    for var in variables:
        low = str(var).lower()
        if var in BLOCKED_EXACT or any(pattern in low for pattern in BLOCKED_PATTERNS):
            hits.append(str(var))
        if low.startswith("y_") or low.startswith("l_") or low.startswith("z_"):
            hits.append(str(var))
    return sorted(set(hits))


def normalise_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).lower().strip()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"^[\W_]+|[\W_]+$", "", line)
    return line


def lexical_features(text: str, supplied_word_count: int) -> dict:
    text = "" if pd.isna(text) else str(text)
    tokens = [t.lower().replace("’", "'") for t in WORD_RE.findall(unicodedata.normalize("NFKC", text))]
    unique = len(set(tokens))
    token_n = len(tokens)
    nonblank_lines = [normalise_line(line) for line in text.splitlines() if normalise_line(line)]
    line_n = len(nonblank_lines)
    repeated_line = (line_n - len(set(nonblank_lines))) / line_n if line_n else 0.0
    average_length = float(np.mean([len(t.replace("'", "")) for t in tokens])) if tokens else 0.0
    lexical_diversity = unique / token_n if token_n else 0.0
    intensity = 100.0 * sum(text.count(ch) for ch in "!?") / max(int(supplied_word_count), 1)
    return {
        "lyric_unique_word_count": unique,
        "lyric_lexical_diversity": lexical_diversity,
        "lyric_average_word_length": average_length,
        "lyric_line_count": line_n,
        "lyric_repeated_line_proportion": repeated_line,
        "lyric_intensity_punctuation_per_100_words": intensity,
        "tokenizer_word_count_check": token_n,
    }


def add_lexical_features(data: pd.DataFrame) -> pd.DataFrame:
    records = [lexical_features(t, w) for t, w in zip(data.lyrics_text, data.lyrics_word_count)]
    lex = pd.DataFrame(records, index=data.index)
    out = pd.concat([data.copy(), lex], axis=1)
    out["lyric_word_count"] = out["lyrics_word_count"].astype(float)
    out["lex_log_word_count"] = np.log1p(out.lyric_word_count)
    out["lex_log_unique_word_count"] = np.log1p(out.lyric_unique_word_count)
    out["lex_log_line_count"] = np.log1p(out.lyric_line_count)
    out["lex_log_intensity_punctuation"] = np.log1p(out.lyric_intensity_punctuation_per_100_words)
    return out


def make_tab_preprocessor(variables: list[str]) -> ColumnTransformer:
    categories = [v for v in variables if v in {"X", "key", "chart_entry_year"}]
    binary = [v for v in variables if v == "mode"]
    continuous = [v for v in variables if v not in set(categories + binary)]
    transforms = []
    if continuous:
        transforms.append(("continuous", StandardScaler(), continuous))
    if categories:
        transforms.append((
            "categorical", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categories,
        ))
    if binary:
        transforms.append(("binary", "passthrough", binary))
    return ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)


def fit_tab_poisson(train, test, variables):
    prep = make_tab_preprocessor(variables)
    Xtr = prep.fit_transform(train[variables])
    Xte = prep.transform(test[variables])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PoissonRegressor(alpha=1e-8, max_iter=1000, tol=1e-8)
        model.fit(Xtr, train[YCOL])
    return model.predict(Xte), prep, model, " | ".join(str(w.message) for w in caught)


def build_tfidf_matrices(train, test, tab_variables):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    train_text = train.lyrics_text.fillna("").astype(str)
    test_text = test.lyrics_text.fillna("").astype(str)
    Xtxt_tr = vectorizer.fit_transform(train_text)
    Xtxt_te = vectorizer.transform(test_text)
    if tab_variables:
        prep = make_tab_preprocessor(tab_variables)
        Xtab_tr = csr_matrix(prep.fit_transform(train[tab_variables]))
        Xtab_te = csr_matrix(prep.transform(test[tab_variables]))
        Xtr = hstack([Xtab_tr, Xtxt_tr], format="csr")
        Xte = hstack([Xtab_te, Xtxt_te], format="csr")
    else:
        prep = None
        Xtr, Xte = Xtxt_tr.tocsr(), Xtxt_te.tocsr()
    return Xtr, Xte, vectorizer, prep, Xtxt_tr, Xtxt_te


def metric_values(y, pred, nmax: int = NMAX) -> dict:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    z = y == 0; pos = y > 0
    return {
        "n": len(y), "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mean_observed_y": float(np.mean(y)), "mean_prediction": float(np.mean(pred)),
        "mean_prediction_bias": float(np.mean(pred - y)),
        "r2_supplementary": float(r2_score(y, pred)),
        "observed_zero_n": int(z.sum()), "observed_zero_pct": float(z.mean() * 100),
        "mae_observed_zero": float(mean_absolute_error(y[z], pred[z])) if z.any() else np.nan,
        "mae_observed_positive": float(mean_absolute_error(y[pos], pred[pos])) if pos.any() else np.nan,
        "mean_prediction_observed_zero": float(pred[z].mean()) if z.any() else np.nan,
        "mean_prediction_observed_positive": float(pred[pos].mean()) if pos.any() else np.nan,
        "predicted_near_zero_pct": float(np.mean(pred < 0.5) * 100),
        "outside_bound_n": int(np.sum((pred < 0) | (pred > nmax))),
        "outside_bound_pct": float(np.mean((pred < 0) | (pred > nmax)) * 100),
    }


def make_prediction(data, index, fold, model_id, model_name, feature_block, pred, source):
    part = data.iloc[index]
    return pd.DataFrame({
        "spotify_track_id": part.spotify_track_id.to_numpy(), "fold": fold,
        "model_id": model_id, "model_name": model_name, "feature_block": feature_block,
        "prediction_source": source, "y_observed": part[YCOL].to_numpy(float),
        "y_pred_raw": np.asarray(pred, float), "y_pred_clipped": np.clip(pred, 0, NMAX),
        "X": part.X.to_numpy(int), "chart_entry_year": part.chart_entry_year.to_numpy(int),
    })


def summarise_regression_predictions(predictions: pd.DataFrame):
    aggregate = []
    fold = []
    for keys, g in predictions.groupby(["model_id", "model_name", "feature_block", "prediction_source"], sort=True):
        aggregate.append({**dict(zip(["model_id", "model_name", "feature_block", "prediction_source"], keys)), **metric_values(g.y_observed, g.y_pred_raw)})
    for keys, g in predictions.groupby(["fold", "model_id", "model_name", "feature_block", "prediction_source"], sort=True):
        fold.append({**dict(zip(["fold", "model_id", "model_name", "feature_block", "prediction_source"], keys)), **metric_values(g.y_observed, g.y_pred_raw)})
    return pd.DataFrame(aggregate), pd.DataFrame(fold)


def calibration_summary(predictions, bins=5):
    rows = []
    for keys, g in predictions.groupby(["model_id", "model_name", "feature_block"], sort=True):
        q = g.copy()
        q["calibration_bin"] = pd.qcut(q.y_pred_raw.rank(method="first"), q=bins, labels=range(1, bins + 1))
        for b, h in q.groupby("calibration_bin", observed=True):
            rows.append({
                **dict(zip(["model_id", "model_name", "feature_block"], keys)),
                "calibration_bin": int(b), "n": len(h),
                "mean_prediction": float(h.y_pred_raw.mean()),
                "mean_observed_y": float(h.y_observed.mean()),
                "prediction_bias": float((h.y_pred_raw - h.y_observed).mean()),
            })
    return pd.DataFrame(rows)


def performance_by_x(predictions):
    rows = []
    for keys, g in predictions.groupby(["model_id", "model_name", "feature_block", "X"], sort=True):
        rows.append({**dict(zip(["model_id", "model_name", "feature_block", "X"], keys)), **metric_values(g.y_observed, g.y_pred_raw)})
    return pd.DataFrame(rows)


def benchmark_references(data, folds):
    bench = pd.read_csv(BENCH_DIR / "grouped_cv_predictions.csv")
    bench = bench[(bench.outcome.eq("Y_52")) & (bench.model_id.eq("M1_poisson")) & bench.feature_block.isin(["F1", "F3", "F4"])].copy()
    names = {"F1": "Reference: X only", "F3": "Reference: X + audio", "F4": "Reference: X + audio + calendar"}
    frames = []
    for block, g in bench.groupby("feature_block"):
        if len(g) != 1014 or g.spotify_track_id.nunique() != 1014:
            raise AssertionError(f"Benchmark reference coverage failed for {block}.")
        q = g[["spotify_track_id", "split", "y_observed", "y_pred_raw", "X", "chart_entry_year"]].rename(columns={"split": "fold"})
        q["model_id"] = f"REF_{block}"
        q["model_name"] = names[block]
        q["feature_block"] = block
        q["prediction_source"] = "frozen_benchmark_v1"
        q["y_pred_clipped"] = np.clip(q.y_pred_raw, 0, NMAX)
        frames.append(q)
    out = pd.concat(frames, ignore_index=True)
    expected = data[["spotify_track_id", YCOL]].rename(columns={YCOL: "expected_y"})
    check = out.merge(expected, on="spotify_track_id", how="left", validate="many_to_one")
    if not np.allclose(check.y_observed, check.expected_y):
        raise AssertionError("Frozen benchmark outcomes do not align with the current cohort.")
    assignment = folds.set_index("spotify_track_id").fold
    if not (out.fold.to_numpy() == out.spotify_track_id.map(assignment).to_numpy()).all():
        raise AssertionError("Frozen benchmark prediction folds do not match frozen assignments.")
    return out


def run_lexical_models(data, folds):
    frames = []
    warnings_rows = []
    for fold in sorted(folds.fold.unique()):
        train_idx = np.flatnonzero(folds.fold.ne(fold).to_numpy())
        test_idx = np.flatnonzero(folds.fold.eq(fold).to_numpy())
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        for model_id, name, block, variables in [
            ("LWC", "F3 + lyric word count", "LWC", LWC),
            ("LEX", "F3 + transparent lexical summaries", "LEX", LEX),
            ("LEX_CAL", "F4 + transparent lexical summaries", "LEX+CAL", LEX_CAL),
        ]:
            pred, prep, model, warning_text = fit_tab_poisson(train, test, variables)
            frames.append(make_prediction(data, test_idx, fold, model_id, name, block, pred, "new_lexical_model"))
            warnings_rows.append({
                "model_stage": "lexical", "outer_fold": fold, "model_id": model_id,
                "status": "warning" if warning_text else "ok", "converged": int(model.n_iter_ < model.max_iter),
                "n_iterations": int(model.n_iter_), "warning_or_failure": warning_text,
            })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(warnings_rows)


def inner_select_tfidf_alpha(outer_train, block, tab_variables, outer_fold, audit_rows, warning_rows):
    groups = outer_train.primary_group_component.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=3)
    result_rows = []
    for alpha in ALPHA_GRID:
        maes, rmses = [], []
        for inner_fold, (itr, iva) in enumerate(splitter.split(outer_train, outer_train[YCOL], groups), start=1):
            tr, va = outer_train.iloc[itr], outer_train.iloc[iva]
            group_overlap = len(set(groups[itr]) & set(groups[iva]))
            if group_overlap:
                raise AssertionError("Inner grouped split has a component crossing.")
            Xtr, Xva, vectorizer, _, _, _ = build_tfidf_matrices(tr, va, tab_variables)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PoissonRegressor(alpha=alpha, max_iter=750, tol=1e-7)
                model.fit(Xtr, tr[YCOL])
            pred = model.predict(Xva)
            maes.append(mean_absolute_error(va[YCOL], pred))
            rmses.append(math.sqrt(mean_squared_error(va[YCOL], pred)))
            warning_text = " | ".join(str(w.message) for w in caught)
            warning_rows.append({
                "model_stage": "tfidf_inner", "outer_fold": outer_fold,
                "inner_fold": inner_fold, "model_id": block, "alpha": alpha,
                "status": "warning" if warning_text else "ok",
                "converged": int(model.n_iter_ < model.max_iter), "n_iterations": int(model.n_iter_),
                "warning_or_failure": warning_text,
            })
            audit_rows.append({
                "audit_stage": "tfidf_inner", "outer_fold": outer_fold, "inner_fold": inner_fold,
                "model_id": block, "condition": "inner training and validation groups are disjoint",
                "status": "PASS", "evidence": f"group_overlap={group_overlap}; train_n={len(tr)}; validation_n={len(va)}; vocabulary_n={len(vectorizer.vocabulary_)}",
            })
        result_rows.append({
            "outer_fold": outer_fold, "model_id": block, "alpha": alpha,
            "inner_cv_folds": 3, "inner_mean_mae": float(np.mean(maes)),
            "inner_sd_mae": float(np.std(maes, ddof=1)), "inner_mean_rmse": float(np.mean(rmses)),
        })
    result = pd.DataFrame(result_rows).sort_values(["inner_mean_mae", "inner_mean_rmse", "alpha"])
    selected = float(result.iloc[0].alpha)
    result["selected_for_outer_fit"] = (result.alpha == selected).astype(int)
    return selected, result


def run_tfidf_models(data, folds, blocks=("T1", "T2", "T3")):
    prediction_frames = []
    regularisation = []
    vocab_rows = []
    audit_rows = []
    warning_rows = []
    for fold in sorted(folds.fold.unique()):
        train_idx = np.flatnonzero(folds.fold.ne(fold).to_numpy())
        test_idx = np.flatnonzero(folds.fold.eq(fold).to_numpy())
        outer_train, outer_test = data.iloc[train_idx], data.iloc[test_idx]
        if set(outer_train.spotify_track_id) & set(outer_test.spotify_track_id):
            raise AssertionError("Outer train/test track overlap.")
        for block in blocks:
            print(f"TF-IDF nested CV: {fold} {block}", flush=True)
            tab_variables = TAB_BLOCKS[block]
            selected, reg = inner_select_tfidf_alpha(
                outer_train, block, tab_variables, fold, audit_rows, warning_rows,
            )
            regularisation.append(reg)
            Xtr, Xte, vectorizer, _, txttr, txtte = build_tfidf_matrices(outer_train, outer_test, tab_variables)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PoissonRegressor(alpha=selected, max_iter=1000, tol=1e-7)
                model.fit(Xtr, outer_train[YCOL])
            pred = model.predict(Xte)
            names = {
                "T1": "TF-IDF lyrics only", "T2": "X + TF-IDF lyrics",
                "T3": "X + audio + TF-IDF lyrics", "T4": "X + audio + calendar + TF-IDF lyrics",
            }
            prediction_frames.append(make_prediction(data, test_idx, fold, block, names[block], block, pred, "new_tfidf_model"))
            warning_text = " | ".join(str(w.message) for w in caught)
            warning_rows.append({
                "model_stage": "tfidf_outer", "outer_fold": fold, "inner_fold": np.nan,
                "model_id": block, "alpha": selected,
                "status": "warning" if warning_text else "ok",
                "converged": int(model.n_iter_ < model.max_iter), "n_iterations": int(model.n_iter_),
                "warning_or_failure": warning_text,
            })
            terms = vectorizer.get_feature_names_out()
            vocab_rows.append({
                "outer_fold": fold, "model_id": block, "training_documents": len(outer_train),
                "heldout_documents": len(outer_test), "vocabulary_size": len(terms),
                "unigram_count": int(sum(" " not in t for t in terms)),
                "bigram_count": int(sum(" " in t for t in terms)),
                "max_features": TFIDF_PARAMS["max_features"], "min_document_frequency": TFIDF_PARAMS["min_df"],
                "max_document_frequency_proportion": TFIDF_PARAMS["max_df"],
                "training_zero_vector_documents": int(np.sum(np.diff(txttr.indptr) == 0)),
                "heldout_zero_vector_documents": int(np.sum(np.diff(txtte.indptr) == 0)),
                "training_track_id_sha256": string_sha256(sorted(outer_train.spotify_track_id)),
                "heldout_track_id_sha256": string_sha256(sorted(outer_test.spotify_track_id)),
                "vocabulary_sha256": string_sha256(sorted(terms)),
                "selected_alpha": selected,
            })
            audit_rows.append({
                "audit_stage": "tfidf_outer", "outer_fold": fold, "inner_fold": np.nan,
                "model_id": block, "condition": "outer held-out tracks excluded from TF-IDF fit and inner tuning",
                "status": "PASS", "evidence": f"track_overlap=0; training_n={len(outer_train)}; heldout_n={len(outer_test)}; selected_alpha={selected}",
            })
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(regularisation, ignore_index=True), pd.DataFrame(vocab_rows),
        pd.DataFrame(audit_rows), pd.DataFrame(warning_rows),
    )


def incremental_table(metrics, fold_metrics):
    comparisons = [
        ("F3_to_LWC", "REF_F3", "LWC"),
        ("F3_to_LEX", "REF_F3", "LEX"),
        ("F4_to_LEX_CAL", "REF_F4", "LEX_CAL"),
        ("F1_to_T1", "REF_F1", "T1"),
        ("F1_to_T2", "REF_F1", "T2"),
        ("F3_to_T3", "REF_F3", "T3"),
    ]
    rows = []
    for label, base, new in comparisons:
        b = metrics[metrics.model_id.eq(base)]
        n = metrics[metrics.model_id.eq(new)]
        if b.empty or n.empty:
            continue
        b, n = b.iloc[0], n.iloc[0]
        bf = fold_metrics[fold_metrics.model_id.eq(base)][["fold", "mae", "rmse"]].rename(columns={"mae":"base_mae","rmse":"base_rmse"})
        nf = fold_metrics[fold_metrics.model_id.eq(new)][["fold", "mae", "rmse"]].rename(columns={"mae":"new_mae","rmse":"new_rmse"})
        fd = bf.merge(nf, on="fold", validate="one_to_one")
        fd["mae_gain"] = fd.base_mae - fd.new_mae
        fd["rmse_gain"] = fd.base_rmse - fd.new_rmse
        rows.append({
            "comparison": label, "baseline_model_id": base, "new_model_id": new,
            "baseline_mae": b.mae, "new_mae": n.mae,
            "mae_improvement_absolute": b.mae - n.mae,
            "mae_improvement_pct": 100 * (b.mae - n.mae) / b.mae,
            "baseline_rmse": b.rmse, "new_rmse": n.rmse,
            "rmse_improvement_absolute": b.rmse - n.rmse,
            "rmse_improvement_pct": 100 * (b.rmse - n.rmse) / b.rmse,
            "folds_with_mae_improvement": int((fd.mae_gain > 0).sum()),
            "folds_with_rmse_improvement": int((fd.rmse_gain > 0).sum()),
            "mean_fold_mae_improvement": float(fd.mae_gain.mean()),
            "sd_fold_mae_improvement": float(fd.mae_gain.std(ddof=1)),
            "min_fold_mae_improvement": float(fd.mae_gain.min()),
            "max_fold_mae_improvement": float(fd.mae_gain.max()),
        })
    return pd.DataFrame(rows)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    cls = (p >= threshold).astype(int)
    negative = y == 0
    specificity = float(np.mean(cls[negative] == 0)) if negative.any() else np.nan
    return {
        "n": len(y), "positive_n": int(y.sum()), "positive_pct": float(y.mean()*100),
        "roc_auc": float(roc_auc_score(y,p)), "pr_auc": float(average_precision_score(y,p)),
        "brier_score": float(brier_score_loss(y,p)), "log_loss": float(log_loss(y,p,labels=[0,1])),
        "accuracy_supplementary": float(accuracy_score(y,cls)),
        "sensitivity_recall_positive": float(recall_score(y,cls)), "specificity_zero": specificity,
        "classification_threshold": threshold, "mean_predicted_positive_probability": float(p.mean()),
    }


def run_hurdle(data, folds, include_h3: bool):
    blocks = ["H1", "H2"] + (["H3"] if include_h3 else [])
    names = {
        "H1": "Hurdle: X only", "H2": "Hurdle: X + audio",
        "H3": "Hurdle: X + audio + lexical summaries",
    }
    predictions=[]; warnings_rows=[]
    for fold in sorted(folds.fold.unique()):
        train_idx=np.flatnonzero(folds.fold.ne(fold).to_numpy())
        test_idx=np.flatnonzero(folds.fold.eq(fold).to_numpy())
        train,test=data.iloc[train_idx],data.iloc[test_idx]
        for block in blocks:
            variables=TAB_BLOCKS[block]
            ybin=(train[YCOL]>0).astype(int).to_numpy()
            binary_prep=make_tab_preprocessor(variables)
            Xtr_bin=binary_prep.fit_transform(train[variables]); Xte_bin=binary_prep.transform(test[variables])
            with warnings.catch_warnings(record=True) as caught_bin:
                warnings.simplefilter("always")
                logit=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,random_state=SEED)
                logit.fit(Xtr_bin,ybin)
            ppos=logit.predict_proba(Xte_bin)[:,1]

            pos_train=train[train[YCOL]>0]
            positive_prep=make_tab_preprocessor(variables)
            Xtr_pos=positive_prep.fit_transform(pos_train[variables]); Xte_pos=positive_prep.transform(test[variables])
            ztp=ZeroTruncatedPoisson().fit(Xtr_pos,pos_train[YCOL].to_numpy(float))
            conditional_mean=ztp.predict_conditional_mean(Xte_pos)
            combined=ppos*conditional_mean
            q=make_prediction(data,test_idx,fold,block,names[block],block,combined,"new_true_hurdle")
            q["p_positive"] = ppos
            q["p_zero"] = 1-ppos
            q["conditional_positive_mean"] = conditional_mean
            predictions.append(q)
            bin_warn=" | ".join(str(w.message) for w in caught_bin)
            warnings_rows.extend([
                {"model_stage":"hurdle_binary","outer_fold":fold,"model_id":block,
                 "status":"warning" if bin_warn else "ok","converged":int(logit.n_iter_[0]<logit.max_iter),
                 "n_iterations":int(logit.n_iter_[0]),"warning_or_failure":bin_warn},
                {"model_stage":"hurdle_positive_ztp","outer_fold":fold,"model_id":block,
                 "status":"ok" if ztp.fit_.converged else "warning","converged":int(ztp.fit_.converged),
                 "n_iterations":ztp.fit_.n_iter,"warning_or_failure":"" if ztp.fit_.converged else ztp.fit_.message},
            ])
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(warnings_rows)


def summarise_hurdle(pred):
    binary_rows=[]; positive_rows=[]; combined_rows=[]; cal_rows=[]
    for keys,g in pred.groupby(["model_id","model_name"],sort=True):
        ybin=(g.y_observed>0).astype(int)
        binary_rows.append({"aggregation":"all_oof",**dict(zip(["model_id","model_name"],keys)),**binary_metrics(ybin,g.p_positive)})
        pos=g.y_observed>0
        positive_rows.append({
            "aggregation":"all_oof",**dict(zip(["model_id","model_name"],keys)),"n_positive":int(pos.sum()),
            "mae_positive_count":float(mean_absolute_error(g.loc[pos,"y_observed"],g.loc[pos,"conditional_positive_mean"])),
            "rmse_positive_count":float(math.sqrt(mean_squared_error(g.loc[pos,"y_observed"],g.loc[pos,"conditional_positive_mean"]))),
            "mean_observed_positive_y":float(g.loc[pos,"y_observed"].mean()),
            "mean_predicted_conditional_positive_y":float(g.loc[pos,"conditional_positive_mean"].mean()),
            "mean_prediction_bias_positive_count":float((g.loc[pos,"conditional_positive_mean"]-g.loc[pos,"y_observed"]).mean()),
        })
        combined_rows.append({"aggregation":"all_oof",**dict(zip(["model_id","model_name"],keys)),**metric_values(g.y_observed,g.y_pred_raw)})
        q=g.copy(); q["binary_calibration_bin"]=pd.qcut(q.p_positive.rank(method="first"),q=5,labels=range(1,6))
        for b,h in q.groupby("binary_calibration_bin",observed=True):
            cal_rows.append({
                "calibration_type":"binary_positive_probability","model_id":keys[0],"model_name":keys[1],
                "calibration_bin":int(b),"n":len(h),"mean_predicted_positive_probability":float(h.p_positive.mean()),
                "observed_positive_proportion":float((h.y_observed>0).mean()),
                "mean_predicted_zero_probability":float(h.p_zero.mean()),"observed_zero_proportion":float((h.y_observed==0).mean()),
            })
        for fold,h in g.groupby("fold"):
            yb=(h.y_observed>0).astype(int)
            binary_rows.append({"aggregation":fold,**dict(zip(["model_id","model_name"],keys)),**binary_metrics(yb,h.p_positive)})
            hp=h.y_observed>0
            positive_rows.append({
                "aggregation":fold,**dict(zip(["model_id","model_name"],keys)),"n_positive":int(hp.sum()),
                "mae_positive_count":float(mean_absolute_error(h.loc[hp,"y_observed"],h.loc[hp,"conditional_positive_mean"])),
                "rmse_positive_count":float(math.sqrt(mean_squared_error(h.loc[hp,"y_observed"],h.loc[hp,"conditional_positive_mean"]))),
                "mean_observed_positive_y":float(h.loc[hp,"y_observed"].mean()),
                "mean_predicted_conditional_positive_y":float(h.loc[hp,"conditional_positive_mean"].mean()),
                "mean_prediction_bias_positive_count":float((h.loc[hp,"conditional_positive_mean"]-h.loc[hp,"y_observed"]).mean()),
            })
            combined_rows.append({"aggregation":fold,**dict(zip(["model_id","model_name"],keys)),**metric_values(h.y_observed,h.y_pred_raw)})
    return pd.DataFrame(binary_rows),pd.DataFrame(positive_rows),pd.DataFrame(combined_rows),pd.DataFrame(cal_rows)


def hurdle_zero_comparison(hurdle_pred, ordinary_predictions):
    rows=[]
    ordinary_map={"H1":"REF_F1","H2":"REF_F3","H3":"LEX"}
    for hid,refid in ordinary_map.items():
        h=hurdle_pred[hurdle_pred.model_id.eq(hid)]
        if h.empty: continue
        o=ordinary_predictions[ordinary_predictions.model_id.eq(refid)].copy()
        o["p_zero"]=np.exp(-np.clip(o.y_pred_raw,0,None))
        for model_type,g,pzero in [
            ("ordinary_poisson_reference",o,o.p_zero),
            ("true_hurdle",h,h.p_zero),
        ]:
            yzero=(g.y_observed==0).astype(int).to_numpy(); pz=np.asarray(pzero,float)
            rows.append({
                "comparison_group":hid,"model_type":model_type,"model_id":refid if model_type.startswith("ordinary") else hid,
                "n":len(g),"observed_zero_pct":float(yzero.mean()*100),
                "mean_predicted_zero_probability":float(pz.mean()),
                "zero_probability_brier_score":float(brier_score_loss(yzero,pz)),
                "zero_probability_log_loss":float(log_loss(yzero,np.clip(pz,1e-12,1-1e-12),labels=[0,1])),
                "predicted_near_zero_y_pct":float(np.mean(g.y_pred_raw<0.5)*100),
                "mean_prediction_observed_zero":float(g.loc[g.y_observed==0,"y_pred_raw"].mean()),
                "mean_prediction_observed_positive":float(g.loc[g.y_observed>0,"y_pred_raw"].mean()),
            })
    return pd.DataFrame(rows)


def run_y26_sensitivity(data26, data52, folds, all_y52_predictions, reg_summary):
    references=pd.read_csv(BENCH_DIR/"grouped_cv_predictions.csv")
    refs=references[(references.outcome.eq("Y_26"))&(references.model_id.eq("M1_poisson"))&references.feature_block.isin(["F1","F3"])].copy()
    frames=[]
    for block,g in refs.groupby("feature_block"):
        q=g[["spotify_track_id","split","y_observed","y_pred_raw","X"]].rename(columns={"split":"fold"})
        q["model_id"]=f"REF_{block}";q["model_name"]={"F1":"Reference: X only","F3":"Reference: X + audio"}[block]
        frames.append(q)

    new_ids=["LWC","LEX","LEX_CAL","T1","T2","T3"]
    agg,_=summarise_regression_predictions(all_y52_predictions[all_y52_predictions.model_id.isin(new_ids)])
    best_id=str(agg.sort_values(["mae","rmse"]).iloc[0].model_id)
    names={"LWC":"F3 + lyric word count","LEX":"F3 + transparent lexical summaries","LEX_CAL":"F4 + transparent lexical summaries",
           "T1":"TF-IDF lyrics only","T2":"X + TF-IDF lyrics","T3":"X + audio + TF-IDF lyrics"}
    d26=data26.set_index("spotify_track_id").loc[data52.spotify_track_id].reset_index()
    lex_cols=["spotify_track_id"]+LEX_RAW+LEX_MODEL+["tokenizer_word_count_check"]
    d26=d26.merge(data52[lex_cols],on="spotify_track_id",how="left",validate="one_to_one")
    d26[YCOL]=d26["Y_26"]
    for fold in sorted(folds.fold.unique()):
        tr_idx=np.flatnonzero(folds.fold.ne(fold).to_numpy());te_idx=np.flatnonzero(folds.fold.eq(fold).to_numpy())
        tr,te=d26.iloc[tr_idx],d26.iloc[te_idx]
        if best_id in {"LWC","LEX","LEX_CAL"}:
            pred,_,_,_=fit_tab_poisson(tr,te,TAB_BLOCKS[best_id])
        else:
            alpha=float(reg_summary[(reg_summary.outer_fold.eq(fold))&(reg_summary.model_id.eq(best_id))&(reg_summary.selected_for_outer_fit.eq(1))].alpha.iloc[0])
            Xtr,Xte,_,_,_,_=build_tfidf_matrices(tr,te,TAB_BLOCKS[best_id])
            model=PoissonRegressor(alpha=alpha,max_iter=1000,tol=1e-7).fit(Xtr,tr[YCOL])
            pred=model.predict(Xte)
        q=pd.DataFrame({"spotify_track_id":te.spotify_track_id.to_numpy(),"fold":fold,"y_observed":te.Y_26.to_numpy(float),"y_pred_raw":pred,"X":te.X.to_numpy(int),"model_id":best_id,"model_name":names[best_id]})
        frames.append(q)
    pred=pd.concat(frames,ignore_index=True)
    rows=[]
    for keys,g in pred.groupby(["model_id","model_name"],sort=True):
        rows.append({"aggregation":"all_oof",**dict(zip(["model_id","model_name"],keys)),"selected_y52_lyric_model":int(keys[0]==best_id),**metric_values(g.y_observed,g.y_pred_raw,nmax=22)})
        for fold,h in g.groupby("fold"):
            rows.append({"aggregation":fold,**dict(zip(["model_id","model_name"],keys)),"selected_y52_lyric_model":int(keys[0]==best_id),**metric_values(h.y_observed,h.y_pred_raw,nmax=22)})
    return pd.DataFrame(rows),best_id,pred


def dataframe_to_markdown(df,decimals=3):
    def val(x):
        if pd.isna(x): return ""
        if isinstance(x,(float,np.floating)): return f"{float(x):.{decimals}f}"
        return str(x).replace("|","\\|")
    lines=["| "+" | ".join(map(str,df.columns))+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]
    lines += ["| "+" | ".join(val(x) for x in row)+" |" for row in df.itertuples(index=False,name=None)]
    return "\n".join(lines)


def make_figures(fig_dir,metrics,byx,cal,hbinary,hzero,hcombined,best_id):
    sns.set_theme(style="whitegrid",context="notebook");fig_dir.mkdir(parents=True,exist_ok=True)
    order=[x for x in ["REF_F1","REF_F3","LWC","LEX","LEX_CAL","T1","T2","T3"] if x in set(metrics.model_id)]
    q=metrics[metrics.model_id.isin(order)].copy();q["model_id"]=pd.Categorical(q.model_id,categories=order,ordered=True);q=q.sort_values("model_id")
    plot=q.melt(id_vars=["model_id","model_name"],value_vars=["mae","rmse"],var_name="metric",value_name="error")
    fig,ax=plt.subplots(figsize=(11,6));sns.barplot(data=plot,x="model_id",y="error",hue="metric",ax=ax)
    ax.set(title="Incremental lyric-model prediction error",xlabel="Model block",ylabel="Out-of-fold error (weeks)")
    fig.tight_layout();fig.savefig(fig_dir/"incremental_lyric_mae_rmse.png",dpi=180);plt.close(fig)

    x4=byx[(byx.X.eq(4))&byx.model_id.isin(["REF_F1","REF_F3","LEX","T2","T3"])].copy()
    p=x4.melt(id_vars=["model_id"],value_vars=["mae","rmse"],var_name="metric",value_name="error")
    fig,ax=plt.subplots(figsize=(9,5.5));sns.barplot(data=p,x="model_id",y="error",hue="metric",ax=ax)
    ax.set(title="Lyric-model performance among X=4 tracks",xlabel="Model block",ylabel="Error (weeks)")
    fig.tight_layout();fig.savefig(fig_dir/"lyric_performance_x4.png",dpi=180);plt.close(fig)

    z=hzero.copy();fig,ax=plt.subplots(figsize=(8,5.5))
    sns.barplot(data=z,x="comparison_group",y="mean_predicted_zero_probability",hue="model_type",ax=ax)
    observed=z.observed_zero_pct.iloc[0]/100;ax.axhline(observed,color="black",linestyle="--",label="Observed zero proportion")
    ax.set(title="Zero calibration: ordinary Poisson versus hurdle",xlabel="Feature set",ylabel="Mean predicted probability of zero")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(fig_dir/"zero_calibration_ordinary_vs_hurdle.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,6))
    for mid,g in hbinary[hbinary.calibration_type.eq("binary_positive_probability")].groupby("model_id"):
        ax.plot(g.mean_predicted_positive_probability,g.observed_positive_proportion,marker="o",label=mid)
    ax.plot([0,1],[0,1],"k--",linewidth=1,label="Ideal");ax.set(xlim=(0,1),ylim=(0,1),title="Hurdle binary calibration",xlabel="Predicted probability of future presence",ylabel="Observed future-presence proportion")
    ax.legend();fig.tight_layout();fig.savefig(fig_dir/"hurdle_binary_calibration.png",dpi=180);plt.close(fig)

    c=cal[cal.model_id.eq(best_id)]
    fig,ax=plt.subplots(figsize=(7,6));ax.plot(c.mean_prediction,c.mean_observed_y,marker="o",label=best_id)
    lim=max(c.mean_prediction.max(),c.mean_observed_y.max())*1.08;ax.plot([0,lim],[0,lim],"k--",linewidth=1,label="Ideal")
    ax.set(xlim=(0,lim),ylim=(0,lim),title=f"Observed versus predicted longevity: {best_id}",xlabel="Mean predicted Y_52",ylabel="Mean observed Y_52")
    ax.legend();fig.tight_layout();fig.savefig(fig_dir/"best_lyric_observed_vs_predicted.png",dpi=180);plt.close(fig)


def write_report(path,definitions,metrics,increment,byx,reg,hbin,hpos,hcomb,hzero,y26,warnings_df,best_id,lex_useful):
    m=metrics.set_index("model_id")
    key=[x for x in ["REF_F1","REF_F3","LWC","LEX","LEX_CAL","T1","T2","T3"] if x in m.index]
    mt=dataframe_to_markdown(metrics[metrics.model_id.isin(key)][["model_id","model_name","mae","rmse","mean_prediction_bias","mae_observed_zero","mae_observed_positive"]],3)
    it=dataframe_to_markdown(increment[["comparison","mae_improvement_absolute","mae_improvement_pct","rmse_improvement_absolute","rmse_improvement_pct","folds_with_mae_improvement","sd_fold_mae_improvement"]],3)
    x4=dataframe_to_markdown(byx[(byx.X.eq(4))&byx.model_id.isin(["REF_F1","REF_F3","LEX","T2","T3"])][["model_id","n","mae","rmse","mean_observed_y","mean_prediction","mean_prediction_bias"]],3)
    hb=dataframe_to_markdown(hbin[hbin.aggregation.eq("all_oof")][["model_id","roc_auc","pr_auc","brier_score","log_loss","sensitivity_recall_positive","specificity_zero"]],3)
    hp=dataframe_to_markdown(hpos[hpos.aggregation.eq("all_oof")][["model_id","n_positive","mae_positive_count","rmse_positive_count","mean_prediction_bias_positive_count"]],3)
    hc=dataframe_to_markdown(hcomb[hcomb.aggregation.eq("all_oof")][["model_id","mae","rmse","mean_prediction_bias","mean_prediction_observed_zero","mean_prediction_observed_positive"]],3)
    yz=dataframe_to_markdown(y26[y26.aggregation.eq("all_oof")][["model_id","model_name","mae","rmse","mean_prediction_bias"]],3)
    selected=reg[reg.selected_for_outer_fit.eq(1)].groupby("model_id").alpha.value_counts().rename("outer_fold_count").reset_index()
    rt=dataframe_to_markdown(selected,3)
    warning_count=int(warnings_df.status.ne("ok").sum())
    text=f"""# Provisional Lyrics and Hurdle Modelling Report

## Status and scope

This focused second modelling round uses the frozen 1,014-track Y_52 cohort and the existing five lead-artist/version-grouped folds. It does not imply supervisor approval of the working outcome, horizon or model family. No source-gap cases, related-ID collapsing, embeddings, transformer models, boosting, zero-inflated model or broad hyperparameter search were introduced.

## Lyric features

All 1,014 cohort tracks have nonblank, non-placeholder lyrics under the existing audit. Seven transparent features were derived from the fixed lyric text: supplied word count, unique token count, lexical diversity, average token length, nonblank line count, exact normalised repeated-line proportion, and exclamation/question-mark intensity per 100 words. Count-like variables were transformed with log(1+x) before fold-local scaling. These variables use no chart outcome or post-entry information.

Word count by itself is treated only as a length control. The LEX block is F3 plus the seven lexical summaries; LEX+CAL adds the same summaries to F4.

## Grouped-CV lyric results

{mt}

The principal comparison is F3 versus T3. Positive improvement values below mean that the lyric-enhanced model reduced held-out error.

{it}

Fold win counts and the standard deviation of fold-level MAE gains are reported to prevent a very small aggregate difference being presented as stable evidence.

## X=4 diagnostic

{x4}

This comparison asks whether lyric content helps discriminate among songs that were present in all four early weeks, the subgroup responsible for most absolute prediction error. Models were not fitted separately by X.

## Fold-local TF-IDF and regularisation

TF-IDF used unigrams and bigrams, no language-specific stop-word list, min_df=3, max_df=0.95 and a 3,000-feature ceiling. For each outer fold, vocabulary, document-frequency filtering and IDF were fit only on the outer training lyrics. Alpha was selected from {{0.01, 0.1, 1.0}} using three-fold inner GroupKFold on the frozen connected grouping component. The outer validation outcomes were not used in selection.

{rt}

T1 contains lyrics only, T2 adds categorical X, and T3 adds X plus audio/song characteristics. T4 was deliberately deferred because T3 directly answers the main incremental question and the task requested a restrained comparison.

## True hurdle sensitivity

The hurdle analysis is a true two-part construction for this sensitivity: Part A is L2-regularised logistic regression for Y_52>0; Part B is a zero-truncated Poisson log-link model fitted only to positive training outcomes. The combined expectation is P(Y>0) multiplied by the zero-truncated conditional count mean. H3 execution status based on the predefined lexical usefulness rule is: **{'included' if lex_useful else 'not included'}**.

### Binary future-presence component

{hb}

### Positive longevity component

{hp}

### Combined prediction

{hc}

The zero-calibration table compares hurdle zero probabilities with the ordinary Poisson implied zero probability exp(-mu). This is more informative than treating a prediction below an arbitrary count threshold as an exact zero.

## Y_26 directional sensitivity

The most informative Y_52 lyric block selected for this limited sensitivity was **{best_id}**. It was compared with the frozen Y_26 F1 and F3 predictions on the same 1,014 tracks and folds.

{yz}

## Leakage, warnings and limitations

All outer folds were used unchanged. No first-credited artist, exact lyric duplicate group or supported related-version group crosses folds. TF-IDF training/held-out ID hashes, vocabulary hashes, zero-vector counts and inner grouping checks are saved. All lexical definitions are target-independent. The blocked-variable audit passed.

There were {warning_count} warning records and no silently discarded fits. Exact convergence messages and iteration counts are retained. Collaborator overlap remains the known limitation of lead-artist grouping. The models are predictive and do not identify causal lyric effects.

## Provisional next step

The next task should consolidate only specifications that show repeatable fold-level improvement, then test temporal robustness and uncertainty. If hurdle calibration improves without worsening count error, it can be retained as a final-model candidate; a zero-inflated model is not justified automatically by the presence of zeros.
"""
    path.write_text(text,encoding="utf-8")


def main():
    if OUT_DIR.exists() or FIG_DIR.parent.exists() or REPORT_PATH.exists():
        raise FileExistsError("Lyrics/hurdle v1 outputs already exist; refusing to overwrite.")
    required=[
        ANALYSIS_DIR/"analysis_ready_550k_52w.csv", ANALYSIS_DIR/"analysis_ready_550k_26w.csv",
        SPEC_DIR/"provisional_grouped_cv_assignments.csv", SPEC_DIR/"provisional_feature_blocks.csv",
        BENCH_DIR/"grouped_cv_predictions.csv", BENCH_DIR/"grouped_cv_aggregate_metrics.csv",
        BENCH_DIR/"benchmark_validation_manifest.json",
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError(missing)
    before={"analysis":directory_digest(ANALYSIS_DIR),"specification":directory_digest(SPEC_DIR),"benchmark":directory_digest(BENCH_DIR)}

    data=add_lexical_features(pd.read_csv(ANALYSIS_DIR/"analysis_ready_550k_52w.csv"))
    data26=pd.read_csv(ANALYSIS_DIR/"analysis_ready_550k_26w.csv")
    frozen_folds=pd.read_csv(SPEC_DIR/"provisional_grouped_cv_assignments.csv")
    data=data.merge(frozen_folds[["spotify_track_id","fold","primary_group_component"]],on="spotify_track_id",how="left",validate="one_to_one")
    if len(data)!=1014 or data.spotify_track_id.duplicated().any() or data.fold.isna().any(): raise AssertionError("Cohort/fold validation failed")
    if frozen_folds[["first_artist_crosses_folds","exact_lyric_group_crosses_folds","related_version_group_crosses_folds"]].to_numpy().sum()!=0: raise AssertionError("Frozen grouping violation")
    # All modelling indices follow the analysis-ready cohort order, not the ordering
    # of the frozen assignment file.
    folds=data[["spotify_track_id","fold"]].copy()
    if not data.lyrics_usable_provisional.eq(1).all(): raise AssertionError("Unexpected unusable lyric in clean cohort")
    for block,vars_ in TAB_BLOCKS.items():
        hits=leakage_hits(vars_)
        if hits: raise AssertionError(f"Leakage in {block}: {hits}")

    definitions=pd.DataFrame([
        {"feature":"lyric_word_count","formula":"Existing audited lyrics_word_count field","reason":"Controls for lyric/document length","model_treatment":"log(1+x), then fold-local scaling"},
        {"feature":"lyric_unique_word_count","formula":"Number of distinct lower-case Unicode word tokens","reason":"Transparent vocabulary-size measure","model_treatment":"log(1+x), then fold-local scaling"},
        {"feature":"lyric_lexical_diversity","formula":"unique word tokens / all word tokens","reason":"Vocabulary variety relative to lyric length","model_treatment":"fold-local scaling"},
        {"feature":"lyric_average_word_length","formula":"Mean letters/digits per token after apostrophes are removed","reason":"Simple lexical-complexity proxy","model_treatment":"fold-local scaling"},
        {"feature":"lyric_line_count","formula":"Count of nonblank lyric lines","reason":"Transparent structural-length measure","model_treatment":"log(1+x), then fold-local scaling"},
        {"feature":"lyric_repeated_line_proportion","formula":"(nonblank normalised lines - unique normalised lines) / nonblank lines","reason":"Captures exact line-level repetition","model_treatment":"fold-local scaling"},
        {"feature":"lyric_intensity_punctuation_per_100_words","formula":"100 * count(! and ?) / audited word count","reason":"Limited punctuation/intensity measure","model_treatment":"log(1+x), then fold-local scaling"},
    ])
    summary=[]
    for c in LEX_RAW+['tokenizer_word_count_check']:
        s=data[c]
        summary.append({"feature":c,"n":len(s),"missing_n":int(s.isna().sum()),"mean":s.mean(),"std":s.std(),"min":s.min(),"p25":s.quantile(.25),"median":s.median(),"p75":s.quantile(.75),"p90":s.quantile(.9),"p95":s.quantile(.95),"max":s.max()})
    lexical_summary=pd.DataFrame(summary)

    print("Validated frozen cohort, folds, lyric coverage, and benchmark references.", flush=True)
    references=benchmark_references(data,folds)
    lexical_pred,lex_warn=run_lexical_models(data,folds)
    print("Completed transparent lexical-summary models.", flush=True)
    tfidf_pred,reg,vocab,tfidf_audit,tfidf_warn=run_tfidf_models(data,folds)
    all_pred=pd.concat([references,lexical_pred,tfidf_pred],ignore_index=True,sort=False)
    metrics,fold_metrics=summarise_regression_predictions(all_pred)
    calibration=calibration_summary(all_pred)
    byx=performance_by_x(all_pred)
    incremental=incremental_table(metrics,fold_metrics)

    lex_comp=incremental[incremental.comparison.eq("F3_to_LEX")].iloc[0]
    lex_useful=bool((lex_comp.mae_improvement_absolute>0) and (lex_comp.folds_with_mae_improvement>=3))
    hurdle_pred,hurdle_warn=run_hurdle(data,folds,include_h3=lex_useful)
    print(f"Completed hurdle models; H3 executed={lex_useful}.", flush=True)
    hbin,hpos,hcomb,hcal=summarise_hurdle(hurdle_pred)
    hzero=hurdle_zero_comparison(hurdle_pred,all_pred)

    y26,best_id,y26_pred=run_y26_sensitivity(data26,data,folds,all_pred,reg)
    print(f"Completed Y_26 sensitivity with selected lyric block {best_id}.", flush=True)
    all_warnings=pd.concat([lex_warn,tfidf_warn,hurdle_warn],ignore_index=True,sort=False)

    audit_rows=[
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"all","condition":"cohort has 1,014 unique Spotify track IDs","status":"PASS","evidence":f"rows={len(data)}; unique_ids={data.spotify_track_id.nunique()}"},
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"all","condition":"frozen artist/lyric/version folds unchanged and crossing-free","status":"PASS","evidence":"first_artist_crossings=0; exact_lyric_crossings=0; supported_related_version_crossings=0"},
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"all","condition":"all selected predictors pass blocked-variable audit","status":"PASS","evidence":"blocked_hits=0"},
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"LEX","condition":"lexical features depend only on fixed lyric text","status":"PASS","evidence":"formulas use lyrics_text and audited lyrics_word_count only"},
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"TFIDF","condition":"all TF-IDF vocabulary/IDF/document-frequency steps occur inside training folds","status":"PASS","evidence":"outer and inner vectorisers instantiated and fit after split; per-fold ID/vocabulary hashes saved"},
        {"audit_stage":"global","outer_fold":"all","inner_fold":np.nan,"model_id":"TFIDF","condition":"regularisation chosen only in inner grouped CV","status":"PASS","evidence":"alpha grid=0.01|0.1|1.0; GroupKFold=3 on primary_group_component; outer outcomes excluded"},
    ]
    leakage=pd.concat([pd.DataFrame(audit_rows),tfidf_audit],ignore_index=True,sort=False)
    if not leakage.status.eq("PASS").all(): raise AssertionError("Leakage audit failure")

    # Coverage and fold alignment assertions.
    for mid,g in all_pred.groupby("model_id"):
        if len(g)!=1014 or g.spotify_track_id.nunique()!=1014: raise AssertionError(f"Prediction coverage failed for {mid}")
        expected=folds.set_index("spotify_track_id").fold
        if not (g.fold.to_numpy()==g.spotify_track_id.map(expected).to_numpy()).all(): raise AssertionError(f"Fold alignment failed for {mid}")
    for mid,g in hurdle_pred.groupby("model_id"):
        if len(g)!=1014 or g.spotify_track_id.nunique()!=1014: raise AssertionError(f"Hurdle coverage failed for {mid}")

    after={"analysis":directory_digest(ANALYSIS_DIR),"specification":directory_digest(SPEC_DIR),"benchmark":directory_digest(BENCH_DIR)}
    if before!=after: raise AssertionError("Frozen input directory changed")

    lyric_specs=pd.DataFrame([
        {"model_id":"REF_F1","feature_block":"F1","features":"categorical X","estimator":"frozen benchmark Poisson","status":"reference"},
        {"model_id":"REF_F3","feature_block":"F3","features":"categorical X + audio/song","estimator":"frozen benchmark Poisson","status":"reference"},
        {"model_id":"LWC","feature_block":"LWC","features":"F3 + log lyric word count","estimator":"Poisson alpha=1e-8","status":"new"},
        {"model_id":"LEX","feature_block":"LEX","features":"F3 + seven transparent lexical summaries","estimator":"Poisson alpha=1e-8","status":"new"},
        {"model_id":"LEX_CAL","feature_block":"LEX+CAL","features":"F4 + seven transparent lexical summaries","estimator":"Poisson alpha=1e-8","status":"new"},
        {"model_id":"T1","feature_block":"T1","features":"TF-IDF lyrics only","estimator":"regularised Poisson; nested alpha selection","status":"new"},
        {"model_id":"T2","feature_block":"T2","features":"categorical X + TF-IDF","estimator":"regularised Poisson; nested alpha selection","status":"new"},
        {"model_id":"T3","feature_block":"T3","features":"categorical X + audio/song + TF-IDF","estimator":"regularised Poisson; nested alpha selection","status":"new"},
        {"model_id":"T4","feature_block":"T4","features":"F4 + TF-IDF","estimator":"not fitted","status":"deferred to keep comparison focused; T3 answers principal increment"},
    ])
    hurdle_specs=pd.DataFrame([
        {"model_id":"H1","features":"categorical X","binary_component":"L2 logistic C=1","positive_count_component":"zero-truncated Poisson; ridge=1e-6","combined_prediction":"P(Y>0)*E(Y|Y>0)","executed":1},
        {"model_id":"H2","features":"categorical X + audio/song","binary_component":"L2 logistic C=1","positive_count_component":"zero-truncated Poisson; ridge=1e-6","combined_prediction":"P(Y>0)*E(Y|Y>0)","executed":1},
        {"model_id":"H3","features":"H2 + transparent lexical summaries","binary_component":"L2 logistic C=1","positive_count_component":"zero-truncated Poisson; ridge=1e-6","combined_prediction":"P(Y>0)*E(Y|Y>0)","executed":int(lex_useful)},
    ])

    stage=Path(tempfile.mkdtemp(prefix="spotify_lyrics_hurdle_v1_"));sd=stage/"data";sf=stage/"figures";sd.mkdir(parents=True)
    outputs={
        "lyric_feature_definitions.csv":definitions,
        "lyric_lexical_feature_summary.csv":lexical_summary,
        "lyric_model_specifications.csv":lyric_specs,
        "lyric_grouped_cv_metrics.csv":metrics,
        "lyric_grouped_cv_fold_metrics.csv":fold_metrics,
        "lyric_incremental_comparison.csv":incremental,
        "lyric_performance_by_X.csv":byx,
        "tfidf_fold_vocabulary_summary.csv":vocab,
        "tfidf_regularisation_summary.csv":reg,
        "hurdle_model_specifications.csv":hurdle_specs,
        "hurdle_binary_metrics.csv":hbin,
        "hurdle_positive_count_metrics.csv":hpos,
        "hurdle_combined_metrics.csv":hcomb,
        "hurdle_zero_calibration.csv":pd.concat([hzero,hcal],ignore_index=True,sort=False),
        "y26_best_lyric_sensitivity.csv":y26,
        "lyrics_hurdle_leakage_audit.csv":leakage,
        "lyric_lexical_features_by_track.csv":data[["spotify_track_id"]+LEX_RAW+LEX_MODEL+["tokenizer_word_count_check"]],
        "lyric_grouped_cv_predictions.csv":all_pred,
        "lyric_calibration_summary.csv":calibration,
        "hurdle_grouped_cv_predictions.csv":hurdle_pred,
        "model_fit_warnings.csv":all_warnings,
    }
    for name,df in outputs.items(): df.to_csv(sd/name,index=False)

    best_new=str(metrics[metrics.prediction_source.str.startswith("new")].sort_values(["mae","rmse"]).iloc[0].model_id)
    make_figures(sf,metrics,byx,calibration,hcal,hzero,hcomb,best_new)
    report=stage/"provisional_lyrics_hurdle_modelling_report.md"
    write_report(report,definitions,metrics,incremental,byx,reg,hbin,hpos,hcomb,hzero,y26,all_warnings,best_id,lex_useful)

    manifest={
        "created_utc":datetime.now(timezone.utc).isoformat(),"working_status":"provisional; not supervisor-approved",
        "random_seed":SEED,"python":sys.version,"platform":platform.platform(),
        "libraries":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
        "cohort":{"outcome":"Y_52","support":[0,48],"n":1014,"fold_sizes":folds.fold.value_counts().sort_index().to_dict()},
        "tfidf":{"parameters":{k:(list(v) if isinstance(v,tuple) else str(v) if v is np.float64 else v) for k,v in TFIDF_PARAMS.items()},"alpha_grid":ALPHA_GRID,"inner_grouped_folds":3,"outer_folds":5,"blocks":["T1","T2","T3"]},
        "hurdle":{"type":"true hurdle sensitivity","binary":"L2 logistic regression","positive_count":"zero-truncated Poisson","H3_executed":bool(lex_useful)},
        "y26":{"grouped_cv_scope":"same 1,014 tracks and frozen folds","selected_lyric_model_from_y52":best_id},
        "fit_summary":{"records":int(len(all_warnings)),"warnings":int(all_warnings.status.ne("ok").sum()),"failures":0},
        "validation_assertions":{"frozen_folds_used":True,"outer_group_crossings_zero":True,"tfidf_fit_training_only":True,"tfidf_inner_selection_training_only":True,"lexical_features_target_independent":True,"blocked_predictors_absent":True,"prediction_coverage_complete":True,"frozen_inputs_unchanged":True},
        "frozen_input_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in required},
        "frozen_directory_digest_before":before,"frozen_directory_digest_after":after,
    }
    manifest["output_sha256"]={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    (sd/"lyrics_hurdle_validation_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    OUT_DIR.parent.mkdir(parents=True,exist_ok=True);FIG_DIR.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(sd),str(OUT_DIR));shutil.move(str(sf),str(FIG_DIR));shutil.move(str(report),str(REPORT_PATH));shutil.rmtree(stage,ignore_errors=True)
    print(json.dumps({"status":"complete","output_directory":str(OUT_DIR),"figure_directory":str(FIG_DIR),"report":str(REPORT_PATH),"best_new_y52_model":best_new,"selected_y26_lyric_model":best_id,"lex_hurdle_executed":bool(lex_useful),"fit_summary":manifest["fit_summary"]},indent=2))


if __name__=="__main__":
    main()
