#!/usr/bin/env python3
"""Create versioned, analysis-ready pre-model cohorts and diagnostics.

This script builds no predictive, count, survival, machine-learning or
hierarchical model. It reads the existing audited sources and fixed-window
diagnostics, applies stricter provisional cohort rules, and writes a new v1
output layer without modifying historical outputs.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "data/processed/fixed_window_count_v1/fixed_window_track_outcomes.csv"
FIRST_SPELL = ROOT / "data/processed/chart_survival/first_spell_survival.csv"
FEATURE_550_EXPORT = ROOT / "data/processed/chart_survival/matched_550k_baseline_features.csv"
SONGS_550 = ROOT / "data/raw/spotify_550k/songs.csv"
ARTISTS_550 = ROOT / "data/raw/spotify_550k/artists.csv"
TRACKS_160 = ROOT / "data/raw/spotify_160k/data.csv"
GENRES_160 = ROOT / "data/raw/spotify_160k/data_w_genres.csv"
RANK200 = ROOT / "data/processed/fixed_window_count_v1/rank_200_weekly_streams.csv"

OUT = ROOT / "data/processed/pre_model_analysis_ready_v1"
FIG = ROOT / "outputs/pre_model_analysis_ready_v1/figures"
REPORT = ROOT / "docs/pre_model_analysis_ready_report.md"

HORIZONS = (26, 52)
AUDIO = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
CONTINUOUS_VIF = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "release_year", "release_age_years", "lyrics_character_count", "lyrics_word_count",
    "X", "chart_entry_year", "rank_200_stream_threshold_at_entry",
]
UNIT_INTERVAL = {
    "danceability", "energy", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence",
}
PLACEHOLDERS = {
    "instrumental", "instrumentals", "no lyrics", "lyrics unavailable",
    "not available", "n a", "na", "none", "unknown",
}
TREE_GUARDS = [
    ROOT / "data/processed/chart_survival",
    ROOT / "data/processed/fixed_window_count_v1",
    ROOT / "data/processed/audits",
    ROOT / "data/frozen/first_spell_survival_2026-07-30",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(sha256(item).encode())
    return h.hexdigest()


def prepare_outputs() -> None:
    collisions = [p for p in (OUT, FIG.parent, REPORT) if p.exists()]
    if collisions:
        joined = ", ".join(str(p) for p in collisions)
        raise RuntimeError(f"Refusing to overwrite existing v1 output(s): {joined}")
    OUT.mkdir(parents=True)
    FIG.mkdir(parents=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def parse_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except (ValueError, SyntaxError):
        pass
    return [text]


def normalise_placeholder(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\[\](){}._-]+", " ", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’']+\b", text, flags=re.UNICODE))


def q(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def distribution_record(cohort: str, horizon: int, variable: str, series: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    variance = float(values.var(ddof=1)) if len(values) > 1 else np.nan
    return {
        "cohort": cohort,
        "horizon_weeks": horizon,
        "variable": variable,
        "n": int(len(values)),
        "mean": mean,
        "median": float(values.median()) if len(values) else np.nan,
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "variance": variance,
        "minimum": float(values.min()) if len(values) else np.nan,
        "p25": q(values, 0.25) if len(values) else np.nan,
        "p50": q(values, 0.50) if len(values) else np.nan,
        "p75": q(values, 0.75) if len(values) else np.nan,
        "p90": q(values, 0.90) if len(values) else np.nan,
        "p95": q(values, 0.95) if len(values) else np.nan,
        "p99": q(values, 0.99) if len(values) else np.nan,
        "maximum": float(values.max()) if len(values) else np.nan,
        "zero_count": int((values == 0).sum()),
        "zero_percent": float(100 * (values == 0).mean()) if len(values) else np.nan,
        "variance_to_mean_ratio": variance / mean if mean and not np.isnan(variance) else np.nan,
        "skewness": float(values.skew()) if len(values) > 2 else np.nan,
    }


def clean_outcome_rows(fixed: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = fixed.loc[fixed["horizon_weeks"].eq(horizon)].copy()
    assert len(frame) == 4247 and frame["spotify_track_id"].is_unique
    frame["uncertain_entry_flag"] = (
        frame["left_truncated"].eq(1) | frame["entry_ambiguous_due_source_gap"].eq(1)
    ).astype(int)
    frame["insufficient_follow_up_flag"] = frame["enough_follow_up"].ne(1).astype(int)
    frame["missing_source_week_contamination_flag"] = frame["source_gap_in_horizon"].eq(1).astype(int)

    conditions = [
        frame["uncertain_entry_flag"].eq(1),
        frame["insufficient_follow_up_flag"].eq(1),
        frame["missing_source_week_contamination_flag"].eq(1),
    ]
    frame["outcome_exclusion_reason_exclusive"] = np.select(
        conditions,
        ["uncertain_entry", "insufficient_follow_up", "missing_source_week_contamination"],
        default="included_clean_outcome",
    )
    frame["included_clean_outcome"] = frame["outcome_exclusion_reason_exclusive"].eq("included_clean_outcome").astype(int)
    clean = frame.loc[frame["included_clean_outcome"].eq(1)].copy()
    for col in ["X_first_4_weeks", "Y_future_chart_weeks", "L_last_active_week_number", "Z_count_presence_blocks", "Z_reentry_more_than_one_block"]:
        clean[col] = pd.to_numeric(clean[col], errors="raise").astype(int)
    return clean, frame


def extract_550_rows(required_ids: set[str]) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with SONGS_550.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ident = row.get("id", "")
            if ident in required_ids:
                if ident in seen:
                    raise RuntimeError(f"Duplicate 550k track ID encountered: {ident}")
                seen.add(ident)
                rows.append(row)
    missing = required_ids - seen
    if missing:
        raise RuntimeError(f"Expected 550k source rows were not found: {len(missing)}")
    return pd.DataFrame(rows)


def lyric_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    texts = result["lyrics"].fillna("").astype(str)
    stripped = texts.str.strip()
    result["lyrics_missing_source_value"] = result["lyrics"].isna().astype(int)
    result["lyrics_blank"] = stripped.eq("").astype(int)
    result["lyrics_placeholder"] = stripped.map(lambda x: int(normalise_placeholder(x) in PLACEHOLDERS))
    result["lyrics_character_count"] = texts.str.len().astype(int)
    result["lyrics_word_count"] = texts.map(word_count).astype(int)
    result["lyrics_extremely_short"] = (
        (result["lyrics_character_count"] < 50) | (result["lyrics_word_count"] < 10)
    ).astype(int)
    result["lyrics_usable_provisional"] = (
        result["lyrics_missing_source_value"].eq(0)
        & result["lyrics_blank"].eq(0)
        & result["lyrics_placeholder"].eq(0)
        & result["lyrics_extremely_short"].eq(0)
    ).astype(int)
    result["lyrics_quality_reason"] = np.select(
        [
            result["lyrics_missing_source_value"].eq(1),
            result["lyrics_blank"].eq(1),
            result["lyrics_placeholder"].eq(1),
            result["lyrics_extremely_short"].eq(1),
        ],
        ["missing_source_value", "blank", "placeholder", "extremely_short"],
        default="usable_under_provisional_policy",
    )
    normalised = stripped.str.lower().str.replace(r"\s+", " ", regex=True)
    hashes = normalised.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest() if x else "")
    counts = hashes[hashes.ne("")].value_counts()
    result["lyrics_normalised_sha256"] = hashes
    result["lyrics_duplicate_text"] = hashes.map(lambda x: int(bool(x) and counts.get(x, 0) > 1))
    result["lyrics_duplicate_group_size"] = hashes.map(lambda x: int(counts.get(x, 0)) if x else 0)
    return result


def base_outcome_columns(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = f"Y_{horizon}"
    l = f"L_{horizon}"
    zc = f"Z_count_{horizon}"
    zr = f"Z_reentry_{horizon}"
    out = frame[[
        "spotify_track_id", "chart_track", "chart_artist", "spotify_track_url",
        "first_observed_chart_entry", "X_first_4_weeks", "Y_future_chart_weeks",
        "L_last_active_week_number", "Z_count_presence_blocks", "Z_reentry_more_than_one_block",
    ]].copy()
    out = out.rename(columns={
        "first_observed_chart_entry": "chart_entry_date",
        "X_first_4_weeks": "X",
        "Y_future_chart_weeks": y,
        "L_last_active_week_number": l,
        "Z_count_presence_blocks": zc,
        "Z_reentry_more_than_one_block": zr,
    })
    out["chart_entry_date"] = pd.to_datetime(out["chart_entry_date"], format="%Y-%m-%d")
    out["chart_entry_year"] = out["chart_entry_date"].dt.year.astype(int)
    out["horizon_weeks"] = horizon
    out["entry_confirmed"] = 1
    out["required_window_source_complete"] = 1
    return out


def add_rank200_threshold(frame: pd.DataFrame, rank200: pd.DataFrame) -> pd.DataFrame:
    thresholds = rank200[["chart_week", "weekly_streams_at_rank_200"]].copy()
    thresholds["chart_week"] = pd.to_datetime(thresholds["chart_week"], format="%Y-%m-%d")
    thresholds = thresholds.rename(columns={
        "chart_week": "chart_entry_date",
        "weekly_streams_at_rank_200": "rank_200_stream_threshold_at_entry",
    })
    before = len(frame)
    merged = frame.merge(thresholds, on="chart_entry_date", how="left", validate="many_to_one")
    assert len(merged) == before
    if merged["rank_200_stream_threshold_at_entry"].isna().any():
        raise RuntimeError("A clean chart-entry week lacks its rank-200 stream threshold.")
    return merged


def build_550(clean: pd.DataFrame, source_550: pd.DataFrame, rank200: pd.DataFrame, horizon: int) -> pd.DataFrame:
    match_ids = set(clean.loc[clean["match_550k_exact_id"].eq(1), "spotify_track_id"])
    outcome = base_outcome_columns(clean.loc[clean["spotify_track_id"].isin(match_ids)], horizon)
    source = source_550.loc[source_550["id"].isin(match_ids)].copy()
    source = lyric_fields(source)
    if not source["id"].is_unique:
        raise RuntimeError("550k source is not one row per exact track ID.")
    source = source.rename(columns={
        "id": "spotify_track_id",
        "name": "feature_track_name",
        "album_name": "feature_album_name",
        "artists": "artists_original",
        "year": "release_year",
        "genre": "genre_raw",
        "artist_ids": "artist_ids_original",
        "niche_genres": "niche_genres_raw",
        "lyrics": "lyrics_text",
    })
    selected = [
        "spotify_track_id", "feature_track_name", "feature_album_name", "artists_original",
        *AUDIO, "release_year", "genre_raw", "artist_ids_original", "niche_genres_raw", "lyrics_text",
        "lyrics_missing_source_value", "lyrics_blank", "lyrics_placeholder", "lyrics_character_count",
        "lyrics_word_count", "lyrics_extremely_short", "lyrics_usable_provisional",
        "lyrics_quality_reason", "lyrics_normalised_sha256", "lyrics_duplicate_text",
        "lyrics_duplicate_group_size",
    ]
    before = len(outcome)
    merged = outcome.merge(source[selected], on="spotify_track_id", how="left", validate="one_to_one")
    assert len(merged) == before == len(match_ids)
    merged = add_rank200_threshold(merged, rank200)
    for col in AUDIO + ["release_year"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["release_age_years"] = merged["chart_entry_year"] - merged["release_year"]
    merged["provisional_cohort_version"] = "pre_model_analysis_ready_v1"
    return merged.sort_values("spotify_track_id").reset_index(drop=True)


def build_160(clean: pd.DataFrame, source_160: pd.DataFrame, rank200: pd.DataFrame, horizon: int) -> pd.DataFrame:
    match_ids = set(clean.loc[clean["match_160k_exact_id"].eq(1), "spotify_track_id"])
    outcome = base_outcome_columns(clean.loc[clean["spotify_track_id"].isin(match_ids)], horizon)
    source = source_160.loc[source_160["id"].isin(match_ids)].copy()
    if not source["id"].is_unique:
        raise RuntimeError("160k source is not one row per exact track ID.")
    source = source.rename(columns={
        "id": "spotify_track_id",
        "name": "feature_track_name",
        "artists": "artists_original",
        "year": "release_year",
    })
    source["release_date_precision"] = source["release_date"].fillna("").astype(str).str.len().map({4: "year", 7: "month", 10: "day"}).fillna("unknown")
    safe = [
        "spotify_track_id", "feature_track_name", "artists_original", *AUDIO,
        "explicit", "release_year", "release_date", "release_date_precision",
    ]
    before = len(outcome)
    merged = outcome.merge(source[safe], on="spotify_track_id", how="left", validate="one_to_one")
    assert len(merged) == before == len(match_ids)
    merged = add_rank200_threshold(merged, rank200)
    for col in AUDIO + ["explicit", "release_year"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["release_age_years"] = merged["chart_entry_year"] - merged["release_year"]
    merged["provisional_cohort_version"] = "pre_model_analysis_ready_v1"
    return merged.sort_values("spotify_track_id").reset_index(drop=True)


def exclusive_reason(row: pd.Series, match_field: str, label: str) -> str:
    outcome_reason = row["outcome_exclusion_reason_exclusive"]
    if outcome_reason != "included_clean_outcome":
        return outcome_reason
    if row[match_field] != 1:
        return f"missing_{label}_exact_id_match"
    return "included_analysis_ready"


def make_exclusion_register(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for horizon, frame in frames.items():
        for _, row in frame.iterrows():
            records.append({
                "horizon_weeks": horizon,
                "spotify_track_id": row["spotify_track_id"],
                "chart_track": row["chart_track"],
                "chart_artist": row["chart_artist"],
                "first_observed_chart_entry": row["first_observed_chart_entry"],
                "left_truncated_flag": int(row["left_truncated"]),
                "entry_ambiguous_due_source_gap_flag": int(row["entry_ambiguous_due_source_gap"]),
                "uncertain_entry_flag": int(row["uncertain_entry_flag"]),
                "insufficient_follow_up_flag": int(row["insufficient_follow_up_flag"]),
                "missing_source_week_contamination_flag": int(row["missing_source_week_contamination_flag"]),
                "match_550k_exact_id": int(row["match_550k_exact_id"]),
                "match_160k_exact_id": int(row["match_160k_exact_id"]),
                "outcome_exclusion_reason_exclusive": row["outcome_exclusion_reason_exclusive"],
                "included_clean_outcome": int(row["included_clean_outcome"]),
                "analysis_550k_status_exclusive": exclusive_reason(row, "match_550k_exact_id", "550k"),
                "analysis_160k_status_exclusive": exclusive_reason(row, "match_160k_exact_id", "160k"),
                "analysis_specific_exclusion": int(row["included_clean_outcome"] == 0 or row["match_550k_exact_id"] == 0 or row["match_160k_exact_id"] == 0),
                "permanent_source_exclusion": 0,
                "other_reason": "",
            })
    return pd.DataFrame(records)


def make_allow_block() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(variable, status, role, timing, applies, rationale, present=True):
        rows.append({
            "variable_or_group": variable, "status": status, "role": role,
            "timing": timing, "applies_to": applies, "rationale": rationale,
            "present_in_analysis_ready_file": int(present),
            "permitted_in_future_predictor_matrix": int(status == "allow"),
        })

    add("spotify_track_id", "audit_only", "join key", "entry", "both", "Identifier only; never encoded as a numeric predictor.")
    for variable in AUDIO:
        add(variable, "allow", "baseline audio descriptor", "pre-entry/static recording", "both", "Exact-track audio descriptor; retain after range validation.")
    add("explicit", "allow", "baseline recording descriptor", "pre-entry/static recording", "160k", "Available in the exact-track 160k source.")
    add("release_year", "allow", "baseline release timing", "pre-entry", "both", "Release year exists before chart entry.")
    add("release_date", "allow", "baseline release timing", "pre-entry", "160k", "Mixed source precision is retained explicitly.")
    add("release_age_years", "allow", "derived baseline release timing", "entry", "both", "Chart-entry year minus release year; do not retain with both component years without addressing exact collinearity.")
    add("genre_raw", "pending", "baseline taxonomy", "timing/provenance unresolved", "550k", "Retained in raw form; not authorised for modelling until taxonomy provenance is accepted.")
    add("niche_genres_raw", "pending", "baseline taxonomy", "timing/provenance unresolved", "550k", "Multi-label artist/track taxonomy snapshot; retain without aggressive collapsing.")
    add("lyrics_text", "allow", "baseline content", "available at release", "550k", "Raw lyrics retained for later train-only text processing; quality flags are preserved.")
    add("lyrics_character_count", "allow", "simple lyric descriptor", "available at release", "550k", "Transparent character count; final NLP remains deferred.")
    add("lyrics_word_count", "allow", "simple lyric descriptor", "available at release", "550k", "Transparent word count; final NLP remains deferred.")
    add("lyrics quality indicators", "audit_only", "quality control", "available at preprocessing", "550k", "Used to document usability; not automatically a substantive predictor.")
    add("X", "allow", "early chart performance", "available after calendar week 4", "both", "Count of appearances in weeks 1–4; not an admission filter.")
    add("chart_entry_year", "allow", "calendar-time control", "entry", "both", "Leakage-safe calendar context; final specification remains undecided.")
    add("rank_200_stream_threshold_at_entry", "allow", "calendar-time/chart-threshold control", "entry week", "both", "Observed in the entry week; final use remains provisional.")
    add("chart artist / grouping artist", "grouping_only", "grouped validation key", "entry", "both", "Use for grouped validation, not automatic one-hot prediction.")
    for variable, rationale in [
        ("track popularity", "Later snapshot may reflect the outcome."),
        ("artist popularity", "Later snapshot may reflect later success."),
        ("artist follower counts", "Follower counts can accumulate after chart entry."),
        ("average artist popularity", "Later snapshot is timing-unsafe."),
        ("future ranks", "Observed during the outcome window."),
        ("future streams", "Observed during the outcome window."),
        ("eventual peak rank", "Uses post-window chart history."),
        ("first-spell duration", "Outcome information when predicting fixed-window Y."),
        ("total chart duration", "Full-history outcome information."),
        ("later spell counts / re-entry summaries / last active date", "Full-history outcome information."),
        ("old two-week/four-week survivor eligibility flags", "They select songs that already survived continuously."),
    ]:
        add(variable, "block", "prohibited predictor", "post-entry or timing-unsafe", "both", rationale, present=False)
    add("Y_52 / Y_26", "outcome_only", "primary/sensitivity outcome", "weeks 5 to horizon", "both", "Target only; never a predictor.")
    add("L_52 / L_26", "descriptive_only", "outcome descriptor", "full fixed window", "both", "Retained for interpretation; never a predictor of Y.")
    add("Z_count_52 / Z_count_26 / Z_reentry_52 / Z_reentry_26", "descriptive_only", "outcome fragmentation", "full fixed window", "both", "Retained for interpretation; never a predictor of Y.")
    return pd.DataFrame(rows)


def lyric_audit(cohorts: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    summaries = []
    duplicate_groups = []
    for horizon, frame in cohorts.items():
        label = f"550k_{horizon}w"
        columns = [
            "spotify_track_id", "chart_track", "chart_artist", "lyrics_missing_source_value",
            "lyrics_blank", "lyrics_placeholder", "lyrics_character_count", "lyrics_word_count",
            "lyrics_extremely_short", "lyrics_usable_provisional", "lyrics_quality_reason",
            "lyrics_normalised_sha256", "lyrics_duplicate_text", "lyrics_duplicate_group_size",
        ]
        temp = frame[columns].copy()
        temp.insert(0, "cohort", label)
        temp.insert(1, "horizon_weeks", horizon)
        records.append(temp)
        char = temp["lyrics_character_count"]
        words = temp["lyrics_word_count"]
        summaries.append({
            "cohort": label, "horizon_weeks": horizon, "tracks": len(temp),
            "missing_lyrics": int(temp["lyrics_missing_source_value"].sum()),
            "blank_lyrics": int(temp["lyrics_blank"].sum()),
            "placeholder_lyrics": int(temp["lyrics_placeholder"].sum()),
            "extremely_short_lyrics": int(temp["lyrics_extremely_short"].sum()),
            "usable_lyrics": int(temp["lyrics_usable_provisional"].sum()),
            "usable_lyrics_percent": float(100 * temp["lyrics_usable_provisional"].mean()),
            "duplicate_text_tracks": int(temp["lyrics_duplicate_text"].sum()),
            "duplicate_text_groups": int(temp.loc[temp["lyrics_duplicate_text"].eq(1), "lyrics_normalised_sha256"].nunique()),
            "characters_min": int(char.min()), "characters_p25": q(char, .25), "characters_median": q(char, .5),
            "characters_mean": float(char.mean()), "characters_p75": q(char, .75), "characters_p95": q(char, .95), "characters_max": int(char.max()),
            "words_min": int(words.min()), "words_p25": q(words, .25), "words_median": q(words, .5),
            "words_mean": float(words.mean()), "words_p75": q(words, .75), "words_p95": q(words, .95), "words_max": int(words.max()),
            "provisional_policy": "Usable if non-missing, non-blank, not a recognised placeholder, at least 50 characters and at least 10 words; no track removed automatically.",
        })
        for digest, group in temp.loc[temp["lyrics_duplicate_text"].eq(1)].groupby("lyrics_normalised_sha256"):
            duplicate_groups.append({
                "cohort": label, "horizon_weeks": horizon, "lyrics_normalised_sha256": digest,
                "tracks_in_group": len(group), "spotify_track_ids": "|".join(sorted(group["spotify_track_id"])),
                "track_titles": " | ".join(group["chart_track"].astype(str).head(8)),
            })
    return pd.concat(records, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(duplicate_groups)


def genre_audit(cohorts: dict[int, pd.DataFrame], artist_ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    artist_lookup = artist_ref.set_index("id").to_dict("index")
    source_descriptions = [
        ("data/raw/spotify_550k/songs.csv", "genre", "song-row single category", "Broad label stored on each song row; derivation and snapshot date are undocumented."),
        ("data/raw/spotify_550k/songs.csv", "niche_genres", "song-row multi-label field", "Broad and niche labels coexist; likely linked to artist taxonomy, but exact derivation is undocumented."),
        ("data/raw/spotify_550k/artists.csv", "genres", "artist-level multi-label field", "Artist-level taxonomy snapshot; timing is undocumented."),
        ("data/raw/spotify_550k/artists.csv", "main_genre", "artist-level single category", "Artist-level broad label; timing is undocumented."),
        ("data/raw/spotify_160k/data.csv", "none", "track-level", "The exact-ID track table has no genre field."),
        ("data/raw/spotify_160k/data_w_genres.csv", "genres", "artist/aggregated table", "Not an exact track-ID keyed field and therefore not forced into the robustness cohort."),
    ]
    for source, field, level, note in source_descriptions:
        rows.append({
            "record_type": "field_inventory", "cohort": "source_inventory", "horizon_weeks": "",
            "source_file": source, "field": field, "level": level, "category": "", "count": "",
            "percent_of_tracks": "", "missing_count": "", "unique_categories": "", "rare_categories_lt10": "",
            "very_rare": "", "notes": note,
        })

    for horizon, frame in cohorts.items():
        label = f"550k_{horizon}w"
        genre = frame["genre_raw"].fillna("").astype(str).str.strip()
        counts = genre[genre.ne("")].value_counts()
        rows.append({
            "record_type": "field_summary", "cohort": label, "horizon_weeks": horizon,
            "source_file": "data/raw/spotify_550k/songs.csv", "field": "genre", "level": "song-row single category",
            "category": "", "count": len(frame), "percent_of_tracks": 100.0,
            "missing_count": int(genre.eq("").sum()), "unique_categories": int(len(counts)),
            "rare_categories_lt10": int((counts < 10).sum()), "very_rare": "", "notes": "Raw broad category preserved; no collapsing applied.",
        })
        for category, count in counts.items():
            rows.append({
                "record_type": "category_frequency", "cohort": label, "horizon_weeks": horizon,
                "source_file": "data/raw/spotify_550k/songs.csv", "field": "genre", "level": "song-row single category",
                "category": category, "count": int(count), "percent_of_tracks": 100 * count / len(frame),
                "missing_count": "", "unique_categories": "", "rare_categories_lt10": "",
                "very_rare": int(count < 10), "notes": "",
            })
        niche_by_track = frame["niche_genres_raw"].map(parse_list)
        niche_counts = Counter(g for values in niche_by_track for g in values)
        rows.append({
            "record_type": "field_summary", "cohort": label, "horizon_weeks": horizon,
            "source_file": "data/raw/spotify_550k/songs.csv", "field": "niche_genres", "level": "song-row multi-label",
            "category": "", "count": len(frame), "percent_of_tracks": 100.0,
            "missing_count": int(sum(not values for values in niche_by_track)), "unique_categories": len(niche_counts),
            "rare_categories_lt10": sum(count < 10 for count in niche_counts.values()), "very_rare": "",
            "notes": "Multiple categories per track; percentages can sum above 100%; no collapsing applied.",
        })
        for category, count in sorted(niche_counts.items(), key=lambda x: (-x[1], x[0])):
            rows.append({
                "record_type": "category_frequency", "cohort": label, "horizon_weeks": horizon,
                "source_file": "data/raw/spotify_550k/songs.csv", "field": "niche_genres", "level": "song-row multi-label",
                "category": category, "count": count, "percent_of_tracks": 100 * count / len(frame),
                "missing_count": "", "unique_categories": "", "rare_categories_lt10": "",
                "very_rare": int(count < 10), "notes": "Track-level incidence; multi-label field.",
            })

        primary_ids = frame["artist_ids_original"].map(parse_list).map(lambda values: values[0] if values else "")
        primary_artist_rows = primary_ids.map(lambda ident: artist_lookup.get(ident, {}))
        main_genres = primary_artist_rows.map(lambda row: str(row.get("main_genre", "")).strip())
        main_counts = main_genres[main_genres.ne("")].value_counts()
        rows.append({
            "record_type": "field_summary", "cohort": label, "horizon_weeks": horizon,
            "source_file": "data/raw/spotify_550k/artists.csv", "field": "main_genre", "level": "primary-artist single category mapped to tracks",
            "category": "", "count": len(frame), "percent_of_tracks": 100.0,
            "missing_count": int(main_genres.eq("").sum()), "unique_categories": int(len(main_counts)),
            "rare_categories_lt10": int((main_counts < 10).sum()), "very_rare": "",
            "notes": "Artist-level snapshot mapped through the first credited Spotify artist ID; timing is undocumented.",
        })
        for category, count in main_counts.items():
            rows.append({
                "record_type": "category_frequency", "cohort": label, "horizon_weeks": horizon,
                "source_file": "data/raw/spotify_550k/artists.csv", "field": "main_genre", "level": "primary-artist single category mapped to tracks",
                "category": category, "count": int(count), "percent_of_tracks": 100 * count / len(frame),
                "missing_count": "", "unique_categories": "", "rare_categories_lt10": "",
                "very_rare": int(count < 10), "notes": "Artist categories are repeated across that artist's tracks.",
            })
        artist_genre_lists = primary_artist_rows.map(lambda row: parse_list(row.get("genres", "")))
        artist_genre_counts = Counter(g for values in artist_genre_lists for g in values)
        rows.append({
            "record_type": "field_summary", "cohort": label, "horizon_weeks": horizon,
            "source_file": "data/raw/spotify_550k/artists.csv", "field": "genres", "level": "primary-artist multi-label mapped to tracks",
            "category": "", "count": len(frame), "percent_of_tracks": 100.0,
            "missing_count": int(sum(not values for values in artist_genre_lists)), "unique_categories": len(artist_genre_counts),
            "rare_categories_lt10": sum(count < 10 for count in artist_genre_counts.values()), "very_rare": "",
            "notes": "Artist-level multi-label snapshot mapped through the primary credited artist; percentages can exceed 100%.",
        })
        for category, count in sorted(artist_genre_counts.items(), key=lambda x: (-x[1], x[0])):
            rows.append({
                "record_type": "category_frequency", "cohort": label, "horizon_weeks": horizon,
                "source_file": "data/raw/spotify_550k/artists.csv", "field": "genres", "level": "primary-artist multi-label mapped to tracks",
                "category": category, "count": count, "percent_of_tracks": 100 * count / len(frame),
                "missing_count": "", "unique_categories": "", "rare_categories_lt10": "",
                "very_rare": int(count < 10), "notes": "Artist categories are repeated across that artist's tracks.",
            })
    return pd.DataFrame(rows)


def track_artist_bridge(frame: pd.DataFrame, artist_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = artist_ref.set_index("id").to_dict("index")
    records = []
    mismatches = 0
    for _, row in frame.iterrows():
        names = parse_list(row["artists_original"])
        ids = parse_list(row["artist_ids_original"])
        if len(names) != len(ids):
            mismatches += 1
        length = max(len(names), len(ids), 1)
        for index in range(length):
            artist_id = ids[index] if index < len(ids) else ""
            credited_name = names[index] if index < len(names) else ""
            ref = lookup.get(artist_id, {})
            records.append({
                "spotify_track_id": row["spotify_track_id"], "chart_track": row["chart_track"],
                "chart_artist_original": row["chart_artist"], "spotify_artists_original": row["artists_original"],
                "spotify_artist_ids_original": row["artist_ids_original"], "credit_order": index + 1,
                "artist_id": artist_id, "credited_artist_name": credited_name,
                "artist_reference_name": ref.get("name", ""), "artist_genres_raw": ref.get("genres", ""),
                "artist_main_genre_raw": ref.get("main_genre", ""), "is_grouping_artist": int(index == 0),
                "proposed_grouping_rule": "Use the first credited Spotify artist ID as the grouping key; retain every credited collaborator as a separate bridge row.",
                "artist_name_id_count_mismatch": int(len(names) != len(ids)),
            })
    bridge = pd.DataFrame(records).sort_values(["spotify_track_id", "credit_order"])
    grouping = bridge.loc[bridge["is_grouping_artist"].eq(1)].copy()
    counts = grouping.groupby(["artist_id", "credited_artist_name"], dropna=False)["spotify_track_id"].nunique().reset_index(name="tracks")
    bins = [
        ("1", lambda x: x == 1), ("2", lambda x: x == 2), ("3-5", lambda x: (x >= 3) & (x <= 5)),
        ("6-10", lambda x: (x >= 6) & (x <= 10)), (">10", lambda x: x > 10),
    ]
    summary = []
    for label, condition in bins:
        subset = counts.loc[condition(counts["tracks"])]
        summary.append({
            "record_type": "track_count_bin", "category": label, "artist_id": "", "artist_name": "",
            "artist_count": len(subset), "track_count": int(subset["tracks"].sum()), "percent_of_tracks": 100 * subset["tracks"].sum() / len(frame),
            "value": "", "notes": "Grouping based on first credited Spotify artist ID in the 52-week 550k cohort.",
        })
    multi = counts.loc[counts["tracks"] >= 2]
    summary.extend([
        {"record_type": "overall", "category": "unique_grouping_artists", "artist_id": "", "artist_name": "", "artist_count": len(counts), "track_count": len(frame), "percent_of_tracks": 100.0, "value": len(counts), "notes": ""},
        {"record_type": "overall", "category": "tracks_with_grouping_artist_repeated", "artist_id": "", "artist_name": "", "artist_count": len(multi), "track_count": int(multi["tracks"].sum()), "percent_of_tracks": 100 * multi["tracks"].sum() / len(frame), "value": int(multi["tracks"].sum()), "notes": "Tracks whose proposed grouping artist has at least two cohort tracks."},
        {"record_type": "overall", "category": "maximum_tracks_per_grouping_artist", "artist_id": "", "artist_name": "", "artist_count": "", "track_count": "", "percent_of_tracks": "", "value": int(counts["tracks"].max()), "notes": ""},
        {"record_type": "quality", "category": "name_id_count_mismatch_tracks", "artist_id": "", "artist_name": "", "artist_count": "", "track_count": "", "percent_of_tracks": "", "value": mismatches, "notes": "No collaborator was discarded; mismatches remain flagged in the bridge."},
    ])
    top = counts.sort_values(["tracks", "credited_artist_name"], ascending=[False, True]).head(20)
    for _, row in top.iterrows():
        summary.append({
            "record_type": "top_20_artist", "category": "", "artist_id": row["artist_id"],
            "artist_name": row["credited_artist_name"], "artist_count": "", "track_count": int(row["tracks"]),
            "percent_of_tracks": 100 * row["tracks"] / len(frame), "value": "", "notes": "",
        })
    return bridge, pd.DataFrame(summary)


def invalid_and_numeric_audit(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    missing_rows = []
    invalid_rows = []
    for cohort_name, frame in cohorts.items():
        horizon = int(re.search(r"(26|52)w", cohort_name).group(1))
        dataset = "550k" if cohort_name.startswith("550k") else "160k"
        numeric_candidates = [c for c in AUDIO + ["release_year", "release_age_years", "chart_entry_year", "rank_200_stream_threshold_at_entry", "X"] if c in frame]
        if dataset == "550k":
            numeric_candidates += ["lyrics_character_count", "lyrics_word_count"]
        for col in frame.columns:
            text = frame[col].astype("string").str.strip()
            missing_mask = frame[col].isna() | text.eq("")
            if col in {"niche_genres_raw", "artists_original"}:
                missing_mask = missing_mask | text.isin(["[]", "{}"])
            missing = int(missing_mask.sum())
            missing_rows.append({
                "cohort": cohort_name, "dataset": dataset, "horizon_weeks": horizon,
                "variable": col, "rows": len(frame), "missing_or_blank": missing,
                "missing_or_blank_percent": 100 * missing / len(frame),
            })
        for col in numeric_candidates:
            values = pd.to_numeric(frame[col], errors="coerce")
            valid = values.dropna()
            invalid_mask = pd.Series(False, index=frame.index)
            suspicious_mask = pd.Series(False, index=frame.index)
            invalid_rule = ""
            suspicious_rule = ""
            if col in UNIT_INTERVAL:
                invalid_mask = values.notna() & ~values.between(0, 1)
                invalid_rule = "outside [0,1]"
            elif col == "key":
                invalid_mask = values.notna() & ~values.between(0, 11)
                invalid_rule = "outside integer keys 0-11"
            elif col == "mode":
                invalid_mask = values.notna() & ~values.isin([0, 1])
                invalid_rule = "not 0 or 1"
            elif col == "tempo":
                invalid_mask = values.notna() & (values < 0)
                suspicious_mask = values.eq(0)
                invalid_rule = "negative tempo"
                suspicious_rule = "zero tempo"
            elif col == "duration_ms":
                invalid_mask = values.notna() & (values <= 0)
                suspicious_mask = values.notna() & ((values < 30000) | (values > 1800000))
                invalid_rule = "non-positive duration"
                suspicious_rule = "duration below 30 seconds or above 30 minutes"
            elif col == "release_year":
                invalid_mask = values.notna() & ((values < 1900) | (values > frame["chart_entry_year"]))
                invalid_rule = "before 1900 or after chart-entry year"
            elif col == "release_age_years":
                invalid_mask = values.notna() & (values < 0)
                suspicious_mask = values.notna() & (values > 100)
                invalid_rule = "negative release age"
                suspicious_rule = "release age above 100 years"
            elif col == "loudness":
                suspicious_mask = values.notna() & ((values < -60) | (values > 0))
                suspicious_rule = "outside typical Spotify range -60 to 0 dB"
            elif col in {"lyrics_character_count", "lyrics_word_count", "X", "chart_entry_year", "rank_200_stream_threshold_at_entry"}:
                invalid_mask = values.notna() & (values < 0)
                invalid_rule = "negative count/year/threshold"
            q1 = valid.quantile(.25) if len(valid) else np.nan
            q3 = valid.quantile(.75) if len(valid) else np.nan
            iqr = q3 - q1 if len(valid) else np.nan
            extreme = valid[(valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)] if len(valid) and iqr > 0 else pd.Series(dtype=float)
            audit_rows.append({
                "cohort": cohort_name, "dataset": dataset, "horizon_weeks": horizon, "variable": col,
                "rows": len(frame), "non_missing": int(values.notna().sum()), "missing": int(values.isna().sum()),
                "minimum": float(valid.min()) if len(valid) else np.nan, "p01": q(valid, .01) if len(valid) else np.nan,
                "p25": q1, "median": q(valid, .5) if len(valid) else np.nan, "mean": float(valid.mean()) if len(valid) else np.nan,
                "p75": q3, "p99": q(valid, .99) if len(valid) else np.nan, "maximum": float(valid.max()) if len(valid) else np.nan,
                "standard_deviation": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                "skewness": float(valid.skew()) if len(valid) > 2 else np.nan, "iqr_extreme_count": len(extreme),
                "invalid_count": int(invalid_mask.sum()), "invalid_rule": invalid_rule,
                "suspicious_count": int(suspicious_mask.sum()), "suspicious_rule": suspicious_rule,
                "outlier_action": "Retained; statistical outliers are not removed automatically.",
            })
            for idx in frame.index[invalid_mask | suspicious_mask]:
                invalid_rows.append({
                    "cohort": cohort_name, "dataset": dataset, "horizon_weeks": horizon,
                    "spotify_track_id": frame.loc[idx, "spotify_track_id"], "variable": col,
                    "observed_value": values.loc[idx], "classification": "invalid" if invalid_mask.loc[idx] else "suspicious",
                    "rule": invalid_rule if invalid_mask.loc[idx] else suspicious_rule,
                    "action": "Retain with flag pending review; no automatic track deletion.",
                })
    return pd.DataFrame(audit_rows), pd.DataFrame(missing_rows), pd.DataFrame(invalid_rows)


def vif_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in frame]
    data = frame[available].apply(pd.to_numeric, errors="coerce").dropna()
    nonconstant = [c for c in available if data[c].nunique() > 1]
    data = data[nonconstant]
    z = (data - data.mean()) / data.std(ddof=0)
    rows = []
    for col in nonconstant:
        y = z[col].to_numpy(float)
        others = [c for c in nonconstant if c != col]
        X = np.column_stack([np.ones(len(z)), z[others].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        sse = float(np.sum(residual ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst else np.nan
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        rows.append({
            "variable": col, "complete_case_n": len(z), "r_squared_against_other_candidates": r2,
            "tolerance": 1 / vif if np.isfinite(vif) and vif else 0,
            "vif": vif, "flag_vif_5plus": int(vif >= 5), "flag_vif_10plus": int(vif >= 10),
            "interpretation": "Exploratory only; reassess after the final predictor set and train/test design are agreed.",
        })
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def correlation_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_predictors = [
        *AUDIO, "release_year", "release_age_years", "lyrics_character_count", "lyrics_word_count",
        "X", "chart_entry_year", "rank_200_stream_threshold_at_entry",
    ]
    numeric_predictors = [c for c in numeric_predictors if c in frame]
    data = frame[numeric_predictors].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    pairs = []
    for method, matrix in (("Pearson", pearson), ("Spearman", spearman)):
        for i, left in enumerate(matrix.columns):
            for right in matrix.columns[i + 1:]:
                value = matrix.loc[left, right]
                if pd.notna(value) and abs(value) >= .70:
                    pairs.append({
                        "method": method, "variable_1": left, "variable_2": right,
                        "correlation": value, "absolute_correlation": abs(value),
                        "above_0_70": int(abs(value) >= .70), "above_0_80": int(abs(value) >= .80),
                        "above_0_90": int(abs(value) >= .90),
                        "action": "Flag for later theory/VIF review; do not remove automatically.",
                    })
    y = "Y_52"
    exploratory_vars = ["X", y, "L_52", "Z_count_52", "Z_reentry_52", *AUDIO]
    exploratory = frame[exploratory_vars].apply(pd.to_numeric, errors="coerce")
    exp_rows = []
    pairs_to_report = [("X", y), (y, "L_52"), (y, "Z_count_52"), (y, "Z_reentry_52")]
    pairs_to_report += [("X", v) for v in AUDIO]
    for a, b in pairs_to_report:
        exp_rows.append({
            "variable_1": a, "variable_2": b,
            "pearson": exploratory[a].corr(exploratory[b], method="pearson"),
            "spearman": exploratory[a].corr(exploratory[b], method="spearman"),
            "role_note": "Outcome/descriptive comparison" if y in {a, b} or a.startswith("L_") or a.startswith("Z_") or b.startswith("L_") or b.startswith("Z_") else "Early X versus baseline audio descriptor",
        })
    return pearson, spearman, pd.DataFrame(pairs).sort_values(["method", "absolute_correlation"], ascending=[True, False]), pd.DataFrame(exp_rows)


def outcome_outputs(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    frequency_rows = []
    for name, frame in cohorts.items():
        horizon = 52 if name.endswith("52w") else 26
        variables = ["X", f"Y_{horizon}", f"L_{horizon}", f"Z_count_{horizon}", f"Z_reentry_{horizon}"]
        for variable in variables:
            summary_rows.append(distribution_record(name, horizon, variable, frame[variable]))
            counts = frame[variable].value_counts(dropna=False).sort_index()
            for value, count in counts.items():
                frequency_rows.append({
                    "cohort": name, "horizon_weeks": horizon, "variable": variable,
                    "value": value, "count": int(count), "percent": 100 * count / len(frame),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(frequency_rows)


def entry_year_rows(clean: dict[int, pd.DataFrame], cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    all_frames = {f"all_clean_{h}w": base_outcome_columns(frame, h) for h, frame in clean.items()}
    all_frames.update(cohorts)
    for name, frame in all_frames.items():
        horizon = 52 if name.endswith("52w") else 26
        counts = frame["chart_entry_year"].value_counts().sort_index()
        for year, count in counts.items():
            rows.append({
                "cohort": name, "horizon_weeks": horizon, "chart_entry_year": int(year),
                "tracks": int(count), "percent_of_cohort": 100 * count / len(frame),
                "cohort_total": len(frame),
            })
    return pd.DataFrame(rows)


def plot_hist(frame: pd.DataFrame, variable: str, title: str, path: Path, max_value: int) -> None:
    values = frame[variable].astype(int)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bins = np.arange(-.5, max_value + 1.5, 1)
    ax.hist(values, bins=bins, color="#2A6496", edgecolor="white", linewidth=.4)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#17365D")
    ax.set_xlabel("Future chart-presence weeks")
    ax.set_ylabel("Tracks")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_small_distributions(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
    variables = [("X", "Early presence X"), ("L_52", "Last active week L"), ("Z_count_52", "Presence blocks Z"), ("Z_reentry_52", "More than one block")]
    for ax, (variable, title) in zip(axes.flat, variables):
        counts = frame[variable].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#2A6496")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#17365D")
        ax.set_ylabel("Tracks")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if variable == "L_52" and len(counts) > 20:
            ax.set_xticks([])
            ax.set_xlabel("Week number (0 = no future presence)")
    fig.suptitle("Early presence and fixed-window outcome descriptors — 550k-linked 52-week cohort", x=.08, ha="left", fontsize=13, fontweight="bold", color="#17365D")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 8.8))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#17365D")
    cbar = fig.colorbar(im, ax=ax, shrink=.78)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_entry_year(entry: pd.DataFrame, path: Path) -> None:
    subset = entry.loc[entry["cohort"].eq("550k_52w")]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(subset["chart_entry_year"].astype(str), subset["tracks"], color="#2A6496")
    ax.set_title("Chart-entry year composition — 550k-linked 52-week cohort", loc="left", fontsize=12.5, fontweight="bold", color="#17365D")
    ax.set_xlabel("Chart-entry year")
    ax.set_ylabel("Tracks")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cohort_summary(register: pd.DataFrame, cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        scoped = register.loc[register["horizon_weeks"].eq(horizon)]
        for cohort, status_col in (("clean_outcome", "outcome_exclusion_reason_exclusive"), ("550k", "analysis_550k_status_exclusive"), ("160k", "analysis_160k_status_exclusive")):
            counts = scoped[status_col].value_counts()
            for reason, count in counts.items():
                rows.append({
                    "horizon_weeks": horizon, "cohort": cohort, "count_type": "mutually_exclusive_sequential_status",
                    "reason_or_status": reason, "tracks": int(count), "categories_mutually_exclusive": 1,
                    "priority_order": "uncertain entry -> insufficient follow-up -> source-gap contamination -> missing feature match -> included",
                })
        for flag in ["uncertain_entry_flag", "insufficient_follow_up_flag", "missing_source_week_contamination_flag"]:
            rows.append({
                "horizon_weeks": horizon, "cohort": "all_chart", "count_type": "overlapping_raw_flag",
                "reason_or_status": flag, "tracks": int(scoped[flag].sum()), "categories_mutually_exclusive": 0,
                "priority_order": "Flags may overlap; use sequential statuses for reconciliation.",
            })
    for name, frame in cohorts.items():
        rows.append({
            "horizon_weeks": 52 if name.endswith("52w") else 26, "cohort": name,
            "count_type": "final_analysis_ready", "reason_or_status": "included_analysis_ready",
            "tracks": len(frame), "categories_mutually_exclusive": 1, "priority_order": "Final exact-ID linked clean cohort.",
        })
    return pd.DataFrame(rows)


def make_comparison(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        a = cohorts[f"550k_{horizon}w"]
        b = cohorts[f"160k_{horizon}w"]
        y = f"Y_{horizon}"
        for name, frame in (("550k", a), ("160k", b)):
            rows.append({
                "horizon_weeks": horizon, "feature_cohort": name, "tracks": len(frame),
                "Y_mean": frame[y].mean(), "Y_median": frame[y].median(), "Y_zero_percent": 100 * frame[y].eq(0).mean(),
                "entry_year_mean": frame["chart_entry_year"].mean(), "release_year_mean": frame["release_year"].mean(),
                "lyrics_available": int(name == "550k"), "genre_available": int(name == "550k"),
                "main_difference": "Richer lyrics and genre fields" if name == "550k" else "Larger exact-ID overlap but no track-level lyrics or genre",
            })
    return pd.DataFrame(rows)


def report_text(
    cohort_counts: pd.DataFrame,
    cohorts: dict[str, pd.DataFrame],
    lyric_summary: pd.DataFrame,
    genre: pd.DataFrame,
    artist_summary: pd.DataFrame,
    numeric: pd.DataFrame,
    outcomes: pd.DataFrame,
    exp_corr: pd.DataFrame,
    high_corr: pd.DataFrame,
    vif: pd.DataFrame,
    entry: pd.DataFrame,
) -> str:
    def n(name):
        return len(cohorts[name])

    lines = [
        "# Pre-model analysis-ready preprocessing and exploratory analysis",
        "",
        "## Scope and provisional status",
        "",
        "This report records a new versioned preprocessing and exploratory-analysis layer. No predictive, count, survival, machine-learning or hierarchical model was fitted. The definitions below are **provisional working assumptions** for review, not final thesis decisions.",
        "",
        "The provisional primary target is fixed-window future chart presence over 52 calendar weeks. X counts chart appearances in weeks 1–4; Y_52 counts chart-presence weeks from weeks 5–52 and includes re-entry. L_52, Z_count_52 and Z_reentry_52 are descriptive outcome variables. The equivalent 26-week variables are retained for sensitivity analysis. Existing first-spell survival outputs remain unchanged for a possible future secondary analysis.",
        "",
        "## Clean cohort construction",
        "",
        f"The final exact-ID linked cohorts contain **{n('550k_52w'):,} 550k tracks at 52 weeks**, **{n('550k_26w'):,} 550k tracks at 26 weeks**, **{n('160k_52w'):,} 160k tracks at 52 weeks**, and **{n('160k_26w'):,} 160k tracks at 26 weeks**.",
        "",
        "Exclusion counts are reported in `cohort_summary.csv` using a mutually exclusive sequence: uncertain entry, insufficient follow-up, missing-source-week contamination, missing exact-ID feature match, then inclusion. The register also retains overlapping raw flags. Unknown source weeks were never treated as absences.",
        "",
    ]
    for horizon in HORIZONS:
        lines += [f"### {horizon}-week reconciliation", ""]
        for feature in ("550k", "160k"):
            scoped = cohort_counts[(cohort_counts["horizon_weeks"] == horizon) & (cohort_counts["cohort"] == feature) & (cohort_counts["count_type"] == "mutually_exclusive_sequential_status")]
            mapping = dict(zip(scoped["reason_or_status"], scoped["tracks"]))
            lines += [
                f"For the {feature} cohort, from 4,247 chart tracks: {mapping.get('uncertain_entry', 0):,} were removed for uncertain entry; {mapping.get('insufficient_follow_up', 0):,} for insufficient follow-up after prior exclusions; {mapping.get('missing_source_week_contamination', 0):,} because the required window crossed a missing source week; {mapping.get(f'missing_{feature}_exact_id_match', 0):,} for no exact {feature} match; and {mapping.get('included_analysis_ready', 0):,} remain. Other exclusions: 0.",
                "",
            ]
    lines += ["## Lyrics quality", ""]
    for _, row in lyric_summary.iterrows():
        lines.append(f"In the {int(row['horizon_weeks'])}-week 550k cohort, {int(row['usable_lyrics']):,} of {int(row['tracks']):,} tracks ({row['usable_lyrics_percent']:.1f}%) meet the provisional usability rule. Missing, blank, placeholder, extremely short and duplicated-text flags are preserved at record level; no track was removed solely because of lyric quality.")
        lines.append("")
    genre_summary = genre[(genre["record_type"] == "field_summary") & (genre["cohort"] == "550k_52w")]
    genre_row = genre_summary[genre_summary["field"] == "genre"].iloc[0]
    niche_row = genre_summary[genre_summary["field"] == "niche_genres"].iloc[0]
    lines += [
        "## Genre fields",
        "",
        f"The 52-week 550k cohort contains {int(genre_row['unique_categories'])} broad `genre` categories and {int(niche_row['unique_categories'])} distinct niche labels. Broad and niche labels coexist, and many niche categories are rare. The song file stores both fields on song rows, while the artist file has separate artist-level `genres` and `main_genre` fields. Their derivation and snapshot timing are undocumented, so the raw categories are preserved and aggressive collapsing is deferred. The exact-ID 160k track table has no track-level genre field.",
        "",
        "## Artist bridge and grouped validation",
        "",
    ]
    overall = artist_summary[artist_summary["record_type"] == "overall"].set_index("category")
    lines += [
        f"The proposed grouping rule uses the first credited Spotify artist ID, while all credited collaborators remain in the bridge. The 52-week 550k cohort has {int(overall.loc['unique_grouping_artists', 'value']):,} grouping artists; {overall.loc['tracks_with_grouping_artist_repeated', 'percent_of_tracks']:.1f}% of tracks belong to grouping artists represented more than once, and the maximum is {int(overall.loc['maximum_tracks_per_grouping_artist', 'value'])} tracks for one artist. Spotify artist IDs are preferable to the chart-source artist string for matched tracks, but this grouping is unavailable for unmatched chart tracks and depends on credited order.",
        "",
        "## Numeric validation and missingness",
        "",
    ]
    serious_invalid = numeric[numeric["invalid_count"] > 0]
    suspicious = numeric[numeric["suspicious_count"] > 0]
    skewed = numeric[numeric["skewness"].abs() > 2]
    lines.append(f"The numeric audit found {int(serious_invalid['invalid_count'].sum()) if len(serious_invalid) else 0} invalid value flags and {int(suspicious['suspicious_count'].sum()) if len(suspicious) else 0} suspicious-value flags across the four cohorts. Suspicious values are logged but retained; statistical outliers were not deleted automatically. Strong skewness is present particularly for instrumentalness and release timing, so transformations may later be considered within training data only.")
    lines += ["", "## Outcome distributions", ""]
    for name in ("550k_52w", "550k_26w"):
        horizon = 52 if name.endswith("52w") else 26
        row = outcomes[(outcomes["cohort"] == name) & (outcomes["variable"] == f"Y_{horizon}")].iloc[0]
        lines.append(f"For {name}, Y has N={int(row['n']):,}, mean={row['mean']:.2f}, median={row['median']:.1f}, variance={row['variance']:.2f}, range {row['minimum']:.0f}–{row['maximum']:.0f}, and {row['zero_percent']:.1f}% zeros. The variance-to-mean ratio is {row['variance_to_mean_ratio']:.2f}, indicating substantial overdispersion descriptively; this is not a fitted-model result.")
        lines.append("")
    lines += ["## Correlation and multicollinearity diagnostics", ""]
    for pair in [("X", "Y_52"), ("Y_52", "L_52"), ("Y_52", "Z_count_52")]:
        row = exp_corr[(exp_corr["variable_1"] == pair[0]) & (exp_corr["variable_2"] == pair[1])].iloc[0]
        lines.append(f"{pair[0]} versus {pair[1]}: Pearson {row['pearson']:.3f}; Spearman {row['spearman']:.3f}.")
    lines.append("")
    if len(high_corr):
        strongest_p = high_corr[high_corr["method"] == "Pearson"].sort_values("absolute_correlation", ascending=False).iloc[0]
        strongest_s = high_corr[high_corr["method"] == "Spearman"].sort_values("absolute_correlation", ascending=False).iloc[0]
        lines.append(f"The strongest flagged predictor pair is {strongest_p['variable_1']}–{strongest_p['variable_2']} for Pearson (r={strongest_p['correlation']:.3f}) and {strongest_s['variable_1']}–{strongest_s['variable_2']} for Spearman (ρ={strongest_s['correlation']:.3f}). Correlations above 0.70, 0.80 and 0.90 are flagged without automatic removal.")
        lines.append("")
    high_vif = vif[vif["vif"] >= 5]
    lines.append(f"The exploratory VIF table flags {len(high_vif)} variables at VIF ≥ 5 and {int((vif['vif'] >= 10).sum())} at VIF ≥ 10. These results may reflect overlapping lyric-length or calendar/release variables and must be reconsidered after the final predictor specification.")
    lines += ["", "## 160k robustness preparation", "", "The 160k exact-ID cohorts use the same clean outcome rules and leakage-safe audio/release fields, without lyrics or forced genre linkage. Cohort-level differences are recorded in `cohort_feature_comparison.csv`; no model was fitted.", "", "## Calendar composition", "", "Entry-year composition is saved for the full clean outcome cohort and both feature cohorts at both horizons. The 52-week requirement excludes later entries and therefore changes the calendar-year composition; final time controls and validation design remain undecided.", "", "## Matters to resolve before modelling", "", "- Supervisor confirmation of the primary outcome and main horizon.", "- Whether genre taxonomy provenance is sufficient for predictive use and how broad/niche labels should be represented.", "- Whether the provisional lyric usability rule should define a text-analysis subset or only a quality indicator.", "- Whether the 550k cohort is primary and the 160k cohort a robustness analysis.", "- The artist-grouped and calendar-time-aware validation design.", "- Treatment of suspicious numeric values and any release-date anomalies after record-level review.", "- Train/test splitting before imputation, scaling, encoding or text vectorisation.", "", "## Output status", "", "All outputs are versioned as `pre_model_analysis_ready_v1`. Historical raw, survival, fixed-window diagnostic and preprocessing-audit directories were checksum-guarded and were not overwritten.", ""]
    return "\n".join(lines)


def main() -> None:
    guards_before = {str(path.relative_to(ROOT)): tree_digest(path) for path in TREE_GUARDS}
    prepare_outputs()

    fixed = read_csv(FIXED)
    first = read_csv(FIRST_SPELL, dtype={"spotify_track_id": str})
    export_550 = read_csv(FEATURE_550_EXPORT, dtype={"spotify_track_id": str})
    rank200 = read_csv(RANK200)
    source_160 = read_csv(TRACKS_160, dtype={"id": str, "release_date": str})
    artist_ref = read_csv(ARTISTS_550, dtype={"id": str})

    assert len(first) == 4247 and first["spotify_track_id"].is_unique
    assert len(export_550) == 1534 and export_550["spotify_track_id"].is_unique
    assert source_160["id"].is_unique
    assert artist_ref["id"].is_unique
    assert len(fixed) == 4247 * 4

    clean: dict[int, pd.DataFrame] = {}
    full_flags: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        clean[horizon], full_flags[horizon] = clean_outcome_rows(fixed, horizon)

    required_550 = set().union(*[
        set(frame.loc[frame["match_550k_exact_id"].eq(1), "spotify_track_id"])
        for frame in clean.values()
    ])
    raw_550 = extract_550_rows(required_550)

    cohorts = {
        "550k_52w": build_550(clean[52], raw_550, rank200, 52),
        "550k_26w": build_550(clean[26], raw_550, rank200, 26),
        "160k_52w": build_160(clean[52], source_160, rank200, 52),
        "160k_26w": build_160(clean[26], source_160, rank200, 26),
    }

    output_names = {
        "550k_52w": "analysis_ready_550k_52w.csv",
        "550k_26w": "analysis_ready_550k_26w.csv",
        "160k_52w": "analysis_ready_160k_52w.csv",
        "160k_26w": "analysis_ready_160k_26w.csv",
    }
    for name, frame in cohorts.items():
        if not frame["spotify_track_id"].is_unique:
            raise RuntimeError(f"Duplicate analysis units in {name}.")
        write_csv(frame, output_names[name])

    register = make_exclusion_register(full_flags)
    write_csv(register, "cohort_exclusion_register.csv")
    allow_block = make_allow_block()
    write_csv(allow_block, "predictor_allowlist_blocklist.csv")

    lyric_records, lyric_summary, lyric_duplicates = lyric_audit({26: cohorts["550k_26w"], 52: cohorts["550k_52w"]})
    write_csv(lyric_records, "lyric_quality_audit.csv")
    write_csv(lyric_summary, "lyric_quality_summary.csv")
    write_csv(lyric_duplicates, "lyric_duplicate_groups.csv")

    genre = genre_audit({26: cohorts["550k_26w"], 52: cohorts["550k_52w"]}, artist_ref)
    write_csv(genre, "genre_audit.csv")

    bridge, artist_summary = track_artist_bridge(cohorts["550k_52w"], artist_ref)
    write_csv(bridge, "track_artist_bridge.csv")
    write_csv(artist_summary, "artist_clustering_summary.csv")

    numeric, missingness, invalid_log = invalid_and_numeric_audit(cohorts)
    write_csv(numeric, "numeric_validation_audit.csv")
    write_csv(missingness, "missingness_summary.csv")
    write_csv(invalid_log, "invalid_value_log.csv")

    pearson, spearman, high_corr, exp_corr = correlation_outputs(cohorts["550k_52w"])
    pearson.to_csv(OUT / "pearson_correlation_matrix_550k_52w.csv", lineterminator="\n")
    spearman.to_csv(OUT / "spearman_correlation_matrix_550k_52w.csv", lineterminator="\n")
    write_csv(high_corr, "high_correlation_pairs.csv")
    write_csv(exp_corr, "exploratory_outcome_correlations.csv")
    vif = vif_table(cohorts["550k_52w"], CONTINUOUS_VIF)
    write_csv(vif, "vif_diagnostic.csv")

    outcomes, frequencies = outcome_outputs(cohorts)
    write_csv(outcomes, "outcome_distribution_summary.csv")
    write_csv(frequencies, "outcome_frequency_distribution.csv")
    entry = entry_year_rows(clean, cohorts)
    write_csv(entry, "entry_year_composition.csv")
    summary = cohort_summary(register, cohorts)
    write_csv(summary, "cohort_summary.csv")
    comparison = make_comparison(cohorts)
    write_csv(comparison, "cohort_feature_comparison.csv")

    # Figures use the clean linked cohorts only and contain no fitted model output.
    plot_hist(cohorts["550k_52w"], "Y_52", "Future chart presence Y — 52-week 550k-linked cohort", FIG / "Y_52_frequency_distribution.png", 48)
    plot_hist(cohorts["550k_26w"], "Y_26", "Future chart presence Y — 26-week 550k-linked cohort", FIG / "Y_26_frequency_distribution.png", 22)
    plot_small_distributions(cohorts["550k_52w"], FIG / "X_L_Z_distributions_550k_52w.png")
    plot_heatmap(pearson, "Pearson correlations among candidate numeric predictors — 550k 52-week cohort", FIG / "pearson_predictor_heatmap_550k_52w.png")
    plot_heatmap(spearman, "Spearman correlations among candidate numeric predictors — 550k 52-week cohort", FIG / "spearman_predictor_heatmap_550k_52w.png")
    plot_entry_year(entry, FIG / "entry_year_composition_550k_52w.png")

    # Column-level validation: outcome/descriptive variables are present, but never in predictor lists.
    predictor_550 = [
        *AUDIO, "release_year", "release_age_years", "lyrics_text", "lyrics_character_count",
        "lyrics_word_count", "X", "chart_entry_year", "rank_200_stream_threshold_at_entry",
    ]
    predictor_160 = [
        *AUDIO, "explicit", "release_year", "release_date", "release_age_years", "X",
        "chart_entry_year", "rank_200_stream_threshold_at_entry",
    ]
    forbidden_exact = {
        "popularity", "track_popularity", "artist_popularity", "total_artist_followers",
        "avg_artist_popularity", "first_spell_observed_weeks", "total_chart_duration",
        "eventual_peak_rank", "last_active_date", "early_2w_prediction_eligible",
        "early_4w_prediction_eligible",
    }
    for name, frame in cohorts.items():
        if forbidden_exact & set(frame.columns):
            raise RuntimeError(f"Blocked variables present in {name}: {forbidden_exact & set(frame.columns)}")
    for variable in predictor_550:
        if variable not in cohorts["550k_52w"].columns:
            raise RuntimeError(f"Missing 550k predictor allowlist field: {variable}")
    for variable in predictor_160:
        if variable not in cohorts["160k_52w"].columns:
            raise RuntimeError(f"Missing 160k predictor allowlist field: {variable}")
    exact_allowed = set(allow_block.loc[allow_block["status"].eq("allow"), "variable_or_group"])
    if not set(predictor_550 + predictor_160).issubset(exact_allowed):
        raise RuntimeError(f"Declared predictors missing exact allowlist rows: {set(predictor_550 + predictor_160) - exact_allowed}")
    leakage_prefixes = ("Y_", "L_", "Z_count_", "Z_reentry_")
    if any(v.startswith(leakage_prefixes) for v in predictor_550 + predictor_160):
        raise RuntimeError("Outcome information leaked into the declared predictor list.")

    guards_after = {str(path.relative_to(ROOT)): tree_digest(path) for path in TREE_GUARDS}
    if guards_before != guards_after:
        raise RuntimeError("A protected historical directory changed during the run.")

    assumptions = """# Provisional working assumptions — pre_model_analysis_ready_v1

These are working assumptions for preprocessing and exploratory analysis only. They are not final thesis decisions and do not imply supervisor approval.

- Primary provisional outcome: fixed-window future chart presence over 52 calendar weeks.
- Early measurement period: calendar weeks 1–4 after confirmed chart entry.
- X: number of those four weeks in which the song is present; X is not a survival-based admission filter.
- Y_52: chart-presence weeks from week 5 through week 52, including re-entry.
- L_52, Z_count_52 and Z_reentry_52: descriptive outcome variables, not predictors of Y_52.
- Sensitivity outcome: equivalent 26-week Y, L and Z variables.
- Entry must not be left-truncated or ambiguous after a source-wide gap.
- The full required horizon must be observable and must not cross a missing source week.
- The 550k exact-ID linked cohort is the provisional primary feature cohort because it contains lyrics.
- The 160k exact-ID linked cohort is prepared as a possible robustness cohort without lyrics.
- First-spell survival outputs remain unchanged for a possible secondary analysis.
- Popularity, follower, future-chart and full-history variables are blocked.
- No modelling, encoding, imputation, scaling or text vectorisation is performed in this version.
"""
    (OUT / "provisional_working_assumptions.md").write_text(assumptions, encoding="utf-8")

    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "pre_model_analysis_ready_v1", "no_models_fitted": True,
        "analysis_ready_rows": {name: len(frame) for name, frame in cohorts.items()},
        "one_row_per_spotify_track_id": {name: bool(frame["spotify_track_id"].is_unique) for name, frame in cohorts.items()},
        "join_row_expansion_detected": False,
        "blocked_variables_absent": True,
        "outcomes_absent_from_declared_predictor_lists": True,
        "unknown_source_weeks_converted_to_absence": False,
        "protected_tree_digests_before": guards_before,
        "protected_tree_digests_after": guards_after,
        "source_files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [FIXED, FIRST_SPELL, FEATURE_550_EXPORT, SONGS_550, ARTISTS_550, TRACKS_160, RANK200]
        },
        "declared_predictor_columns": {"550k": predictor_550, "160k": predictor_160},
        "missing_source_weeks": ["2017-06-02", "2017-06-09", "2021-02-19"],
    }
    (OUT / "validation_manifest.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    REPORT.write_text(report_text(summary, cohorts, lyric_summary, genre, artist_summary, numeric, outcomes, exp_corr, high_corr, vif, entry), encoding="utf-8")

    print(json.dumps({
        "output_directory": str(OUT), "figure_directory": str(FIG), "report": str(REPORT),
        "sample_sizes": {name: len(frame) for name, frame in cohorts.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
