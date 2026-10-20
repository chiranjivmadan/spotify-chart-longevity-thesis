#!/usr/bin/env python3
"""Build remaining pre-model EDA and methodological evidence (no models)."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/pre_model_analysis_ready_v1"
OUT = ROOT / "data/processed/pre_model_remaining_eda_v2"
FIG = ROOT / "outputs/pre_model_remaining_eda_v2/figures"
REPORT = ROOT / "docs/pre_model_remaining_eda_report.md"

AR550_52 = BASE / "analysis_ready_550k_52w.csv"
AR550_26 = BASE / "analysis_ready_550k_26w.csv"
AR160_52 = BASE / "analysis_ready_160k_52w.csv"
RAW160 = ROOT / "data/raw/spotify_160k/data.csv"
ARTISTS550 = ROOT / "data/raw/spotify_550k/artists.csv"
CHART = ROOT / "data/raw/charts/extracted/bartomiejczyewski_top200_global/Spotify Top 200 Global (2017-2021).csv"
FIXED = ROOT / "data/processed/fixed_window_count_v1/fixed_window_track_outcomes.csv"
FIRST = ROOT / "data/processed/chart_survival/first_spell_survival.csv"
FEATURE550 = ROOT / "data/processed/chart_survival/matched_550k_baseline_features.csv"
BRIDGE = BASE / "track_artist_bridge.csv"

PROTECTED = [
    ROOT / "data/processed/chart_survival",
    ROOT / "data/processed/fixed_window_count_v1",
    ROOT / "data/processed/audits",
    ROOT / "data/frozen/first_spell_survival_2026-07-30",
    BASE,
]
MISSING_WEEKS = {date(2017, 6, 2), date(2017, 6, 9), date(2021, 2, 19)}
AUDIO = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
VERSION_PATTERN = re.compile(
    r"\b(remix|live|acoustic|remaster(?:ed)?|radio edit|edit|version|stripped|piano|orchestral|sped up|slowed)\b",
    re.I,
)


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


def prepare() -> None:
    existing = [p for p in (OUT, FIG.parent, REPORT) if p.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    OUT.mkdir(parents=True)
    FIG.mkdir(parents=True)


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, lineterminator="\n")


def parse_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def norm(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def base_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\[(](?:feat\.?|featuring|with)\b[^\])]*[\])]", " ", text, flags=re.I)
    text = re.sub(r"\s+-\s+.*\b(remix|live|acoustic|remaster(?:ed)?|edit|version|stripped|sped up|slowed)\b.*$", " ", text, flags=re.I)
    text = re.sub(r"[\[(][^\])]*\b(remix|live|acoustic|remaster(?:ed)?|edit|version|stripped|sped up|slowed)\b[^\])]*[\])]", " ", text, flags=re.I)
    return norm(text)


def group_summary(frame: pd.DataFrame, group_cols: list[str], y: str = "Y_52") -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group[y].astype(float)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "N": len(group), "Y_mean": values.mean(), "Y_median": values.median(),
            "Y_standard_deviation": values.std(ddof=1), "Y_zero_count": int(values.eq(0).sum()),
            "Y_zero_percent": 100 * values.eq(0).mean(), "Y_p25": values.quantile(.25),
            "Y_p50": values.quantile(.50), "Y_p75": values.quantile(.75),
            "Y_min": values.min(), "Y_max": values.max(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def smd(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
    return float((left.mean() - right.mean()) / pooled) if pooled else np.nan


def duplicate_lyrics(art550_26: pd.DataFrame, raw160: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate = art550_26.loc[art550_26["lyrics_duplicate_text"].eq(1)].copy()
    explicit_map = raw160.set_index("id")["explicit"].to_dict()
    release_map = raw160.set_index("id")["release_date"].to_dict()
    member_rows = []
    group_rows = []
    for digest, group in duplicate.groupby("lyrics_normalised_sha256"):
        group = group.sort_values("spotify_track_id")
        titles = group["feature_track_name"].fillna(group["chart_track"]).astype(str)
        norm_titles = titles.map(norm)
        bases = titles.map(base_title)
        artist_ids = group["artist_ids_original"].map(lambda value: tuple(parse_list(value)))
        artist_names = group["artists_original"].map(lambda value: tuple(parse_list(value)))
        titles_identical = norm_titles.nunique() == 1
        base_titles_identical = bases.nunique() == 1
        artists_identical = artist_ids.nunique() == 1 if all(artist_ids.map(bool)) else artist_names.nunique() == 1
        release_years_identical = group["release_year"].nunique() == 1
        albums_identical = group["feature_album_name"].astype(str).map(norm).nunique() == 1
        version_labels = titles.map(lambda value: "|".join(sorted(set(m.group(1).lower() for m in VERSION_PATTERN.finditer(value)))))
        version_labels_differ = version_labels.nunique() > 1
        audio_matrix = group[AUDIO].astype(float).to_numpy()
        audio_identical = bool(np.all(np.isclose(audio_matrix, audio_matrix[0], rtol=0, atol=1e-12)))
        max_duration_difference = float(group["duration_ms"].max() - group["duration_ms"].min())
        explicit_values = [explicit_map.get(ident, np.nan) for ident in group["spotify_track_id"]]
        known_explicit = [int(v) for v in explicit_values if pd.notna(v)]
        explicit_differs = len(set(known_explicit)) > 1

        if explicit_differs and base_titles_identical and artists_identical:
            classification = "clean/explicit variants"
            confidence = "high"
            rationale = "The 160k source provides differing explicit flags for otherwise related records."
        elif base_titles_identical and version_labels_differ:
            classification = "remix/live/acoustic/remaster versions"
            confidence = "high"
            rationale = "Titles share a base title but contain different version descriptors."
        elif titles_identical and artists_identical and audio_identical:
            classification = "same track/version duplicated in the feature source"
            confidence = "high"
            rationale = "Distinct Spotify IDs have identical title, credited artists and exact audio descriptors."
        elif base_titles_identical and (artists_identical or norm_titles.nunique() == 1):
            classification = "different recordings of the same underlying song"
            confidence = "medium"
            rationale = "The underlying title and credits are related, but audio descriptors or release metadata differ."
        elif not base_titles_identical and not artists_identical:
            classification = "genuinely different songs sharing lyrics"
            confidence = "low"
            rationale = "Both titles and credited artists differ; manual review is still required because exact shared text may also reflect source matching error."
        else:
            classification = "unresolved cases"
            confidence = "low"
            rationale = "The available metadata do not distinguish duplicate release, alternate master or version reliably."

        if classification in {
            "same track/version duplicated in the feature source", "clean/explicit variants",
            "remix/live/acoustic/remaster versions", "different recordings of the same underlying song",
        }:
            leakage = "high: keep the group together in validation"
        elif classification == "genuinely different songs sharing lyrics":
            leakage = "high for lyric/text models; lower but non-zero for audio-only models"
        else:
            leakage = "possible: conservatively keep the group together"

        group_rows.append({
            "lyrics_normalised_sha256": digest, "tracks_in_group": len(group),
            "spotify_track_ids": "|".join(group["spotify_track_id"]),
            "track_titles": " | ".join(titles), "artist_strings": " | ".join(group["artists_original"].astype(str)),
            "artist_ids": " | ".join(group["artist_ids_original"].astype(str)),
            "release_years": "|".join(group["release_year"].astype(int).astype(str)),
            "release_dates_where_160k_available": "|".join(str(release_map.get(i, "")) for i in group["spotify_track_id"]),
            "broad_genres": "|".join(group["genre_raw"].astype(str)),
            "titles_identical": int(titles_identical), "base_titles_identical": int(base_titles_identical),
            "artists_identical": int(artists_identical), "release_years_identical": int(release_years_identical),
            "albums_identical": int(albums_identical), "audio_descriptors_identical": int(audio_identical),
            "maximum_duration_difference_ms": max_duration_difference,
            "known_explicit_values": "|".join(map(str, known_explicit)), "explicit_values_differ": int(explicit_differs),
            "version_labels_differ": int(version_labels_differ), "provisional_classification": classification,
            "classification_confidence": confidence, "classification_rationale": rationale,
            "potential_validation_leakage": leakage,
            "recommended_grouping_rule": "Keep all tracks sharing this exact normalised lyric hash in the same validation partition; do not merge or delete records.",
        })
        for (_, row), title, artist_tuple, explicit in zip(group.iterrows(), titles, artist_ids, explicit_values):
            member_rows.append({
                "lyrics_normalised_sha256": digest, "spotify_track_id": row["spotify_track_id"],
                "track_title": title, "chart_title": row["chart_track"], "album_name": row["feature_album_name"],
                "artist_string": row["artists_original"], "artist_ids": row["artist_ids_original"],
                "release_year": int(row["release_year"]), "release_date_160k_if_available": release_map.get(row["spotify_track_id"], ""),
                "broad_genre": row["genre_raw"], "explicit_160k_if_available": explicit,
                "danceability": row["danceability"], "energy": row["energy"], "loudness": row["loudness"],
                "tempo": row["tempo"], "duration_ms": row["duration_ms"],
                "provisional_group_classification": classification,
            })
    groups = pd.DataFrame(group_rows).sort_values(["provisional_classification", "lyrics_normalised_sha256"])
    members = pd.DataFrame(member_rows).sort_values(["lyrics_normalised_sha256", "spotify_track_id"])
    summary = groups.groupby("provisional_classification").agg(
        duplicate_groups=("lyrics_normalised_sha256", "count"),
        tracks=("tracks_in_group", "sum"),
        high_confidence_groups=("classification_confidence", lambda s: int((s == "high").sum())),
    ).reset_index()
    summary["percent_of_duplicate_groups"] = 100 * summary["duplicate_groups"] / len(groups)
    return groups, members, summary


def additional_eda(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_x = group_summary(frame, ["X"])
    by_x["small_sample_flag"] = ""

    by_genre = group_summary(frame, ["genre_raw"])
    by_genre["sample_size_flag"] = np.select(
        [by_genre["N"] < 10, by_genre["N"] < 30],
        ["very small: do not infer a genre effect", "small: interpret cautiously"],
        default="adequate for description only",
    )
    by_genre = by_genre.sort_values("N", ascending=False)

    by_year = group_summary(frame, ["chart_entry_year"])
    by_year["percent_of_cohort"] = 100 * by_year["N"] / len(frame)
    by_year["follow_up_note"] = np.where(
        by_year["chart_entry_year"].eq(by_year["chart_entry_year"].max()),
        "Latest entry year is strongly truncated by the 52-week follow-up requirement.", "",
    )

    release_rows = []
    release_exact = group_summary(frame, ["release_year"])
    release_exact.insert(0, "release_timing_dimension", "release_year_exact")
    release_exact["category"] = release_exact["release_year"].astype(int).astype(str)
    release_rows.append(release_exact.drop(columns=["release_year"]))

    age_exact = group_summary(frame, ["release_age_years"])
    age_exact.insert(0, "release_timing_dimension", "release_age_exact")
    age_exact["category"] = age_exact["release_age_years"].astype(int).astype(str)
    release_rows.append(age_exact.drop(columns=["release_age_years"]))

    age_bins = pd.cut(
        frame["release_age_years"], bins=[-0.1, 0, 1, 2, 5, 10, np.inf],
        labels=["0 years", "1 year", "2 years", "3-5 years", "6-10 years", ">10 years"],
    )
    temp = frame.assign(release_age_band=age_bins)
    age_band = group_summary(temp, ["release_age_band"])
    age_band.insert(0, "release_timing_dimension", "release_age_band")
    age_band["category"] = age_band["release_age_band"].astype(str)
    release_rows.append(age_band.drop(columns=["release_age_band"]))
    release = pd.concat(release_rows, ignore_index=True)
    return {"x": by_x, "genre": by_genre, "entry_year": by_year, "release": release}


def composition_rows(frame: pd.DataFrame, cohort: str, dimension: str, column: str) -> list[dict[str, object]]:
    counts = frame[column].value_counts(dropna=False).sort_index()
    return [{
        "comparison_section": "composition", "cohort": cohort, "dimension": dimension,
        "variable": column, "category": value, "N": int(count), "percent": 100 * count / len(frame),
        "mean": "", "median": "", "standard_deviation": "", "p25": "", "p75": "", "standardised_mean_difference_52_minus_26": "",
    } for value, count in counts.items()]


def compare_horizons(f26: pd.DataFrame, f52: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids26, ids52 = set(f26["spotify_track_id"]), set(f52["spotify_track_id"])
    membership = pd.DataFrame([
        {"membership": "both_26w_and_52w", "tracks": len(ids26 & ids52)},
        {"membership": "26w_only", "tracks": len(ids26 - ids52)},
        {"membership": "52w_only", "tracks": len(ids52 - ids26)},
    ])
    rows = []
    for name, frame, y in (("550k_26w", f26, "Y_26"), ("550k_52w", f52, "Y_52")):
        rows.append({
            "comparison_section": "overall", "cohort": name, "dimension": "sample",
            "variable": "tracks", "category": "all", "N": len(frame), "percent": 100,
            "mean": frame[y].mean(), "median": frame[y].median(), "standard_deviation": frame[y].std(ddof=1),
            "p25": frame[y].quantile(.25), "p75": frame[y].quantile(.75),
            "standardised_mean_difference_52_minus_26": "",
        })
        rows.extend(composition_rows(frame, name, "entry_year", "chart_entry_year"))
        rows.extend(composition_rows(frame, name, "genre", "genre_raw"))
        rows.extend(composition_rows(frame, name, "X", "X"))
    for feature in AUDIO:
        a, b = f26[feature].astype(float), f52[feature].astype(float)
        difference = smd(b, a)
        for name, values in (("550k_26w", a), ("550k_52w", b)):
            rows.append({
                "comparison_section": "audio_distribution", "cohort": name, "dimension": "audio",
                "variable": feature, "category": "all", "N": len(values), "percent": 100,
                "mean": values.mean(), "median": values.median(), "standard_deviation": values.std(ddof=1),
                "p25": values.quantile(.25), "p75": values.quantile(.75),
                "standardised_mean_difference_52_minus_26": difference,
            })
    return membership, pd.DataFrame(rows)


def compare_feature_cohorts(f550: pd.DataFrame, f160: pd.DataFrame, first: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    debut = first[["spotify_track_id", "early_2w_debut_rank"]].rename(columns={"early_2w_debut_rank": "debut_rank"})
    a = f550.merge(debut, on="spotify_track_id", how="left", validate="one_to_one")
    b = f160.merge(debut, on="spotify_track_id", how="left", validate="one_to_one")
    rows = []
    for name, frame in (("550k_52w", a), ("160k_52w", b)):
        for variable in ("Y_52", "X", "debut_rank"):
            values = pd.to_numeric(frame[variable], errors="coerce")
            rows.append({
                "comparison_section": "numeric_summary", "cohort": name, "dimension": variable,
                "variable": variable, "category": "all", "N": int(values.notna().sum()),
                "percent": 100 * values.notna().mean(), "mean": values.mean(), "median": values.median(),
                "standard_deviation": values.std(ddof=1), "p25": values.quantile(.25), "p75": values.quantile(.75),
                "zero_percent": 100 * values.eq(0).mean() if variable == "Y_52" else "",
                "standardised_mean_difference_550_minus_160": smd(a[variable], b[variable]),
            })
        rows.extend({**row, "zero_percent": "", "standardised_mean_difference_550_minus_160": ""} for row in composition_rows(frame, name, "entry_year", "chart_entry_year"))
        rows.extend({**row, "zero_percent": "", "standardised_mean_difference_550_minus_160": ""} for row in composition_rows(frame, name, "X", "X"))
    for feature in AUDIO:
        difference = smd(a[feature], b[feature])
        for name, frame in (("550k_52w", a), ("160k_52w", b)):
            values = frame[feature].astype(float)
            rows.append({
                "comparison_section": "common_audio_distribution", "cohort": name, "dimension": "audio",
                "variable": feature, "category": "all", "N": len(values), "percent": 100,
                "mean": values.mean(), "median": values.median(), "standard_deviation": values.std(ddof=1),
                "p25": values.quantile(.25), "p75": values.quantile(.75), "zero_percent": "",
                "standardised_mean_difference_550_minus_160": difference,
            })
    ids_a, ids_b = set(a.spotify_track_id), set(b.spotify_track_id)
    membership = pd.DataFrame([
        {"membership": "both_550k_and_160k", "tracks": len(ids_a & ids_b)},
        {"membership": "550k_only", "tracks": len(ids_a - ids_b)},
        {"membership": "160k_only", "tracks": len(ids_b - ids_a)},
        {"membership": "union", "tracks": len(ids_a | ids_b)},
    ])
    overlap = a.loc[a.spotify_track_id.isin(ids_a & ids_b), ["spotify_track_id", *AUDIO]].merge(
        b.loc[b.spotify_track_id.isin(ids_a & ids_b), ["spotify_track_id", *AUDIO]],
        on="spotify_track_id", suffixes=("_550k", "_160k"), validate="one_to_one",
    )
    consistency = []
    for feature in AUDIO:
        diff = overlap[f"{feature}_550k"].astype(float) - overlap[f"{feature}_160k"].astype(float)
        consistency.append({
            "feature": feature, "overlap_tracks": len(overlap), "exact_equal_count": int(diff.eq(0).sum()),
            "exact_equal_percent": 100 * diff.eq(0).mean(), "mean_550_minus_160": diff.mean(),
            "mean_absolute_difference": diff.abs().mean(), "maximum_absolute_difference": diff.abs().max(),
        })
    return pd.DataFrame(rows), membership, pd.DataFrame(consistency)


def extract_id(link: str) -> str:
    match = re.search(r"(?:open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]{22})", link or "")
    return match.group(1) if match else ""


def sensitivity_variants(fixed: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = fixed.loc[fixed["horizon_weeks"].eq(52)].copy()
    base = base.loc[base["match_550k_exact_id"].eq(1) & base["enough_follow_up"].eq(1)].copy()
    feature_map = features.set_index("spotify_track_id")[["genre"]]
    present: dict[str, set[date]] = defaultdict(set)
    observed_weeks: set[date] = set()
    with CHART.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ident = extract_id(row["Link"])
            week = datetime.strptime(row["Week"], "%Y-%m-%d").date()
            present[ident].add(week)
            observed_weeks.add(week)
    first_week, last_week = min(observed_weeks), max(observed_weeks)
    calendar = []
    current = first_week
    while current <= last_week:
        calendar.append(current)
        current += timedelta(days=7)
    index = {week: i for i, week in enumerate(calendar)}

    track_rows = []
    for _, row in base.iterrows():
        entry = datetime.strptime(row["first_observed_chart_entry"], "%Y-%m-%d").date()
        weeks = calendar[index[entry]:index[entry] + 52]
        states = [None if week in MISSING_WEEKS else int(week in present[row["spotify_track_id"]]) for week in weeks]
        x_states, y_states = states[:4], states[4:]
        x_lower = sum(v == 1 for v in x_states)
        x_unknown = sum(v is None for v in x_states)
        y_lower = sum(v == 1 for v in y_states)
        y_unknown = sum(v is None for v in y_states)
        uncertain = int(row["left_truncated"] == 1 or row["entry_ambiguous_due_source_gap"] == 1)
        gap = int(any(v is None for v in states))
        track_rows.append({
            "spotify_track_id": row["spotify_track_id"], "chart_track": row["chart_track"],
            "chart_artist": row["chart_artist"], "chart_entry_date": entry.isoformat(), "chart_entry_year": entry.year,
            "genre_raw": feature_map.loc[row["spotify_track_id"], "genre"], "uncertain_entry": uncertain,
            "missing_source_week_in_window": gap, "X_lower": x_lower, "X_upper": x_lower + x_unknown,
            "X_unknown_weeks": x_unknown, "Y_lower": y_lower, "Y_upper": y_lower + y_unknown,
            "Y_unknown_weeks": y_unknown,
        })
    tracks = pd.DataFrame(track_rows)
    definitions = {
        "A_strict_clean": lambda x: (x.uncertain_entry == 0) & (x.missing_source_week_in_window == 0),
        "B_plus_uncertain_entry": lambda x: x.missing_source_week_in_window == 0,
        "C_plus_missing_week_affected": lambda x: x.uncertain_entry == 0,
        "D_plus_both": lambda x: pd.Series(True, index=x.index),
    }
    summaries = []
    compositions = []
    memberships = []
    for variant, rule in definitions.items():
        subset = tracks.loc[rule(tracks)].copy()
        summaries.append({
            "variant": variant, "N": len(subset), "uncertain_entry_tracks": int(subset.uncertain_entry.sum()),
            "missing_week_affected_tracks": int(subset.missing_source_week_in_window.sum()),
            "Y_mean_lower_bound": subset.Y_lower.mean(), "Y_mean_upper_bound": subset.Y_upper.mean(),
            "Y_median_lower_bound": subset.Y_lower.median(), "Y_median_upper_bound": subset.Y_upper.median(),
            "Y_definite_zero_percent": 100 * subset.Y_upper.eq(0).mean(),
            "Y_possible_zero_percent": 100 * subset.Y_lower.eq(0).mean(),
            "X_mean_lower_bound": subset.X_lower.mean(), "X_mean_upper_bound": subset.X_upper.mean(),
            "X_exact_tracks": int(subset.X_unknown_weeks.eq(0).sum()), "X_interval_tracks": int(subset.X_unknown_weeks.gt(0).sum()),
            "interpretation_note": "Bounds preserve unknown source weeks as unknown. Lower bounds count observed presences only; upper bounds add every unknown week as potentially present.",
        })
        for dimension, column in (("X_lower", "X_lower"), ("X_upper", "X_upper"), ("entry_year", "chart_entry_year"), ("genre", "genre_raw")):
            for category, count in subset[column].value_counts().sort_index().items():
                compositions.append({
                    "variant": variant, "dimension": dimension, "category": category,
                    "tracks": int(count), "percent_of_variant": 100 * count / len(subset),
                })
        member = subset[["spotify_track_id"]].copy()
        member["variant"] = variant
        memberships.append(member)
    return pd.DataFrame(summaries), pd.DataFrame(compositions), pd.concat(memberships, ignore_index=True)


def genre_provenance() -> pd.DataFrame:
    rows = [
        ("genre", "data/raw/spotify_550k/songs.csv", "stored once per song row", "Raw header and song-row placement", "No derivation or collection timestamp found", "not established"),
        ("niche_genres", "data/raw/spotify_550k/songs.csv", "stored as a list on each song row", "Raw header and song-row placement", "No definition of whether labels are inherited from artists; no collection timestamp", "not established"),
        ("main_genre", "data/raw/spotify_550k/artists.csv", "stored once per artist row", "Raw header and artist-row placement", "No rule for selecting the main genre and no collection timestamp", "not established"),
        ("genres", "data/raw/spotify_550k/artists.csv", "stored as a list on each artist row", "Raw header and artist-row placement", "No taxonomy derivation, update rule or collection timestamp", "not established"),
    ]
    return pd.DataFrame([{
        "field": field, "source_file": source, "observed_storage_level": level,
        "evidence_available": evidence, "missing_provenance": missing,
        "derivation_documented": 0, "collection_time_documented": 0,
        "static_or_snapshot_status": "single untimestamped value in the supplied file; cannot establish whether it is static or a later snapshot",
        "overall_provenance_status": status,
        "recommended_interim_treatment": "Preserve raw field; use only provisionally and discuss timing/derivation before modelling.",
        "evidence_paths": "archive (4).zip; docs/dataset_audit_summary.md; data/processed/audit/raw_dataset_audit.json; data/processed/audits/predictor_timing_and_leakage_audit.csv",
    } for field, source, level, evidence, missing, status in rows])


def loudness_review(raw160: pd.DataFrame, analysis160: pd.DataFrame) -> pd.DataFrame:
    ident = "5mqzhMuUpvnMfwNz6iepmO"
    raw = raw160.loc[raw160.id.eq(ident)].iloc[0]
    processed = analysis160.loc[analysis160.spotify_track_id.eq(ident)].iloc[0]
    loud = raw160.loudness.astype(float)
    fields_normal = (
        0 <= raw.danceability <= 1 and 0 <= raw.energy <= 1 and 0 <= raw.acousticness <= 1
        and 0 <= raw.instrumentalness <= 1 and 0 <= raw.liveness <= 1 and 0 <= raw.valence <= 1
        and raw.tempo > 0 and raw.duration_ms > 0 and int(raw["key"]) in range(12) and int(raw["mode"]) in (0, 1)
    )
    return pd.DataFrame([{
        "spotify_track_id": ident, "track_title": raw["name"], "artists": raw["artists"],
        "raw_source_file": "data/raw/spotify_160k/data.csv", "raw_loudness_db": raw.loudness,
        "analysis_ready_loudness_db": processed.loudness,
        "value_preserved_exactly": int(float(raw.loudness) == float(processed.loudness)),
        "raw_positive_loudness_tracks": int((loud > 0).sum()), "raw_total_tracks": len(raw160),
        "raw_loudness_min": loud.min(), "raw_loudness_median": loud.median(), "raw_loudness_max": loud.max(),
        "other_source_fields_within_basic_ranges": int(fields_normal),
        "selected_other_fields": json.dumps({
            "danceability": raw.danceability, "energy": raw.energy, "acousticness": raw.acousticness,
            "instrumentalness": raw.instrumentalness, "liveness": raw.liveness, "valence": raw.valence,
            "tempo": raw.tempo, "duration_ms": int(raw.duration_ms), "key": int(raw["key"]), "mode": int(raw["mode"]),
            "explicit": int(raw.explicit), "release_date": raw.release_date,
        }),
        "assessment": "Exact source value and not a parse error. It is an unusual positive loudness value but remains within the source's observed range; treat as a plausible extreme/source anomaly pending external verification.",
        "later_recommendation": "Retain with a flag; use robust/train-only scaling if needed and run a sensitivity check excluding this one record rather than silently changing it.",
    }])


def redundant_review(frame: pd.DataFrame) -> pd.DataFrame:
    residual = frame.chart_entry_year - frame.release_year - frame.release_age_years
    char_word_p = frame.lyrics_character_count.corr(frame.lyrics_word_count, method="pearson")
    char_word_s = frame.lyrics_character_count.corr(frame.lyrics_word_count, method="spearman")
    ratio = frame.lyrics_character_count / frame.lyrics_word_count.replace(0, np.nan)
    return pd.DataFrame([
        {
            "issue": "release/calendar variables", "variables": "release_year; chart_entry_year; release_age_years",
            "mathematical_relationship": "release_age_years = chart_entry_year - release_year",
            "evidence_1": f"Maximum absolute dependency residual: {abs(residual).max():.0f}",
            "evidence_2": f"Pearson(release_year, release_age_years)={frame.release_year.corr(frame.release_age_years):.6f}",
            "redundancy_assessment": "Exact three-variable collinearity; all three cannot appear together in a standard design matrix.",
            "candidate_specifications": "(1) release age + chart-entry year; (2) release year + chart-entry year; (3) release age only plus a separately justified calendar control such as rank-200 threshold.",
            "provisional_recommendation": "Release age plus one calendar-time control is interpretable, but the final specification still requires methodological confirmation.",
        },
        {
            "issue": "lyric length variables", "variables": "lyrics_character_count; lyrics_word_count",
            "mathematical_relationship": "Both measure the amount of lyric text; no exact identity but near-linear redundancy.",
            "evidence_1": f"Pearson={char_word_p:.6f}; Spearman={char_word_s:.6f}",
            "evidence_2": f"Characters per word: median={ratio.median():.2f}, p25={ratio.quantile(.25):.2f}, p75={ratio.quantile(.75):.2f}",
            "redundancy_assessment": "Severe redundancy and VIF above 80 when both are used.",
            "candidate_specifications": "Retain one simple length control, or omit both once train-only NLP representations are used.",
            "provisional_recommendation": "Word count is easier to interpret as a simple length control; keep character count for audit traceability.",
        },
    ])


def primary_artist_map(bridge: pd.DataFrame) -> dict[str, str]:
    primary = bridge.loc[bridge.is_grouping_artist.eq(1), ["spotify_track_id", "artist_id"]]
    return dict(zip(primary.spotify_track_id, primary.artist_id))


def greedy_group_folds(groups: pd.Series, n_folds: int = 5) -> pd.Series:
    sizes = groups.value_counts().sort_values(ascending=False)
    totals = [0] * n_folds
    assignment = {}
    for group, size in sizes.items():
        fold = int(np.argmin(totals))
        assignment[group] = fold + 1
        totals[fold] += int(size)
    return groups.map(assignment)


def connected_components(frame: pd.DataFrame) -> pd.Series:
    # Connect tracks sharing either primary artist or an exact normalised lyric hash.
    neighbours: dict[str, set[str]] = defaultdict(set)
    for column in ("grouping_artist_id", "lyrics_normalised_sha256"):
        for _, group in frame.groupby(column):
            ids = list(group.spotify_track_id)
            if len(ids) > 1:
                root = ids[0]
                for ident in ids[1:]:
                    neighbours[root].add(ident)
                    neighbours[ident].add(root)
    component = {}
    for ident in frame.spotify_track_id:
        if ident in component:
            continue
        queue = deque([ident])
        members = []
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component[current] = "pending"
            members.append(current)
            queue.extend(neighbours[current] - set(component))
        label = min(members)
        for member in members:
            component[member] = label
    return frame.spotify_track_id.map(component)


def prospective_validation(frame: pd.DataFrame, bridge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artist_map = primary_artist_map(bridge)
    data = frame[["spotify_track_id", "chart_entry_year", "lyrics_normalised_sha256"]].copy()
    data["grouping_artist_id"] = data.spotify_track_id.map(artist_map)
    if data.grouping_artist_id.isna().any():
        raise RuntimeError("Missing primary artist ID in validation design input.")
    data["artist_fold"] = greedy_group_folds(data.grouping_artist_id)
    data["artist_lyric_component"] = connected_components(data)
    data["component_fold"] = greedy_group_folds(data.artist_lyric_component)

    assignments = []
    for _, row in data.iterrows():
        assignments.append({
            "strategy": "artist_grouped_5fold", "spotify_track_id": row.spotify_track_id,
            "partition": f"fold_{int(row.artist_fold)}", "chart_entry_year": row.chart_entry_year,
            "grouping_artist_id": row.grouping_artist_id, "related_version_group": row.lyrics_normalised_sha256,
        })
        assignments.append({
            "strategy": "artist_plus_exact_lyric_component_5fold", "spotify_track_id": row.spotify_track_id,
            "partition": f"fold_{int(row.component_fold)}", "chart_entry_year": row.chart_entry_year,
            "grouping_artist_id": row.grouping_artist_id, "related_version_group": row.lyrics_normalised_sha256,
        })

    early = data.chart_entry_year <= 2018
    later = data.chart_entry_year >= 2019
    for _, row in data.iterrows():
        partition = "train_2017_2018" if row.chart_entry_year <= 2018 else ("validation_2019" if row.chart_entry_year == 2019 else "test_2020")
        assignments.append({
            "strategy": "time_aware_2017_2018_train_2019_validation_2020_test", "spotify_track_id": row.spotify_track_id,
            "partition": partition, "chart_entry_year": row.chart_entry_year,
            "grouping_artist_id": row.grouping_artist_id, "related_version_group": row.lyrics_normalised_sha256,
        })

    later_artists = set(data.loc[later, "grouping_artist_id"])
    later_lyrics = set(data.loc[later, "lyrics_normalised_sha256"])
    purged_train = early & ~data.grouping_artist_id.isin(later_artists) & ~data.lyrics_normalised_sha256.isin(later_lyrics)
    for _, row in data.loc[purged_train | later].iterrows():
        assignments.append({
            "strategy": "combined_time_holdout_with_artist_and_lyric_purge", "spotify_track_id": row.spotify_track_id,
            "partition": "train_2017_2018_purged" if row.chart_entry_year <= 2018 else "test_2019_2020",
            "chart_entry_year": row.chart_entry_year, "grouping_artist_id": row.grouping_artist_id,
            "related_version_group": row.lyrics_normalised_sha256,
        })

    assign = pd.DataFrame(assignments)
    summary_rows = []
    for (strategy, partition), group in assign.groupby(["strategy", "partition"]):
        all_strategy = assign.loc[assign.strategy.eq(strategy)]
        others = all_strategy.loc[~all_strategy.partition.eq(partition)]
        artist_overlap = len(set(group.grouping_artist_id) & set(others.grouping_artist_id))
        lyric_overlap = len(set(group.related_version_group) & set(others.related_version_group))
        summary_rows.append({
            "strategy": strategy, "partition": partition, "tracks": len(group),
            "unique_grouping_artists": group.grouping_artist_id.nunique(),
            "unique_related_version_groups": group.related_version_group.nunique(),
            "entry_year_counts": json.dumps(group.chart_entry_year.value_counts().sort_index().to_dict()),
            "artist_groups_overlapping_other_partitions": artist_overlap,
            "exact_lyric_groups_overlapping_other_partitions": lyric_overlap,
        })
    summary = pd.DataFrame(summary_rows)

    strategy_rows = [
        {
            "strategy": "artist_grouped cross-validation", "formation": "Assign each primary Spotify artist wholly to one of five approximately balanced folds; optionally use connected artist-plus-exact-lyric groups for stricter version control.",
            "artists_on_both_sides": "No for artist-only folds; no artist or exact-lyric group overlap for the conservative component variant.",
            "later_years_held_out": "No; years are mixed across folds.", "advantages": "Measures generalisation to songs by artists not represented in training and prevents artist memorisation.",
            "disadvantages": "Does not directly test future calendar drift; conservative components may create uneven folds.",
            "sample_size_implication": "Uses all 1,014 tracks across five folds.",
            "score_interpretation": "Expected performance for charting tracks by previously unseen artists within the observed 2017-2020 calendar mixture.",
        },
        {
            "strategy": "time-aware holdout", "formation": "Train on 2017-2018 entries, validate on 2019 and retain 2020 as a small final stress-test set.",
            "artists_on_both_sides": "Yes unless explicitly purged.", "later_years_held_out": "Yes: 2019 and 2020.",
            "advantages": "Tests calendar drift and matches deployment from earlier to later chart periods.",
            "disadvantages": "Artist and related-version leakage can remain; the 2020 set is very small because 52-week follow-up removes later entries.",
            "sample_size_implication": "628 training, 341 validation and 45 test tracks before any artist purge.",
            "score_interpretation": "Expected performance when applying an earlier-period model to later chart entrants, including recurring artists.",
        },
        {
            "strategy": "combined artist-grouped and time-aware holdout", "formation": "Use 2019-2020 as the test period and purge from the 2017-2018 training set any primary artist or exact-lyric group appearing later.",
            "artists_on_both_sides": "No after the proposed purge.", "later_years_held_out": "Yes: 2019-2020.",
            "advantages": "Simultaneously addresses calendar drift, artist repetition and exact-lyric/version leakage.",
            "disadvantages": "Reduces training size and changes the training population; scores a harder novel-artist/novel-version question.",
            "sample_size_implication": "Prospective counts are recorded in the split summary and should be reviewed before adoption.",
            "score_interpretation": "Expected performance for later chart entrants without training exposure to their primary artist or exact lyric/version group.",
        },
    ]
    return assign, summary, pd.DataFrame(strategy_rows)


def decision_status() -> pd.DataFrame:
    rows = [
        ("Fixed-window Y as provisional primary outcome", "Includes re-entry and avoids treating first exit as absorbing; Y is zero-heavy and overdispersed.", "Supports fixed-window Y as a strong primary candidate, with first-spell survival secondary.", "Partly", "Yes"),
        ("52 versus 26 weeks", "52 weeks gives longer interpretation but 1,014 versus 1,220 550k tracks and much less 2020 coverage.", "52 remains defensible, but composition as well as sample size changes.", "No", "Yes"),
        ("Uncertain-entry exclusions", "Sensitivity variant B can quantify changes without altering the strict cohort.", "Use sensitivity evidence; strict exclusion remains the cleaner time origin.", "Partly", "Yes"),
        ("Missing-week exclusions", "Exact Y is not observed across source gaps; only lower/upper bounds are defensible.", "Strict exclusion is technically clean; bounded sensitivity can show robustness.", "Partly", "Yes"),
        ("550k versus 160k roles", "160k is larger but lacks lyrics/track genre and has a different Y distribution.", "550k as rich-feature primary and 160k as robustness remains sensible.", "Partly", "Yes"),
        ("Genre use", "Fields exist and are complete/partly complete, but derivation and collection timing are undocumented.", "Preserve raw labels; predictive use remains provisional.", "No", "Yes"),
        ("Lyric-quality rule", "All primary tracks pass the simple threshold, but exact duplicate texts remain.", "Use quality indicators and grouped validation; do not exclude automatically.", "Partly", "Yes"),
        ("Release/calendar variable choice", "Release age is exactly entry year minus release year.", "Use only two independent terms; release age plus one calendar control is a candidate.", "Yes technically", "Yes for final specification"),
        ("Lyric character versus word count", "Pearson and Spearman correlations are approximately 0.99; VIF exceeds 80.", "Word count is the more interpretable single length control.", "Yes technically", "Advisable"),
        ("Artist-grouped validation", "87.1% of primary tracks belong to repeated grouping artists.", "Artist grouping is necessary for at least one validation analysis.", "Yes technically", "Yes for primary design"),
        ("Calendar-time-aware validation", "Feature coverage and rank threshold change over time; only 45 primary tracks enter in 2020.", "A time-aware analysis is important but the exact holdout needs discussion.", "No", "Yes"),
        ("Suspicious loudness value", "The 0.175 value is exact in the source and other fields parse normally.", "Retain with a flag and later sensitivity check; do not edit it.", "Yes technically", "No, unless exclusion is contemplated"),
        ("Train/test strategy", "Artist, time and version leakage imply different estimands and sample-size trade-offs.", "Compare grouped, time-aware and combined strategies before selecting the primary score.", "No", "Yes"),
    ]
    return pd.DataFrame(rows, columns=["unresolved_issue", "current_evidence", "evidence_presently_suggests", "resolvable_from_data_alone", "supervisor_confirmation_advisable"])


def plot_y_by_x(summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    x = summary.X.astype(int)
    ax1.bar(x - .14, summary.Y_mean, width=.28, label="Mean Y", color="#2A6496")
    ax1.bar(x + .14, summary.Y_median, width=.28, label="Median Y", color="#7AA6C2")
    ax1.set_xlabel("X: chart appearances in weeks 1–4")
    ax1.set_ylabel("Future chart-presence weeks")
    ax2 = ax1.twinx()
    ax2.plot(x, summary.Y_zero_percent, color="#B54A4A", marker="o", label="Y = 0")
    ax2.set_ylabel("Y = 0 (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    ax1.set_title("Future chart presence rises sharply with early chart presence X", loc="left", fontweight="bold", color="#17365D")
    ax1.grid(axis="y", alpha=.2)
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Y52_by_X.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_genre(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("N", ascending=False).genre_raw)
    data = [frame.loc[frame.genre_raw.eq(g), "Y_52"] for g in order]
    labels = [f"{g}\nN={len(values)}" for g, values in zip(order, data)]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "#7AA6C2"}, medianprops={"color": "#17365D", "linewidth": 1.6})
    ax.set_ylabel("Y_52: future chart-presence weeks")
    ax.set_title("Y_52 by broad genre — small genres are descriptive only", loc="left", fontweight="bold", color="#17365D")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Y52_by_broad_genre.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_entry_year(summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    years = summary.chart_entry_year.astype(int)
    ax1.bar(years, summary.N, color="#D9E8F2", edgecolor="#2A6496", label="Tracks")
    ax1.set_ylabel("Tracks")
    ax1.set_xlabel("Chart-entry year")
    ax1.set_xticks(years)
    ax2 = ax1.twinx()
    ax2.plot(years, summary.Y_mean, color="#2A6496", marker="o", label="Mean Y")
    ax2.plot(years, summary.Y_median, color="#B54A4A", marker="s", label="Median Y")
    ax2.set_ylabel("Future chart-presence weeks")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax1.set_title("Entry-year composition and Y_52", loc="left", fontweight="bold", color="#17365D")
    ax1.grid(axis="y", alpha=.18)
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Y52_and_composition_by_entry_year.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_release(release: pd.DataFrame) -> None:
    band = release.loc[release.release_timing_dimension.eq("release_age_band")].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(band))
    ax.bar(x, band.Y_mean, color="#2A6496", label="Mean Y")
    ax.plot(x, band.Y_median, color="#B54A4A", marker="o", label="Median Y")
    ax.set_xticks(x, band.category, rotation=25, ha="right")
    ax.set_ylabel("Future chart-presence weeks")
    ax.set_title("Y_52 by release age at chart entry", loc="left", fontweight="bold", color="#17365D")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Y52_by_release_age.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_composition(comparison: pd.DataFrame) -> None:
    subset = comparison.loc[(comparison.comparison_section.eq("composition")) & (comparison.dimension.eq("entry_year"))]
    pivot = subset.pivot(index="category", columns="cohort", values="percent").fillna(0)
    pivot = pivot[["550k_26w", "550k_52w"]]
    pivot.plot(kind="bar", figsize=(8.2, 4.8), color=["#7AA6C2", "#2A6496"])
    ax = plt.gca()
    ax.set_title("Entry-year composition changes between 26- and 52-week cohorts", loc="left", fontweight="bold", color="#17365D")
    ax.set_xlabel("Chart-entry year")
    ax.set_ylabel("Share of cohort (%)")
    ax.legend(["26 weeks", "52 weeks"], frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "26w_vs_52w_entry_year_composition.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_feature_cohorts(comparison: pd.DataFrame) -> None:
    y = comparison.loc[(comparison.comparison_section.eq("numeric_summary")) & (comparison.variable.eq("Y_52"))]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(y.cohort, y["mean"], color=["#2A6496", "#7AA6C2"])
    axes[0].set_title("Mean Y_52", loc="left", fontweight="bold", color="#17365D")
    axes[0].set_ylabel("Future chart-presence weeks")
    axes[1].bar(y.cohort, y.zero_percent, color=["#2A6496", "#7AA6C2"])
    axes[1].set_title("Zero proportion", loc="left", fontweight="bold", color="#17365D")
    axes[1].set_ylabel("Y_52 = 0 (%)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("The 550k- and 160k-linked subsets have different outcome distributions", x=.08, ha="left", fontweight="bold", color="#17365D")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(FIG / "550k_vs_160k_Y52_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(summary))
    lower = summary.Y_mean_lower_bound
    upper = summary.Y_mean_upper_bound
    ax.errorbar(x, (lower + upper) / 2, yerr=(upper - lower) / 2, fmt="o", color="#2A6496", capsize=5)
    ax.set_xticks(x, ["A strict", "B + uncertain", "C + source gaps", "D + both"])
    ax.set_ylabel("Mean Y_52 (identified interval)")
    ax.set_title("Exclusion sensitivity — unknown weeks remain interval-valued", loc="left", fontweight="bold", color="#17365D")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "exclusion_sensitivity_Y52_bounds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(
    duplicate_summary: pd.DataFrame, by: dict[str, pd.DataFrame], membership_h: pd.DataFrame,
    compare_h: pd.DataFrame, feature_compare: pd.DataFrame, feature_membership: pd.DataFrame,
    sensitivity: pd.DataFrame, provenance: pd.DataFrame, loudness: pd.DataFrame,
    redundant: pd.DataFrame, split_summary: pd.DataFrame,
) -> str:
    dx = by["x"].set_index("X")
    genre = by["genre"]
    year = by["entry_year"].set_index("chart_entry_year")
    classification = dict(zip(duplicate_summary.provisional_classification, duplicate_summary.duplicate_groups))
    y550 = feature_compare[(feature_compare.comparison_section == "numeric_summary") & (feature_compare.cohort == "550k_52w") & (feature_compare.variable == "Y_52")].iloc[0]
    y160 = feature_compare[(feature_compare.comparison_section == "numeric_summary") & (feature_compare.cohort == "160k_52w") & (feature_compare.variable == "Y_52")].iloc[0]
    s = sensitivity.set_index("variant")
    lines = [
        "# Remaining pre-model exploratory analysis and methodological evidence",
        "", "## Scope", "",
        "This versioned analysis investigates empirical questions that can be addressed before the supervisor is available. No predictive, count, survival, machine-learning or hierarchical model was fitted. Existing analysis-ready and historical outputs were not overwritten.",
        "", "## Duplicate lyrics and related song versions", "",
        f"The 26-week 550k cohort contains {duplicate_summary.duplicate_groups.sum()} exact normalised-lyric groups covering {duplicate_summary.tracks.sum()} tracks. The rule-based audit identifies {classification.get('same track/version duplicated in the feature source', 0)} groups as probable duplicate listings of the same recording/version, {classification.get('clean/explicit variants', 0)} clean/explicit groups, {classification.get('remix/live/acoustic/remaster versions', 0)} version-labelled groups, {classification.get('different recordings of the same underlying song', 0)} groups of related recordings, {classification.get('genuinely different songs sharing lyrics', 0)} possible different-song groups, and {classification.get('unresolved cases', 0)} unresolved groups.",
        "",
        "These classifications use titles, Spotify artist IDs, release information, exact audio descriptors and 160k explicit flags where available. They are provisional and no tracks were merged or removed. Exact lyric duplicates can leak text information across train and test sets. The conservative future rule is to keep every exact normalised-lyric group in one validation partition, in addition to artist grouping.",
        "", "## Y_52 and early chart presence X", "",
    ]
    for x in (1, 2, 3, 4):
        row = dx.loc[x]
        lines.append(f"- X={x}: N={int(row.N)}, mean Y={row.Y_mean:.2f}, median={row.Y_median:.1f}, zero proportion={row.Y_zero_percent:.1f}%, IQR={row.Y_p25:.1f}-{row.Y_p75:.1f}.")
    lines += ["", "The relationship is strong descriptively: songs present in more of the first four calendar weeks have substantially more subsequent chart weeks. This remains an association within already-charting songs, not a model result or causal effect.", "", "## Genre", ""]
    top = genre.sort_values("Y_mean", ascending=False).head(3)
    bottom = genre.sort_values("Y_mean").head(3)
    lines.append("Broad genres differ descriptively, but the cohort is dominated by Hip-Hop, Pop and Electronic. The highest and lowest sample means are documented in the table and plot; Blues, Classical, Folk and Jazz are too small for strong genre conclusions. No separate small-genre models are justified.")
    lines += ["", "## Entry year and follow-up composition", ""]
    for entry_year, row in year.iterrows():
        lines.append(f"- {int(entry_year)}: N={int(row.N)}, mean Y={row.Y_mean:.2f}, median={row.Y_median:.1f}, zero proportion={row.Y_zero_percent:.1f}%, cohort share={row.percent_of_cohort:.1f}%.")
    lines += ["", "Only 45 tracks enter in 2020 because the 52-week horizon requires a full year of follow-up. Consequently, apparent year differences combine calendar effects, match coverage and changing eligibility composition.", "", "## Release timing", "", "Y varies across release years and release-age bands, but release year, chart-entry year and release age are algebraically dependent. The descriptive tables are retained without choosing the final calendar specification.", "", "## 26-week versus 52-week cohorts", ""]
    mh = dict(zip(membership_h.membership, membership_h.tracks))
    lines.append(f"All {mh.get('both_26w_and_52w', 0)} 52-week tracks also appear in the 26-week cohort; {mh.get('26w_only', 0)} appear only at 26 weeks and none appear only at 52 weeks. The removed tracks are concentrated in later entry years, especially 2020, so moving to 52 weeks changes sample composition rather than sample size alone. Genre, X and audio-feature differences are reported with standardised differences.")
    lines += ["", "## 550k versus 160k at 52 weeks", ""]
    fm = dict(zip(feature_membership.membership, feature_membership.tracks))
    lines.append(f"The 550k cohort has 1,014 tracks and the 160k cohort 1,216; {fm.get('both_550k_and_160k', 0)} tracks overlap. Mean Y is {y550['mean']:.2f} in 550k and {y160['mean']:.2f} in 160k, while zero proportions are {y550.zero_percent:.1f}% and {y160.zero_percent:.1f}%. Entry year, X, debut rank and common audio distributions also differ. The 160k subset is therefore not simply a larger copy of the 550k subset.")
    lines += ["", "## Exclusion sensitivity", ""]
    for variant in sensitivity.variant:
        row = s.loc[variant]
        lines.append(f"- {variant}: N={int(row.N)}, mean-Y interval {row.Y_mean_lower_bound:.2f}-{row.Y_mean_upper_bound:.2f}, median interval {row.Y_median_lower_bound:.1f}-{row.Y_median_upper_bound:.1f}, definite/possible zero proportions {row.Y_definite_zero_percent:.1f}%-{row.Y_possible_zero_percent:.1f}%.")
    lines += ["", "Missing-week variants are interval-valued because unknown chart weeks were not converted to absences. The table and figure show whether conclusions are materially sensitive without pretending that exact Y is observed.", "", "## Genre provenance", "", "The supplied 550k archive contains only `songs.csv` and `artists.csv`; it includes no README or field dictionary. File placement establishes that `genre` and `niche_genres` are stored on song rows, while `main_genre` and `genres` are stored on artist rows. The project contains no evidence establishing their derivation, collection date, update rule or whether the song fields were inherited from artist taxonomy. Genre provenance therefore remains unresolved.", "", "## Loudness review", ""]
    lr = loudness.iloc[0]
    lines.append(f"Track `{lr.spotify_track_id}` ({lr.track_title}) has loudness 0.175 dB in the raw 160k file and the same value in the analysis-ready file. Other fields pass basic range checks. The raw source contains {int(lr.raw_positive_loudness_tracks)} positive-loudness tracks in total. This is not a parsing error; it is best treated as a plausible extreme or source anomaly, retained with a flag and checked in a later sensitivity analysis.")
    lines += ["", "## Redundant predictors", "", "Release age equals chart-entry year minus release year exactly. Only two independent calendar/release terms can be used together. Candidate specifications are saved, with release age plus one calendar-time control as an interpretable option. Lyric character and word counts correlate at approximately 0.99; word count is the clearer single length control, while character count can remain for audit traceability.", "", "## Prospective validation strategies", "", "1. **Artist-grouped cross-validation:** five balanced folds, with each primary artist confined to one fold. A stricter artist-plus-exact-lyric connected-component variant also prevents duplicate-text leakage. This uses all tracks but does not hold out future years.", "", "2. **Time-aware holdout:** train on 2017-2018, validate on 2019 and use 2020 as a small stress test. This tests calendar transfer but permits recurring artists unless purged.", "", "3. **Combined time-aware and grouped holdout:** test on 2019-2020 and remove from the early training set any artist or exact-lyric group appearing later. This prevents artist/version leakage but reduces training data and answers a harder new-artist future-period question.", "", "Prospective counts are saved for review; no split was adopted and no model was fitted.", "", "## Decisions still requiring confirmation", "", "The evidence can resolve exact technical redundancy and confirm that the loudness value is source-preserved. Supervisor confirmation remains advisable for the primary outcome, 52-versus-26-week horizon, strict exclusion rules, 550k/160k roles, genre use, lyric policy, calendar specification and the primary validation estimand.", ""]
    return "\n".join(lines)


def main() -> None:
    guards_before = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}
    prepare()

    f550_52 = pd.read_csv(AR550_52, low_memory=False)
    f550_26 = pd.read_csv(AR550_26, low_memory=False)
    f160_52 = pd.read_csv(AR160_52, low_memory=False)
    raw160 = pd.read_csv(RAW160, low_memory=False, dtype={"id": str, "release_date": str})
    fixed = pd.read_csv(FIXED, low_memory=False)
    first = pd.read_csv(FIRST, low_memory=False)
    features = pd.read_csv(FEATURE550, low_memory=False)
    bridge = pd.read_csv(BRIDGE, low_memory=False)

    groups, members, duplicate_summary = duplicate_lyrics(f550_26, raw160)
    write(groups, "duplicate_lyric_version_group_audit.csv")
    write(members, "duplicate_lyric_version_member_audit.csv")
    write(duplicate_summary, "duplicate_lyric_version_summary.csv")

    by = additional_eda(f550_52)
    write(by["x"], "Y52_by_X.csv")
    write(by["genre"], "Y52_by_genre.csv")
    write(by["entry_year"], "Y52_by_entry_year.csv")
    write(by["release"], "Y52_by_release_timing.csv")

    horizon_membership, horizon_compare = compare_horizons(f550_26, f550_52)
    write(horizon_membership, "cohort_26w_vs_52w_membership.csv")
    write(horizon_compare, "cohort_26w_vs_52w_comparison.csv")

    feature_compare, feature_membership, overlap_consistency = compare_feature_cohorts(f550_52, f160_52, first)
    write(feature_compare, "cohort_550k_vs_160k_52w_comparison.csv")
    write(feature_membership, "cohort_550k_vs_160k_52w_membership.csv")
    write(overlap_consistency, "cohort_550k_vs_160k_overlap_audio_consistency.csv")

    sensitivity, sensitivity_composition, sensitivity_membership = sensitivity_variants(fixed, features)
    write(sensitivity, "exclusion_sensitivity_summary.csv")
    write(sensitivity_composition, "exclusion_sensitivity_composition.csv")
    write(sensitivity_membership, "exclusion_sensitivity_membership.csv")

    provenance = genre_provenance()
    write(provenance, "genre_provenance_audit.csv")
    loud = loudness_review(raw160, f160_52)
    write(loud, "suspicious_loudness_review.csv")
    redundant = redundant_review(f550_52)
    write(redundant, "redundant_predictor_review.csv")

    assignments, split_summary, strategies = prospective_validation(f550_52, bridge)
    write(assignments, "prospective_validation_assignments.csv")
    write(split_summary, "prospective_validation_split_group_summary.csv")
    write(strategies, "validation_strategy_options.csv")
    decisions = decision_status()
    write(decisions, "methodological_decision_status.csv")

    plot_y_by_x(by["x"])
    plot_genre(f550_52, by["genre"])
    plot_entry_year(by["entry_year"])
    plot_release(by["release"])
    plot_horizon_composition(horizon_compare)
    plot_feature_cohorts(feature_compare)
    plot_sensitivity(sensitivity)

    guards_after = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}
    if guards_before != guards_after:
        raise RuntimeError("Protected outputs changed during remaining-EDA construction.")

    REPORT.write_text(report(duplicate_summary, by, horizon_membership, horizon_compare, feature_compare, feature_membership, sensitivity, provenance, loud, redundant, split_summary), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "version": "pre_model_remaining_eda_v2",
        "no_models_fitted": True, "existing_outputs_overwritten": False,
        "protected_tree_digests_before": guards_before, "protected_tree_digests_after": guards_after,
        "source_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in [AR550_52, AR550_26, AR160_52, RAW160, CHART, FIXED, FIRST, FEATURE550, BRIDGE]},
        "output_csv_count": len(list(OUT.glob("*.csv"))), "figure_count": len(list(FIG.glob("*.png"))),
        "method_note": "Descriptive analysis and prospective split counts only; unknown chart-source weeks remain unknown and are represented by bounds in sensitivity variants.",
    }
    (OUT / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(OUT), "report": str(REPORT), "figures": len(list(FIG.glob("*.png"))),
        "duplicate_groups": len(groups), "strict_550k_52w": len(f550_52),
    }, indent=2))


if __name__ == "__main__":
    main()
