#!/usr/bin/env python3
"""Close targeted pre-model questions and freeze a provisional specification.

This script is descriptive only. It does not fit, tune, or evaluate any model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "data/processed/pre_model_analysis_ready_v1"
EDA = ROOT / "data/processed/pre_model_remaining_eda_v2"
OUT = ROOT / "data/processed/final_pre_model_specification_v1"
REPORT = ROOT / "docs/final_pre_model_design_report.md"

F550_52 = READY / "analysis_ready_550k_52w.csv"
F550_26 = READY / "analysis_ready_550k_26w.csv"
F160_52 = READY / "analysis_ready_160k_52w.csv"
F160_26 = READY / "analysis_ready_160k_26w.csv"
BRIDGE = READY / "track_artist_bridge.csv"
PANEL = ROOT / "data/processed/chart_survival/song_week_panel.csv"
DUP_GROUP = EDA / "duplicate_lyric_version_group_audit.csv"
DUP_MEMBER = EDA / "duplicate_lyric_cohort_membership.csv"
AUDIO_AGREEMENT = EDA / "cohort_550k_vs_160k_overlap_audio_consistency.csv"
OLD_ALLOW_BLOCK = READY / "predictor_allowlist_blocklist.csv"

PROTECTED = [READY, EDA]
MISSING_SOURCE_WEEKS = {"2017-06-02", "2017-06-09", "2021-02-19"}
AUDIO = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(sha256(item).encode("ascii"))
    return digest.hexdigest()


def parse_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (list, tuple)):
            return [str(item).strip() for item in result if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [text]


class UnionFind:
    def __init__(self, items: list[str] | set[str]):
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def greedy_folds(groups: pd.Series, folds: int = 5) -> pd.Series:
    sizes = groups.value_counts().sort_values(ascending=False)
    totals = [0] * folds
    assignment: dict[str, int] = {}
    for group, size in sizes.items():
        target = int(np.argmin(totals))
        assignment[str(group)] = target + 1
        totals[target] += int(size)
    return groups.astype(str).map(assignment)


def distribution(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="raise")
    return {
        "N": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "variance": values.var(ddof=1),
        "standard_deviation": values.std(ddof=1),
        "minimum": values.min(),
        "p25": values.quantile(.25),
        "p50": values.quantile(.50),
        "p75": values.quantile(.75),
        "p90": values.quantile(.90),
        "p95": values.quantile(.95),
        "p99": values.quantile(.99),
        "maximum": values.max(),
        "zero_count": int(values.eq(0).sum()),
        "zero_percent": 100 * values.eq(0).mean(),
    }


def first_artist_map(bridge: pd.DataFrame) -> dict[str, str]:
    first = bridge.loc[bridge.is_grouping_artist.eq(1), ["spotify_track_id", "artist_id"]]
    if first.spotify_track_id.duplicated().any():
        raise RuntimeError("First-credited artist is not unique per track.")
    return dict(zip(first.spotify_track_id, first.artist_id))


def chart_version_audit(
    cohort: pd.DataFrame,
    panel: pd.DataFrame,
    groups: pd.DataFrame,
    members: pd.DataFrame,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = set(cohort.spotify_track_id)
    member = members.loc[members.spotify_track_id.isin(ids)].copy()
    counts = member.groupby("lyrics_normalised_sha256").spotify_track_id.nunique()
    duplicate_hashes = set(counts[counts.ge(2)].index)
    member = member.loc[member.lyrics_normalised_sha256.isin(duplicate_hashes)].copy()
    group_meta = groups.set_index("lyrics_normalised_sha256")
    first_map = first_artist_map(bridge)
    present = panel.loc[
        panel.spotify_track_id.isin(member.spotify_track_id) & panel.chart_state.eq("present"),
        ["spotify_track_id", "chart_week"],
    ].copy()
    present.chart_week = pd.to_datetime(present.chart_week)
    weeks_by_track = {
        track: set(values.chart_week.dt.strftime("%Y-%m-%d"))
        for track, values in present.groupby("spotify_track_id")
    }

    rows = []
    for digest, frame in member.groupby("lyrics_normalised_sha256", sort=True):
        track_ids = sorted(frame.spotify_track_id.unique())
        week_sets = {track: weeks_by_track.get(track, set()) for track in track_ids}
        all_weeks = sorted(set().union(*week_sets.values()))
        simultaneous_weeks = [week for week in all_weeks if sum(week in week_sets[t] for t in track_ids) > 1]
        identical_sets = len({tuple(sorted(week_sets[t])) for t in track_ids}) == 1
        ranges = {
            track: (min(week_sets[track]), max(week_sets[track])) if week_sets[track] else (None, None)
            for track in track_ids
        }
        range_overlap = False
        pair_relations = []
        gap_only_uncertainty = False
        for i, left in enumerate(track_ids):
            for right in track_ids[i + 1:]:
                shared = week_sets[left] & week_sets[right]
                lmin, lmax = ranges[left]
                rmin, rmax = ranges[right]
                overlap_range = bool(lmin and rmin and max(lmin, rmin) <= min(lmax, rmax))
                range_overlap = range_overlap or overlap_range
                if shared:
                    relation = "simultaneous"
                elif overlap_range:
                    relation = "alternating_with_overlapping_date_ranges"
                elif lmax and rmin and lmax < rmin:
                    between = pd.date_range(pd.Timestamp(lmax) + pd.Timedelta(days=7), pd.Timestamp(rmin) - pd.Timedelta(days=7), freq="7D")
                    labels = {week.strftime("%Y-%m-%d") for week in between}
                    uncertain = bool(labels) and labels.issubset(MISSING_SOURCE_WEEKS)
                    gap_only_uncertainty = gap_only_uncertainty or uncertain
                    relation = "left_then_right_source_gap_uncertain" if uncertain else "left_then_right"
                elif rmax and lmin and rmax < lmin:
                    between = pd.date_range(pd.Timestamp(rmax) + pd.Timedelta(days=7), pd.Timestamp(lmin) - pd.Timedelta(days=7), freq="7D")
                    labels = {week.strftime("%Y-%m-%d") for week in between}
                    uncertain = bool(labels) and labels.issubset(MISSING_SOURCE_WEEKS)
                    gap_only_uncertainty = gap_only_uncertainty or uncertain
                    relation = "right_then_left_source_gap_uncertain" if uncertain else "right_then_left"
                else:
                    relation = "unresolved"
                pair_relations.append({
                    "left": left, "right": right, "shared_present_weeks": len(shared),
                    "date_ranges_overlap": int(overlap_range), "relation": relation,
                })

        if any(not week_sets[track] for track in track_ids):
            primary = "unresolved"
        elif simultaneous_weeks and identical_sets:
            primary = "simultaneous_fully_coincident"
        elif simultaneous_weeks or range_overlap:
            primary = "partially_overlapping_histories"
        elif gap_only_uncertainty:
            primary = "unresolved_across_source_gap"
        elif all("then" in item["relation"] for item in pair_relations):
            primary = "sequential_non_overlapping"
        else:
            primary = "unresolved"

        metadata_class = group_meta.loc[digest, "provisional_classification"]
        metadata_separate = metadata_class in {
            "different recordings of the same underlying song",
            "remix/live/acoustic/remaster versions",
            "clean/explicit variants",
        }
        first_artists = sorted({first_map.get(track, "") for track in track_ids})
        supported_grouping = metadata_class != "genuinely different songs sharing lyrics" and metadata_class != "unresolved cases"
        rows.append({
            "lyrics_normalised_sha256": digest,
            "tracks_in_clean_52w_group": len(track_ids),
            "spotify_track_ids": "|".join(track_ids),
            "track_titles": " | ".join(frame.sort_values("spotify_track_id").track_title.astype(str)),
            "artist_strings": " | ".join(frame.sort_values("spotify_track_id").artist_string.astype(str)),
            "first_credited_artist_ids": "|".join(first_artists),
            "single_first_credited_artist_group": int(len(first_artists) == 1 and first_artists[0] != ""),
            "metadata_version_classification": metadata_class,
            "metadata_supports_separate_release_or_recording": int(metadata_separate),
            "first_chart_week": min(all_weeks) if all_weeks else "",
            "last_chart_week": max(all_weeks) if all_weeks else "",
            "group_present_track_weeks": sum(len(week_sets[t]) for t in track_ids),
            "unique_group_present_weeks": len(all_weeks),
            "simultaneous_chart_weeks": len(simultaneous_weeks),
            "has_simultaneous_chart_presence": int(bool(simultaneous_weeks)),
            "date_ranges_overlap": int(range_overlap),
            "primary_chart_history_relation": primary,
            "pairwise_relations_json": json.dumps(pair_relations, sort_keys=True),
            "fragmentation_risk": (
                "high" if supported_grouping and primary in {"sequential_non_overlapping", "simultaneous_fully_coincident"}
                else "moderate" if supported_grouping and primary == "partially_overlapping_histories"
                else "unresolved"
            ),
            "eligible_for_later_underlying_song_sensitivity_group": int(supported_grouping and len(first_artists) == 1),
            "sensitivity_grouping_rule": (
                "Union weekly presence for exact lyric hash + same first-credited artist + supported audit class; count a calendar week once; "
                "use earliest-charting member features, with lexicographic track-ID tie-break."
            ),
        })
    audit = pd.DataFrame(rows).sort_values(["primary_chart_history_relation", "lyrics_normalised_sha256"])
    relation_counts = audit.primary_chart_history_relation.value_counts()
    summary_rows = [
        ("affected_duplicate_groups", len(audit), "Mutually exclusive unit: groups."),
        ("affected_tracks", int(audit.tracks_in_clean_52w_group.sum()), "Distinct Spotify track IDs across groups."),
        ("groups_with_any_simultaneous_chart_presence", int(audit.has_simultaneous_chart_presence.sum()), "Not mutually exclusive with partially overlapping classification."),
        ("simultaneous_fully_coincident_groups", int(relation_counts.get("simultaneous_fully_coincident", 0)), "Mutually exclusive primary relation."),
        ("sequential_non_overlapping_groups", int(relation_counts.get("sequential_non_overlapping", 0)), "Mutually exclusive primary relation."),
        ("partially_overlapping_history_groups", int(relation_counts.get("partially_overlapping_histories", 0)), "Mutually exclusive primary relation; includes alternating IDs within overlapping date ranges."),
        ("unresolved_groups", int(relation_counts.get("unresolved", 0) + relation_counts.get("unresolved_across_source_gap", 0)), "Mutually exclusive unresolved primary relations combined."),
        ("metadata_separate_release_or_recording_groups", int(audit.metadata_supports_separate_release_or_recording.sum()), "Metadata classification; not mutually exclusive with chart-history relation."),
        ("metadata_identity_unresolved_groups", int(audit.metadata_version_classification.eq("unresolved cases").sum()), "Identity/version interpretation unresolved even when chart timing is observable."),
        ("groups_with_single_first_credited_artist", int(audit.single_first_credited_artist_group.sum()), "Must equal all affected groups for the requested containment check."),
        ("groups_eligible_for_later_grouped_sensitivity", int(audit.eligible_for_later_underlying_song_sensitivity_group.sum()), "Excludes unresolved or genuinely different-song metadata cases."),
        ("tracks_in_supported_sensitivity_groups", int(audit.loc[audit.eligible_for_later_underlying_song_sensitivity_group.eq(1), "tracks_in_clean_52w_group"].sum()), "Track IDs that would contribute to supported grouped units."),
        ("potential_unit_reduction_under_grouped_sensitivity", int(audit.loc[audit.eligible_for_later_underlying_song_sensitivity_group.eq(1), "tracks_in_clean_52w_group"].sum() - audit.eligible_for_later_underlying_song_sensitivity_group.sum()), "Reduction from replacing each supported group by one underlying-song unit."),
        ("potential_grouped_sensitivity_cohort_size", int(len(cohort) - (audit.loc[audit.eligible_for_later_underlying_song_sensitivity_group.eq(1), "tracks_in_clean_52w_group"].sum() - audit.eligible_for_later_underlying_song_sensitivity_group.sum())), "Illustrative later sensitivity size; primary dataset remains unchanged."),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "category_relationship_note"])
    return audit, summary


def y26_y52_comparison(
    cohort52: pd.DataFrame,
    cohort26: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    left = cohort52[["spotify_track_id", "Y_52"]].copy()
    right = cohort26[["spotify_track_id", "Y_26"]].copy()
    same = left.merge(right, on="spotify_track_id", how="left", validate="one_to_one")
    if same.Y_26.isna().any() or len(same) != 1014:
        raise RuntimeError("The same-track horizon comparison did not reconcile to 1,014 tracks.")
    same["increment"] = same.Y_52 - same.Y_26

    panel52 = panel.loc[
        panel.spotify_track_id.isin(same.spotify_track_id) & panel.week_index_since_first_entry.between(0, 51),
        ["spotify_track_id", "week_index_since_first_entry", "chart_state"],
    ].copy()
    reentry_rows = []
    for track, frame in panel52.groupby("spotify_track_id", sort=False):
        states = frame.sort_values("week_index_since_first_entry").set_index("week_index_since_first_entry").chart_state.to_dict()
        if any(states.get(index) == "unknown_source_gap" for index in range(52)):
            raise RuntimeError("Strict 52-week cohort unexpectedly crosses a missing source week.")
        block = 0
        was_present = False
        extra = 0
        reentry_extra = 0
        for index in range(52):
            present = states.get(index) == "present"
            if present and not was_present:
                block += 1
            if index >= 26 and present:
                extra += 1
                if block > 1:
                    reentry_extra += 1
            was_present = present
        reentry_rows.append((track, extra, reentry_extra))
    reentry = pd.DataFrame(reentry_rows, columns=["spotify_track_id", "panel_increment", "increment_weeks_in_reentry_blocks"])
    same = same.merge(reentry, on="spotify_track_id", validate="one_to_one")
    if not same.increment.eq(same.panel_increment).all():
        raise RuntimeError("Y_52 - Y_26 does not match chart-panel weeks 27-52.")

    rows = []
    for name in ("Y_26", "Y_52"):
        values = distribution(same[name])
        for metric, value in values.items():
            rows.append({"section": "outcome_distribution", "metric": metric, "Y_26": value if name == "Y_26" else np.nan, "Y_52": value if name == "Y_52" else np.nan, "note": "Same 1,014 clean 550k-linked tracks."})
    comparison = pd.DataFrame(rows).groupby(["section", "metric", "note"], as_index=False, dropna=False).agg({"Y_26": "max", "Y_52": "max"})
    pearson = same.Y_26.corr(same.Y_52, method="pearson")
    spearman = same.Y_26.corr(same.Y_52, method="spearman")
    extra_total = int(same.increment.sum())
    reentry_extra_total = int(same.increment_weeks_in_reentry_blocks.sum())
    facts = {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "Y26_zero_Y52_positive_count": int((same.Y_26.eq(0) & same.Y_52.gt(0)).sum()),
        "Y26_zero_Y52_positive_percent": 100 * (same.Y_26.eq(0) & same.Y_52.gt(0)).mean(),
        "Y26_at_ceiling_count": int(same.Y_26.eq(22).sum()),
        "Y26_at_ceiling_percent": 100 * same.Y_26.eq(22).mean(),
        "Y52_at_ceiling_count": int(same.Y_52.eq(48).sum()),
        "Y52_at_ceiling_percent": 100 * same.Y_52.eq(48).mean(),
        "total_additional_weeks_27_52": extra_total,
        "additional_weeks_in_reentry_blocks": reentry_extra_total,
        "percent_additional_weeks_in_reentry_blocks": 100 * reentry_extra_total / extra_total if extra_total else 0,
        "tracks_with_positive_increment": int(same.increment.gt(0).sum()),
        "tracks_with_positive_increment_percent": 100 * same.increment.gt(0).mean(),
        "tracks_with_any_reentry_associated_increment": int(same.increment_weeks_in_reentry_blocks.gt(0).sum()),
        "tracks_with_any_reentry_associated_increment_percent": 100 * same.increment_weeks_in_reentry_blocks.gt(0).mean(),
    }
    for metric, value in facts.items():
        comparison.loc[len(comparison)] = {
            "section": "cross_horizon_comparison", "metric": metric,
            "note": "Single shared statistic for the same-track comparison.",
            "Y_26": value, "Y_52": value,
        }

    bins = pd.cut(
        same.increment,
        bins=[-0.1, 0, 2, 5, 10, np.inf],
        labels=["0 additional weeks", "1-2", "3-5", "6-10", ">10"],
        include_lowest=True,
    )
    increment = bins.value_counts(sort=False).rename_axis("increment_category").reset_index(name="tracks")
    increment["percent_of_1014"] = 100 * increment.tracks / len(same)
    increment["categories_mutually_exclusive"] = 1
    return comparison, increment, facts


def variable_roles() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str, str, str, str, str]] = []

    def add(name: str, source: str, available: str, role: str, dtype: str, treatment: str, leakage: str, notes: str) -> None:
        rows.append((name, source, available, role, dtype, treatment, leakage, notes))

    add("spotify_track_id", "chart + feature link", "yes", "grouping-only", "categorical identifier", "analysis-unit key; never numeric", "safe but identifier memorisation risk", "Primary analysis unit; related-version sensitivity is separate.")
    add("chart_entry_date", "chart panel", "yes", "descriptor", "date", "retain for splitting/audit; do not feed raw date", "safe", "Time origin.")
    add("chart_entry_year", "derived from chart entry", "yes", "predictor", "categorical", "categorical calendar control in main candidate", "safe", "Do not assume a linear trend automatically.")
    add("X", "chart weeks 1-4", "yes at end of week 4", "predictor", "categorical", "four-level categorical in main regression; linear 1-4 only as sensitivity", "safe", "Strongly nonlinear EDA relationship.")
    add("Y_26", "chart weeks 5-26", "no", "outcome", "bounded count", "outcome with support 0-22", "not a predictor", "Sensitivity horizon outcome.")
    add("Y_52", "chart weeks 5-52", "no", "outcome", "bounded count", "outcome with support 0-48", "not a predictor", "Provisional primary outcome.")
    for name in ["L_26", "L_52", "Z_count_26", "Z_count_52", "Z_reentry_26", "Z_reentry_52"]:
        add(name, "full fixed-window chart history", "no", "descriptor", "numeric", "descriptive only", "blocked: outcome-window information", "Never predict Y with L or Z.")
    for name in ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo"]:
        add(name, "550k/160k audio features", "yes", "predictor", "numeric", "continuous; train-fold scaling if model needs it", "safe baseline feature", "Inspect source agreement and transformations inside training folds.")
    add("instrumentalness", "550k/160k audio features", "yes", "predictor", "numeric", "continuous; flag zero concentration/skew; possible fold-local transform", "safe baseline feature", "Do not transform before splitting.")
    add("duration_ms", "550k/160k audio features", "yes", "predictor", "numeric", "continuous; possible fold-local log transform", "safe baseline feature", "Retain milliseconds in audit data.")
    add("key", "550k/160k audio features", "yes", "predictor", "categorical", "12-level categorical; do not treat 0-11 as ordinary magnitude", "safe baseline feature", "Optional cyclic representation would require musical justification.")
    add("mode", "550k/160k audio features", "yes", "predictor", "binary", "binary indicator", "safe baseline feature", "Major/minor coding as supplied.")
    add("release_age_years", "release year and chart-entry year", "yes", "predictor", "numeric", "continuous candidate; inspect log1p/nonlinearity inside training folds", "safe but collinear with release and entry year", "Use with entry-year control, not with both source years.")
    add("release_year", "feature source", "yes", "predictor", "numeric or categorical", "alternative specification only", "safe but exactly dependent with entry year and release age", "Exclude when main release-age + entry-year specification is used.")
    add("release_date", "160k feature source", "nominally yes", "descriptor", "date", "avoid exact-day common-cohort use", "timing anomaly present", "Track 6VpQdig9pdpTSIFItgkJV5 has recorded release two days after chart entry.")
    add("rank_200_stream_threshold_at_entry", "chart panel", "yes", "predictor", "numeric", "alternative continuous calendar-intensity control", "safe at entry", "Alternative to categorical entry year, not an outcome stream for the song.")
    add("genre_raw", "550k song row", "yes but provenance/timing unclear", "predictor", "categorical", "optional one-hot/categorical sensitivity", "provenance-flagged", "Derivation and snapshot timing are undocumented.")
    add("niche_genres_raw", "550k song row", "yes but provenance/timing unclear", "descriptor", "multi-label categorical", "preserve raw; defer representation", "provenance-flagged", "Do not aggressively collapse before a rule is approved.")
    add("first_credited_artist_id", "track-artist bridge", "yes", "grouping-only", "categorical identifier", "grouped validation and clustered inference only", "safe but memorisation risk", "Never use as an ordinary numeric predictor.")
    add("all_credited_artist_ids", "track-artist bridge", "yes", "grouping-only", "multi-label identifier", "collaborator-overlap audit only", "safe but memorisation risk", "Complete all-collaborator separation may be impractical.")
    add("lyrics_text", "550k feature source", "yes", "predictor source", "text", "fold-local TF-IDF or transparent lexical derivation only", "safe if transformer fitted in training fold", "Never vectorise on the full dataset.")
    add("lyrics_word_count", "550k lyrics", "yes", "predictor", "numeric", "single simple length control; possible fold-local log1p", "safe", "Not a substantive lyrics-content representation by itself.")
    add("lyrics_character_count", "550k lyrics", "yes", "blocked", "numeric", "retain only for audit", "blocked due near-redundancy with word count", "Do not include alongside word count.")
    add("lyrics_normalised_sha256", "derived lyrics", "yes", "grouping-only", "categorical hash", "keep related text/version groups within fold", "safe grouping value; predictor prohibited", "Not a predictor.")
    add("lyrics_quality_flags", "derived lyrics", "yes", "descriptor", "binary", "audit/stratification only initially", "safe", "No automatic exclusion or content claim.")
    add("debut_rank", "entry chart week", "yes at week 1", "descriptor", "numeric", "not included in frozen blocks pending design confirmation", "timing-safe but outside current X-only early block", "May be reconsidered later without using future peak rank.")
    add("explicit", "160k only", "yes", "descriptor", "binary", "exclude from principal common-cohort robustness specification", "safe source-specific field", "Optional source-specific model only if later justified.")

    blocked = {
        "track_popularity": "later snapshot may reflect outcome",
        "artist_popularity": "later snapshot may reflect outcome",
        "artist_followers": "later snapshot may reflect outcome",
        "average_artist_popularity": "later snapshot may reflect outcome",
        "future_ranks": "outcome-window information",
        "future_streams": "outcome-window information",
        "eventual_peak_rank": "full-history information",
        "total_chart_duration": "full-history outcome",
        "first_spell_duration": "outcome information when predicting fixed-window Y",
        "later_spell_information": "outcome-window information",
        "later_reentry_descriptors": "outcome-window information",
        "old_landmark_eligibility_flags": "selection on already-survived songs",
        "any_feature_derived_after_week_4": "outside prediction landmark",
    }
    for name, reason in blocked.items():
        add(name, "historical/derived", "no or timing-unsafe", "blocked", "various", "exclude from every predictor matrix", f"blocked: {reason}", "Preserve only for outcome construction or audit where needed.")
    return pd.DataFrame(rows, columns=[
        "variable_name", "source", "available_by_prediction_time", "variable_role", "variable_type",
        "proposed_initial_modelling_treatment", "leakage_status", "notes",
    ])


def feature_blocks() -> pd.DataFrame:
    audio_treatment = {
        "key": "categorical 12 levels", "mode": "binary",
        "instrumentalness": "continuous; skew/zero flag; possible fold-local transform",
        "duration_ms": "continuous; possible fold-local log transform",
    }
    for name in AUDIO:
        audio_treatment.setdefault(name, "continuous; fold-local scaling if required")
    definitions: list[tuple[str, str, str, list[str], str]] = [
        ("F0", "trivial benchmark", "550k and common 160k", ["__intercept_only__"], "Naive future benchmark; no song predictors."),
        ("F1", "X only", "550k and common 160k", ["X"], "Early chart presence only."),
        ("F2", "song characteristics only", "550k and common 160k", AUDIO, "Leakage-safe audio/song characteristics; excludes X."),
        ("F3", "X + audio/song characteristics", "550k and common 160k", ["X", *AUDIO], "Tests addition of song characteristics beyond X."),
        ("F4", "X + audio/song + calendar/release", "550k and common 160k", ["X", *AUDIO, "release_age_years", "chart_entry_year"], "Main non-collinear calendar candidate."),
        ("F5a", "full safe 550k without genre", "550k", ["X", *AUDIO, "release_age_years", "chart_entry_year", "lyrics_word_count"], "Adds a simple lyric-length control, not lyrics content."),
        ("F5b", "full safe 550k with broad genre", "550k", ["X", *AUDIO, "release_age_years", "chart_entry_year", "lyrics_word_count", "genre_raw"], "Genre-provenance sensitivity."),
        ("F6a", "future fold-local lyric content", "550k", ["X", *AUDIO, "release_age_years", "chart_entry_year", "lyrics_word_count", "lyrics_tfidf_fold_local"], "Future TF-IDF extension; not created in this task."),
        ("F6b", "future transparent lexical sensitivity", "550k", ["X", *AUDIO, "release_age_years", "chart_entry_year", "lyrics_word_count", "transparent_lexical_summaries_fold_local"], "Future low-dimensional lexical extension; not created in this task."),
    ]
    rows = []
    for block, label, scope, variables, note in definitions:
        for variable in variables:
            if variable == "__intercept_only__":
                treatment = "none; estimate training-outcome mean only"
            elif variable == "X":
                treatment = "categorical 4 levels"
            elif variable in audio_treatment:
                treatment = audio_treatment[variable]
            elif variable in {"chart_entry_year", "genre_raw"}:
                treatment = "categorical"
            elif variable in {"release_age_years", "lyrics_word_count"}:
                treatment = "continuous; inspect nonlinearity/fold-local transformation"
            else:
                treatment = "fit representation inside each training fold only"
            rows.append({
                "feature_block": block, "feature_block_name": label, "cohort_scope": scope,
                "variable": variable, "initial_treatment": treatment,
                "provisional_status": "frozen for next benchmark task; pending supervisor confirmation",
                "block_note": note,
            })
    return pd.DataFrame(rows)


def lyric_plan() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "option": "regularised TF-IDF", "provisional_role": "primary lyric-content candidate",
            "representation": "Word 1-2 grams with training-fold vocabulary, document frequencies and feature limits; regularisation required.",
            "interpretability": "Moderate: influential terms/ngrams can be inspected, but correlated terms make individual coefficients unstable.",
            "dimensionality": "High; provisionally cap at roughly 2,000-5,000 training-derived features.",
            "leakage_control": "Fit tokenizer/vocabulary/IDF and any feature selection on each training fold only; transform validation/test afterwards.",
            "computational_burden": "Moderate and feasible for N≈1,014 with sparse matrices.",
            "suitability": "Good thesis-scale primary text comparison if kept regularised and modest.",
        },
        {
            "option": "transparent lexical summaries", "provisional_role": "simpler sensitivity candidate",
            "representation": "Small preregistered set such as lexical diversity, average word length, repetition, line count and dictionary-based counts where a documented lexicon is available.",
            "interpretability": "High; each feature has a direct definition.",
            "dimensionality": "Low, preferably fewer than 20 features.",
            "leakage_control": "Definitions fixed before evaluation; scaling and any data-dependent thresholds fitted within training folds.",
            "computational_burden": "Low.",
            "suitability": "Useful sensitivity analysis; less expressive than TF-IDF.",
        },
        {
            "option": "fold-local TF-IDF plus truncated SVD", "provisional_role": "optional computational sensitivity",
            "representation": "TF-IDF followed by a small latent semantic representation, with both stages fitted inside training folds.",
            "interpretability": "Lower because components combine many terms.",
            "dimensionality": "Moderate, for example tens rather than thousands of components; exact number must be tuned within training only.",
            "leakage_control": "Vocabulary, IDF and SVD fitted only on training-fold lyrics.",
            "computational_burden": "Moderate and feasible, but adds tuning and explanation burden.",
            "suitability": "Optional; not preferred over regularised TF-IDF for the main thesis comparison.",
        },
    ])


def calendar_specs() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "specification_id": "C1_main", "included_variables": "release_age_years + categorical chart_entry_year",
            "excluded_dependent_variable": "release_year", "interpretation": "Separates age at entry from broad calendar period without exact collinearity.",
            "advantages": "Simple and interpretable; common to 550k and 160k.",
            "limitations": "Only four entry-year categories under Y_52; release age is strongly skewed and may need a fold-local nonlinear treatment.",
            "release_date_anomaly_treatment": "Use year-level age; do not alter raw exact date for 6VpQdig9pdpTSIFItgkJV5.",
            "provisional_status": "preferred candidate pending supervisor confirmation",
        },
        {
            "specification_id": "C2_calendar_intensity", "included_variables": "release_age_years + rank_200_stream_threshold_at_entry",
            "excluded_dependent_variable": "release_year and raw chart_entry_year", "interpretation": "Uses contemporaneous chart-entry intensity instead of year categories.",
            "advantages": "Continuous measure of changing chart environment and common to both feature cohorts.",
            "limitations": "Threshold may reflect platform growth and seasonality; functional form requires checking.",
            "release_date_anomaly_treatment": "Use year-level age; retain exact source value only for audit.",
            "provisional_status": "alternative sensitivity candidate",
        },
        {
            "specification_id": "C3_two_years", "included_variables": "release_year + categorical chart_entry_year",
            "excluded_dependent_variable": "release_age_years", "interpretation": "Directly controls source release cohort and chart-entry calendar cohort.",
            "advantages": "Avoids the exact three-variable dependency.",
            "limitations": "Old release years are sparse and may require grouping; less direct interpretation than release age.",
            "release_date_anomaly_treatment": "Use release year only.",
            "provisional_status": "secondary alternative, not main",
        },
    ])


def collaboration_and_cv(
    cohort: pd.DataFrame,
    bridge: pd.DataFrame,
    version_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    ids = set(cohort.spotify_track_id)
    b = bridge.loc[bridge.spotify_track_id.isin(ids)].sort_values(["spotify_track_id", "credit_order"]).copy()
    if b.spotify_track_id.nunique() != len(cohort):
        raise RuntimeError("Artist bridge does not cover every 52-week track.")
    track_artists = b.groupby("spotify_track_id").artist_id.apply(lambda s: tuple(dict.fromkeys(s.astype(str)))).to_dict()
    first = b.loc[b.is_grouping_artist.eq(1)].set_index("spotify_track_id").artist_id.astype(str).to_dict()
    single = sum(len(track_artists[track]) == 1 for track in ids)
    multi = len(ids) - single
    all_artists = {artist for values in track_artists.values() for artist in values}

    graph = UnionFind(ids)
    tracks_by_artist: dict[str, list[str]] = defaultdict(list)
    for track, artists in track_artists.items():
        for artist in artists:
            tracks_by_artist[artist].append(track)
    for tracks in tracks_by_artist.values():
        root = tracks[0]
        for track in tracks[1:]:
            graph.union(root, track)
    graph_components: dict[str, list[str]] = defaultdict(list)
    for track in ids:
        graph_components[graph.find(track)].append(track)
    component_sizes = sorted((len(values) for values in graph_components.values()), reverse=True)

    # Primary grouping components: first-credited artist, strengthened by every supported exact-lyric/version group.
    primary = UnionFind(ids)
    tracks_by_first: dict[str, list[str]] = defaultdict(list)
    for track, artist in first.items():
        tracks_by_first[artist].append(track)
    for tracks in tracks_by_first.values():
        for track in tracks[1:]:
            primary.union(tracks[0], track)
    supported = version_audit.loc[version_audit.eligible_for_later_underlying_song_sensitivity_group.eq(1)]
    for value in supported.spotify_track_ids:
        tracks = [track for track in str(value).split("|") if track in ids]
        for track in tracks[1:]:
            primary.union(tracks[0], track)
    group_label = {track: min(t for t in ids if primary.find(t) == primary.find(track)) for track in ids}
    assign = cohort[["spotify_track_id", "chart_entry_year", "lyrics_normalised_sha256"]].copy()
    assign["first_credited_artist_id"] = assign.spotify_track_id.map(first)
    assign["all_credited_artist_ids"] = assign.spotify_track_id.map(lambda track: "|".join(track_artists[track]))
    assign["primary_group_component"] = assign.spotify_track_id.map(group_label)
    assign["fold"] = greedy_folds(assign.primary_group_component).map(lambda value: f"fold_{int(value)}")
    assign["exact_lyric_group_crosses_folds"] = assign.groupby("lyrics_normalised_sha256").fold.transform("nunique").gt(1).astype(int)
    assign["first_artist_crosses_folds"] = assign.groupby("first_credited_artist_id").fold.transform("nunique").gt(1).astype(int)
    supported_hashes = set(supported.lyrics_normalised_sha256)
    assign["supported_related_version_group"] = assign.lyrics_normalised_sha256.isin(supported_hashes).astype(int)
    assign["related_version_group_crosses_folds"] = (
        assign.groupby("lyrics_normalised_sha256").fold.transform("nunique").gt(1) & assign.supported_related_version_group.eq(1)
    ).astype(int)

    collaborator_fold_count: dict[str, set[str]] = defaultdict(set)
    for _, row in assign.iterrows():
        for artist in track_artists[row.spotify_track_id]:
            collaborator_fold_count[artist].add(row.fold)
    crossing_collaborators = {artist for artist, folds in collaborator_fold_count.items() if len(folds) > 1}
    assign["has_any_credited_artist_cross_fold_overlap"] = assign.spotify_track_id.map(
        lambda track: int(any(artist in crossing_collaborators for artist in track_artists[track]))
    )

    fold_counts = assign.fold.value_counts().sort_index()
    diagnostic_rows = [
        ("analysis_unit", "cohort_tracks", len(ids), "One row per Spotify track ID."),
        ("artist_credits", "single_artist_tracks", single, "Mutually exclusive with multi-artist tracks."),
        ("artist_credits", "multi_artist_tracks", multi, "Mutually exclusive with single-artist tracks."),
        ("artist_credits", "distinct_first_credited_artists", len(set(first.values())), "Primary grouping identities."),
        ("artist_credits", "distinct_all_credited_artists", len(all_artists), "Includes collaborators."),
        ("collaboration_graph", "connected_components", len(graph_components), "Tracks connected when they share any credited artist, transitively."),
        ("collaboration_graph", "largest_component_tracks", component_sizes[0], "All-collaborator connected component."),
        ("collaboration_graph", "largest_component_percent", 100 * component_sizes[0] / len(ids), "Percentage of cohort in largest all-collaborator component."),
        ("primary_grouped_cv", "group_components", assign.primary_group_component.nunique(), "First artist strengthened by supported exact lyric/version groups."),
        ("primary_grouped_cv", "collaborators_crossing_folds", len(crossing_collaborators), "Reported limitation; first-artist grouping does not prevent this."),
        ("primary_grouped_cv", "tracks_with_any_cross_fold_collaborator", int(assign.has_any_credited_artist_cross_fold_overlap.sum()), "Includes tracks whose lead or collaborator appears in another fold as any credit."),
    ]
    for fold, count in fold_counts.items():
        diagnostic_rows.append(("primary_grouped_cv", f"{fold}_tracks", int(count), "Prospective assignment only; no model fitted."))
    for size, count in Counter(component_sizes).most_common():
        diagnostic_rows.append(("collaboration_component_size_distribution", f"components_with_{size}_tracks", count, "All-collaborator graph."))
    diagnostics = pd.DataFrame(diagnostic_rows, columns=["section", "metric", "value", "note"])
    facts = {
        "single_artist_tracks": single, "multi_artist_tracks": multi,
        "distinct_first_artists": len(set(first.values())), "distinct_all_artists": len(all_artists),
        "collaboration_components": len(graph_components), "largest_component_tracks": component_sizes[0],
        "largest_component_percent": 100 * component_sizes[0] / len(ids),
        "cross_fold_collaborators": len(crossing_collaborators),
        "tracks_with_cross_fold_collaborator": int(assign.has_any_credited_artist_cross_fold_overlap.sum()),
    }
    return diagnostics, assign.sort_values(["fold", "spotify_track_id"]), facts


def temporal_assignments(
    cohort: pd.DataFrame,
    bridge: pd.DataFrame,
    version_audit: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    ids = set(cohort.spotify_track_id)
    b = bridge.loc[bridge.spotify_track_id.isin(ids)].copy()
    first = b.loc[b.is_grouping_artist.eq(1)].set_index("spotify_track_id").artist_id.astype(str).to_dict()
    data = cohort[["spotify_track_id", "chart_entry_year", "lyrics_normalised_sha256"]].copy()
    data["first_credited_artist_id"] = data.spotify_track_id.map(first)
    data["temporal_partition"] = np.where(
        data.chart_entry_year.le(2018), "train_2017_2018",
        np.where(data.chart_entry_year.eq(2019), "validation_2019", "test_2020"),
    )
    later = data.chart_entry_year.ge(2019)
    later_artists = set(data.loc[later, "first_credited_artist_id"])
    supported_hashes = set(version_audit.loc[version_audit.eligible_for_later_underlying_song_sensitivity_group.eq(1), "lyrics_normalised_sha256"])
    later_related = set(data.loc[later & data.lyrics_normalised_sha256.isin(supported_hashes), "lyrics_normalised_sha256"])
    early = data.chart_entry_year.le(2018)
    keep_early = early & ~data.first_credited_artist_id.isin(later_artists) & ~data.lyrics_normalised_sha256.isin(later_related)
    data["combined_time_grouped_status"] = np.where(
        later, "strict_test_2019_2020",
        np.where(keep_early, "strict_train_2017_2018", "purged_from_strict_train"),
    )
    data["horizon_weeks"] = horizon
    counts = data.temporal_partition.value_counts().to_dict()
    facts = {
        "total": len(data),
        "train_2017_2018": int(counts.get("train_2017_2018", 0)),
        "validation_2019": int(counts.get("validation_2019", 0)),
        "test_2020": int(counts.get("test_2020", 0)),
        "strict_combined_train": int(data.combined_time_grouped_status.eq("strict_train_2017_2018").sum()),
        "strict_combined_test": int(data.combined_time_grouped_status.eq("strict_test_2019_2020").sum()),
        "purged_from_strict_train": int(data.combined_time_grouped_status.eq("purged_from_strict_train").sum()),
    }
    return data.sort_values(["temporal_partition", "spotify_track_id"]), facts


def common_spec(agreement: pd.DataFrame) -> pd.DataFrame:
    variables = ["X", *AUDIO, "release_age_years", "chart_entry_year"]
    agreement_map = agreement.set_index("feature").to_dict("index")
    rows = []
    for variable in variables:
        source_agreement = agreement_map.get(variable, {})
        max_diff = source_agreement.get("maximum_absolute_difference", np.nan)
        notable = ""
        if variable == "tempo":
            notable = "Typical difference is small, but maximum disagreement is 77.642 BPM."
        elif variable == "key":
            notable = "Five of 638 overlaps differ; maximum categorical difference is 7 key codes."
        elif variable == "mode":
            notable = "Three of 638 overlaps differ; maximum difference changes the binary mode."
        elif variable == "danceability":
            notable = "Mean absolute difference is small, but maximum disagreement is 0.25."
        elif variable == "liveness":
            notable = "Mean absolute difference is small, but maximum disagreement is 0.155."
        elif variable == "loudness":
            notable = "Mean absolute difference is small, but maximum disagreement is 1.018 dB."
        rows.append({
            "variable": variable,
            "included_in_principal_common_specification": 1,
            "550k_available": 1,
            "160k_available": 1,
            "initial_treatment": (
                "categorical 4 levels" if variable == "X"
                else "categorical 12 levels" if variable == "key"
                else "binary" if variable == "mode"
                else "categorical" if variable == "chart_entry_year"
                else "continuous; inspect fold-local transformation" if variable == "release_age_years"
                else "continuous"
            ),
            "overlap_tracks_in_agreement_audit": source_agreement.get("overlap_tracks", np.nan),
            "exact_equal_percent": source_agreement.get("exact_equal_percent", np.nan),
            "mean_absolute_difference": source_agreement.get("mean_absolute_difference", np.nan),
            "maximum_absolute_difference": max_diff,
            "agreement_interpretation": (
                "Derived identically from chart/entry information; source agreement audit not applicable."
                if not source_agreement else "Average agreement is close, but record-level equality is not assumed."
            ),
            "notable_outlier_disagreement": notable,
            "robustness_role": "Refit the same specification in each feature-linked cohort; do not pool source rows.",
        })
    rows.append({
        "variable": "explicit", "included_in_principal_common_specification": 0,
        "550k_available": 0, "160k_available": 1, "initial_treatment": "source-specific optional only",
        "overlap_tracks_in_agreement_audit": np.nan, "exact_equal_percent": np.nan,
        "mean_absolute_difference": np.nan, "maximum_absolute_difference": np.nan,
        "agreement_interpretation": "Not conceptually comparable because it is absent from the 550k analysis table.",
        "notable_outlier_disagreement": "", "robustness_role": "Exclude from principal common robustness comparison.",
    })
    return pd.DataFrame(rows)


def leakage_audit(blocks: pd.DataFrame) -> pd.DataFrame:
    prohibited = {
        "track_popularity", "artist_popularity", "artist_followers", "average_artist_popularity",
        "future_ranks", "future_streams", "eventual_peak_rank", "L_26", "L_52", "Z_count_26",
        "Z_count_52", "Z_reentry_26", "Z_reentry_52", "later_reentry_descriptors",
        "total_chart_duration", "first_spell_duration", "later_spell_information",
        "old_landmark_eligibility_flags", "any_feature_derived_after_week_4",
    }
    rows = []
    for _, row in blocks.iterrows():
        variable = row.variable
        blocked = variable in prohibited
        rows.append({
            "audit_type": "feature_block_variable", "feature_block": row.feature_block,
            "variable": variable, "prohibited_or_leakage_prone": int(blocked),
            "present_in_feature_block": 1, "audit_pass": int(not blocked),
            "note": "Blocked variable must never appear in a future predictor matrix." if blocked else "Allowed by provisional timing/role specification.",
        })
    in_blocks = set(blocks.variable)
    for variable in sorted(prohibited):
        rows.append({
            "audit_type": "mandatory_blocklist_presence_check", "feature_block": "ALL",
            "variable": variable, "prohibited_or_leakage_prone": 1,
            "present_in_feature_block": int(variable in in_blocks), "audit_pass": int(variable not in in_blocks),
            "note": "Required blocked variable is absent from all frozen feature blocks.",
        })
    return pd.DataFrame(rows)


def decision_status(facts: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("Spotify track ID as primary analysis unit", facts["analysis_unit_evidence"], "Retain as primary; add grouped underlying-song sensitivity later.", "partly", "yes"),
        ("Related-version fragmentation", facts["version_relation_evidence"], "Does not invalidate track-level outcomes, but affected groups require fold grouping and sensitivity aggregation.", "partly", "yes"),
        ("52 versus 26 weeks", facts["horizon_evidence"], "52 adds later information for a minority and reduces temporal test coverage; retain both provisionally.", "no", "yes"),
        ("Bounded outcome family", "Y_26 has support 0-22 and Y_52 support 0-48.", "Compare unbounded diagnostic count baselines with at least one bounded formulation later.", "partly", "yes"),
        ("X representation", "EDA is strongly nonlinear across X=1,2,3,4.", "Categorical X in main interpretable specification; linear coding only as sensitivity.", "yes technically", "advisable"),
        ("Musical key representation", "Key codes denote pitch classes, not an ordered magnitude.", "Use categorical encoding initially.", "yes technically", "advisable"),
        ("Genre use", "Broad genre is complete but derivation and timing are undocumented.", "Only F5b provenance sensitivity; not required in the main no-genre block.", "no", "yes"),
        ("Lyric-content representation", "N≈1,014 and duplicate text groups require fold-local processing.", "Regularised fold-local TF-IDF primary candidate; transparent lexical summaries sensitivity.", "partly", "yes"),
        ("Calendar/release specification", "Release age equals entry year minus release year; exact 160k date anomaly exists.", "C1 release age + categorical entry year is preferred; C2/C3 retained as alternatives.", "partly", "yes"),
        ("Primary validation", facts["cv_evidence"], "Five-fold lead-artist + related-version grouped CV is feasible.", "yes technically", "yes for primary estimand"),
        ("All-collaborator separation", facts["collaboration_evidence"], "Report collaborator overlap; do not make strict all-collaborator separation the default.", "partly", "yes"),
        ("Temporal robustness", facts["temporal_evidence"], "Use both; Y_26 provides a much larger 2020 stress-test sample.", "yes technically", "yes for interpretation"),
        ("Combined time + grouped sensitivity", facts["combined_evidence"], "Retain as strict sensitivity only because training size contracts sharply.", "yes technically", "advisable"),
        ("550k versus 160k roles", "Cohorts differ in outcome composition and overlap audio values are close on average but not identical.", "Refit one common specification separately; keep source-specific models optional.", "partly", "yes"),
        ("Leakage blocklist", "Programmatic audit reports zero prohibited variables in frozen blocks.", "Keep frozen blocklist unchanged.", "yes technically", "no unless scope changes"),
        ("Prediction versus inference", "Grouped/time-aware prediction and artist-clustered inference answer different questions.", "Report them separately; do not interpret predictive CV as coefficient inference.", "yes technically", "advisable"),
    ]
    return pd.DataFrame(rows, columns=[
        "decision_issue", "current_evidence", "provisional_specification", "resolvable_from_data_alone",
        "supervisor_confirmation_advisable",
    ])


def bounded_note() -> str:
    return """# Bounded fixed-window outcomes: provisional modelling note

`Y_26` counts chart appearances from calendar weeks 5 through 26 and therefore has fixed support from 0 to 22. `Y_52` counts appearances from weeks 5 through 52 and has fixed support from 0 to 48. Re-entry is included, and each observed calendar week can contribute at most one appearance.

A Poisson model may be used later as a transparent diagnostic baseline. A negative-binomial model may better accommodate observed overdispersion. Both distributions are nevertheless unbounded and can assign probability or predictions above the outcome ceiling. Future comparison should therefore include at least one statistically appropriate bounded-count formulation, such as a binomial or beta-binomial treatment of observed chart weeks out of 22 or 48 opportunities. The dependence between adjacent weeks and re-entry structure must be acknowledged when judging that formulation.

The large number of zero outcomes does not by itself prove that a separate zero-generating mechanism exists. A hurdle or zero-inflated model should only be added if its substantive mechanism, diagnostics and out-of-sample performance justify it. No model or distribution was fitted in this task.
"""


def prediction_inference_note() -> str:
    return """# Prediction and inference are separate aims

Predictive evaluation will estimate out-of-sample performance using the frozen grouped and time-aware assignments. Its purpose is to assess how accurately a model predicts future chart presence for held-out tracks under a stated generalisation setting.

Inferential or interpretable regression has a different purpose: estimating and describing adjusted associations under explicit model assumptions. Because multiple tracks can share a first-credited artist, uncertainty estimates should account for repeated artists where appropriate, for example through artist-clustered standard errors or an explicitly grouped model. A good predictive validation score does not establish a causal or statistically significant feature effect, and clustered coefficient inference does not replace out-of-sample validation. Neither type of model was fitted in this task.
"""


def make_report(
    version_summary: pd.DataFrame,
    horizon: pd.DataFrame,
    increment: pd.DataFrame,
    horizon_facts: dict[str, float | int],
    collab: dict[str, int | float],
    temporal52: dict[str, int],
    temporal26: dict[str, int],
    leakage: pd.DataFrame,
) -> str:
    metrics = dict(zip(version_summary.metric, version_summary.value))
    get = lambda outcome, metric: horizon.loc[
        (horizon.section.eq("outcome_distribution")) & horizon.metric.eq(metric), outcome
    ].iloc[0]
    inc_lines = "\n".join(
        f"- {row.increment_category}: {int(row.tracks)} tracks ({row.percent_of_1014:.1f}%)."
        for _, row in increment.iterrows()
    )
    return f"""# Final provisional pre-model design report

## Scope and status

This report closes the targeted questions that can be answered before modelling. It uses the validated `pre_model_analysis_ready_v1` and authoritative `pre_model_remaining_eda_v2` outputs as immutable inputs. All specifications remain provisional where supervisor confirmation is appropriate. No predictive, count, survival, machine-learning, hierarchical, text or NLP model was fitted.

## 1. Analysis unit and related Spotify versions

The clean 52-week 550k cohort contains {int(metrics['affected_duplicate_groups'])} exact normalised-lyric duplicate groups involving {int(metrics['affected_tracks'])} track IDs. {int(metrics['groups_with_any_simultaneous_chart_presence'])} groups have at least one week in which multiple IDs chart simultaneously. The mutually exclusive chart-history classification contains {int(metrics['simultaneous_fully_coincident_groups'])} fully coincident groups, {int(metrics['sequential_non_overlapping_groups'])} sequential non-overlapping groups, {int(metrics['partially_overlapping_history_groups'])} partially overlapping group and {int(metrics['unresolved_groups'])} timing-unresolved groups. Metadata support suggests separate releases or recordings for {int(metrics['metadata_separate_release_or_recording_groups'])} groups, while the identity/version interpretation remains unresolved for {int(metrics['metadata_identity_unresolved_groups'])} group.

Every one of the {int(metrics['affected_duplicate_groups'])} duplicate groups is contained within a single first-credited-artist group. Spotify track ID remains defensible as the primary unit because the source chart and exact feature linkage are URI-specific. However, related IDs can fragment one underlying song or recording. They must remain in the same validation fold. The 70 supported groups cover {int(metrics['tracks_in_supported_sensitivity_groups'])} IDs; replacing each supported group by one union-presence unit would reduce the illustrative sensitivity cohort by {int(metrics['potential_unit_reduction_under_grouped_sensitivity'])} units, from 1,014 to {int(metrics['potential_grouped_sensitivity_cohort_size'])}. A later sensitivity analysis may take the union of member-ID chart presence and count each calendar week once. The primary dataset is unchanged.

## 2. What 52 weeks adds beyond 26 weeks

The same 1,014 tracks are used for both outcomes.

| Statistic | Y_26 | Y_52 |
|---|---:|---:|
| Mean | {get('Y_26','mean'):.2f} | {get('Y_52','mean'):.2f} |
| Median | {get('Y_26','median'):.1f} | {get('Y_52','median'):.1f} |
| Standard deviation | {get('Y_26','standard_deviation'):.2f} | {get('Y_52','standard_deviation'):.2f} |
| Variance | {get('Y_26','variance'):.2f} | {get('Y_52','variance'):.2f} |
| Zero proportion | {get('Y_26','zero_percent'):.1f}% | {get('Y_52','zero_percent'):.1f}% |
| P90 | {get('Y_26','p90'):.1f} | {get('Y_52','p90'):.1f} |
| P95 | {get('Y_26','p95'):.1f} | {get('Y_52','p95'):.1f} |
| P99 | {get('Y_26','p99'):.1f} | {get('Y_52','p99'):.1f} |
| Maximum possible | 22 | 48 |

Pearson correlation is {horizon_facts['pearson_correlation']:.3f} and Spearman correlation is {horizon_facts['spearman_correlation']:.3f}.

{inc_lines}

There are {horizon_facts['Y26_zero_Y52_positive_count']} tracks ({horizon_facts['Y26_zero_Y52_positive_percent']:.1f}%) with no chart presence in weeks 5-26 but positive presence in weeks 27-52. Y_26 reaches its ceiling for {horizon_facts['Y26_at_ceiling_count']} tracks ({horizon_facts['Y26_at_ceiling_percent']:.1f}%), compared with {horizon_facts['Y52_at_ceiling_count']} tracks ({horizon_facts['Y52_at_ceiling_percent']:.1f}%) at the Y_52 ceiling. Of all {horizon_facts['total_additional_weeks_27_52']} additional observed chart weeks, {horizon_facts['additional_weeks_in_reentry_blocks']} ({horizon_facts['percent_additional_weeks_in_reentry_blocks']:.1f}%) occur in unambiguously identified re-entry blocks.

The 52-week outcome adds meaningful information, but the gain is concentrated in the {horizon_facts['tracks_with_positive_increment']} tracks ({horizon_facts['tracks_with_positive_increment_percent']:.1f}%) with any additional presence rather than being spread evenly across the cohort. The horizon is therefore not selected automatically.

## 3. Bounded outcomes

Y_26 is bounded from 0 to 22 and Y_52 from 0 to 48. Poisson and negative-binomial models may later serve as diagnostic count baselines, but both are unbounded. At least one bounded-count formulation should be compared if statistically appropriate. Many zeros alone do not establish the need for a hurdle or zero-inflated model.

## 4. Frozen variable treatment and feature blocks

X will be categorical with four levels in the main interpretable specification. A linear 1-4 coding is reserved for sensitivity analysis. Musical key will be categorical, mode binary, entry year categorical in the main calendar candidate, artist ID grouping-only, and lyric word count the sole simple length control. Instrumentalness, release age and lyric-length measures are flagged for possible transformations fitted inside training folds only.

The frozen incremental blocks are F0 trivial, F1 X only, F2 song characteristics only, F3 X plus song characteristics, F4 F3 plus release age and entry-year control, F5a F4 plus simple lyric length without genre, and F5b F5a plus provenance-flagged broad genre. F6a and F6b reserve future fold-local lyric-content comparisons.

## 5. Future lyric-content comparison

The provisional primary option is regularised TF-IDF with vocabulary and IDF fitted inside each training fold. A small transparent lexical-summary block is the simpler sensitivity option. Fold-local TF-IDF plus truncated SVD is optional but less interpretable. Word count alone is not described as a lyrics-content model.

## 6. Calendar and release information

The preferred candidate is release age plus categorical chart-entry year. Alternatives are release age plus the rank-200 stream threshold at entry, or release year plus categorical entry year while omitting release age. Exact day-level release information is avoided in the common robustness specification because track `6VpQdig9pdpTSIFItgkJV5` is recorded as released two days after chart entry. The raw value is retained unchanged.

## 7. Artist and collaborator grouping

The cohort contains {collab['single_artist_tracks']} single-artist and {collab['multi_artist_tracks']} multi-artist tracks, {collab['distinct_first_artists']} distinct first-credited artists and {collab['distinct_all_artists']} distinct credited artists including collaborators. The all-collaborator graph has {collab['collaboration_components']} connected components; its largest component contains {collab['largest_component_tracks']} tracks ({collab['largest_component_percent']:.1f}% of the cohort). Complete all-collaborator separation can therefore create impractically large validation units.

The primary grouping key is the first-credited artist, strengthened by exact lyric/version groups. It guarantees that no first artist or supported related-version group crosses folds. It does not guarantee zero collaborator overlap: {collab['cross_fold_collaborators']} credited artists appear in more than one fold, affecting {collab['tracks_with_cross_fold_collaborator']} tracks. This limitation is reported explicitly.

## 8. Prospective validation

Primary provisional validation uses five approximately balanced lead-artist plus version-group folds. No first-credited artist, exact lyric group or supported related-version group crosses folds.

For Y_52, the temporal design contains {temporal52['train_2017_2018']} training tracks from 2017-2018, {temporal52['validation_2019']} validation tracks from 2019 and only {temporal52['test_2020']} 2020 stress-test tracks. For Y_26 the corresponding counts are {temporal26['train_2017_2018']}, {temporal26['validation_2019']} and {temporal26['test_2020']}; its larger 2020 sample makes it more useful for temporal generalisation checks.

The combined time-plus-grouped design remains a strict sensitivity. It retains only {temporal52['strict_combined_train']} Y_52 training tracks and {temporal26['strict_combined_train']} Y_26 training tracks after purging lead-artist/version overlap with later periods, so it is not the default.

## 9. Common 550k/160k robustness specification

The principal robustness question will refit the same specification separately in each linked cohort: categorical X, the twelve common audio descriptors with categorical key and binary mode, release age, and categorical chart-entry year. Lyrics, genre and the 160k-only explicit flag are excluded from this common comparison.

For the 638 overlapping tracks, average audio-feature agreement is generally close, but the sources are not identical. Notable maximum disagreements include tempo (77.642 BPM), key (7 codes), mode (1), danceability (0.25), liveness (0.155) and loudness (1.018 dB). Source-specific records must not be treated as interchangeable duplicates.

## 10. Leakage protection and methodological purpose

The programmatic audit contains {int(leakage.audit_pass.eq(0).sum())} failed leakage checks. Popularity/follower snapshots, future ranks and streams, eventual peak, L, Z, re-entry summaries, full-duration variables, first-spell duration, old eligibility flags and every feature derived after week 4 remain absent from all frozen feature blocks.

Predictive validation and interpretable regression remain distinct. Prediction will use held-out metrics under grouped/time-aware assignments. Later coefficient inference should account for repeated artists, for example through artist-clustered standard errors where appropriate. Neither was performed here.

## 11. Remaining supervisor decisions

Supervisor confirmation remains advisable for the primary horizon and outcome, the bounded model family comparison, genre use, lyric-content representation, final calendar specification, primary validation estimand, strict related-version sensitivity rule, and the final roles of the 550k and 160k cohorts. Technically, the project is now ready for a versioned provisional benchmark-modelling task after these frozen specifications are reviewed.
"""


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise RuntimeError("Refusing to overwrite existing final specification outputs.")
    guards_before = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}

    f550_52 = pd.read_csv(F550_52)
    f550_26 = pd.read_csv(F550_26)
    f160_52 = pd.read_csv(F160_52)
    f160_26 = pd.read_csv(F160_26)
    bridge = pd.read_csv(BRIDGE)
    panel = pd.read_csv(PANEL)
    dup_group = pd.read_csv(DUP_GROUP)
    dup_member = pd.read_csv(DUP_MEMBER)
    agreement = pd.read_csv(AUDIO_AGREEMENT)

    version_audit, version_summary = chart_version_audit(f550_52, panel, dup_group, dup_member, bridge)
    horizon_compare, increment, horizon_facts = y26_y52_comparison(f550_52, f550_26, panel)
    roles = variable_roles()
    blocks = feature_blocks()
    lyrics = lyric_plan()
    calendars = calendar_specs()
    collaboration, grouped_cv, collab_facts = collaboration_and_cv(f550_52, bridge, version_audit)
    temporal52, temporal52_facts = temporal_assignments(f550_52, bridge, version_audit, 52)
    temporal26, temporal26_facts = temporal_assignments(f550_26, bridge, version_audit, 26)
    common = common_spec(agreement)
    leakage = leakage_audit(blocks)

    facts = {
        "analysis_unit_evidence": f"{len(version_audit)} related lyric groups cover {int(version_audit.tracks_in_clean_52w_group.sum())} of 1,014 track IDs; chart records remain ID-specific.",
        "version_relation_evidence": "; ".join(f"{row.metric}={int(row.value)}" for _, row in version_summary.iloc[2:7].iterrows()),
        "horizon_evidence": f"{horizon_facts['tracks_with_positive_increment']} tracks gain weeks after week 26; {horizon_facts['Y26_zero_Y52_positive_count']} return after Y_26=0.",
        "cv_evidence": f"Five folds contain {grouped_cv.fold.value_counts().min()}-{grouped_cv.fold.value_counts().max()} tracks with zero first-artist or supported-version crossings.",
        "collaboration_evidence": f"Largest all-collaborator component contains {collab_facts['largest_component_tracks']} tracks; {collab_facts['cross_fold_collaborators']} credited artists cross lead-grouped folds.",
        "temporal_evidence": f"2020 test N is {temporal52_facts['test_2020']} for Y_52 and {temporal26_facts['test_2020']} for Y_26.",
        "combined_evidence": f"Strict combined training retains {temporal52_facts['strict_combined_train']} Y_52 and {temporal26_facts['strict_combined_train']} Y_26 tracks.",
    }
    decisions = decision_status(facts)

    # Internal validation before any output is created.
    assert len(f550_52) == 1014 and f550_52.spotify_track_id.is_unique
    assert len(f550_26) == 1220 and f550_26.spotify_track_id.is_unique
    assert set(f550_52.spotify_track_id).issubset(set(f550_26.spotify_track_id))
    assert len(version_audit) == 71
    assert int(version_summary.loc[version_summary.metric.eq("affected_tracks"), "value"].iloc[0]) == 145
    assert version_audit.single_first_credited_artist_group.eq(1).all()
    assert grouped_cv.first_artist_crosses_folds.eq(0).all()
    assert grouped_cv.exact_lyric_group_crosses_folds.eq(0).all()
    assert grouped_cv.related_version_group_crosses_folds.eq(0).all()
    assert grouped_cv.fold.value_counts().max() - grouped_cv.fold.value_counts().min() <= 1
    assert leakage.audit_pass.eq(1).all()
    assert temporal52.temporal_partition.value_counts().sum() == 1014
    assert temporal26.temporal_partition.value_counts().sum() == 1220
    assert f160_52.spotify_track_id.is_unique and f160_26.spotify_track_id.is_unique
    assert horizon_facts["Y26_at_ceiling_count"] == int((f550_26.set_index("spotify_track_id").loc[f550_52.spotify_track_id, "Y_26"] == 22).sum())

    report_text = make_report(
        version_summary, horizon_compare, increment, horizon_facts, collab_facts,
        temporal52_facts, temporal26_facts, leakage,
    )
    guards_after_compute = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}
    if guards_before != guards_after_compute:
        raise RuntimeError("Immutable inputs changed during computation.")

    OUT.mkdir(parents=True)
    outputs = {
        "analysis_unit_version_audit.csv": version_audit,
        "analysis_unit_version_summary.csv": version_summary,
        "Y26_Y52_same_tracks_comparison.csv": horizon_compare,
        "Y26_Y52_increment_distribution.csv": increment,
        "provisional_variable_roles.csv": roles,
        "provisional_feature_blocks.csv": blocks,
        "provisional_calendar_specifications.csv": calendars,
        "provisional_lyric_feature_plan.csv": lyrics,
        "collaboration_grouping_diagnostic.csv": collaboration,
        "provisional_grouped_cv_assignments.csv": grouped_cv,
        "provisional_temporal_validation_52w.csv": temporal52,
        "provisional_temporal_validation_26w.csv": temporal26,
        "validation_leakage_audit.csv": leakage,
        "common_550k_160k_predictor_specification.csv": common,
        "final_pre_model_decision_status.csv": decisions,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, lineterminator="\n")
    (OUT / "bounded_outcome_model_note.md").write_text(bounded_note(), encoding="utf-8")
    (OUT / "prediction_vs_inference_note.md").write_text(prediction_inference_note(), encoding="utf-8")
    REPORT.write_text(report_text, encoding="utf-8")

    guards_after_write = {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}
    if guards_before != guards_after_write:
        raise RuntimeError("Immutable inputs changed while outputs were written.")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "final_pre_model_specification_v1",
        "provisional_pending_supervisor_confirmation": True,
        "models_fitted": False,
        "nlp_or_text_vectorisation_run": False,
        "immutable_input_digests_before": guards_before,
        "immutable_input_digests_after": guards_after_write,
        "input_files": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in [F550_52, F550_26, F160_52, F160_26, BRIDGE, PANEL, DUP_GROUP, DUP_MEMBER, AUDIO_AGREEMENT, OLD_ALLOW_BLOCK]},
        "created_files": sorted([*outputs.keys(), "bounded_outcome_model_note.md", "prediction_vs_inference_note.md"]),
        "report": str(REPORT.relative_to(ROOT)),
        "validation_assertions_passed": True,
    }
    (OUT / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(OUT), "report": str(REPORT),
        "version_groups": len(version_audit), "affected_tracks": int(version_audit.tracks_in_clean_52w_group.sum()),
        "Y26_Y52_pearson": horizon_facts["pearson_correlation"],
        "Y26_Y52_spearman": horizon_facts["spearman_correlation"],
        "all_checks_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
