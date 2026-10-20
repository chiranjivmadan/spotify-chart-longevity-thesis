#!/usr/bin/env python3
"""Build a gap-aware weekly Spotify Global Top 200 survival dataset.

This script does target construction and exact-ID matching only.  It does not
fit a model and does not create a discounted-stream target.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHART_PATH = ROOT / "data/raw/charts/extracted/bartomiejczyewski_top200_global/Spotify Top 200 Global (2017-2021).csv"
SONGS_550K_PATH = ROOT / "data/raw/spotify_550k/songs.csv"
SONGS_160K_PATH = ROOT / "data/raw/spotify_160k/data.csv"
OUTPUT = ROOT / "data/processed/chart_survival"
MATCHING_OUTPUT = ROOT / "data/processed/matching"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def spotify_id(link: str) -> str:
    """Extract a standard Spotify track ID from an open.spotify.com track URL."""
    match = re.search(r"(?:open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]{22})", link or "")
    return match.group(1) if match else ""


def csv_write(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def optional(value):
    return "" if value is None else value


def read_reference(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    blank_ids = 0
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            track_id = row.get("id", "").strip()
            if not track_id:
                blank_ids += 1
                continue
            counts[track_id] += 1
            rows_by_id.setdefault(track_id, row)
    return rows_by_id, {
        "rows": rows,
        "unique_ids": len(rows_by_id),
        "blank_ids": blank_ids,
        "duplicate_id_rows_beyond_first": sum(value - 1 for value in counts.values() if value > 1),
    }


def contiguous_segments(states: list[dict]) -> list[dict]:
    """Return observed-present segments, splitting at confirmed absence or source gaps."""
    segments: list[dict] = []
    current: list[dict] = []
    entry_type = "first_observed_entry"
    prior_state = None
    for record in states:
        state = record["chart_state"]
        if state == "present":
            if not current:
                if prior_state == "confirmed_absent":
                    entry_type = "confirmed_reentry"
                elif prior_state == "unknown_source_gap":
                    entry_type = "unknown_gap_continuation_or_entry"
                else:
                    entry_type = "first_observed_entry" if not segments else "observed_reappearance"
                current = [record]
            else:
                current.append(record)
        elif current:
            boundary = "confirmed_exit" if state == "confirmed_absent" else "source_gap_unknown"
            segments.append({"records": current, "entry_type": entry_type, "boundary_type": boundary, "boundary_date": record["chart_week"]})
            current = []
        prior_state = state
    if current:
        segments.append({"records": current, "entry_type": entry_type, "boundary_type": "right_censored_end_of_observation", "boundary_date": ""})
    return segments


def mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def population_sd(values: list[int]) -> float | None:
    if len(values) < 2:
        return None
    centre = sum(values) / len(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / len(values))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MATCHING_OUTPUT.mkdir(parents=True, exist_ok=True)

    observations: dict[tuple[str, date], dict[str, str]] = {}
    dates_by_track: defaultdict[str, list[date]] = defaultdict(list)
    track_metadata: dict[str, dict[str, str]] = {}
    raw_link_counts: Counter[str] = Counter()
    extracted_id_counts: Counter[str] = Counter()
    missing_or_invalid_links: list[dict[str, str]] = []
    duplicate_track_weeks: list[dict[str, str]] = []
    observed_dates: set[date] = set()

    with CHART_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row_number, row in enumerate(reader, start=2):
            week = parse_date(row["Week"])
            link = row["Link"].strip()
            raw_link_counts[link] += 1
            track_id = spotify_id(link)
            if not track_id:
                missing_or_invalid_links.append({"source_row_number": source_row_number, "reason": "missing_or_invalid_spotify_track_url", **row})
                continue
            extracted_id_counts[track_id] += 1
            key = (track_id, week)
            if key in observations:
                duplicate_track_weeks.append({"source_row_number": source_row_number, "reason": "duplicate_track_week", "spotify_track_id": track_id, **row})
                continue
            observation = {
                "spotify_track_id": track_id,
                "chart_week": week.isoformat(),
                "rank": int(row["Rank"]),
                "weekly_streams": int(row["Streams"]),
                "track": row["Track"],
                "artist": row["Artist"],
                "spotify_track_url": link,
            }
            observations[key] = observation
            dates_by_track[track_id].append(week)
            track_metadata.setdefault(track_id, observation)
            observed_dates.add(week)

    ordered_observed_dates = sorted(observed_dates)
    first_observed_date, final_observed_date = ordered_observed_dates[0], ordered_observed_dates[-1]
    all_calendar_dates: list[date] = []
    current_date = first_observed_date
    while current_date <= final_observed_date:
        all_calendar_dates.append(current_date)
        current_date += timedelta(days=7)
    missing_source_dates = sorted(set(all_calendar_dates) - observed_dates)
    calendar_index = {week: index for index, week in enumerate(all_calendar_dates)}
    track_ids = sorted(track_metadata)

    reference_550k, reference_550k_stats = read_reference(SONGS_550K_PATH)
    reference_160k, reference_160k_stats = read_reference(SONGS_160K_PATH)
    chart_ids_set = set(track_ids)
    matched_550k = sorted(chart_ids_set & set(reference_550k))
    matched_160k = sorted(chart_ids_set & set(reference_160k))
    unmatched_550k = sorted(chart_ids_set - set(reference_550k))
    unmatched_160k = sorted(chart_ids_set - set(reference_160k))

    unmatched_fields = ["spotify_track_id", "reason", "track", "artist", "spotify_track_url", "first_observed_chart_entry"]
    for reference_name, unmatched_ids in (("550k", unmatched_550k), ("160k", unmatched_160k)):
        rows = []
        for track_id in unmatched_ids:
            meta = track_metadata[track_id]
            first_entry = min(dates_by_track[track_id])
            rows.append({
                "spotify_track_id": track_id,
                "reason": "spotify_track_id_not_present_in_reference_dataset",
                "track": meta["track"],
                "artist": meta["artist"],
                "spotify_track_url": meta["spotify_track_url"],
                "first_observed_chart_entry": first_entry.isoformat(),
            })
        csv_write(MATCHING_OUTPUT / f"chart_tracks_unmatched_{reference_name}_exact_id.csv", rows, unmatched_fields)

    # Build one record per track-week from first observed entry to the final chart week.
    panel_rows: list[dict] = []
    spell_rows: list[dict] = []
    first_spell_rows: list[dict] = []
    reentry_rows: list[dict] = []
    exclusion_rows: list[dict] = []
    source_gap_boundary_affected: set[str] = set()
    first_spell_gap_affected: set[str] = set()
    source_gap_entry_ambiguous: set[str] = set()
    left_truncated_tracks: set[str] = set()
    event_tracks: set[str] = set()
    censored_tracks: set[str] = set()
    outcome_unknown_tracks: set[str] = set()
    confirmed_reentry_count = 0
    tracks_with_confirmed_reentry: set[str] = set()
    unknown_gap_segment_count = 0
    tracks_with_unknown_gap_segment: set[str] = set()

    base_panel_fields = [
        "spotify_track_id", "track", "artist", "spotify_track_url", "chart_week", "week_index_since_first_entry",
        "source_week_status", "chart_state", "rank", "weekly_streams", "is_first_observed_entry",
    ]
    spell_fields = [
        "spotify_track_id", "track", "artist", "spell_number", "entry_type", "spell_start_date", "spell_end_date",
        "observed_chart_weeks", "debut_rank", "best_rank", "last_rank", "weekly_streams_sum", "boundary_type", "boundary_date",
        "match_550k_exact_id", "match_160k_exact_id",
    ]
    first_spell_fields = [
        "spotify_track_id", "track", "artist", "spotify_track_url", "first_observed_chart_entry", "last_known_active_date",
        "first_confirmed_exit_date", "first_spell_observed_weeks", "event_indicator", "outcome_status", "right_censored",
        "left_truncated", "left_truncation_reason", "entry_ambiguous_due_source_gap", "first_spell_source_gap_affected",
        "survival_outcome_eligible", "match_550k_exact_id", "match_160k_exact_id",
    ]
    early_fields = []
    for window in (2, 4):
        early_fields += [
            f"early_{window}w_landmark_date", f"early_{window}w_window_complete", f"early_{window}w_landmark_active",
            f"early_{window}w_continuous_first_spell_through_landmark", f"early_{window}w_prediction_eligible",
            f"early_{window}w_chart_weeks", f"early_{window}w_unknown_source_weeks",
            f"early_{window}w_debut_rank", f"early_{window}w_best_rank", f"early_{window}w_last_observed_rank",
            f"early_{window}w_rank_change_last_minus_debut", f"early_{window}w_rank_volatility_sd",
            f"early_{window}w_streams_sum", f"early_{window}w_streams_mean",
        ]
    first_spell_fields += early_fields

    for track_id in track_ids:
        meta = track_metadata[track_id]
        present_dates = sorted(dates_by_track[track_id])
        entry_date = present_dates[0]
        start_index = calendar_index[entry_date]
        states: list[dict] = []
        for week in all_calendar_dates[start_index:]:
            observation = observations.get((track_id, week))
            if week in missing_source_dates:
                source_status, chart_state = "source_week_missing", "unknown_source_gap"
                rank, streams = None, None
            elif observation:
                source_status, chart_state = "source_week_observed", "present"
                rank, streams = observation["rank"], observation["weekly_streams"]
            else:
                source_status, chart_state = "source_week_observed", "confirmed_absent"
                rank, streams = None, None
            state_record = {
                "spotify_track_id": track_id,
                "track": meta["track"],
                "artist": meta["artist"],
                "spotify_track_url": meta["spotify_track_url"],
                "chart_week": week.isoformat(),
                "week_index_since_first_entry": calendar_index[week] - start_index,
                "source_week_status": source_status,
                "chart_state": chart_state,
                "rank": rank,
                "weekly_streams": streams,
                "is_first_observed_entry": int(week == entry_date),
            }
            states.append(state_record)
            panel_rows.append({field: optional(state_record[field]) for field in base_panel_fields})

        # A boundary presence makes continuity or the timing of exit/re-entry unknowable across that gap.
        for missing_week in missing_source_dates:
            missing_index = calendar_index[missing_week]
            if missing_index <= start_index:
                continue
            previous_week = all_calendar_dates[missing_index - 1]
            next_week = all_calendar_dates[missing_index + 1]
            if (track_id, previous_week) in observations or (track_id, next_week) in observations:
                source_gap_boundary_affected.add(track_id)

        if entry_date == first_observed_date:
            left_truncated_tracks.add(track_id)
            left_truncation_reason = "present_in_first_available_chart_week"
        elif all_calendar_dates[calendar_index[entry_date] - 1] in missing_source_dates:
            left_truncated_tracks.add(track_id)
            source_gap_entry_ambiguous.add(track_id)
            left_truncation_reason = "first_observed_entry_immediately_after_missing_source_week"
        else:
            left_truncation_reason = ""

        segments = contiguous_segments(states)
        for spell_number, segment in enumerate(segments, start=1):
            records = segment["records"]
            ranks = [record["rank"] for record in records]
            streams = [record["weekly_streams"] for record in records]
            spell_rows.append({
                "spotify_track_id": track_id,
                "track": meta["track"],
                "artist": meta["artist"],
                "spell_number": spell_number,
                "entry_type": segment["entry_type"],
                "spell_start_date": records[0]["chart_week"],
                "spell_end_date": records[-1]["chart_week"],
                "observed_chart_weeks": len(records),
                "debut_rank": ranks[0],
                "best_rank": min(ranks),
                "last_rank": ranks[-1],
                "weekly_streams_sum": sum(streams),
                "boundary_type": segment["boundary_type"],
                "boundary_date": segment["boundary_date"],
                "match_550k_exact_id": int(track_id in reference_550k),
                "match_160k_exact_id": int(track_id in reference_160k),
            })
            if segment["entry_type"] == "confirmed_reentry":
                confirmed_reentry_count += 1
                tracks_with_confirmed_reentry.add(track_id)
            if segment["entry_type"] == "unknown_gap_continuation_or_entry":
                unknown_gap_segment_count += 1
                tracks_with_unknown_gap_segment.add(track_id)

        first_segment = segments[0]
        first_records = first_segment["records"]
        boundary = first_segment["boundary_type"]
        if boundary == "confirmed_exit":
            event_indicator, outcome_status, right_censored, exit_date = 1, "confirmed_exit", 0, first_segment["boundary_date"]
            event_tracks.add(track_id)
        elif boundary == "right_censored_end_of_observation":
            event_indicator, outcome_status, right_censored, exit_date = 0, "right_censored", 1, ""
            censored_tracks.add(track_id)
        else:
            event_indicator, outcome_status, right_censored, exit_date = "", "source_gap_unknown", "", ""
            outcome_unknown_tracks.add(track_id)
            first_spell_gap_affected.add(track_id)

        row = {
            "spotify_track_id": track_id,
            "track": meta["track"],
            "artist": meta["artist"],
            "spotify_track_url": meta["spotify_track_url"],
            "first_observed_chart_entry": entry_date.isoformat(),
            "last_known_active_date": first_records[-1]["chart_week"],
            "first_confirmed_exit_date": exit_date,
            "first_spell_observed_weeks": len(first_records),
            "event_indicator": event_indicator,
            "outcome_status": outcome_status,
            "right_censored": right_censored,
            "left_truncated": int(track_id in left_truncated_tracks),
            "left_truncation_reason": left_truncation_reason,
            "entry_ambiguous_due_source_gap": int(track_id in source_gap_entry_ambiguous),
            "first_spell_source_gap_affected": int(track_id in first_spell_gap_affected),
            "survival_outcome_eligible": int(track_id not in left_truncated_tracks and outcome_status in {"confirmed_exit", "right_censored"}),
            "match_550k_exact_id": int(track_id in reference_550k),
            "match_160k_exact_id": int(track_id in reference_160k),
        }

        entry_index = calendar_index[entry_date]
        first_spell_dates = {parse_date(record["chart_week"]) for record in first_records}
        for window in (2, 4):
            window_dates = all_calendar_dates[entry_index: entry_index + window]
            horizon_available = len(window_dates) == window
            records_until_landmark = [record for record in states if parse_date(record["chart_week"]) in set(window_dates)]
            known_window = horizon_available and all(record["chart_state"] != "unknown_source_gap" for record in records_until_landmark)
            # Use the first spell only. A later re-entry within the early window must not
            # supply post-exit information to a first-spell survival prediction task.
            present_records = [
                record for record in records_until_landmark
                if record["chart_state"] == "present" and parse_date(record["chart_week"]) in first_spell_dates
            ]
            ranks = [record["rank"] for record in present_records]
            streams = [record["weekly_streams"] for record in present_records]
            landmark_record = next((record for record in records_until_landmark if parse_date(record["chart_week"]) == window_dates[-1]), None) if horizon_available else None
            landmark_active = int(landmark_record is not None and landmark_record["chart_state"] == "present")
            continuous_first_spell = int(known_window and len(present_records) == window)
            rank_change = ranks[-1] - ranks[0] if len(ranks) >= 2 else None
            row.update({
                f"early_{window}w_landmark_date": window_dates[-1].isoformat() if horizon_available else "",
                f"early_{window}w_window_complete": int(known_window),
                f"early_{window}w_landmark_active": landmark_active if horizon_available else "",
                f"early_{window}w_continuous_first_spell_through_landmark": continuous_first_spell,
                f"early_{window}w_prediction_eligible": continuous_first_spell,
                f"early_{window}w_chart_weeks": len(present_records),
                f"early_{window}w_unknown_source_weeks": sum(record["chart_state"] == "unknown_source_gap" for record in records_until_landmark),
                f"early_{window}w_debut_rank": optional(ranks[0] if ranks else None),
                f"early_{window}w_best_rank": optional(min(ranks) if ranks else None),
                f"early_{window}w_last_observed_rank": optional(ranks[-1] if ranks else None),
                f"early_{window}w_rank_change_last_minus_debut": optional(rank_change),
                f"early_{window}w_rank_volatility_sd": optional(population_sd(ranks)),
                f"early_{window}w_streams_sum": optional(sum(streams) if streams else None),
                f"early_{window}w_streams_mean": optional(mean(streams)),
            })
        first_spell_rows.append(row)

        reentry_rows.append({
            "spotify_track_id": track_id,
            "track": meta["track"],
            "artist": meta["artist"],
            "observed_present_segments": len(segments),
            "confirmed_reentry_count": sum(segment["entry_type"] == "confirmed_reentry" for segment in segments),
            "has_confirmed_reentry": int(any(segment["entry_type"] == "confirmed_reentry" for segment in segments)),
            "unknown_gap_segment_count": sum(segment["entry_type"] == "unknown_gap_continuation_or_entry" for segment in segments),
            "source_gap_boundary_affected": int(track_id in source_gap_boundary_affected),
        })

        if track_id not in reference_550k:
            exclusion_rows.append({"spotify_track_id": track_id, "scope": "550k_feature_matching", "reason": "spotify_track_id_not_present_in_550k_dataset"})
        if track_id in left_truncated_tracks:
            exclusion_rows.append({"spotify_track_id": track_id, "scope": "first_entry_duration_analysis", "reason": left_truncation_reason})
        if track_id in first_spell_gap_affected:
            exclusion_rows.append({"spotify_track_id": track_id, "scope": "first_spell_survival_outcome", "reason": "source_gap_before_exit_or_censoring_resolution"})

    # Keep compact, leakage-safe baseline data for exact 550k matches. Lyrics stay in the immutable raw source.
    feature_fields = [
        "spotify_track_id", "chart_track", "chart_artist", "lyrics_available", "lyrics_character_count",
        "name", "album_name", "artists", "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "year", "genre",
        "popularity", "total_artist_followers", "avg_artist_popularity", "artist_ids", "niche_genres",
    ]
    feature_rows = []
    for track_id in matched_550k:
        source = reference_550k[track_id]
        feature_rows.append({
            "spotify_track_id": track_id,
            "chart_track": track_metadata[track_id]["track"],
            "chart_artist": track_metadata[track_id]["artist"],
            "lyrics_available": int(bool(source.get("lyrics", "").strip())),
            "lyrics_character_count": len(source.get("lyrics", "")),
            **{field: source.get(field, "") for field in feature_fields if field not in {"spotify_track_id", "chart_track", "chart_artist", "lyrics_available", "lyrics_character_count"}},
        })

    csv_write(OUTPUT / "song_week_panel.csv", panel_rows, base_panel_fields)
    csv_write(OUTPUT / "chart_spells.csv", spell_rows, spell_fields)
    csv_write(OUTPUT / "first_spell_survival.csv", first_spell_rows, first_spell_fields)
    csv_write(OUTPUT / "reentry_summary.csv", reentry_rows, list(reentry_rows[0]) if reentry_rows else [])
    csv_write(OUTPUT / "exclusions_table.csv", exclusion_rows, ["spotify_track_id", "scope", "reason"])
    csv_write(OUTPUT / "matched_550k_baseline_features.csv", feature_rows, feature_fields)
    csv_write(OUTPUT / "invalid_or_missing_chart_track_urls.csv", missing_or_invalid_links, list(missing_or_invalid_links[0]) if missing_or_invalid_links else ["source_row_number", "reason"])
    csv_write(OUTPUT / "duplicate_chart_track_week_records.csv", duplicate_track_weeks, list(duplicate_track_weeks[0]) if duplicate_track_weeks else ["source_row_number", "reason"])

    eligible_rows = [row for row in first_spell_rows if row["survival_outcome_eligible"] == 1 and row["match_550k_exact_id"] == 1]
    csv_write(OUTPUT / "first_spell_survival_matched_550k_eligible.csv", eligible_rows, first_spell_fields)

    quality = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chart_source": str(CHART_PATH),
        "chart_observation_rows": len(observations),
        "unique_chart_tracks": len(track_ids),
        "weekly_chart_dates_observed": len(observed_dates),
        "minimum_observed_week": first_observed_date.isoformat(),
        "maximum_observed_week": final_observed_date.isoformat(),
        "missing_source_weeks": [week.isoformat() for week in missing_source_dates],
        "chart_fields": {"weekly_dates": "Week", "weekly_rank": "Rank", "weekly_streams": "Streams", "spotify_track_url": "Link", "extracted_spotify_track_id": "spotify_track_id"},
        "chart_id_quality": {
            "missing_or_invalid_track_urls": len(missing_or_invalid_links),
            "duplicate_chart_track_week_records": len(duplicate_track_weeks),
            "duplicate_unique_track_ids": 0,
            "raw_url_occurrences_beyond_first": sum(count - 1 for count in raw_link_counts.values() if count > 1),
            "track_id_observations_beyond_first": sum(count - 1 for count in extracted_id_counts.values() if count > 1),
        },
        "exact_id_matching": {
            "550k": {
                "reference": reference_550k_stats,
                "exact_matches": len(matched_550k),
                "unmatched_chart_tracks": len(unmatched_550k),
                "exact_match_rate": len(matched_550k) / len(track_ids),
                "final_usable_exact_id_sample": len(matched_550k),
            },
            "160k": {
                "reference": reference_160k_stats,
                "exact_matches": len(matched_160k),
                "unmatched_chart_tracks": len(unmatched_160k),
                "exact_match_rate": len(matched_160k) / len(track_ids),
            },
            "matched_in_both_references": len(set(matched_550k) & set(matched_160k)),
            "matching_method": "exact extracted Spotify track ID only; no fuzzy matching",
        },
        "survival_construction": {
            "all_chart_tracks": len(track_ids),
            "first_spells_with_confirmed_exit_event": len(event_tracks),
            "first_spells_right_censored": len(censored_tracks),
            "first_spells_outcome_unknown_due_source_gap": len(outcome_unknown_tracks),
            "left_truncated_tracks": len(left_truncated_tracks),
            "left_truncated_at_initial_dataset_boundary": sum(row["left_truncation_reason"] == "present_in_first_available_chart_week" for row in first_spell_rows),
            "left_truncated_due_source_gap_entry_ambiguity": len(source_gap_entry_ambiguous),
            "survival_outcome_eligible_before_feature_matching": sum(row["survival_outcome_eligible"] == 1 for row in first_spell_rows),
            "survival_outcome_eligible_with_550k_exact_match": len(eligible_rows),
            "source_gap_boundary_affected_songs": len(source_gap_boundary_affected),
            "first_spell_source_gap_affected_songs": len(first_spell_gap_affected),
            "definition_note": "Event=first confirmed observed absence after the first contiguous observed chart spell. Right censoring=present in the final observed source week. Source gaps give an unknown outcome, not an exit or re-entry.",
        },
        "reentry": {
            "confirmed_reentry_spells": confirmed_reentry_count,
            "tracks_with_confirmed_reentry": len(tracks_with_confirmed_reentry),
            "segments_after_source_gap_with_unknown_continuity": unknown_gap_segment_count,
            "tracks_with_unknown_gap_segments": len(tracks_with_unknown_gap_segment),
            "definition_note": "A confirmed re-entry begins only after at least one observed source week of confirmed absence. A source-wide missing week never establishes an exit or re-entry.",
        },
        "early_prediction_preparation": {
            "windows_weeks": [2, 4],
            "two_week_prediction_eligible": sum(row["early_2w_prediction_eligible"] == 1 for row in first_spell_rows),
            "four_week_prediction_eligible": sum(row["early_4w_prediction_eligible"] == 1 for row in first_spell_rows),
            "leakage_rule": "All early-window fields use only first-spell chart weeks from entry through the stated landmark; prediction eligibility requires continuous first-spell presence. No later rank, stream, peak, total duration, re-entry, or outcome field is used as an early predictor.",
        },
    }
    (OUTPUT / "survival_construction_audit.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    dictionary_rows = [
        {"file": "song_week_panel.csv", "field": "chart_state", "definition": "present, confirmed_absent, or unknown_source_gap; unknown source weeks are not song exits."},
        {"file": "song_week_panel.csv", "field": "rank / weekly_streams", "definition": "Observed weekly Top-200 rank and stream count; blank for non-present and source-gap rows."},
        {"file": "chart_spells.csv", "field": "boundary_type", "definition": "confirmed_exit, source_gap_unknown, or right_censored_end_of_observation."},
        {"file": "first_spell_survival.csv", "field": "event_indicator", "definition": "1 first confirmed exit; 0 right-censored; blank where a source gap prevents outcome resolution."},
        {"file": "first_spell_survival.csv", "field": "left_truncated", "definition": "1 when the first observed entry is at the dataset start or immediately after a missing source week."},
        {"file": "first_spell_survival.csv", "field": "survival_outcome_eligible", "definition": "1 only when entry is not left-truncated and the first-spell event or censoring status is known."},
        {"file": "first_spell_survival.csv", "field": "early_2w_* / early_4w_*", "definition": "Leakage-safe fields available by the two- or four-week landmark only. Prediction eligibility requires continuous first-spell presence through an observed landmark; post-exit re-entry information is excluded."},
        {"file": "reentry_summary.csv", "field": "confirmed_reentry_count", "definition": "Observed return after a confirmed absence; source gaps are not counted as re-entry."},
        {"file": "matched_550k_baseline_features.csv", "field": "lyrics_available / lyrics_character_count", "definition": "Availability indicators only. Raw lyrics remain in the original 550k source and no lyric feature was constructed here."},
    ]
    csv_write(OUTPUT / "data_dictionary.csv", dictionary_rows, ["file", "field", "definition"])


if __name__ == "__main__":
    main()
