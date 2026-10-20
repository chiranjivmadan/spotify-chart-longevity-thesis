#!/usr/bin/env python3
"""Build separate pre-model diagnostics requested in supervisor feedback.

The script deliberately preserves existing chart-survival outputs. It writes a
new fixed-window diagnostic version and freezes copies of the historical
first-spell artefacts with checksums. It fits no statistical or predictive
model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "data/raw/charts/extracted/bartomiejczyewski_top200_global/Spotify Top 200 Global (2017-2021).csv"
FIRST = ROOT / "data/processed/chart_survival/first_spell_survival.csv"
FEATURES = ROOT / "data/processed/chart_survival/matched_550k_baseline_features.csv"
OUT = ROOT / "data/processed/fixed_window_count_v1"
DOC = ROOT / "docs/professor_feedback_pre_model_diagnostics.md"
PLOT_DIR = ROOT / "outputs/fixed_window_diagnostics"
FROZEN = ROOT / "data/frozen/first_spell_survival_2026-07-30"
HORIZONS = (26, 52, 78, 104)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def track_id(link: str) -> str:
    match = re.search(r"(?:open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]{22})", link or "")
    return match.group(1) if match else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    return {
        "n": len(values), "min": min(values), "p25": quantile(values, 0.25), "median": quantile(values, 0.5),
        "mean": sum(values) / len(values), "p75": quantile(values, 0.75), "max": max(values),
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((value - mx) ** 2 for value in x)
    sy = sum((value - my) ** 2 for value in y)
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def count_blocks(states: list[int]) -> int:
    blocks = 0
    prior = 0
    for state in states:
        if state and not prior:
            blocks += 1
        prior = state
    return blocks


def write_rank200_svg(rows: list[dict[str, object]], destination: Path) -> None:
    """Write a compact self-contained line plot without an external plotting dependency."""
    width, height = 1100, 520
    left, right, top, bottom = 86, 34, 52, 72
    values = [int(row["weekly_streams_at_rank_200"]) for row in rows]
    lo, hi = min(values), max(values)
    pad = max(1, (hi - lo) * 0.06)
    y0, y1 = lo - pad, hi + pad
    def x(index: int) -> float:
        return left + (width - left - right) * index / max(1, len(rows) - 1)
    def y(value: float) -> float:
        return top + (height - top - bottom) * (y1 - value) / (y1 - y0)
    points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
    year_first_index = {}
    for index, row in enumerate(rows):
        year_first_index.setdefault(str(row["chart_week"])[:4], index)
    grid = []
    for frac in range(5):
        value = y0 + (y1 - y0) * frac / 4
        yy = y(value)
        grid.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d1d5db" stroke-width="1"/>')
        grid.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{value:,.0f}</text>')
    years = []
    for year, index in year_first_index.items():
        xx = x(index)
        years.append(f'<line x1="{xx:.1f}" y1="{height-bottom}" x2="{xx:.1f}" y2="{height-bottom+5}" stroke="#374151"/>')
        years.append(f'<text x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="12" fill="#374151">{year}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Weekly streams at rank 200, Spotify Global Top 200</title>
<desc id="desc">A weekly line chart from 2017 to 2021 of streams received by the song at rank 200.</desc>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="27" font-family="Arial, sans-serif" font-size="18" font-weight="600" fill="#111827">Weekly streams at rank 200 — Spotify Global Top 200</text>
<text x="{left}" y="46" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">Observed weekly threshold, 2017-01-06 to 2021-04-16</text>
{''.join(grid)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#374151" stroke-width="1.2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#374151" stroke-width="1.2"/>
{''.join(years)}
<polyline fill="none" stroke="#2563eb" stroke-width="1.6" points="{points}"/>
<text x="{left}" y="{height-12}" font-family="Arial, sans-serif" font-size="12" fill="#374151">Chart week</text>
<text x="16" y="{top}" font-family="Arial, sans-serif" font-size="12" fill="#374151" transform="rotate(-90 16,{top})">Weekly streams</text>
</svg>'''
    destination.write_text(svg, encoding="utf-8")


def freeze_history() -> list[dict[str, object]]:
    FROZEN.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "data/processed/chart_survival"
    files = [
        "first_spell_survival.csv", "first_spell_survival_matched_550k_eligible.csv", "chart_spells.csv",
        "reentry_summary.csv", "exclusions_table.csv", "data_dictionary.csv", "survival_construction_audit.json",
    ]
    manifest = []
    for name in files:
        source, dest = source_dir / name, FROZEN / name
        source_hash = sha256(source)
        if dest.exists():
            if sha256(dest) != source_hash:
                raise RuntimeError(f"Frozen destination differs from source and will not be overwritten: {dest}")
        else:
            shutil.copy2(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "frozen_copy": str(dest.relative_to(ROOT)), "bytes": source.stat().st_size, "sha256": source_hash})
    manifest_path = FROZEN / "freeze_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("files") != manifest:
            raise RuntimeError("Existing freeze manifest differs; no overwrite was performed.")
    else:
        manifest_path.write_text(json.dumps({"frozen_utc": datetime.now(timezone.utc).isoformat(), "purpose": "Historical first-spell survival artefacts preserved before fixed-window diagnostics.", "files": manifest}, indent=2), encoding="utf-8")
    return manifest


def prepare_output_dir() -> None:
    if OUT.exists():
        raise RuntimeError(f"Versioned output directory already exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    freeze_manifest = freeze_history()
    prepare_output_dir()

    first_rows = {row["spotify_track_id"]: row for row in read_csv(FIRST)}
    feature_rows = {row["spotify_track_id"]: row for row in read_csv(FEATURES)}
    if len(first_rows) != 4247:
        raise RuntimeError("Unexpected number of first-spell rows; construction stopped.")

    present_by_track: defaultdict[str, set[date]] = defaultdict(set)
    metadata: dict[str, dict[str, str]] = {}
    observed_weeks: set[date] = set()
    rank200: list[dict[str, object]] = []
    invalid_ids = 0
    with CHART.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            week = parse_date(row["Week"])
            ident = track_id(row["Link"])
            if not ident:
                invalid_ids += 1
                continue
            present_by_track[ident].add(week)
            metadata.setdefault(ident, {"chart_track": row["Track"], "chart_artist": row["Artist"], "spotify_track_url": row["Link"]})
            observed_weeks.add(week)
            if int(row["Rank"]) == 200:
                rank200.append({"chart_week": week.isoformat(), "weekly_streams_at_rank_200": int(row["Streams"]), "track": row["Track"], "artist": row["Artist"], "spotify_track_id": ident})
    if invalid_ids or len(metadata) != len(first_rows):
        raise RuntimeError("Chart ID or track count mismatch; construction stopped.")

    ordered_weeks = sorted(observed_weeks)
    full_calendar: list[date] = []
    current = ordered_weeks[0]
    while current <= ordered_weeks[-1]:
        full_calendar.append(current)
        current += timedelta(days=7)
    calendar_index = {week: index for index, week in enumerate(full_calendar)}
    missing_weeks = sorted(set(full_calendar) - observed_weeks)
    final_calendar_index = len(full_calendar) - 1

    allow_rows = [
        {"variable_or_group": "spotify_track_id", "role": "join key only", "availability": "at chart entry", "permitted_use": "Identity, joins and duplicate checks; never a numeric predictive feature."},
        {"variable_or_group": "audio descriptors: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms", "role": "baseline song characteristics", "availability": "at chart entry", "permitted_use": "Candidate baseline predictors after range checks and train-only transformations."},
        {"variable_or_group": "release year / release date", "role": "baseline song characteristic", "availability": "at chart entry", "permitted_use": "Candidate baseline predictor; preserve the 160k mixed date precision."},
        {"variable_or_group": "X: chart appearances in first four calendar weeks", "role": "early chart performance", "availability": "after week 4", "permitted_use": "Candidate early predictor for the fixed-window count formulation only; never an admission filter."},
        {"variable_or_group": "chart artist identity", "role": "grouping identifier", "availability": "at chart entry", "permitted_use": "Artist-grouped validation or clustered inference after a documented grouping rule; not automatically a one-hot feature."},
    ]
    block_rows = [
        ("track popularity", "Static snapshot can reflect later listening/chart success."),
        ("artist popularity", "Static snapshot can reflect later success."),
        ("artist follower counts / total_artist_followers", "Follower count can accumulate after chart entry."),
        ("average artist popularity", "Later snapshot, potentially outcome-influenced."),
        ("future ranks", "Observed after the early window and part of the outcome period."),
        ("future streams", "Observed after the early window and part of the outcome period."),
        ("total chart duration / total chart weeks", "Full-history outcome information."),
        ("eventual peak rank / full-spell best rank", "Uses post-window observations."),
        ("re-entry count / later spell count / last active date", "Full-history outcome information."),
        ("event indicator / censoring / left-truncation flags", "Outcome-quality flags, not predictive features."),
        ("early_2w_prediction_eligible / early_4w_prediction_eligible", "Old flags require uninterrupted first-spell survival and must not define the new cohort."),
        ("old early rank/stream landmark summaries", "Not included in the minimal X-plus-baseline formulation; may only be reconsidered under a separately specified post-entry design."),
    ]
    write_csv(OUT / "predictor_allowlist.csv", allow_rows, ["variable_or_group", "role", "availability", "permitted_use"])
    write_csv(OUT / "predictor_blocklist.csv", [{"variable_or_group": a, "reason_blocked": b, "rule": "Exclude from every future modelling dataset unless separately time-stamped and approved."} for a, b in block_rows], ["variable_or_group", "reason_blocked", "rule"])

    outcome_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    horizon_summaries: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        eligible: list[dict[str, object]] = []
        reasons = Counter()
        for ident in sorted(metadata):
            first = first_rows[ident]
            entry = parse_date(first["first_observed_chart_entry"])
            start = calendar_index[entry]
            dates = full_calendar[start: start + horizon]
            base = {
                "horizon_weeks": horizon, "spotify_track_id": ident, **metadata[ident],
                "first_observed_chart_entry": entry.isoformat(), "left_truncated": first["left_truncated"],
                "entry_ambiguous_due_source_gap": first["entry_ambiguous_due_source_gap"],
                "match_550k_exact_id": first["match_550k_exact_id"], "match_160k_exact_id": first["match_160k_exact_id"],
            }
            if len(dates) < horizon:
                row = {**base, "enough_follow_up": 0, "source_gap_in_horizon": 0, "fixed_window_eligible": 0, "exclusion_reason": "insufficient_follow_up", "X_first_4_weeks": "", "Y_future_chart_weeks": "", "L_last_active_week_number": "", "Z_count_presence_blocks": "", "Z_reentry_more_than_one_block": ""}
                outcome_rows.append(row); exclusion_rows.append({**base, "exclusion_reason": "insufficient_follow_up"}); reasons["insufficient_follow_up"] += 1
                continue
            gaps = [week for week in dates if week in missing_weeks]
            if gaps:
                row = {**base, "enough_follow_up": 1, "source_gap_in_horizon": 1, "fixed_window_eligible": 0, "exclusion_reason": "source_gap_in_horizon", "X_first_4_weeks": "", "Y_future_chart_weeks": "", "L_last_active_week_number": "", "Z_count_presence_blocks": "", "Z_reentry_more_than_one_block": ""}
                outcome_rows.append(row); exclusion_rows.append({**base, "exclusion_reason": "source_gap_in_horizon", "source_gap_dates": ";".join(week.isoformat() for week in gaps)}); reasons["source_gap_in_horizon"] += 1
                continue
            first_four = [int(week in present_by_track[ident]) for week in dates[:4]]
            future = [int(week in present_by_track[ident]) for week in dates[4:]]
            x = sum(first_four)
            y = sum(future)
            l = max((index + 5 for index, state in enumerate(future) if state), default=0)
            z = count_blocks(future)
            row = {**base, "enough_follow_up": 1, "source_gap_in_horizon": 0, "fixed_window_eligible": 1, "exclusion_reason": "", "X_first_4_weeks": x, "Y_future_chart_weeks": y, "L_last_active_week_number": l, "Z_count_presence_blocks": z, "Z_reentry_more_than_one_block": int(z > 1)}
            outcome_rows.append(row); eligible.append(row)
        xs = [float(r["X_first_4_weeks"]) for r in eligible]
        ys = [float(r["Y_future_chart_weeks"]) for r in eligible]
        ls = [float(r["L_last_active_week_number"]) for r in eligible]
        zs = [float(r["Z_count_presence_blocks"]) for r in eligible]
        row = {
            "horizon_weeks": horizon, "total_chart_tracks": len(metadata), "eligible_chart_tracks": len(eligible),
            "eligible_550k_linked_tracks": sum(r["match_550k_exact_id"] == "1" for r in eligible),
            "eligible_160k_linked_tracks": sum(r["match_160k_exact_id"] == "1" for r in eligible),
            "excluded_insufficient_follow_up": reasons["insufficient_follow_up"], "excluded_source_gap_in_horizon": reasons["source_gap_in_horizon"],
            "excluded_total": sum(reasons.values()), "Y_zero_count": sum(value == 0 for value in ys),
            "Y_zero_percent": 100 * sum(value == 0 for value in ys) / len(ys) if ys else None,
            "Y_L_pearson_correlation": pearson(ys, ls), "Z_reentry_count": sum(value > 1 for value in zs),
            "Z_reentry_percent": 100 * sum(value > 1 for value in zs) / len(zs) if zs else None,
        }
        for label, values in (("X", xs), ("Y", ys), ("L", ls), ("Z_count", zs)):
            stat = summary(values)
            for key, value in stat.items():
                row[f"{label}_{key}"] = value
            counts = Counter(int(value) for value in values)
            for value, count in sorted(counts.items()):
                distribution_rows.append({"horizon_weeks": horizon, "variable": label, "value": value, "count": count, "percent_of_eligible": 100 * count / len(values) if values else None})
        for value, count in sorted(Counter(int(r["Z_reentry_more_than_one_block"]) for r in eligible).items()):
            distribution_rows.append({"horizon_weeks": horizon, "variable": "Z_reentry", "value": value, "count": count, "percent_of_eligible": 100 * count / len(eligible) if eligible else None})
        horizon_summaries.append(row)

    outcome_fields = ["horizon_weeks", "spotify_track_id", "chart_track", "chart_artist", "spotify_track_url", "first_observed_chart_entry", "left_truncated", "entry_ambiguous_due_source_gap", "match_550k_exact_id", "match_160k_exact_id", "enough_follow_up", "source_gap_in_horizon", "fixed_window_eligible", "exclusion_reason", "X_first_4_weeks", "Y_future_chart_weeks", "L_last_active_week_number", "Z_count_presence_blocks", "Z_reentry_more_than_one_block"]
    write_csv(OUT / "fixed_window_track_outcomes.csv", outcome_rows, outcome_fields)
    write_csv(OUT / "fixed_window_exclusions.csv", exclusion_rows, ["horizon_weeks", "spotify_track_id", "chart_track", "chart_artist", "spotify_track_url", "first_observed_chart_entry", "left_truncated", "entry_ambiguous_due_source_gap", "match_550k_exact_id", "match_160k_exact_id", "exclusion_reason", "source_gap_dates"])
    summary_fields = list(horizon_summaries[0])
    write_csv(OUT / "fixed_window_horizon_summary.csv", horizon_summaries, summary_fields)
    write_csv(OUT / "fixed_window_variable_distributions.csv", distribution_rows, ["horizon_weeks", "variable", "value", "count", "percent_of_eligible"])

    # Artist clustering using the chart source's primary Artist label. Only two labels contain commas.
    artist_tracks: defaultdict[str, set[str]] = defaultdict(set)
    for ident, meta in metadata.items():
        artist_tracks[meta["chart_artist"]].add(ident)
    artist_rows = [{"chart_artist": artist, "unique_chart_tracks": len(ids), "match_550k_exact_tracks": sum(first_rows[i]["match_550k_exact_id"] == "1" for i in ids), "match_160k_exact_tracks": sum(first_rows[i]["match_160k_exact_id"] == "1" for i in ids)} for artist, ids in artist_tracks.items()]
    artist_rows.sort(key=lambda row: (-row["unique_chart_tracks"], row["chart_artist"]))
    write_csv(OUT / "artist_track_counts.csv", artist_rows, ["chart_artist", "unique_chart_tracks", "match_550k_exact_tracks", "match_160k_exact_tracks"])
    artists_multi = [row for row in artist_rows if row["unique_chart_tracks"] >= 2]
    artist_summary = [{
        "artist_grouping_field": "chart Artist source field", "unique_artist_labels": len(artist_rows), "chart_tracks": len(metadata),
        "artists_with_2plus_tracks": len(artists_multi), "tracks_in_artists_with_2plus_tracks": sum(row["unique_chart_tracks"] for row in artists_multi),
        "percent_tracks_in_artists_with_2plus_tracks": 100 * sum(row["unique_chart_tracks"] for row in artists_multi) / len(metadata),
        "max_tracks_for_one_artist": max(row["unique_chart_tracks"] for row in artist_rows),
        "artist_labels_containing_comma": sum("," in row["chart_artist"] for row in artist_rows),
    }]
    write_csv(OUT / "artist_cluster_summary.csv", artist_summary, list(artist_summary[0]))

    # Exact-ID coverage and composition diagnostic.
    combo_counts = Counter((row["match_550k_exact_id"], row["match_160k_exact_id"]) for row in first_rows.values())
    match_rows = []
    for combo, count in sorted(combo_counts.items()):
        match_rows.append({"scope": "all_chart_tracks", "match_550k_exact_id": combo[0], "match_160k_exact_id": combo[1], "tracks": count, "percent_of_scope": 100 * count / len(first_rows), "interpretation": "both" if combo == ("1", "1") else "550k_only" if combo == ("1", "0") else "160k_only" if combo == ("0", "1") else "neither"})
    for year in sorted({row["first_observed_chart_entry"][:4] for row in first_rows.values()}):
        scoped = [row for row in first_rows.values() if row["first_observed_chart_entry"].startswith(year)]
        match_rows.append({"scope": f"entry_year_{year}", "match_550k_exact_id": "any", "match_160k_exact_id": "any", "tracks": len(scoped), "percent_of_scope": 100.0, "interpretation": "all"})
        for name, field in (("550k", "match_550k_exact_id"), ("160k", "match_160k_exact_id")):
            count = sum(row[field] == "1" for row in scoped)
            match_rows.append({"scope": f"entry_year_{year}", "match_550k_exact_id": "1" if name == "550k" else "", "match_160k_exact_id": "1" if name == "160k" else "", "tracks": count, "percent_of_scope": 100 * count / len(scoped), "interpretation": f"{name}_coverage"})
    write_csv(OUT / "match_rate_composition_diagnostic.csv", match_rows, ["scope", "match_550k_exact_id", "match_160k_exact_id", "tracks", "percent_of_scope", "interpretation"])

    coverage_rows = []
    for year in sorted({row["first_observed_chart_entry"][:4] for row in first_rows.values()}):
        scoped = [row for row in first_rows.values() if row["first_observed_chart_entry"].startswith(year)]
        strict = [row for row in scoped if row["survival_outcome_eligible"] == "1"]
        coverage_rows.append({"entry_year": int(year), "all_chart_tracks": len(scoped), "550k_exact_matches": sum(r["match_550k_exact_id"] == "1" for r in scoped), "550k_coverage_percent": 100 * sum(r["match_550k_exact_id"] == "1" for r in scoped) / len(scoped), "160k_exact_matches": sum(r["match_160k_exact_id"] == "1" for r in scoped), "160k_coverage_percent": 100 * sum(r["match_160k_exact_id"] == "1" for r in scoped) / len(scoped), "both_exact_matches": sum(r["match_550k_exact_id"] == "1" and r["match_160k_exact_id"] == "1" for r in scoped), "strict_survival_tracks": len(strict), "strict_550k_matches": sum(r["match_550k_exact_id"] == "1" for r in strict), "strict_160k_matches": sum(r["match_160k_exact_id"] == "1" for r in strict)})
    write_csv(OUT / "feature_coverage_by_chart_entry_year.csv", coverage_rows, list(coverage_rows[0]))

    # Rank-200 threshold time series and annual summary.
    rank200.sort(key=lambda row: row["chart_week"])
    if len(rank200) != len(ordered_weeks):
        raise RuntimeError("Expected exactly one rank-200 track per observed week.")
    write_csv(OUT / "rank_200_weekly_streams.csv", rank200, ["chart_week", "weekly_streams_at_rank_200", "track", "artist", "spotify_track_id"])
    annual = []
    for year in sorted({r["chart_week"][:4] for r in rank200}):
        vals = [r["weekly_streams_at_rank_200"] for r in rank200 if r["chart_week"].startswith(year)]
        stat = summary([float(v) for v in vals])
        annual.append({"year": int(year), "observed_weeks": len(vals), **{f"streams_{key}": value for key, value in stat.items()}})
    write_csv(OUT / "rank_200_annual_stream_summary.csv", annual, list(annual[0]))
    plot_path = PLOT_DIR / "rank_200_weekly_streams_2017_2021.svg"
    write_rank200_svg(rank200, plot_path)

    # Quantify old landmark selection using strict survival files only.
    landmark_rows = []
    strict_rows = [row for row in first_rows.values() if row["survival_outcome_eligible"] == "1"]
    groups = [
        ("strict_all", strict_rows),
        ("strict_2w_continuous_eligible", [r for r in strict_rows if r["early_2w_prediction_eligible"] == "1"]),
        ("strict_4w_continuous_eligible", [r for r in strict_rows if r["early_4w_prediction_eligible"] == "1"]),
        ("strict_550k_all", [r for r in strict_rows if r["match_550k_exact_id"] == "1"]),
        ("strict_550k_2w_continuous_eligible", [r for r in strict_rows if r["match_550k_exact_id"] == "1" and r["early_2w_prediction_eligible"] == "1"]),
        ("strict_550k_4w_continuous_eligible", [r for r in strict_rows if r["match_550k_exact_id"] == "1" and r["early_4w_prediction_eligible"] == "1"]),
    ]
    for name, rows in groups:
        durations = [float(r["first_spell_observed_weeks"]) for r in rows]
        stat = summary(durations)
        landmark_rows.append({"cohort": name, "tracks": len(rows), **{f"first_spell_weeks_{key}": value for key, value in stat.items()}, "percent_with_first_spell_4plus_weeks": 100 * sum(v >= 4 for v in durations) / len(durations), "percent_with_first_spell_8plus_weeks": 100 * sum(v >= 8 for v in durations) / len(durations)})
    write_csv(OUT / "old_landmark_selection_audit.csv", landmark_rows, list(landmark_rows[0]))

    # Method comparison and audit metadata.
    fixed_by_horizon = {row["horizon_weeks"]: row for row in horizon_summaries}
    comparison_rows = [{
        "approach": "first_spell_survival", "cohort_definition": "Strict: observed first entry, known first exit/censoring, excludes left truncation and first-spell source-gap ambiguity.",
        "question_answered": "How long does the first contiguous observed chart spell last before its first confirmed exit?",
        "reentry_treatment": "Recorded separately but does not extend the first-spell duration.", "all_chart_tracks_retained": len(strict_rows),
        "550k_linked_tracks_retained": sum(r["match_550k_exact_id"] == "1" for r in strict_rows), "advantages": "Explicit entry, exit, censoring and source-gap handling.",
        "limitations": "First exit is non-absorbing for songs that re-enter; only 69 strict cases are right-censored; weekly durations are discrete with many ties.",
    }]
    for horizon in HORIZONS:
        row = fixed_by_horizon[horizon]
        comparison_rows.append({
            "approach": f"fixed_window_count_{horizon}w", "cohort_definition": f"All tracks with {horizon} observable calendar weeks from first observed entry and no source-wide gap inside the horizon.",
            "question_answered": f"After early four-week chart presence X, how many chart weeks Y occur in weeks 5–{horizon}?",
            "reentry_treatment": "All later appearances, including re-entries, are counted in Y; Z describes fragmentation.", "all_chart_tracks_retained": row["eligible_chart_tracks"],
            "550k_linked_tracks_retained": row["eligible_550k_linked_tracks"], "advantages": "Retains one-week songs when follow-up exists; avoids treating first exit as absorbing; yields a fully observed fixed-horizon count.",
            "limitations": "Excludes late entrants without follow-up and source-gap-affected horizons; horizon choice must be justified; count may be zero-inflated/skewed.",
        })
    write_csv(OUT / "approach_comparison.csv", comparison_rows, list(comparison_rows[0]))

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "version": "fixed_window_count_v1", "no_model_fitted": True,
        "chart_source": str(CHART.relative_to(ROOT)), "first_spell_source": str(FIRST.relative_to(ROOT)),
        "historical_freeze_manifest": str((FROZEN / "freeze_manifest.json").relative_to(ROOT)),
        "calendar_definition": "Weekly Friday calendar from first to final observed chart date; source-wide missing weeks are unknown, never absence.",
        "fixed_window_definition": {"X": "Number of present chart weeks in calendar weeks 1–4 from first observed entry.", "Y": "Number of present chart weeks in calendar weeks 5–H, including re-entries.", "L": "Last numbered chart week with presence in weeks 5–H; 0 if Y=0.", "Z_count": "Number of contiguous present blocks in weeks 5–H.", "Z_reentry": "1 if Z_count > 1, otherwise 0."},
        "eligibility_definition": "H calendar weeks must be available after first observed entry and no source-wide missing week may fall within those H weeks. Left-truncated entry is flagged but not excluded.",
        "missing_source_weeks": [week.isoformat() for week in missing_weeks], "horizons_weeks": list(HORIZONS),
        "horizon_summary": horizon_summaries, "artist_summary": artist_summary[0],
        "matching_composition": {"both": combo_counts[("1", "1")], "550k_only": combo_counts[("1", "0")], "160k_only": combo_counts[("0", "1")], "neither": combo_counts[("0", "0")]},
        "rank_200_overall": summary([float(r["weekly_streams_at_rank_200"]) for r in rank200]),
        "feature_file_rows": len(feature_rows), "frozen_files": freeze_manifest,
    }
    (OUT / "fixed_window_construction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    first_h = horizon_summaries[0]
    last_h = horizon_summaries[-1]
    overall_threshold = audit["rank_200_overall"]
    docs = [
        "# Pre-model diagnostics responding to supervisor feedback",
        "",
        "## Scope",
        "",
        "This document records target-construction and diagnostic work only. No regression, survival, machine-learning or hierarchical model has been fitted. The existing chart panel and first-spell outputs remain unchanged; a checksum-verified historical copy is stored separately.",
        "",
        "## Corrections made",
        "",
        "- Historical first-spell artefacts were frozen before new target work.",
        "- Explicit predictor allowlist and blocklist files were created. Snapshot popularity and follower variables, future chart information, total duration and re-entry summaries are blocked from future modelling datasets.",
        "- The three missing chart weeks remain unknown states. A fixed-window record is eligible only if its full horizon has no source-wide missing week.",
        "- The former two- and four-week continuous-survival eligibility fields are retained as historical diagnostics, not as admission filters for the fixed-window formulation.",
        "",
        "## Fixed-window count formulation",
        "",
        "For a horizon H, X is the number of chart appearances in calendar weeks 1–4 after first observed entry. Y is the total number of chart appearances in weeks 5–H, including re-entries. L is the last numbered active week in the outcome window; it is zero when Y is zero. Z_count is the number of separate presence blocks in weeks 5–H and Z_reentry indicates more than one such block. Entries at the source start or immediately after a source gap are flagged but retained, because the new formulation is based on first observed entry rather than first-spell survival.",
        "",
        "## Interpretation and recommendation",
        "",
        "First-spell survival remains a valid descriptive target for the duration of the initial uninterrupted chart spell. It handles its defined source gaps, censoring and left-truncation correctly, but it treats first exit as the endpoint even though later re-entry is observed. The fixed-window formulation instead asks how much chart presence follows an early four-week period and includes later returns. It avoids the old survivor-only landmark admission rule and gives a fully observed count where the horizon is available.",
        "",
        "Recommendation: use the fixed-window count formulation as the proposed primary empirical outcome, subject to supervisor confirmation of the selected horizon. Retain first-spell survival as a secondary descriptive or sensitivity analysis rather than silently discarding it. The main horizon should balance a meaningful follow-up period with the loss of later entries; the accompanying horizon summary provides this trade-off.",
        "",
        "## Diagnostics completed",
        "",
        f"- Four horizons were created. At 26 weeks, {first_h['eligible_chart_tracks']} chart tracks are eligible ({first_h['eligible_550k_linked_tracks']} linked to 550k); at 104 weeks, {last_h['eligible_chart_tracks']} are eligible ({last_h['eligible_550k_linked_tracks']} linked to 550k).",
        f"- The rank-200 threshold series contains {len(rank200)} observed weeks. Overall weekly streams at rank 200 have median {overall_threshold['median']:.0f}, minimum {overall_threshold['min']:.0f} and maximum {overall_threshold['max']:.0f}; annual values and the plot are saved separately.",
        f"- Artist grouping based on the chart Artist field yields {artist_summary[0]['unique_artist_labels']} labels. {artist_summary[0]['tracks_in_artists_with_2plus_tracks']} of {artist_summary[0]['chart_tracks']} tracks ({artist_summary[0]['percent_tracks_in_artists_with_2plus_tracks']:.1f}%) belong to an artist label with at least two chart tracks, so artist grouping is material for later validation.",
        "- Exact IDs are valid and unique in the three relevant files. The lower 550k rate is therefore not caused by duplicate IDs or fuzzy matching: its catalogue overlaps 1,534 chart IDs, compared with 1,981 for 160k. The data establish a catalogue-composition/snapshot-coverage difference, not a definitive causal reason; the sources do not provide enough collection documentation to prove why particular IDs are absent.",
        "",
        "## Output locations",
        "",
        "All numerical outputs are in `data/processed/fixed_window_count_v1/`; the chart is in `outputs/fixed_window_diagnostics/`; frozen historical artefacts are in `data/frozen/first_spell_survival_2026-07-30/`.",
    ]
    DOC.write_text("\n".join(docs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
