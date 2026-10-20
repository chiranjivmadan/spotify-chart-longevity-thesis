#!/usr/bin/env python3
"""Reproducible audit of the raw Spotify datasets used in this thesis project.

The script uses the Python standard library so that the audit does not depend
on a notebook state.  It reads the large lyric-bearing CSV as a stream.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import pickle
import re
import unicodedata
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


PLACEHOLDER_PATTERNS = (
    "lyrics not available",
    "lyrics unavailable",
    "no lyrics available",
    "no lyrics",
    "instrumental",
    "[instrumental]",
    "not available",
    "n/a",
)
NUMERIC_SONG_COLUMNS = {
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "year", "popularity", "total_artist_followers",
    "avg_artist_popularity", "lyrics_length",
}
NUMERIC_160K_COLUMNS = {
    "valence", "year", "acousticness", "danceability", "duration_ms", "energy",
    "explicit", "instrumentalness", "key", "liveness", "loudness", "mode",
    "popularity", "speechiness", "tempo",
}


def blank(value: str | None) -> bool:
    return value is None or not value.strip()


def normalise_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"\b(feat(?:uring)?|ft)\.?\b.*$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def row_digest(row: dict[str, str], fields: list[str]) -> str:
    payload = "\x1f".join(row.get(field, "") for field in fields)
    return hashlib.blake2b(payload.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def value_type(value: str) -> str:
    """Return a conservative inferred scalar type for a nonblank CSV value."""
    try:
        int(value)
        return "integer"
    except ValueError:
        try:
            float(value)
            return "float"
        except ValueError:
            return "string"


def parse_artists(value: str | None) -> list[str]:
    """Parse the list-like artist field used by both supplied song datasets."""
    if blank(value):
        return []
    try:
        parsed = json.loads(value or "")
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value or "")
        except (SyntaxError, ValueError):
            return [value.strip()] if value else []
    return [str(item).strip() for item in parsed if str(item).strip()] if isinstance(parsed, list) else [str(parsed).strip()]


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None,
                "mean": None, "p75": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "mean": mean(values),
        "p75": quantile(values, 0.75),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_song_chunks(chunks: list[tuple]) -> dict[str, object]:
    """Combine bounded song-audit chunks without loading the source CSV again."""
    first = chunks[0][0]
    columns = first["columns"]
    missing = Counter()
    years = Counter()
    genres = Counter()
    ids = Counter()
    row_hashes = Counter()
    artist_values = Counter()
    individual_artist_names: set[str] = set()
    numeric_values = {column: array("d") for column in NUMERIC_SONG_COLUMNS}
    type_sets = {column: set() for column in columns}
    lyrics_counts = Counter()
    placeholder_examples = Counter()
    total_rows = 0

    for detail, numeric, chunk_years, chunk_genres, chunk_ids, _, chunk_hashes, chunk_artists, chunk_individuals, chunk_lyric_lengths in chunks:
        total_rows += detail["rows"]
        missing.update(detail["missing_values"])
        years.update(chunk_years)
        genres.update(chunk_genres)
        ids.update(chunk_ids)
        row_hashes.update(chunk_hashes)
        artist_values.update(chunk_artists)
        individual_artist_names.update(chunk_individuals)
        for column, values in numeric.items():
            numeric_values[column].extend(values)
        for column, inferred_type in detail["inferred_value_types"].items():
            type_sets[column].add(inferred_type)
        quality = detail["lyrics_quality"]
        for key in ("empty_or_whitespace", "placeholder_exact", "under_20_characters_nonblank", "under_100_characters_nonblank", "contains_newline"):
            lyrics_counts[key] += quality[key]
        for example in quality["placeholder_examples"]:
            placeholder_examples[example["text"]] += example["count"]

    lyrics_lengths = array("d")
    for chunk in chunks:
        lyrics_lengths.extend(chunk[9])

    inferred_types = {}
    for column, types in type_sets.items():
        inferred_types[column] = "string" if "string" in types else ("float" if "float" in types else ("integer" if "integer" in types else "empty"))

    return {
        "file": first["file"],
        "rows": total_rows,
        "columns": columns,
        "column_count": len(columns),
        "inferred_value_types": inferred_types,
        "missing_values": {column: missing[column] for column in columns},
        "duplicate_rows_beyond_first": sum(count - 1 for count in row_hashes.values() if count > 1),
        "duplicate_id_values_beyond_first": sum(count - 1 for count in ids.values() if count > 1),
        "unique_nonblank_ids": len(ids),
        "year_distribution": dict(sorted(years.items(), key=lambda pair: pair[0])),
        "numeric_summaries": {column: numeric_summary(values) for column, values in numeric_values.items()},
        "distinct_raw_artist_values": len(artist_values),
        "distinct_individual_artist_names": len(individual_artist_names),
        "genre_coverage": {
            "distinct_nonblank_genres": len(genres),
            "top_20": [{"genre": key, "count": value} for key, value in genres.most_common(20)],
        },
        "lyrics_quality": {
            "empty_or_whitespace": lyrics_counts["empty_or_whitespace"],
            "placeholder_exact": lyrics_counts["placeholder_exact"],
            "under_20_characters_nonblank": lyrics_counts["under_20_characters_nonblank"],
            "under_100_characters_nonblank": lyrics_counts["under_100_characters_nonblank"],
            "contains_newline": lyrics_counts["contains_newline"],
            "character_length_summary": numeric_summary(lyrics_lengths),
            "placeholder_examples": [{"text": text, "count": count} for text, count in placeholder_examples.most_common(20)],
        },
    }


def inspect_csv(
    path: Path,
    numeric_columns: set[str],
    id_column: str = "id",
    collect_lyrics: bool = False,
    track_key: bool = False,
    start_record: int = 0,
    max_records: int | None = None,
) -> tuple[dict[str, object], dict[str, array], Counter, Counter, Counter, Counter, Counter, Counter, set[str], array]:
    missing = Counter()
    ids: Counter = Counter()
    row_hashes: Counter = Counter()
    years = Counter()
    numeric_values = {column: array("d") for column in numeric_columns}
    inferred_types: dict[str, set[str]] = {column: set() for column in []}
    genres = Counter()
    normalized_keys = Counter()
    artist_values = Counter()
    individual_artist_names: set[str] = set()
    blank_match_components = 0
    lyrics = {
        "empty_or_whitespace": 0,
        "placeholder_exact": 0,
        "under_20_characters_nonblank": 0,
        "under_100_characters_nonblank": 0,
        "contains_newline": 0,
        "character_lengths": array("d"),
        "placeholder_examples": Counter(),
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        inferred_types = {column: set() for column in columns}
        rows = 0
        for record_index, row in enumerate(reader):
            if record_index < start_record:
                continue
            if max_records is not None and rows >= max_records:
                break
            rows += 1
            if max_records is not None and rows % 10000 == 0:
                print(f"Processed {rows} rows in current chunk", flush=True)
            for column in columns:
                if blank(row.get(column)):
                    missing[column] += 1
                else:
                    inferred_types[column].add(value_type(row[column]))
            row_hashes[row_digest(row, columns)] += 1
            song_id = row.get(id_column, "")
            if not blank(song_id):
                ids[song_id] += 1
            year = row.get("year", "")
            if not blank(year):
                years[year] += 1
            for column in numeric_columns:
                value = row.get(column, "")
                if not blank(value):
                    try:
                        numeric_values[column].append(float(value))
                    except ValueError:
                        pass
            genre = row.get("genre", "") or row.get("main_genre", "")
            if not blank(genre):
                genres[genre.strip()] += 1
            artist = row.get("artists", "") or row.get("name", "")
            if not blank(artist):
                artist_values[artist.strip()] += 1
            for artist_name in parse_artists(row.get("artists")):
                individual_artist_names.add(artist_name)
            if track_key:
                artist_key = " ".join(normalise_text(item) for item in parse_artists(row.get("artists")))
                key = (normalise_text(row.get("name")), artist_key)
                if not key[0] or not key[1]:
                    blank_match_components += 1
                if key != ("", ""):
                    normalized_keys[key] += 1
            if collect_lyrics:
                text = row.get("lyrics", "") or ""
                stripped = text.strip()
                if not stripped:
                    lyrics["empty_or_whitespace"] += 1
                else:
                    lowered = stripped.lower()
                    if lowered in PLACEHOLDER_PATTERNS:
                        lyrics["placeholder_exact"] += 1
                        lyrics["placeholder_examples"][stripped[:120]] += 1
                    if len(stripped) < 20:
                        lyrics["under_20_characters_nonblank"] += 1
                    if len(stripped) < 100:
                        lyrics["under_100_characters_nonblank"] += 1
                    if "\n" in text or "\r" in text:
                        lyrics["contains_newline"] += 1
                    lyrics["character_lengths"].append(len(stripped))

    result = {
        "file": str(path),
        "rows": rows,
        "columns": columns,
        "column_count": len(columns),
        "inferred_value_types": {
            column: "string" if "string" in types else ("float" if "float" in types else ("integer" if "integer" in types else "empty"))
            for column, types in inferred_types.items()
        },
        "missing_values": {column: missing[column] for column in columns},
        "duplicate_rows_beyond_first": sum(count - 1 for count in row_hashes.values() if count > 1),
        "duplicate_id_values_beyond_first": sum(count - 1 for count in ids.values() if count > 1),
        "unique_nonblank_ids": len(ids),
        "year_distribution": dict(sorted(years.items(), key=lambda pair: pair[0])),
        "numeric_summaries": {column: numeric_summary(values) for column, values in numeric_values.items()},
        "distinct_raw_artist_values": len(artist_values),
        "distinct_individual_artist_names": len(individual_artist_names),
        "genre_coverage": {
            "distinct_nonblank_genres": len(genres),
            "top_20": [{"genre": key, "count": value} for key, value in genres.most_common(20)],
        },
    }
    if collect_lyrics:
        result["lyrics_quality"] = {
            "empty_or_whitespace": lyrics["empty_or_whitespace"],
            "placeholder_exact": lyrics["placeholder_exact"],
            "under_20_characters_nonblank": lyrics["under_20_characters_nonblank"],
            "under_100_characters_nonblank": lyrics["under_100_characters_nonblank"],
            "contains_newline": lyrics["contains_newline"],
            "character_length_summary": numeric_summary(lyrics["character_lengths"]),
            "placeholder_examples": [
                {"text": text, "count": count}
                for text, count in lyrics["placeholder_examples"].most_common(20)
            ],
        }
    if track_key:
        result["matching_key_quality"] = {
            "blank_normalized_name_or_artist": blank_match_components,
            "unique_normalized_name_artist_keys": len(normalized_keys),
            "duplicate_normalized_name_artist_rows_beyond_first": sum(
                count - 1 for count in normalized_keys.values() if count > 1
            ),
        }
    return result, numeric_values, years, genres, ids, normalized_keys, row_hashes, artist_values, individual_artist_names, lyrics["character_lengths"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--songs-chunk-start", type=int)
    parser.add_argument("--songs-chunk-size", type=int)
    parser.add_argument("--merge-song-chunks", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    raw550 = root / "data/raw/spotify_550k"
    raw160 = root / "data/raw/spotify_160k"
    out = root / "data/processed/audit"
    out.mkdir(parents=True, exist_ok=True)

    chunks_dir = out / "song_chunks"
    if args.songs_chunk_start is not None:
        if args.songs_chunk_size is None:
            parser.error("--songs-chunk-size is required when creating a song chunk")
        chunks_dir.mkdir(exist_ok=True)
        chunk = inspect_csv(
            raw550 / "songs.csv", NUMERIC_SONG_COLUMNS, collect_lyrics=True,
            start_record=args.songs_chunk_start, max_records=args.songs_chunk_size,
        )
        filename = chunks_dir / f"songs_{args.songs_chunk_start:07d}_{args.songs_chunk_size:07d}.pkl"
        with filename.open("wb") as handle:
            pickle.dump(chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {filename}")
        return

    if args.merge_song_chunks:
        chunk_files = sorted(chunks_dir.glob("songs_*.pkl"))
        if not chunk_files:
            parser.error("No song chunks found to merge")
        chunks = []
        for filename in chunk_files:
            with filename.open("rb") as handle:
                chunks.append(pickle.load(handle))
        songs = merge_song_chunks(chunks)
    else:
        songs = None

    if songs is None:
        songs, song_numeric, song_years, song_genres, song_ids, _, _, _, _, _ = inspect_csv(
            raw550 / "songs.csv", NUMERIC_SONG_COLUMNS, collect_lyrics=True
        )
    else:
        song_numeric = {column: array("d") for column in NUMERIC_SONG_COLUMNS}
        song_years = Counter(songs["year_distribution"])
        song_genres = Counter()
        song_ids = Counter()

    artists, artist_numeric, artist_years, artist_genres, artist_ids, _, _, _, _, _ = inspect_csv(
        raw550 / "artists.csv", {"followers", "popularity"}, id_column="id"
    )
    tracks160, tracks160_numeric, tracks160_years, _, _, keys160, _, _, _, _ = inspect_csv(
        raw160 / "data.csv", NUMERIC_160K_COLUMNS, track_key=True
    )

    cohort_rows = sum(count for year, count in tracks160_years.items() if year.isdigit() and 2016 <= int(year) <= 2019)
    key_blank = 0
    key_dupes = 0
    for (name, artist), count in keys160.items():
        if count > 1:
            key_dupes += count - 1
    tracks160["matching_key_quality"] = {
        "blank_normalized_name_or_artist": tracks160["matching_key_quality"]["blank_normalized_name_or_artist"],
        "unique_normalized_name_artist_keys": len(keys160),
        "duplicate_normalized_name_artist_rows_beyond_first": key_dupes,
        "2016_2019_cohort_rows": cohort_rows,
    }

    audit = {
        "audit_metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Streamed CSV audit; duplicate rows are exact full-row duplicates; matching keys are accent-insensitive lower-case name+artist keys with trailing feat./ft. text removed.",
            "source_archives": ["archive (4).zip", "archive (5).zip"],
        },
        "spotify_550k_songs": songs,
        "spotify_550k_artists": artists,
        "spotify_160k_tracks": tracks160,
    }
    (out / "raw_dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    for dataset_name, detail in (("550k_songs", songs), ("550k_artists", artists), ("160k_tracks", tracks160)):
        summary_rows.append({
            "dataset": dataset_name,
            "rows": detail["rows"],
            "columns": detail["column_count"],
            "duplicate_rows_beyond_first": detail["duplicate_rows_beyond_first"],
            "unique_nonblank_ids": detail["unique_nonblank_ids"],
            "duplicate_id_values_beyond_first": detail["duplicate_id_values_beyond_first"],
        })
    write_csv(out / "dataset_overview.csv", summary_rows, list(summary_rows[0]))

    missing_rows = []
    for dataset_name, detail in (("550k_songs", songs), ("550k_artists", artists), ("160k_tracks", tracks160)):
        for column, count in detail["missing_values"].items():
            missing_rows.append({"dataset": dataset_name, "column": column, "missing_values": count})
    write_csv(out / "missing_values.csv", missing_rows, ["dataset", "column", "missing_values"])

    distribution_rows = []
    for dataset_name, detail in (("550k_songs", songs), ("550k_artists", artists), ("160k_tracks", tracks160)):
        for column, values in detail["numeric_summaries"].items():
            distribution_rows.append({"dataset": dataset_name, "variable": column, **values})
    write_csv(out / "numeric_distributions.csv", distribution_rows,
              ["dataset", "variable", "count", "min", "p25", "median", "mean", "p75", "max"])

    year_rows = []
    for dataset_name, counts in (("550k_songs", song_years), ("160k_tracks", tracks160_years)):
        for year, count in sorted(counts.items(), key=lambda pair: pair[0]):
            year_rows.append({"dataset": dataset_name, "year": year, "count": count})
    write_csv(out / "year_distributions.csv", year_rows, ["dataset", "year", "count"])


if __name__ == "__main__":
    main()
