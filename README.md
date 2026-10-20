# Predicting post-entry chart longevity

**Master's thesis:** *Predicting Post-Entry Chart Longevity in the Spotify Global Top 200: Early Chart Performance and Song Characteristics* — Chiranjiv Madan, University of Pisa.

**Research question:** Among songs that have already entered the Spotify Global Top 200, how well do early chart performance, audio characteristics and lyric characteristics predict subsequent chart presence? This is post-entry forecasting, not pre-release hit prediction or causal inference.

## Project overview

The study links a [Spotify Global Weekly Top 200 archive (2017–2021)](https://www.kaggle.com/datasets/bartomiejczyewski/spotify-top-200-weekly-global-20172021) to [550K Spotify Songs: Audio, Lyrics & Genres](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres) by exact Spotify ID. The primary analysis covers **1,014 eligible tracks**. A separate cohort linked to [Spotify Dataset 1921–2020, 160k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks) supports robustness checks; it is not pooled with the primary cohort.

The early predictor **X** counts Top-200 appearances in calendar weeks 1–4 after confirmed entry. The primary outcome **Y52** counts appearances in weeks 5–52, including later presence after an absence. Y52 is a future-presence count, **not** time to first chart exit.

## Methods

The principal comparison adds information in stages: **F1_YEAR** uses X and chart-entry year; **F3_YEAR** adds selected audio/song characteristics; **LEX_YEAR** adds seven transparent lyric summaries. **H3_YEAR** is a separate two-part companion model for any future presence and, among tracks with future presence, its positive duration.

The main evaluation uses five-fold protected grouped cross-validation. A retrospective entry-year transfer check trains on 2017–2018 entrants and evaluates 2019 entrants, with 2020 treated as a small stress test. It is not a rolling-origin deployment simulation.

## Key findings

Prediction errors below are measured in **chart weeks**; lower MAE and RMSE are better.

| Model | MAE (chart weeks) | RMSE (chart weeks) |
| --- | ---: | ---: |
| Training-fold median (zero) | 6.599 | 14.164 |
| F1_YEAR | 6.319 | 10.735 |
| F3_YEAR | 6.229 | 10.837 |
| LEX_YEAR | 6.072 | 10.818 |
| H3_YEAR (companion expected count) | 6.074 | 10.753 |

Early chart persistence supplies most of the examined predictive information. Audio lowers MAE only slightly after X and year. The observed F3_YEAR-to-LEX_YEAR MAE gain is **0.1572 chart weeks**, but its protected-component 95% bootstrap interval **[−0.0014, 0.3162]** reaches just below zero, and the lyric advantage does not repeat in the 2019 check. The H3_YEAR binary component distinguishes any from no future presence strongly; predicting exact positive duration remains difficult. These are predictive findings, not causal effects.

## Repository guide

| Location | Contents |
| --- | --- |
| [`notebooks/final_thesis_v2/`](notebooks/final_thesis_v2/) | Six final analysis notebooks and their run order |
| [`scripts/`](scripts/) | Notebook runner, helpers, preflight and upstream generators |
| [`examples/`](examples/) | Independent synthetic workflow illustration |
| [`data/README.md`](data/README.md) | External source links, filenames and acquisition instructions |
| [`provenance/notebook_input_map.md`](provenance/notebook_input_map.md) | Processed inputs required by each notebook |
| [`provenance/reproducibility_modes.md`](provenance/reproducibility_modes.md) | Setup, execution modes and validation boundaries |
| [`provenance/processed_artifact_manifest.csv`](provenance/processed_artifact_manifest.csv) | Original artifact provenance and checksums |

From the repository root, run the tested **synthetic demonstration** with `python examples/synthetic_workflow.py`. It uses invented tracks, needs no thesis data or third-party packages, and does not reproduce thesis results. The six thesis notebooks require separately obtained external data and 61 omitted processed artifacts; **a complete public rebuild has not been tested**. Setup and execution instructions are in the linked supporting documentation.

Raw datasets, lyrics, processed track-level files, saved predictions and the thesis PDF are excluded from this repository. The thesis PDF is intentionally not included here. A direct University of Pisa ETD record link can be added once the thesis record is published.

## Licence

Repository-authored code and documentation are licensed under the [MIT License](LICENSE). This licence does not grant rights to third-party datasets, lyrics, Spotify content or other upstream materials. Applicable provider and source terms still apply; third-party source labels and provenance are preserved in [`data/README.md`](data/README.md).

**Scope limits:** The results concern selected, already-charted tracks in one global chart during 2017–2021. Rich-feature linkage is selective, historical availability/version identity of every lyric cannot be established, and the 2020 stress cohort is small. The study does not identify the mechanism causing later chart longevity.
