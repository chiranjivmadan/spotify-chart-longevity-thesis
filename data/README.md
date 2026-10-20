# Obtain the external data locally

No third-party raw dataset is included in this public candidate. Download the files from their source pages and put them at these exact project-relative paths. Create the parent directories if your Git clone does not contain them. The SHA-256 values identify the local files used in the thesis; a different dataset version is not automatically equivalent.

| Source | Required filename and destination | Local size | Local SHA-256 |
| --- | --- | ---: | --- |
| [Spotify Top 200 Weekly Global 2017–2021](https://www.kaggle.com/datasets/bartomiejczyewski/spotify-top-200-weekly-global-20172021), Bartłomiej Czyżewski | `data/raw/charts/original/spotify_top_200_weekly_global_2017_2021.zip` | 2,781,634 bytes | `642244e89b797b5de46248bd8cc12d8da37e3c2e638698f5e571164a203e1544` |
| [550K Spotify Songs: Audio, Lyrics & Genres](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres), Serkan Tüysüz | `data/raw/spotify_550k/songs.csv` | 856,462,614 bytes | `ffdfee29c9cb93cd9d118d8857990334d2db20f3face4b4b54369fc3acec7eab` |
| [Spotify Dataset 1921–2020, 160k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks), Yamac Eren Ay | `data/raw/spotify_160k/data.csv` | 29,654,587 bytes | `c97c42349c2e97339ff908ab01163ec5a3264f02ec65e78cb961c85077ded7f8` |

The chart archive's project-authored source metadata is included at `data/raw/charts/source_metadata/bartomiejczyewski_top200_global_2017_2021.json`; its original spelling is part of the established workflow. Rename the downloaded chart archive to the required filename if necessary, but do not recompress it: Notebook 01 checks its identity.

The optional full rebuild **requires** two further local files that the three-file quick-mode table does not supply by itself:

| Additional full-rebuild file | Original-project size | Original-project SHA-256 |
| --- | ---: | --- |
| `data/raw/spotify_550k/artists.csv` (from the same 550k source download) | 5,429,571 bytes | `559fc644b532208546aac98d6eb58623cfba8a170235f5ad0f7ef89fca91732e` |
| `data/raw/charts/extracted/bartomiejczyewski_top200_global/Spotify Top 200 Global (2017-2021).csv` (extract the named member of the chart ZIP) | 8,885,853 bytes | `66e668f540f0b4c6c592c0ffb9b2fa2f06775f6f65a6cebc9a5b28fa6d7acd37` |

Upstream scripts open these files. The name `data/raw/spotify_160k/data_w_genres.csv` appears in genre-provenance descriptions but is not opened by the scanned generator code. The exact full-rebuild steps have not been clean-machine tested for this public candidate.

The 550k Kaggle hosting page labels its dataset CC BY-NC-SA 4.0; the 160k repost page labels it Community Data License Agreement – Sharing – Version 1.0; the chart metadata records a CC0 hosting-page label. These labels alone do **not** establish rights to redistribute every upstream Spotify, chart or lyric element. Each user should obtain the sources from their providers and comply with the applicable terms. No raw download or full lyric text should be committed here.

Once the external files **and** the 61 processed artifacts listed in `../provenance/processed_artifact_manifest.csv` are in place, run `python scripts/check_environment.py` from the repository root before the notebooks. The public clone alone is deliberately incomplete for execution.
