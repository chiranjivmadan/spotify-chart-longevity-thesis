# Processed artifacts are intentionally absent

The final six-notebook workflow expects 61 processed/intermediate files at the relative paths in `../../provenance/processed_artifact_manifest.csv`. The manifest's sizes and SHA-256 values identify the **original saved copies**; regenerated files need not have identical bytes. The existing private portable release includes the originals, but this public candidate does not: some contain lyric text, and many contain third-party-derived track-level data. Do not commit regenerated copies without a separate redistribution review.

The upstream generators and intended sequence are documented in `../../provenance/reproducibility_modes.md`. The full rebuild is computationally expensive and has not been clean-machine tested from this public candidate.
