#!/usr/bin/env python3
"""Tiny synthetic illustration of timing and fold-local baseline logic.

No thesis data, fitted thesis model, saved artifact or numerical result is used.
This script has no third-party dependencies and writes no files.
"""

from __future__ import annotations

from statistics import median


# (illustrative label, weeks with a chart appearance, illustrative fold)
TOY_TRACKS = [
    ("Toy A", (1,), 0),
    ("Toy B", (1, 2, 6), 1),  # week 6 counts despite weeks 3-5 being absent
    ("Toy C", (1, 2, 3, 4, 5, 6, 8), 2),
    ("Toy D", (1, 4, 12), 0),
    ("Toy E", (1, 2, 3, 5, 11), 1),
    ("Toy F", (1, 2, 3, 4), 2),
]


def outcomes(weeks: tuple[int, ...]) -> tuple[int, int]:
    if 1 not in weeks or len(set(weeks)) != len(weeks):
        raise ValueError("Each toy track must enter in week 1 with unique weeks")
    if any(week < 1 or week > 52 for week in weeks):
        raise ValueError("Toy weeks must lie between 1 and 52")
    x = sum(1 <= week <= 4 for week in weeks)
    y52 = sum(5 <= week <= 52 for week in weeks)
    return x, y52


def main() -> None:
    print("SYNTHETIC DEMONSTRATION — NOT THESIS DATA OR RESULTS")
    print("Weeks 1–4 -> X; weeks 5–52 -> Y52 (later presence counts).")
    records = [(name, fold, *outcomes(weeks)) for name, weeks, fold in TOY_TRACKS]
    for name, fold, x, y52 in records:
        print(f"{name}: toy fold={fold}, X={x}, Y52={y52}")

    print("\nIllustrative fold-local training-median benchmark:")
    for fold in sorted({record[1] for record in records}):
        training_y = [y52 for _, group, _, y52 in records if group != fold]
        heldout_names = [name for name, group, _, _ in records if group == fold]
        prediction = median(training_y)
        print(f"fold {fold}: train median={prediction:g}; held out={', '.join(heldout_names)}")
    print("This is not the thesis's protected grouping or a model-performance test.")


if __name__ == "__main__":
    main()
