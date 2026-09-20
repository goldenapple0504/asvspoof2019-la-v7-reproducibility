"""Language-neutral V7 boundary-specificity cell contrast specification."""

from __future__ import annotations

from collections.abc import Mapping


CELL_ORDER = (
    ("TTS", "Boundary"),
    ("BF", "Boundary"),
    ("TTS", "Interior_A"),
    ("BF", "Interior_A"),
    ("TTS", "Interior_B"),
    ("BF", "Interior_B"),
)
CELL_WEIGHTS = (1.0, -1.0, -0.5, 0.5, -0.5, 0.5)


def boundary_specificity(cell_means: Mapping[tuple[str, str], float]) -> float:
    missing = set(CELL_ORDER) - set(cell_means)
    if missing:
        raise ValueError(f"Missing cell means: {sorted(missing)}")
    return sum(weight * float(cell_means[cell]) for cell, weight in zip(CELL_ORDER, CELL_WEIGHTS, strict=True))


def direct_definition(cell_means: Mapping[tuple[str, str], float]) -> float:
    boundary = cell_means[("TTS", "Boundary")] - cell_means[("BF", "Boundary")]
    interior_a = cell_means[("TTS", "Interior_A")] - cell_means[("BF", "Interior_A")]
    interior_b = cell_means[("TTS", "Interior_B")] - cell_means[("BF", "Interior_B")]
    return float(boundary - 0.5 * (interior_a + interior_b))

