"""Banded entropic optimal transport for dense grid correspondence.

The pairing between two image-aligned splat grids is the solution of an
optimal transport problem whose ground cost combines appearance and
elevation features (with Gaussian neighborhood context), a gentle quadratic
position term, and an exponential barrier past the horizontal separation a
morph trajectory can organically absorb. `anti=True` pairs peaks with
troughs (the height cost becomes a squared difference against a
height-flipped counterpart — one sign flip; the machinery is unchanged).

Device-agnostic: runs on CUDA when available, CPU otherwise.
"""

from .transport import (
    coarse_features,
    transport_plan,
    barycentric_maps,
    optimal_pairing_bidir,
    sample_grid,
)

__all__ = [
    "coarse_features",
    "transport_plan",
    "barycentric_maps",
    "optimal_pairing_bidir",
    "sample_grid",
]
