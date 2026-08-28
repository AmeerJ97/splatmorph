"""CPU unit tests for the transport stage — the verification gates in
miniature. No GPU, no SHARP, no rendering."""

import numpy as np
import torch

from splatmorph_transport import coarse_features, transport_plan, barycentric_maps
from splatmorph_transport import transport as T


def _synthetic_scene(seed, side=64):
    rng = np.random.default_rng(seed)
    elev = rng.random((side, side)).astype(np.float32)
    col = rng.random((side, side, 3)).astype(np.float32)
    roi = np.ones((side, side), bool)
    return elev, col, roi


def _feats(seed, coarse=16):
    elev, col, roi = _synthetic_scene(seed)
    return coarse_features(elev, col, roi, coarse=coarse, device=torch.device("cpu"))


def test_sinkhorn_marginals_converge():
    fA, mA = _feats(0)
    fB, mB = _feats(1)
    P, info = transport_plan(fA, mA, fB, mB)
    assert info["converged"], f"marginal err {info['marginal_err']:.2e}"
    assert info["marginal_err"] <= 1e-3


def test_anti_is_height_flip():
    """anti=True must equal match-mode against a height-flipped B (with the
    anti height weight) — one sign flip, same machinery."""
    fA, mA = _feats(0)
    fB, mB = _feats(1)
    _, anti = transport_plan(fA, mA, fB, mB, anti=True)
    fB2 = fB.clone()
    fB2[:, 2] = 1.0 - fB2[:, 2]
    fB2[:, 6] = 1.0 - fB2[:, 6]
    old = T.W_ELEV
    T.W_ELEV = T.W_ANTI
    try:
        _, match = transport_plan(fA, mA, fB2, mB, anti=False)
    finally:
        T.W_ELEV = old
    assert abs(anti["cost"] - match["cost"]) < 1e-4


def test_band_barrier_is_zero_within_band():
    d = torch.tensor([0.0, T.BAND_R * 0.99, T.BAND_R, T.BAND_R + 0.1])
    barrier = T.W_BAND * torch.clamp(torch.exp(T.BAND_ALPHA * (d - T.BAND_R)) - 1, min=0)
    assert barrier[0] == 0 and barrier[1] == 0 and barrier[2] == 0
    assert barrier[3] > 0


def test_identical_scenes_map_to_identity():
    """Self-pairing must be (approximately) the identity map."""
    fA, mA = _feats(7)
    P, _ = transport_plan(fA, mA, fA, mA)
    dest_AB, dest_BA = barycentric_maps(P, full=64, coarse=16)
    ident = torch.stack(torch.meshgrid(
        torch.arange(64.0), torch.arange(64.0), indexing="ij"), dim=-1)
    err = (dest_AB - ident).norm(dim=-1).mean()
    assert err < 4.0, f"self-map drifted {err:.2f} cells"


def test_perfect_anti_partner_has_zero_h_residual():
    """A scene and its exact height-flip anti-match with ~zero residual."""
    elev, col, roi = _synthetic_scene(3)
    cpu = torch.device("cpu")
    fA, mA = coarse_features(elev, col, roi, coarse=16, device=cpu)
    fB, mB = coarse_features(1.0 - elev, col, roi, coarse=16, device=cpu)
    _, info = transport_plan(fA, mA, fB, mB, anti=True)
    assert info["h_residual"] < 0.05, info["h_residual"]
