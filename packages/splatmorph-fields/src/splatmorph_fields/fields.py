"""Field extraction from a SHARP splat .ply + its source image."""

from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

GRID = (2, 768, 768)          # SHARP's two-layer image-aligned splat grid
SIGMA, WIN, LAM = 6.0, 49, 0.35
BANDS = [(0.5, 1.5), (1.5, 4.0), (4.0, 10.0)]


def norm01(x, mask=None):
    ref = x[mask] if mask is not None else x
    lo, hi = np.percentile(ref, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _local_range(x, win):
    return gaussian_filter(maximum_filter(x, win) - minimum_filter(x, win), win / 4)


def _bandpass(x, s_lo, s_hi):
    return gaussian_filter(x, s_lo) - gaussian_filter(x, s_hi)


def fields_from_ply(ply_path: Path, image_path: Path):
    """Base fields: (z_layer0, luma, roi). ROI is the luminance support; a
    frame with no dark void treats the whole frame as ROI."""
    v = PlyData.read(str(ply_path))["vertex"]
    z = np.asarray(v["z"], dtype=np.float32).reshape(GRID)[0]
    luma = np.asarray(
        Image.open(image_path).convert("L").resize((768, 768), Image.BILINEAR),
        np.float32) / 255.0
    roi = gaussian_filter(luma, 12) > 0.05
    if roi.mean() < 0.05:
        roi = np.ones_like(roi, dtype=bool)
    return z, luma, roi


def analyze(ply_path: Path, image_path: Path):
    """Grid fields for one scene.

    Returns (elev, D, span, roi):
    - elev: normalized relative elevation in [0,1] * roi (peak=1, trough=0,
      self-calibrating from the 2-98 depth percentiles inside the ROI)
    - D: glare-fused shard displacement field — multi-band image detail,
      unit-normalized per band, carried by the depth band's local energy
      and bounded by the local elevation range
    - span: the scene's depth span (world units)
    - roi: luminance support mask
    """
    z, luma_raw, roi = fields_from_ply(ply_path, image_path)
    lo, hi = np.percentile(z[roi], [2, 98])
    span = float(hi - lo)
    elev = np.clip((hi - z) / (span + 1e-9), 0, 1)
    luma = norm01(luma_raw, roi)
    detail = np.zeros_like(elev)
    for s_lo, s_hi in BANDS:
        d_band = _bandpass(elev, s_lo, s_hi)
        g_band = _bandpass(luma, s_lo, s_hi)
        w = np.sqrt(gaussian_filter(d_band ** 2, 12))
        g_energy = np.sqrt(gaussian_filter(g_band ** 2, 12))
        floor = np.percentile(g_energy[roi], 30)
        g_unit = np.where(g_energy > floor, g_band / (g_energy + 1e-3), 0.0)
        detail += w * g_unit
    amp = _local_range(elev, WIN)
    amp = np.minimum(amp, np.percentile(amp, 95))
    D = LAM * (detail / (_local_range(detail, WIN) + 1e-3)) * amp * roi
    return (elev * roi).astype(np.float32), D.astype(np.float32), span, roi


def curl_field(elev):
    """Solenoidal (divergence-free) flow along depth contours: rotated
    gradient of the smoothed elevation."""
    h = gaussian_filter(elev, 5)
    gy, gx = np.gradient(h)
    return np.stack([gy, -gx], axis=-1)
