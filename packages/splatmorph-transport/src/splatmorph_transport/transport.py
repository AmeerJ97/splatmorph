"""Coarse-grid entropic OT with a trajectory-band position barrier."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


COARSE = 96          # transport nodes per side
W_POS = 1.0          # gentle within-band spatial regularity
W_ELEV = 4.0         # crests pair with crests (match mode)
W_ANTI = 8.0         # anti mode: height term dominates
W_COL = 4.0          # appearance similarity
SINKHORN_ITERS = 300
EPS_FRAC = 0.02      # entropic reg as a fraction of median FEATURE cost
MARGINAL_GATE = 1e-3

# The morph trajectory organically absorbs horizontal separation up to
# BAND_R (normalized units); beyond it the penalty grows exponentially.
BAND_R = 0.15
BAND_ALPHA = 20.0
W_BAND = 2.0


def coarse_features(elev, color_grid, roi, coarse: int = COARSE, device=None):
    """(coarse^2, 10): x, y, elev, r, g, b + Gaussian neighborhood-context
    copies of elev/r/g/b — each cell matches on itself AND its surroundings.

    Returns (features, roi_mass) tensors on `device`.
    """
    device = device or get_device()
    s = elev.shape[0] // coarse

    def pool(x):
        return x.reshape(coarse, s, coarse, s).mean(axis=(1, 3))

    ys, xs = np.mgrid[0:coarse, 0:coarse].astype(np.float32) / (coarse - 1)
    e = pool(elev)
    cols = [pool(color_grid[..., c]) for c in range(3)]
    ctx = [gaussian_filter(a, 2.0) for a in [e] + cols]
    m = pool(roi.astype(np.float32))
    feats = np.stack([xs, ys, e] + cols + ctx, axis=-1).reshape(-1, 10)
    return (torch.from_numpy(feats).float().to(device),
            torch.from_numpy(m.reshape(-1)).float().to(device))


def transport_plan(fA, mA, fB, mB, tol: float = 1e-4, anti: bool = False):
    """Converged entropic plan between coarse feature clouds.

    anti=True: peaks pair with troughs — the height cost becomes
    (h_A + h_B)^2 in signed coordinates, which in [0,1] elevation space is
    exactly a squared difference against a height-FLIPPED B. Height is the
    dominant term; position/color break the 1-D degeneracy.

    Early-stops once the marginal error drops below tol (checked every 25
    iterations). Returns (P, info): info reports eps, transport cost <C,P>,
    max marginal error, and in anti mode h_residual (mass-weighted
    signed-height mismatch; 0 = perfect anti-match).
    """
    device = fA.device
    w_e = W_ANTI if anti else W_ELEV
    w = torch.tensor([W_POS, W_POS, w_e, W_COL, W_COL, W_COL,
                      w_e * 0.5, W_COL * 0.5, W_COL * 0.5, W_COL * 0.5],
                     device=device).sqrt()
    if anti:
        fB = fB.clone()
        fB[:, 2] = 1.0 - fB[:, 2]
        fB[:, 6] = 1.0 - fB[:, 6]
    C = torch.cdist(fA * w, fB * w) ** 2
    # eps is scaled to the FEATURE cost landscape; the barrier is added
    # after (its exponential magnitudes would otherwise inflate the median
    # and smear the plan — measured failure mode, kept out by construction)
    eps = EPS_FRAC * C.median()
    d_xy = torch.cdist(fA[:, :2], fB[:, :2])
    C = C + W_BAND * torch.clamp(torch.exp(BAND_ALPHA * (d_xy - BAND_R)) - 1.0, min=0.0)

    mu = mA + 0.05
    mu = mu / mu.sum()
    nu = mB + 0.05
    nu = nu / nu.sum()
    log_mu, log_nu = mu.log(), nu.log()

    f = torch.zeros_like(mu)
    g = torch.zeros_like(nu)
    for it in range(SINKHORN_ITERS):
        f = eps * (log_mu - torch.logsumexp((g[None, :] - C) / eps, dim=1))
        g = eps * (log_nu - torch.logsumexp((f[:, None] - C) / eps, dim=0))
        if (it + 1) % 25 == 0:
            P = torch.exp((f[:, None] + g[None, :] - C) / eps)
            if float((P.sum(1) - mu).abs().max()) < tol and \
               float((P.sum(0) - nu).abs().max()) < tol:
                break
    P = torch.exp((f[:, None] + g[None, :] - C) / eps)

    err = max(float((P.sum(1) - mu).abs().max()), float((P.sum(0) - nu).abs().max()))
    info = {"eps": float(eps), "cost": float((P * C).sum()),
            "marginal_err": err, "converged": err <= MARGINAL_GATE}
    if anti:
        d = (fA[:, 2][:, None] - fB[None, :, 2]) ** 2 * 4.0
        info["h_residual"] = float((P * d).sum())
    return P, info


def _lift(dest_coarse, full, coarse):
    d = dest_coarse.permute(2, 0, 1)[None]
    d = torch.nn.functional.interpolate(d, size=(full, full), mode="bilinear",
                                        align_corners=True)[0]
    s = full / coarse
    return d.permute(1, 2, 0) * s + (s - 1) / 2


def barycentric_maps(P, full: int = 768, coarse: int = COARSE):
    """Both barycentric projections of one plan, lifted to the full grid.

    dest_AB[r,c] = expected B-location of A-cell (r,c); dest_BA the reverse.
    """
    device = P.device
    coords = torch.stack(torch.meshgrid(
        torch.arange(coarse, device=device), torch.arange(coarse, device=device),
        indexing="ij"), dim=-1).float().reshape(-1, 2)
    dAB = (P @ coords) / (P.sum(dim=1, keepdim=True) + 1e-12)
    dBA = (P.T @ coords) / (P.sum(dim=0).unsqueeze(-1) + 1e-12)
    return (_lift(dAB.reshape(coarse, coarse, 2), full, coarse),
            _lift(dBA.reshape(coarse, coarse, 2), full, coarse))


def optimal_pairing_bidir(elevA, colA, roiA, elevB, colB, roiB,
                          anti: bool = False, full: int = 768, device=None):
    """features -> plan -> both lifted maps. Returns (dest_AB, dest_BA, info)."""
    device = device or get_device()
    fA, mA = coarse_features(elevA, colA, roiA, device=device)
    fB, mB = coarse_features(elevB, colB, roiB, device=device)
    P, info = transport_plan(fA, mA, fB, mB, anti=anti)
    dest_AB, dest_BA = barycentric_maps(P, full=full)
    return dest_AB, dest_BA, info


def sample_grid(field, dest, channels: int, layers: int = 2, side: int = 768):
    """Bilinear-sample a (layers, side, side, channels) tensor at dest
    coordinates, identically per layer. Returns flat (layers*side*side, ch)."""
    g = field.reshape(layers, side, side, channels).permute(0, 3, 1, 2)
    grid = torch.stack([dest[..., 1] / (side - 1) * 2 - 1,
                        dest[..., 0] / (side - 1) * 2 - 1], dim=-1)
    grid = grid[None].expand(layers, -1, -1, -1)
    out = torch.nn.functional.grid_sample(g, grid, mode="bilinear",
                                          padding_mode="border", align_corners=True)
    return out.permute(0, 2, 3, 1).reshape(-1, channels)
