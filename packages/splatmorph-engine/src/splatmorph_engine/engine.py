"""The symmetric two-set morph state and its choreography layers.

Design lineage (verified empirically, gate-checked): transport pairing with
home-owned attributes -> endpoint purity by construction; quadratic Bezier
along the transport map (control point = transport midpoint + elevation
overshoot; the 2(1-a)a weight vanishes at both endpoints); tp-gated shard
tear with per-splat mismatch modulation (tearing concentrates where the
correspondence is strained); curl-of-depth detour; velocity-aligned streak;
fracture-masked opacity handoff with an endpoint-pure schedule.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from splatmorph_fields import norm01
from splatmorph_transport import sample_grid

K_OVERSHOOT = 0.25
EDGE_BIAS = 1.0
SHARD_CYCLES = 2
SHARD_BASE, SHARD_OVER, LEAD = 0.35, 1.6, 1.2
CURL_AMP = 0.06
STREAK = 3.0
COUPLE_SIGMA = 2.5

# --- absolute choreography scaling -------------------------------------
# Every distortion amplitude used to be normalized by its OWN 90th
# percentile, which made it scale-invariant: a hop between near-identical
# scenes got the same storm as a hop between unrelated ones. Measured on
# real scenes (near tour hop meshy_01->meshy_29 vs the emerald->mesh demo
# pair): travel differed 5.5x while shard displacement differed only 1.24x,
# so distortion-per-unit-motion was 4x HIGHER for the near pair.
# These reference values are taken from that measurement.
STREAK_REF = 25.0      # per-frame speed, in splat widths, for full streaking
#                        (measured: near hop 3.5, far pair 27.7)
TRAVEL_REF = 0.25      # mean transport travel for full shard amplitude
#                        (measured: near hop 0.046, far pair 0.254)
SHARD_FLOOR = 0.0      # tearing when correspondence strain is zero
ABSOLUTE_SCALING = True   # False restores the original self-normalized behaviour

_ZDIR = [0.0, 0.0, -1.0]
_EX = [1.0, 0.0, 0.0]


def _kernel(device):
    r = int(3 * COUPLE_SIGMA)
    k = torch.exp(-torch.arange(-r, r + 1, device=device, dtype=torch.float32) ** 2
                  / (2 * COUPLE_SIGMA ** 2))
    return k / k.sum(), r


def radial_couple(f, side: int = 768):
    """Separable Gaussian coupling applied AFTER nonlinearities, so adjacent
    splats get spatially continuous tear timing."""
    k1d, r = _kernel(f.device)
    x = f.reshape(2, 1, side, side)
    x = torch.nn.functional.conv2d(x, k1d.view(1, 1, 1, -1), padding=(0, r))
    x = torch.nn.functional.conv2d(x, k1d.view(1, 1, -1, 1), padding=(r, 0))
    return x.reshape(-1)


def _grid_field(a, device):
    return torch.from_numpy(np.tile(a.astype(np.float32)[None], (2, 1, 1))
                            .reshape(-1)).to(device)


def _vec2_field(a, device):
    return torch.from_numpy(np.tile(a.astype(np.float32)[None], (2, 1, 1, 1))
                            ).reshape(-1, 2).to(device)


def quat_from_x_to(d, ex):
    c = (d @ ex).clamp(-1 + 1e-6, 1 - 1e-6)
    axis = torch.linalg.cross(ex.expand_as(d), d)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    half = torch.acos(c) * 0.5
    return torch.cat([torch.cos(half)[:, None], torch.sin(half)[:, None] * axis], dim=-1)


def build_set(home, away, dest, forward: bool, cn: float, anti: bool = False):
    """One traveling set: home splats with their OWN attributes; away-side
    positions and fields sampled at the home splats' optimal destinations."""
    dv = home.device
    s = {"forward": forward}
    s["p_home"] = home.gs.mean_vectors[0]
    s["p_away"] = sample_grid(away.gs.mean_vectors[0], dest, 3)
    s["q"] = home.gs.quaternions[0]
    s["sv"] = home.gs.singular_values[0]
    s["op"] = home.gs.opacities[0]
    s["col"] = home.gs.colors[0]
    s["elev_home"] = _grid_field(home.elev, dv)
    s["elev_away"] = sample_grid(torch.from_numpy(np.tile(away.elev[None], (2, 1, 1)))
                                 .float().reshape(-1, 1).to(dv), dest, 1)[:, 0]
    s["D_home"] = _grid_field(home.D, dv)
    s["D_away"] = sample_grid(torch.from_numpy(np.tile(away.D[None], (2, 1, 1))).float()
                              .reshape(-1, 1).to(dv), dest, 1)[:, 0]
    s["curl_home"] = _vec2_field(home.curl / cn, dv)
    s["curl_away"] = sample_grid(_vec2_field(away.curl / cn, dv), dest, 2)
    s["span_home"], s["span_away"] = home.span, away.span
    s["sp_phase"] = _grid_field(np.pi * norm01(gaussian_filter(home.elev, 4)), dv)
    D_sum = torch.abs(s["D_home"] + s["D_away"])
    s["disp_scale"] = torch.quantile(D_sum, 0.90) + 1e-9
    s["shard_w"] = torch.clamp(D_sum / s["disp_scale"], 0, 1)
    # nonhomogeneous distortion: per-splat strain vs the matched counterpart
    # (deviation from perfect ANTI-match when anti), neighbor-coupled
    col_away = sample_grid(away.gs.colors[0], dest, 3)
    d_e = (s["elev_home"] + s["elev_away"] - 1.0) if anti \
        else (s["elev_home"] - s["elev_away"])
    d_c = (s["col"] - col_away).pow(2).sum(-1)
    mm = radial_couple(torch.sqrt(4.0 * d_e ** 2 + 4.0 * d_c))
    s["mm"] = torch.clamp(mm / (torch.quantile(mm, 0.90) + 1e-9), 0, 1)
    # absolute gate: how far matter ACTUALLY travels on this hop, against a
    # fixed reference — near-identical scenes barely move, so they barely tear
    travel = (s["p_away"] - s["p_home"]).norm(dim=-1).mean()
    s["travel_gate"] = float(torch.clamp(travel / TRAVEL_REF, 0, 1)) \
        if ABSOLUTE_SCALING else 1.0
    return s


def drive_periodic(ph: int, T: int):
    """A -> B -> A exact loop: m and the shard wave share the period."""
    m = 0.5 - 0.5 * float(np.cos(2 * np.pi * ph / T))
    return m, 2 * np.pi * SHARD_CYCLES * ph / T


def drive_oneway(i: int, T: int):
    """One-way segment, zero velocity at both ends (chain joins)."""
    sfrac = i / T
    m = 0.5 - 0.5 * float(np.cos(np.pi * sfrac))
    return m, 2 * np.pi * SHARD_CYCLES * sfrac


def set_state(s, m: float, wave_phase: float, span_mix: float, rough: float = 1.0):
    """Position + coupled shard displacement for one set at morph time m."""
    dv = s["p_home"].device
    zdir = torch.tensor(_ZDIR, device=dv)
    tp = 1 - abs(2 * m - 1)
    mt = torch.tensor(m, device=dv, dtype=torch.float32)
    a = mt if s["forward"] else (1 - mt)
    # quadratic Bezier along the transport map: control point = transport
    # midpoint lifted by the elevation overshoot; endpoint purity is native
    lift = 0.5 * (s["elev_home"] * s["span_home"] + s["elev_away"] * s["span_away"])
    ctrl = 0.5 * (s["p_home"] + s["p_away"]) \
        + (lift * K_OVERSHOOT * 2.0 * rough).unsqueeze(-1) * zdir
    pos = (1 - a) ** 2 * s["p_home"] + 2 * (1 - a) * a * ctrl + a ** 2 * s["p_away"]
    if s["forward"]:
        D_m = s["D_home"] * (1 - mt) + s["D_away"] * mt
        curl_m = s["curl_home"] * (1 - mt) + s["curl_away"] * mt
    else:
        D_m = s["D_away"] * (1 - mt) + s["D_home"] * mt
        curl_m = s["curl_away"] * (1 - mt) + s["curl_home"] * mt
    wave = torch.sin(torch.tensor(wave_phase, device=dv, dtype=torch.float32)
                     + LEAD * tp + s["sp_phase"])
    floor = SHARD_FLOOR if ABSOLUTE_SCALING else 0.35
    disp = radial_couple(D_m * wave) * ((SHARD_BASE + SHARD_OVER * tp) * tp) \
        * span_mix * rough * (floor + (1.0 - floor) * s["mm"]) \
        * s.get("travel_gate", 1.0)
    pos = pos + disp.unsqueeze(-1) * zdir
    pos = pos + torch.cat([curl_m * (CURL_AMP * rough * tp * span_mix),
                           torch.zeros_like(disp)[:, None]], dim=-1)
    return pos, disp, tp, mt


def set_frame(s, m, wave_phase, m_prev, wave_phase_prev, span_mix, rough: float = 1.0):
    """Full per-frame attributes for one set (opacity handoff, streaks)."""
    dv = s["p_home"].device
    ex = torch.tensor(_EX, device=dv)
    pos, disp, tp, mt = set_state(s, m, wave_phase, span_mix, rough)
    edge = torch.tanh(2.0 * torch.abs(disp) / (s["disp_scale"] * span_mix))
    k = torch.clamp(mt * (1 + EDGE_BIAS * 0.6) - (1 - edge) * (EDGE_BIAS * 0.6), 0, 1)
    k = k * k * (3 - 2 * k)                        # endpoint-pure handoff
    op = s["op"] * ((1 - k) if s["forward"] else k) * (1 - 0.15 * min(rough, 1.0) * tp)
    q, sv = s["q"], s["sv"]
    pos_prev, _, _, _ = set_state(s, m_prev, wave_phase_prev, span_mix, rough)
    v = pos - pos_prev
    vmag = v.norm(dim=-1)
    s_geo = sv.prod(dim=-1).pow(1 / 3)
    # a splat streaks in proportion to how far it moves relative to its OWN
    # width — an absolute criterion. The old form divided by the frame's own
    # 90th-percentile speed, which discarded exactly that information and
    # saturated the streak no matter how slowly anything was moving.
    speed = vmag / (s_geo * STREAK_REF + 1e-12) if ABSOLUTE_SCALING \
        else vmag / (torch.quantile(vmag, 0.90) + 1e-9)
    w = tp * s["shard_w"] * torch.clamp(speed, 0, 1)
    vdir = v / (vmag.unsqueeze(-1) + 1e-9)
    q_vel = quat_from_x_to(vdir, ex)
    q_vel = torch.where((q * q_vel).sum(-1, keepdim=True) < 0, -q_vel, q_vel)
    q = torch.nn.functional.normalize(q * (1 - w)[:, None] + q_vel * w[:, None], dim=-1)
    sv_streak = torch.stack([s_geo * (1 + STREAK), s_geo * 0.6, s_geo * 0.6], dim=-1)
    sv = sv * (1 - w)[:, None] + sv_streak * w[:, None]
    return pos, q, sv, s["col"], op


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


class Renderer:
    """gsplat renderer bound to a scene's camera. Requires CUDA + sharp."""

    def __init__(self, meta, device):
        from sharp.utils import gsplat as sharp_gsplat   # lazy: CUDA
        W, H = meta.resolution_px
        f_px = meta.focal_length_px
        self.W, self.H = W, H
        self.intrinsics = torch.tensor(
            [[f_px, 0, (W - 1) / 2, 0], [0, f_px, (H - 1) / 2, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32)
        self.extrinsics = torch.eye(4, device=device, dtype=torch.float32)
        self._r = sharp_gsplat.GSplatRenderer(color_space=meta.color_space)

    def __call__(self, gs):
        out = self._r(gs, extrinsics=self.extrinsics[None],
                      intrinsics=self.intrinsics[None],
                      image_width=self.W, image_height=self.H)
        return (out.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    def render_sets(self, sets, m, wave_phase, m_prev, wp_prev, span_mix, rough=1.0):
        from sharp.utils.gaussians import Gaussians3D    # lazy
        parts = [set_frame(s, m, wave_phase, m_prev, wp_prev, span_mix, rough)
                 for s in sets]
        gs = Gaussians3D(
            mean_vectors=torch.cat([p[0] for p in parts])[None],
            singular_values=torch.cat([p[2] for p in parts])[None],
            quaternions=torch.cat([p[1] for p in parts])[None],
            colors=torch.cat([p[3] for p in parts])[None],
            opacities=torch.cat([p[4] for p in parts])[None])
        return self(gs)
