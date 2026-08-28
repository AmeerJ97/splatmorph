"""End-to-end pipelines: two-image loop, and the full chain
(filtration -> ordering -> frame allocation -> two-set rendering), every
stage driven by the same transport functional and gate-verified.
"""

import itertools
from pathlib import Path

import numpy as np
import torch

from splatmorph_engine import (
    Scene, build_set, drive_oneway, drive_periodic, mirror_image, psnr, Renderer,
)
from splatmorph_transport import optimal_pairing_bidir, transport_plan

TS_MIN, TS_MAX = 24, 160
ROUGH_LO, ROUGH_HI = 0.35, 1.6


def _pair_ctx(A, B):
    cn = max(np.percentile(np.abs(A.curl), 98), np.percentile(np.abs(B.curl), 98)) + 1e-9
    return cn, 0.5 * (A.span + B.span)


def render_two(image_a: Path, image_b: Path, cache: Path, frames: int = 96,
               anti: bool = False, weights: Path | None = None, log=print):
    """A -> B -> A exact periodic loop. Returns (frames array, gates dict)."""
    A = Scene(image_a, cache, weights)
    B = Scene(image_b, cache, weights, device=A.device)
    dest_AB, dest_BA, info = optimal_pairing_bidir(
        A.elev, A.col_grid, A.roi, B.elev, B.col_grid, B.roi, anti=anti)
    cn, span_mix = _pair_ctx(A, B)
    sets = [build_set(A, B, dest_AB, True, cn, anti),
            build_set(B, A, dest_BA, False, cn, anti)]
    r = Renderer(A.meta, A.device)
    pure_a, pure_b = r(A.gs), r(B.gs)
    T = frames
    out = []
    for i in range(T + 1):
        m, wp = drive_periodic(i % T, T)
        mp, wpp = drive_periodic((i % T - 1) % T, T)
        out.append(r.render_sets(sets, m, wp, mp, wpp, span_mix))
        if i % 24 == 0:
            log(f"  frame {i}/{T}")
    out = np.stack(out)
    m_mid, wp_mid = drive_periodic(T // 2, T)
    gates = {
        "solver_marginal_err": info["marginal_err"],
        "loop_psnr": psnr(out[0], out[T]),
        "endpoint_a_psnr": psnr(out[0], pure_a),
        "endpoint_b_psnr": psnr(out[T // 2], pure_b),
        "mid_control_psnr": psnr(out[0], out[T // 2]),
    }
    return out[:T], gates


def _cycle_order(dist, n):
    if n <= 9:
        best, best_c = None, np.inf
        for perm in itertools.permutations(range(1, n)):
            order = (0,) + perm
            c = sum(dist[order[i], order[(i + 1) % n]] for i in range(n))
            if c < best_c:
                best_c, best = c, list(order)
        return best
    order = [0]
    todo = set(range(1, n))
    while todo:
        nxt = min(todo, key=lambda j: dist[order[-1], j])
        order.append(nxt)
        todo.discard(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c_, e = order[j], order[(j + 1) % n]
                if dist[a, c_] + dist[b, e] < dist[a, b] + dist[c_, e] - 1e-12:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def _membership(scenes, D, anti, log):
    """Holistic unanimity: two independent channels — transport isolation
    (computed under the SAME pairing mode the chain uses) and perceptual
    LPIPS — an image drops only when unanimously worst, iterated (< n/3).
    Requires the lpips extra; without it, filtration is skipped."""
    n = len(scenes)
    try:
        import lpips as lpips_mod
        from PIL import Image
    except ImportError:
        log("  (lpips not installed — filtration skipped; pip install splatmorph[lpips])")
        return []
    dv = scenes[0].device
    lp = lpips_mod.LPIPS(net="alex", verbose=False).to(dv)
    th = []
    for s in scenes:
        a = np.asarray(Image.open(s.image_path).convert("RGB").resize((256, 256)),
                       np.float32) / 255
        th.append(torch.from_numpy(a).permute(2, 0, 1)[None].to(dv) * 2 - 1)
    L = np.zeros((n, n))
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                L[i, j] = L[j, i] = float(lp(th[i], th[j]))
    excluded, active = [], list(range(n))
    while len(excluded) < n // 3:
        sub = np.ix_(active, active)
        iso = (D[sub] + np.eye(len(active)) * 1e9).min(axis=1)
        lmin = (L[sub] + np.eye(len(active)) * 1e9).min(axis=1)
        wi, wl = int(np.argmax(iso)), int(np.argmax(lmin))
        if wi != wl:
            break
        excluded.append(active[wi])
        active.pop(wi)
    for i in range(n):
        im = (D[i] + (np.arange(n) == i) * 1e9).min()
        lm = (L[i] + (np.arange(n) == i) * 1e9).min()
        flag = "  EXCLUDED (unanimous)" if i in excluded else ""
        log(f"  {scenes[i].name}: transport-iso {im:.3f}  lpips-min {lm:.3f}{flag}")
    return excluded


def render_chain(images: list[Path], cache: Path, frames_per_seg: int = 64,
                 anti: bool = False, auto_mirror: bool = False,
                 weights: Path | None = None, log=print):
    """Full pipeline. Returns (frames array, gates dict, order names)."""
    images = list(images)
    if auto_mirror:
        mdir = images[0].parent / "_mirrors"
        images += [mirror_image(p, mdir) for p in images]
        log(f"auto-mirror: cast doubled to {len(images)} (peak-trough counterparts)")

    log("== scenes")
    scenes = []
    for p in images:
        scenes.append(Scene(p, cache, weights,
                            device=scenes[0].device if scenes else None))
        log(f"  {p.name}")

    log("== pairwise transport" + (" (anti)" if anti else ""))
    n = len(scenes)
    D = np.zeros((n, n))
    max_err = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            _, info = transport_plan(*scenes[i].feats, *scenes[j].feats, anti=anti)
            D[i, j] = D[j, i] = info["cost"]
            max_err = max(max_err, info["marginal_err"])

    log("== membership (holistic unanimity)")
    excluded = _membership(scenes, D, anti, log)
    keep = [i for i in range(n) if i not in excluded]
    scenes = [scenes[i] for i in keep]
    D = D[np.ix_(keep, keep)]
    n = len(scenes)

    log("== ordering (shortest Hamiltonian cycle on transport cost)")
    order = _cycle_order(D, n)
    seg_cost = [float(D[order[k], order[(k + 1) % n]]) for k in range(n)]
    scenes = [scenes[i] for i in order]
    log("  " + " -> ".join(s.name for s in scenes))

    mean_c = float(np.mean(seg_cost)) + 1e-12
    med_c = float(np.median(seg_cost)) + 1e-12
    ts_list = [int(np.clip(round(frames_per_seg * c / mean_c), TS_MIN, TS_MAX))
               for c in seg_cost]

    r = Renderer(scenes[0].meta, scenes[0].device)
    pure = [r(s.gs) for s in scenes]

    all_frames, arrivals, bsteps, seg_meds = [], [], [], []
    for seg in range(n):
        A, B = scenes[seg], scenes[(seg + 1) % n]
        Tseg = ts_list[seg]
        rough = float(np.clip(seg_cost[seg] / med_c, ROUGH_LO, ROUGH_HI))
        log(f"== segment {seg}: {A.name} -> {B.name} ({Tseg} frames, rough {rough:.2f})")
        dest_AB, dest_BA, _ = optimal_pairing_bidir(
            A.elev, A.col_grid, A.roi, B.elev, B.col_grid, B.roi, anti=anti)
        cn, span_mix = _pair_ctx(A, B)
        sets = [build_set(A, B, dest_AB, True, cn, anti),
                build_set(B, A, dest_BA, False, cn, anti)]
        i0 = 0 if seg == 0 else 1
        intra = []
        for i in range(i0, Tseg + 1):
            m, wp = drive_oneway(i, Tseg)
            mp, wpp = drive_oneway(max(i - 1, 0), Tseg)
            frame = r.render_sets(sets, m, wp, mp, wpp, span_mix, rough)
            if i == i0 and seg > 0:
                bsteps.append(psnr(all_frames[-1], frame))
            elif i > i0:
                intra.append(psnr(all_frames[-1], frame))
            if i == Tseg:
                arrivals.append(psnr(frame, pure[(seg + 1) % n]))
            all_frames.append(frame)
        seg_meds.append(float(np.median(intra)))
        del sets

    closure = psnr(all_frames[-1], all_frames[0])
    all_frames = all_frames[:-1]                    # drop duplicate of frame 0
    gates = {
        "solver_marginal_err": max_err,
        "arrival_purity_min": min(arrivals),
        "boundaries_quieter_than_motion": all(
            b >= seg_meds[k + 1] for k, b in enumerate(bsteps)),
        "closure_psnr": closure,
        "excluded": [images[i].name for i in excluded] if excluded else [],
    }
    return np.stack(all_frames), gates, [s.name for s in scenes]


def verify(gates: dict, log=print) -> bool:
    """The frozen quality gates, printed and judged. Returns overall pass."""
    checks = []
    if "loop_psnr" in gates:                        # two-image mode
        checks = [
            ("solver marginals <= 1e-3", gates["solver_marginal_err"] <= 1e-3),
            ("loop exact (>= 85 dB)", gates["loop_psnr"] >= 85),
            ("endpoint A pure (>= 50 dB)", gates["endpoint_a_psnr"] >= 50),
            ("endpoint B pure (>= 50 dB)", gates["endpoint_b_psnr"] >= 50),
        ]
    else:                                           # chain mode
        checks = [
            ("solver marginals <= 1e-3", gates["solver_marginal_err"] <= 1e-3),
            ("arrival purity (>= 80 dB)", gates["arrival_purity_min"] >= 80),
            ("boundaries quieter than motion", gates["boundaries_quieter_than_motion"]),
            ("loop closure (>= 80 dB)", gates["closure_psnr"] >= 80),
        ]
    ok = True
    for name, passed in checks:
        log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    return ok
