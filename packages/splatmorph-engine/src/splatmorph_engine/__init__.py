"""Symmetric two-set Gaussian splat morph engine.

One transport plan yields both barycentric maps; scene A's splats travel
out along one while scene B's travel home along the other, each set keeping
its own attributes — so BOTH endpoints reproduce the pure scenes exactly,
by construction, and the crossover is carried by a fracture-masked opacity
handoff rather than color blending. Choreography (shard tear, curl detour,
velocity streaks) rides the transport as loop-periodic decoration.

Rendering requires CUDA (gsplat via the vendored SHARP runtime); the state
mathematics are device-agnostic and unit-testable on CPU.
"""

from .scene import Scene, ensure_gaussians, mirror_image
from .engine import (
    build_set,
    set_state,
    set_frame,
    drive_periodic,
    drive_oneway,
    radial_couple,
    psnr,
    Renderer,
)

__all__ = [
    "Scene", "ensure_gaussians", "mirror_image",
    "build_set", "set_state", "set_frame",
    "drive_periodic", "drive_oneway", "radial_couple", "psnr", "Renderer",
]
