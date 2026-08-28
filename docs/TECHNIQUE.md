# The technique, one page

## Lift

Each image becomes ~1.18M image-aligned 3D Gaussians (two 768×768 layers)
via SHARP — position, scale, rotation, color, opacity per splat, real
monocular depth. From the depth and the image we derive per-scene fields
(`splatmorph-fields`): normalized relative elevation (peak = 1, trough = 0,
self-calibrating), a glare-fused shard displacement field (band-passed image
detail carried by depth-band energy), and a curl-of-depth solenoidal flow.

## Pair

Correspondence is optimal transport, not grid location
(`splatmorph-transport`). Ground cost per coarse cell: appearance ⊕
elevation (each with a Gaussian neighborhood-context copy — cells match on
themselves *and* their surroundings) ⊕ a gentle quadratic position term ⊕
an **exponential barrier** past the horizontal separation a trajectory can
organically absorb. Solved with log-domain Sinkhorn (marginal convergence
is a hard gate); both barycentric projections of the one plan give the
A→B and B→A maps.

**Anti-pairing** (`--wave`): peaks pair with troughs. In signed height the
cost is (h_A + h_B)², which in [0,1] elevation space is a squared
difference against a height-flipped counterpart — one sign flip, machinery
unchanged. Mirrored copies of each scene supply every crest a trough at
its own position; the ordering optimizer then *discovers* crest↔trough
alternation because it is the cheapest transport.

## Move

The symmetric two-set formulation (`splatmorph-engine`) is what makes
endpoints exact: scene A's splats travel outward along the A→B map while
scene B's travel home along B→A, each set keeping its **own** attributes.
At either end of a segment exactly one set is visible and it *is* the pure
scene — endpoint purity by construction, verified at ≥ 104 dB.

Each splat's base path is a quadratic Bézier whose control point is the
transport midpoint lifted by the elevation overshoot; the 2(1−a)a weight
vanishes at both endpoints, so no gating is needed. On top ride the
choreography layers, all periodic and endpoint-silent: the radially-coupled
shard tear (amplitude ∝ per-splat correspondence strain — tearing
concentrates exactly where matter must genuinely change), the curl detour,
velocity-aligned streaking, and a fracture-masked **opacity** handoff
(colors never blend; identity is carried by geometry).

## Compose

For an image set (`splatmorph` CLI): membership by holistic unanimity
(transport isolation under the pairing mode in use + perceptual LPIPS —
drop only what both channels independently rank worst), ordering by
shortest Hamiltonian cycle on transport cost, frame budget per hop ∝ its
cost (constant perceptual speed — long hops diffuse, near-twins snap), and
per-segment distortion ∝ hop roughness. Segments render *through* their
arrival frame, so joins are exact and the grand loop closes bit-identically.

## Verify

Every render prints its gates: solver marginals ≤ 1e-3, loop exactness,
endpoint/arrival purity, boundaries quieter than each segment's own median
motion. `--strict` turns any failure into a nonzero exit — the same checks
run as CPU-miniature CI on every commit.
