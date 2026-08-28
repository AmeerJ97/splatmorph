"""Depth-derived field analysis for image-aligned splat grids.

Every downstream stage — transport pairing, choreography, membership —
consumes only the scalar/vector fields computed here: normalized relative
elevation (peak = 1, trough = 0, self-calibrating per image), the
glare-fused shard displacement field (band-passed image detail carried by
depth-band energy), curl-of-depth solenoidal flow, and the luminance ROI.
"""

from .fields import analyze, curl_field, norm01, fields_from_ply

__all__ = ["analyze", "curl_field", "norm01", "fields_from_ply"]
