"""CPU tests for the engine's loop/endpoint mathematics."""

import torch

from splatmorph_engine import drive_oneway, drive_periodic


def test_periodic_drive_is_exactly_periodic():
    T = 96
    m0, w0 = drive_periodic(0, T)
    mT, wT = drive_periodic(T, T)
    assert m0 == 0.0
    assert abs(mT - m0) < 1e-12
    m_half, _ = drive_periodic(T // 2, T)
    assert abs(m_half - 1.0) < 1e-12


def test_oneway_drive_has_still_endpoints():
    T = 64
    assert drive_oneway(0, T)[0] == 0.0
    assert abs(drive_oneway(T, T)[0] - 1.0) < 1e-12
    # velocity ~ 0 at both ends (half-cosine)
    v0 = drive_oneway(1, T)[0] - drive_oneway(0, T)[0]
    vmid = drive_oneway(T // 2 + 1, T)[0] - drive_oneway(T // 2, T)[0]
    assert v0 < vmid / 5


def test_quadratic_bezier_endpoint_purity():
    """The 2(1-a)a control weight vanishes at both endpoints regardless of
    the control point — endpoint purity needs no gating."""
    p0 = torch.randn(10, 3)
    p1 = torch.randn(10, 3)
    ctrl = 0.5 * (p0 + p1) + torch.randn(10, 3) * 5.0    # arbitrary lift
    for a, expect in [(0.0, p0), (1.0, p1)]:
        pos = (1 - a) ** 2 * p0 + 2 * (1 - a) * a * ctrl + a ** 2 * p1
        assert torch.allclose(pos, expect)


def test_endpoint_pure_opacity_handoff():
    """The fracture-masked handoff k must be exactly 0 at m=0 and 1 at m=1
    for every edge value — the schedule that makes both endpoints pure."""
    edge = torch.linspace(0, 1, 11)
    for m, expect in [(0.0, 0.0), (1.0, 1.0)]:
        k = torch.clamp(torch.tensor(m) * (1 + 0.6) - (1 - edge) * 0.6, 0, 1)
        k = k * k * (3 - 2 * k)
        assert torch.allclose(k, torch.full_like(k, expect))
