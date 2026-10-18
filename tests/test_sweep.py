import numpy as np
import pytest

from meshkit import Mesh, group
from meshkit.mesh import signed_volume
from meshkit.sweep import catmull_rom, sweep, transport_frames
from conftest import approx, bad_winding_edges, boundary_edges


def closed_ok(m):
    return not boundary_edges(m) and not bad_winding_edges(m) and signed_volume(m.verts, m.faces) > 0


def test_catmull_rom_passes_ends_and_is_even():
    pts = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (3, 2, 1)]
    C, u = catmull_rom(pts, resolution=6)
    assert len(C) == 3 * 6 + 1 and approx(C[0], pts[0]) and approx(C[-1], pts[-1])
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    assert seg.std() / seg.mean() < 0.02
    assert u[0] == 0 and approx(u[-1], 3)
    line, _ = catmull_rom([(0, 0, 0), (0, 0, 4)], resolution=4)
    assert approx(line[:, 2], [0, 1, 2, 3, 4])


def test_straight_sweep_caps_closed():
    m, cr = sweep(Mesh(), [(0, 0, 0), (2, 0, 0)], radius=0.5, segments=8, resolution=4, caps=True, group="rod")
    assert len(cr.rings) == 5 and len(cr.start) == 8 and cr.start.closed
    assert closed_ok(m) and len(m.faces) == 4 * 8 + 2
    octagon = 0.5 * 8 * 0.25 * np.sin(2 * np.pi / 8)
    assert approx(signed_volume(m.verts, m.faces), octagon * 2)
    assert group(m, "rod") == tuple(range(len(m.faces)))


def test_frames_perpendicular_and_twist_free_on_helix():
    t = np.linspace(0, 4 * np.pi, 12)
    pts = np.c_[np.cos(t), np.sin(t), 0.3 * t]
    C, _ = catmull_rom(pts, 8)
    T, R = transport_frames(C)
    assert np.allclose((T * R).sum(1), 0, atol=1e-9)
    assert np.all((R[1:] * R[:-1]).sum(1) > 0.95)  # no flips between consecutive frames
    m, cr = sweep(Mesh(), pts, radius=0.1, segments=6, resolution=8, caps=True)
    assert closed_ok(m)
    for ring, c, tan in zip(cr.rings[::7], C[::7], T[::7]):
        assert np.allclose((m.verts[list(ring)] - c) @ tan, 0, atol=1e-9)


def test_taper_to_point_and_per_point_list():
    m, cr = sweep(Mesh(), [(0, 0, 0), (0, 0, 1), (0, 0.5, 2)], radius=0.2, segments=8, taper=0.0, caps=True)
    assert len(cr.end) == 1 and closed_ok(m)
    m, cr = sweep(Mesh(), [(0, 0, 0), (0, 0, 1), (0, 0, 2)], radius=1.0, segments=8, resolution=2,
                  taper=[1.0, 0.5, 1.0])
    r = [np.linalg.norm(m.verts[list(ring)][:, :2], axis=1).mean() for ring in cr.rings]
    assert approx(r[0], 1) and approx(r[2], 0.5) and approx(r[4], 1)
    with pytest.raises(ValueError):
        sweep(Mesh(), [(0, 0, 0), (0, 0, 1)], taper=[1, 2, 3])


def test_custom_profile_and_up():
    square_cw = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    m, cr = sweep(Mesh(), [(0, 0, 0), (3, 0, 0)], radius=0.1, profile=square_cw, caps=True, resolution=2)
    assert closed_ok(m) and len(cr.start) == 4
    tri = [(0, 0), (1, 0), (0, 1)]  # y follows up: apex at +z
    m, cr = sweep(Mesh(), [(0, 0, 0), (1, 0, 0)], radius=1.0, profile=tri, up="z", resolution=1)
    assert approx(m.verts[list(cr.start)][:, 2].max(), 1.0)


def test_closed_curve_sweep_is_seamless():
    th = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts = np.c_[3 * np.cos(th), 3 * np.sin(th), np.sin(2 * th)]
    m, cr = sweep(Mesh(), pts, radius=0.3, segments=8, resolution=6, closed=True)
    assert closed_ok(m)
    assert len(m.verts) - len(m.edges) + len(m.faces) == 0
    C, _ = catmull_rom(pts, 6, closed=True)
    T, R = transport_frames(C, closed=True)
    assert (R[-1] * R[0]).sum() > 0.95


def test_sweep_appends_to_existing_mesh():
    base, _ = sweep(Mesh(), [(0, 0, 0), (1, 0, 0)], radius=0.1, segments=4, resolution=1)
    m, cr = sweep(base, [(0, 1, 0), (1, 1, 0)], radius=0.1, segments=4, resolution=1)
    assert min(cr.verts) == len(base.verts) and len(m.faces) == 2 * len(base.faces)
