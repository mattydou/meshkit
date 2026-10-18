import numpy as np
import pytest

from meshkit import Mesh, check, sweep
from meshkit.sweep import joint_path


def line_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    v = P - a
    return np.linalg.norm(v - (v @ d)[:, None] * d, axis=1)


L_PATH = [(0, 0, 0), (2, 0, 0), (2, 2, 0)]


def test_mitered_joint_is_exact():
    m, cr = sweep(Mesh(), L_PATH, radius=0.2, segments=8, resolution=3, caps=True, path="joints")
    assert check(m, closed=True).ok
    ring = m.verts[list(cr.joints[1])]
    assert np.allclose(line_dist(ring, (0, 0, 0), (2, 0, 0)), 0.2)  # on the incoming tube
    assert np.allclose(line_dist(ring, (2, 0, 0), (2, 2, 0)), 0.2)  # and on the outgoing tube
    bis = np.array([1.0, 1.0, 0]) / np.sqrt(2)
    assert np.allclose((ring - (2, 0, 0)) @ bis, 0)  # in the bisector plane
    seg = [m.verts[list(r)] for r in cr.rings[:3]]
    assert all(np.allclose(line_dist(r, (0, 0, 0), (2, 0, 0)), 0.2) for r in seg)  # straight, constant section


def test_rounded_joint_arc():
    m, cr = sweep(Mesh(), L_PATH, radius=0.1, segments=8, resolution=2, path="joints", joint_radius=0.5, caps=True)
    assert check(m, closed=True).ok
    C, T, u, miter = joint_path(L_PATH, 2, joint_radius=0.5)
    arc = C[(np.abs(C[:, 0] - 2) > 1e-9) & (np.abs(C[:, 1]) > 1e-9)]
    assert len(arc) >= 2 and np.allclose(np.linalg.norm(arc - (1.5, 0.5, 0), axis=1), 0.5)
    assert np.all((T[1:] * T[:-1]).sum(1) > 0.9)  # tangents turn gradually
    assert all(mi is None for mi in miter)
    C2, *_ = joint_path(L_PATH, 2, joint_radius=50.0)  # clamped to half the shorter segment
    assert np.isfinite(C2).all() and len(C2) > len(joint_path(L_PATH, 2)[0])


def test_closed_jointed_frame_and_taper_per_joint():
    square = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    m, cr = sweep(Mesh(), square, radius=0.1, segments=6, resolution=1, closed=True, path="joints")
    r = check(m, closed=True)
    assert r.ok and r.stats["chi"] == 0 and len(cr.joints) == 4
    m, cr = sweep(Mesh(), [(0, 0, 0), (0, 0, 1), (0, 1, 2)], radius=1.0, segments=8, resolution=2,
                  taper=[1.0, 0.5, 0.25], path="joints")
    widths = [np.linalg.norm(m.verts[list(j)] - m.verts[list(j)].mean(0), axis=1).min() for j in cr.joints]
    assert np.allclose(widths, [1.0, 0.5, 0.25], atol=1e-9)


def test_joint_errors():
    with pytest.raises(ValueError, match="turns back"):
        sweep(Mesh(), [(0, 0, 0), (1, 0, 0), (0, 0.01, 0)], path="joints")
    with pytest.raises(ValueError, match="coincide"):
        sweep(Mesh(), [(0, 0, 0), (0, 0, 0), (1, 0, 0)], path="joints")
    with pytest.raises(ValueError):
        sweep(Mesh(), L_PATH, path="wiggly")
