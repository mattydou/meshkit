import numpy as np
import pytest

from meshkit import (add_group, box, capsule, check, cone, decimate, displace_sdf, ellipsoid, grid, group,
                     intersect, noise, quad_sphere, revolve, select_by_plane, smooth_union, sphere, subtract,
                     to_mesh, union)
from meshkit.mesh import signed_volume
from meshkit.query import boundary_loops


def test_primitive_distances():
    assert np.isclose(sphere(1, center=1)((1, 1, 3)), 1.0)
    assert np.allclose(box(1)(np.array([(1, 0, 0), (0, 0, 0), (1, 1, 0)])), [0.5, -0.5, np.hypot(0.5, 0.5)])
    assert np.isclose(box(2, round=0.5)((1, 1, 0)), np.hypot(0.5, 0.5) - 0.5)
    e = ellipsoid((2, 1, 0.5))
    assert np.allclose(e(np.array([(2, 0, 0), (0, 1, 0), (0, 0, 0.5)])), 0, atol=1e-12) and e((0, 0, 0)) < 0
    c = capsule((0, 0, 0), (0, 0, 2), 0.5)
    assert np.allclose(c(np.array([(1, 0, 1), (0, 0, 3), (0, 0, 1)])), [0.5, 0.5, -0.5])
    assert np.isclose(capsule((0, 0, 0), (0, 0, 2), 1.0, radius_b=0.0)((0, 0, 3)), 1.0)
    k = cone((0, 0, 0), (0, 0, 1), 1.0, 0.0)
    assert np.isclose(k((0, 0, 2)), 1.0) and np.isclose(k((0, 0, -1)), 1.0) and k((0, 0, 0.2)) < 0
    assert np.isclose(k((2, 0, 0)), 1.0)


def test_csg_semantics_and_bounds():
    a, b = sphere(1), sphere(1, center=(1.5, 0, 0))
    P = np.array([(-0.5, 0, 0), (2, 0, 0), (0.75, 0, 0), (0.75, 0.75, 0)])
    assert np.allclose(union(a, b)(P), np.minimum(a(P), b(P)))
    assert np.all(smooth_union(a, b, k=0.3)(P) <= union(a, b)(P) + 1e-12)
    assert smooth_union(a, b, k=0.3)((0.75, 0.75, 0)) < 0 < union(a, b)((0.75, 0.75, 0))  # fillet adds material
    assert np.isclose(smooth_union(a, b, k=0.3)((-0.9, 0, 0)), a((-0.9, 0, 0)))  # far from the seam: unchanged
    d = subtract(a, b)
    assert d((0.75, 0, 0)) > 0 and d((-0.5, 0, 0)) < 0
    assert intersect(a, b)((0.75, 0, 0)) < 0 and intersect(a, b)((-0.5, 0, 0)) > 0
    assert np.allclose(union(a, b).lo, (-1, -1, -1)) and np.allclose(union(a, b).hi, (2.5, 1, 1))
    assert np.allclose(smooth_union(a, b, k=0.3).hi, (2.8, 1.3, 1.3))
    n = displace_sdf(a, noise(strength=0.2))
    assert np.allclose(n.hi, 1.2)
    with pytest.raises(ValueError):
        displace_sdf(a, lambda P: P[:, 0])
    assert repr(a).startswith("SDF(sphere bounds=")


def test_to_mesh_sphere_closed_accurate_deterministic():
    m = to_mesh(sphere(1.0), resolution=40)
    r = check(m, closed=True)
    assert r.ok and r.stats["comps"] == 1 and r.stats["chi"] == 2
    h = 2.0 / 40
    assert np.all(np.abs(np.linalg.norm(m.verts, axis=1) - 1) < 0.1 * h)
    assert abs(signed_volume(m.verts, m.faces) / (4 / 3 * np.pi) - 1) < 0.02
    m2 = to_mesh(sphere(1.0), resolution=40)
    assert np.array_equal(m.verts, m2.verts) and m.faces == m2.faces


def test_to_mesh_keeps_largest_piece_and_checks_bounds():
    two = union(sphere(1), sphere(0.5, center=(3, 0, 0)))
    one = to_mesh(two, 40, group="blob")
    assert check(one).stats["comps"] == 1 and one.verts[:, 0].max() < 1.1
    assert len(group(one, "blob")) == len(one.faces)
    assert check(to_mesh(two, 40, keep="all"), contacts=None).stats["comps"] == 2
    with pytest.raises(ValueError, match="sampling bounds"):
        to_mesh(sphere(1), 20, bounds=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    with pytest.raises(ValueError, match="no inside"):
        to_mesh(sphere(1, center=10), 20, bounds=((-1, -1, -1), (1, 1, 1)))


def test_organic_blend_with_noise_is_one_closed_shell():
    body = smooth_union(ellipsoid((1.2, 0.7, 0.6)), sphere(0.45, center=(1.3, 0, 0.35)),
                        capsule((-0.9, 0, 0), (-1.8, 0, 0.6), 0.25, 0.08), k=0.25)
    m = to_mesh(displace_sdf(body, noise(seed=3, scale=0.4, strength=0.05)), resolution=56)
    r = check(m, closed=True)
    assert r.ok and r.stats["comps"] == 1


def test_decimate_keeps_closed_topology_and_groups():
    s = quad_sphere(1, 12)
    s, _ = add_group(s, "east", select_by_plane(s, 0, "x", kind="f"))
    s, _ = add_group(s, "pole", [int(np.argmax(s.verts[:, 2]))], kind="v")
    d, cr = decimate(s, 300)
    r = check(d, closed=True)
    assert r.ok and len(d.faces) <= 300 and r.stats["chi"] == 2 and r.stats["tris"] == len(d.faces)
    assert cr.faces_before == len(s.faces) and cr.faces_after == len(d.faces)
    assert abs(signed_volume(d.verts, d.faces) / signed_volume(s.verts, s.faces) - 1) < 0.05
    east = group(d, "east")
    assert 0.3 * len(d.faces) < len(east) < 0.7 * len(d.faces) and np.all(d.face_centers[list(east)][:, 0] > -0.2)
    assert len(group(d, "pole")) == 1 and cr.vmap.max() == len(d.verts) - 1
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    torus = revolve(np.c_[2 + 0.5 * np.cos(th), 0.5 * np.sin(th)], segments=24, closed=True)
    t, _ = decimate(torus, 150)
    assert check(t, closed=True).ok and check(t).stats["chi"] == 0


def test_decimate_locks_open_borders():
    g = grid(2, 10)
    d, cr = decimate(g, 40)
    (before,), (after,) = boundary_loops(g), boundary_loops(d)
    assert len(after) == len(before) and np.allclose(sorted(map(tuple, d.verts[list(after)])),
                                                     sorted(map(tuple, g.verts[list(before)])))
    assert check(d).ok and len(d.faces) <= 2 * len(g.faces)
