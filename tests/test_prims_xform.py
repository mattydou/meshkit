import numpy as np
import pytest

from meshkit import Mesh, Sel, group
from meshkit.mesh import signed_volume
from meshkit.prims import basis, circle, cube, cylinder, grid, quad_sphere, revolve, ring
from meshkit.xform import rotate, scale, translate
from conftest import approx, bad_winding_edges, boundary_edges


def closed_ok(m):
    return not boundary_edges(m) and not bad_winding_edges(m) and signed_volume(m.verts, m.faces) > 0


def chi(m):
    return len(m.verts) - len(m.edges) + len(m.faces)


def test_basis_right_handed():
    for ax in ("x", "y", "z", "-x", "-y", "-z", (1, 1, 0)):
        U, V, W = basis(ax)
        assert approx(np.cross(U, V), W)


def test_grid():
    g = grid((2, 3), (2, 3), group="floor")
    assert len(g.verts) == 12 and len(g.faces) == 6
    assert np.allclose(g.face_normals, (0, 0, 1)) and approx(g.face_areas.sum(), 6)
    assert group(g, "floor") == tuple(range(6))
    assert np.allclose(grid(axis="x").face_normals, (1, 0, 0))
    assert np.allclose(grid(axis="-y").face_normals, (0, -1, 0))


def test_cube():
    c = cube()
    assert (len(c.verts), len(c.faces)) == (8, 6) and closed_ok(c) and chi(c) == 2
    assert approx(signed_volume(c.verts, c.faces), 1.0)
    c2 = cube((2, 1, 0.5), center=(1, 2, 3), divisions=2)
    assert (len(c2.verts), len(c2.faces)) == (26, 24) and closed_ok(c2)
    assert approx(signed_volume(c2.verts, c2.faces), 1.0)
    assert approx(c2.verts.mean(axis=0), (1, 2, 3))


def test_quad_sphere():
    s = quad_sphere(2.0, 4)
    assert (len(s.verts), len(s.faces)) == (98, 96) and closed_ok(s) and chi(s) == 2
    assert np.allclose(np.linalg.norm(s.verts, axis=1), 2.0)
    assert all(len(f) == 4 for f in s.faces)
    assert 0.9 * 4 / 3 * np.pi * 8 < signed_volume(s.verts, s.faces) < 4 / 3 * np.pi * 8


def test_cylinder_variants():
    c = cylinder(1.0, 2.0, segments=8, rings=2)
    assert (len(c.verts), len(c.faces)) == (24, 18) and closed_ok(c)
    assert len(boundary_edges(cylinder(segments=8, caps=False))) == 16
    cone = cylinder(1.0, 1.0, segments=8, radius_top=0)
    assert len(cone.verts) == 9 and closed_ok(cone)
    lo, hi = cylinder(0.5, 4.0, axis="x").verts.min(0), cylinder(0.5, 4.0, axis="x").verts.max(0)
    assert approx(hi - lo, (4, 1, 1))


def test_circle_revolve_ring():
    w = circle(1.0, 8)
    assert len(w.faces) == 0 and len(w.loose_edges) == 8
    assert np.allclose(circle(1.0, 8, axis="y", fill=True).face_normals, (0, 1, 0))
    vase = revolve([(0, 0), (1, 0), (1, 2), (0.5, 3)], segments=12)
    assert len(boundary_edges(vase)) == 12 and not bad_winding_edges(vase)
    assert closed_ok(revolve([(0, 0), (1, 0), (1, 2), (0.5, 3)], segments=12, caps=True))
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    torus = revolve(np.c_[2 + 0.5 * np.cos(th), 0.5 * np.sin(th)][::-1], segments=16, closed=True)
    assert closed_ok(torus) and chi(torus) == 0
    flat = ring(1.0, 0.5, 12)
    assert np.allclose(flat.face_normals, (0, 0, 1)) and len(boundary_edges(flat)) == 24
    washer = ring(1.0, 0.5, 12, depth=0.2)
    assert closed_ok(washer) and chi(washer) == 0


def test_transforms_on_subsets():
    c = Mesh(cube().verts, cube().faces, groups={"top": ("f", [i for i, n in enumerate(cube().face_normals) if n[2] > .9])})
    m, r = translate(c, (0, 0, 1), sel="top")
    assert len(r.moved) == 4 and approx(m.verts[:, 2].max(), 1.5) and approx(m.verts[:, 2].min(), -0.5)
    assert approx(c.verts[:, 2].max(), 0.5)  # input untouched
    p = Mesh([(1, 0, 0), (2, 0, 0)])
    m, _ = rotate(p, "z", 90, pivot=(0, 0, 0))
    assert np.array_equal(m.verts, [[0, 1, 0], [0, 2, 0]])  # exact for multiples of 90
    m, _ = rotate(translate(c, (5, 0, 0))[0], "z", 45)
    assert approx(m.verts.mean(axis=0), (5, 0, 0))  # default pivot = bbox center
    m, _ = scale(c, (2, 1, 1), sel=Sel("f", [0]))  # face Sel -> its verts
    assert m.groups == c.groups
    m, _ = scale(c, 2)
    assert approx(signed_volume(m.verts, m.faces), 8.0)
