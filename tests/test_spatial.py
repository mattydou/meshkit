import numpy as np

from meshkit import (Mesh, closest_point, cube, inside, quad_sphere, raycast, revolve, select_by_normal,
                     winding_number)


def test_closest_point_regions_on_cube():
    c = cube()
    r = closest_point(c, [(2, 0, 0), (0.1, 0, 0), (2, 2, 2), (2, 2, 0), (0, 0, 0.5)])
    assert np.allclose(r.points, [(0.5, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0, 0, 0.5)])
    assert np.allclose(r.dists, [1.5, 0.4, np.sqrt(3 * 1.5 ** 2), np.sqrt(2 * 1.5 ** 2), 0])
    assert np.allclose(c.face_normals[r.faces[0]], (1, 0, 0))
    top = select_by_normal(c, "z", 1)
    r = closest_point(c, (2, 0, 0), faces=top)
    assert r.faces[0] == top[0] and np.isclose(r.dists[0], np.hypot(1.5, 0.5))
    assert np.isclose(closest_point(c, 0).dists[0], 0.5)  # scalar broadcast


def test_closest_point_matches_dense_sampling_on_sphere():
    s = quad_sphere(1.0, 8)
    P = np.random.default_rng(0).normal(size=(300, 3)) * 2
    r = closest_point(s, P)
    assert np.all(np.abs(r.dists - np.abs(np.linalg.norm(P, axis=1) - 1)) < 0.02)
    T, _ = s.triangles
    bary = np.random.default_rng(1).dirichlet((1, 1, 1), size=(len(T), 20))
    S = np.einsum("tki,tij->tkj", bary, s.verts[T]).reshape(-1, 3)
    brute = np.sqrt(((P[:, None] - S[None]) ** 2).sum(-1)).min(1)
    assert np.all(r.dists <= brute + 1e-12)
    assert np.allclose(np.linalg.norm(r.points - P, axis=1), r.dists)


def test_raycast_hits_misses_and_limits():
    c = cube()
    h = raycast(c, [(0, 0, 5), (3, 3, 5), (0, 0, 0)], [(0, 0, -1), (0, 0, -1), (1, 0, 0)])
    assert list(h.hit) == [True, False, True]
    assert np.allclose(h.points[0], (0, 0, 0.5)) and np.isclose(h.dists[0], 4.5)
    assert np.allclose(c.face_normals[h.faces[0]], (0, 0, 1)) and h.faces[1] == -1 and np.isnan(h.points[1]).all()
    assert np.isclose(h.dists[2], 0.5)  # from inside: exits through +x
    assert not raycast(c, (0, 0, 5), "-z", max_dist=4).hit[0]
    many = raycast(c, (0, 0, 5), [(0, 0, -1), (0.05, 0, -1), (0, 0, 1)])  # one origin, many dirs
    assert list(many.hit) == [True, True, False]
    assert np.allclose(raycast(c, (0, 0, 5), (0, 0, -10)).dists, 4.5)  # dirs are normalized


def test_inside_and_winding_number():
    c = cube()
    assert list(inside(c, [(0, 0, 0), (0.6, 0, 0), (0.49, 0.49, -0.49)])) == [True, False, True]
    assert np.isclose(winding_number(c, (0, 0, 0))[0], 1) and np.isclose(winding_number(c, (5, 0, 0))[0], 0)
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    torus = revolve(np.c_[2 + 0.5 * np.cos(th), 0.5 * np.sin(th)], segments=24, closed=True)
    assert list(inside(torus, [(0, 0, 0), (2, 0, 0)])) == [False, True]
