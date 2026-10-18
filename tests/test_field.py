import numpy as np
import pytest

from meshkit import Mesh, Sel, cube, displace, grid, map_verts, noise, quad_sphere
from meshkit.field import gradient_noise


def pts(n=2000, seed=3, span=10):
    return np.random.default_rng(seed).uniform(-span, span, (n, 3))


def test_noise_deterministic_bounded_seeded():
    P = pts()
    a, b = noise(seed=1)(P), noise(seed=1)(P)
    assert np.array_equal(a, b)
    assert not np.allclose(a, noise(seed=2)(P))
    v = noise(seed=5, strength=0.3)(P)
    assert np.abs(v).max() <= 0.3 and np.abs(v).max() > 0.1 and abs(v.mean()) < 0.03
    assert np.isscalar(noise()(np.array([0.3, 0.2, 0.1])))
    assert noise(strength=0.4).amplitude == 0.4


def test_gradient_noise_zero_on_lattice_and_continuous():
    lat = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)], float)
    assert np.allclose(gradient_noise(lat, seed=9), 0)
    P = pts(500)
    assert np.abs(gradient_noise(P + 1e-4) - gradient_noise(P)).max() < 1e-2


def test_noise_scale_sets_feature_size():
    x = np.linspace(0, 20, 2001)
    P = np.c_[x, 0 * x + 0.37, 0 * x + 0.71]

    def crossings(scale):
        v = noise(seed=4, scale=scale, octaves=1)(P)
        return int((np.diff(np.sign(v)) != 0).sum())
    assert crossings(0.5) > 3 * crossings(4.0)


def test_displace_normal_axis_scalar_vectors_and_sel():
    s = quad_sphere(1.0, 6)
    m, cr = displace(s, 0.25)
    assert np.allclose(np.linalg.norm(m.verts, axis=1), 1.25, atol=1e-3) and len(cr.moved) == len(s.verts)
    m, _ = displace(s, noise(seed=2, strength=0.1))
    r = np.linalg.norm(m.verts, axis=1)
    assert 0.9 <= r.min() and r.max() <= 1.1 and r.std() > 0.01
    g = grid(1, 2)
    m, _ = displace(g, lambda P: P[:, 0], along="z", sel=Sel("v", [0, 1]))
    assert np.allclose(m.verts[[0, 1], 2], g.verts[[0, 1], 0]) and np.allclose(m.verts[2:, 2], 0)
    m, _ = displace(g, lambda P: np.tile([0, 0, 1.0], (len(P), 1)), along=None)
    assert np.allclose(m.verts[:, 2], 1)
    with pytest.raises(ValueError):
        displace(g, lambda P: P, along="z")


def test_map_verts():
    c = cube()
    m, cr = map_verts(c, lambda P: P * (2, 1, 1))
    assert np.allclose(np.ptp(m.verts, axis=0), (2, 1, 1))
    twist = lambda P: np.c_[P[:, 0] * np.cos(P[:, 2]) - P[:, 1] * np.sin(P[:, 2]),
                            P[:, 0] * np.sin(P[:, 2]) + P[:, 1] * np.cos(P[:, 2]), P[:, 2]]
    m, _ = map_verts(c, twist, sel=Sel("v", [0]))
    assert np.allclose(m.verts[1:], c.verts[1:])
    with pytest.raises(ValueError):
        map_verts(c, lambda P: P[:, :2])
