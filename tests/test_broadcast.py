import numpy as np
import pytest

from meshkit import (Mesh, circle, cube, cylinder, grid, nearest_verts, quad_sphere, revolve, ring, rotate, scale,
                     select_by_bbox, select_by_plane, sweep, translate, vec)


def test_vec_broadcasts_scalars():
    assert np.array_equal(vec(0), [0, 0, 0]) and np.array_equal(vec(2.5), [2.5] * 3)
    assert np.array_equal(vec(np.float64(1)), [1, 1, 1])
    with pytest.raises(ValueError):
        vec((1, 2))


def test_scalar_centers_everywhere():
    for m in (quad_sphere(center=0), cube(center=0), grid(center=0), cylinder(center=0), circle(center=0),
              revolve([(1, 0), (1, 1)], center=0), ring(center=0)):
        assert len(m.verts)
    assert np.allclose(cube(center=1).verts.mean(0), 1)
    assert np.allclose(quad_sphere(center=2).verts.mean(0), 2)


def test_scalar_points_offsets_pivots_bounds():
    c = cube()
    assert np.allclose(translate(c, 1)[0].verts.mean(0), 1)
    assert np.allclose(rotate(c, "z", 90, pivot=0)[0].verts.mean(0), 0)
    assert np.allclose(scale(c, 2, pivot=0)[0].verts.max(0), 1)
    assert len(select_by_bbox(c, lo=0)) == 1 and len(select_by_bbox(c, hi=0)) == 1
    assert len(select_by_plane(c, 0, "z")) == 4
    assert len(nearest_verts(c, 0, k=8)) == 8
    assert len(cube(np.float64(2), divisions=np.int64(2)).faces) == 24
    m, cr = sweep(Mesh(), [(0, 0, 0), (0, 0, 1)], taper=np.float64(0.5), resolution=1)
    assert len(cr.rings) == 2
