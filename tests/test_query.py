import numpy as np
import pytest

from meshkit import Mesh, Sel, add_edges, add_verts, delete
from meshkit.query import (bbox, boundary_loops, centroid, edge_loop, edge_ring, nearest_verts, select_by_bbox,
                           select_by_normal, select_by_plane)
from conftest import hand_grid


def vid(i, j, nx=4):
    return j * (nx + 1) + i


def hand_tube(seg=8, rings=3):
    """Open tube of quads around z (outward normals), rings x seg faces."""
    th = 2 * np.pi * np.arange(seg) / seg
    V = [(np.cos(t), np.sin(t), float(r)) for r in range(rings + 1) for t in th]
    F = [(r * seg + j, r * seg + (j + 1) % seg, (r + 1) * seg + (j + 1) % seg, (r + 1) * seg + j)
         for r in range(rings) for j in range(seg)]
    return Mesh(V, F)


def test_boundary_loops_grid_and_hole():
    g = hand_grid(4, 4)
    (loop,) = boundary_loops(g)
    assert loop.closed and len(loop) == 16 and loop.kind == "v"
    holed, _ = delete(g, [5])  # an interior face
    loops = boundary_loops(holed)
    assert sorted(len(l) for l in loops) == [4, 16]


def test_boundary_loop_follows_face_winding():
    g = hand_grid(1, 1)
    (loop,) = boundary_loops(g)
    f = g.faces[0]
    k = f.index(loop[0])
    assert f[(k + 1) % 4] == loop[1]


def test_edge_loop_open_closed_boundary_wire():
    g = hand_grid(4, 4)
    loop = edge_loop(g, (vid(1, 1), vid(2, 1)))
    assert not loop.closed and list(loop) == [vid(i, 1) for i in range(5)]
    b = edge_loop(g, (vid(0, 0), vid(1, 0)))
    assert b.closed and len(b) == 16
    t = hand_tube(8, 3)
    ring = edge_loop(t, (8, 9))  # around the tube
    assert ring.closed and list(ring) == list(range(8, 16))
    m, c = add_verts(Mesh(), [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 0)])
    m, _ = add_edges(m, c.verts)
    w = edge_loop(m, (1, 2))
    assert list(w) == [0, 1, 2, 3] and not w.closed


def test_edge_ring_open_and_closed():
    g = hand_grid(4, 4)
    r = edge_ring(g, (vid(1, 1), vid(2, 1)))
    assert r.kind == "e" and len(r) == 5 and not r.closed
    t = hand_tube(8, 3)
    r = edge_ring(t, (8, 16))  # vertical edge -> ring goes around
    assert r.closed and len(r) == 8
    with pytest.raises(IndexError):
        edge_ring(g, (0, 24))


def test_select_by_bbox_normal_plane_nearest():
    g = hand_grid(4, 4)
    s = select_by_bbox(g, lo=(None, None, None), hi=(1, 1, None))
    assert list(s) == [vid(0, 0), vid(1, 0), vid(0, 1), vid(1, 1)]
    f = select_by_bbox(g, (0, 0, -1), (2, 1, 1), kind="f")
    assert f.kind == "f" and list(f) == [0, 1]
    t = hand_tube(8, 1)
    side = select_by_normal(t, "x", angle=30)
    assert len(side) == 2  # faces centered at +-22.5 degrees
    assert len(select_by_normal(g, "-z")) == 0 and len(select_by_normal(g, (0, 0, 2))) == 16
    above = select_by_plane(g, (0, 2, 0), "y", side="+")
    assert len(above) == 10 and len(select_by_plane(g, (0, 2, 0), "y", side="on")) == 5
    assert list(nearest_verts(g, (0.1, 0.1, 0), k=2)) == [0, 1]
    assert list(nearest_verts(g, (10, 10, 0), k=3, radius=1)) == []


def test_bbox_and_centroid():
    g = hand_grid(2, 2)
    lo, hi = bbox(g)
    assert np.allclose(lo, 0) and np.allclose(hi, (2, 2, 0))
    assert np.allclose(centroid(g, Sel("f", [0])), (0.5, 0.5, 0))
