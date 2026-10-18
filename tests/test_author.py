import numpy as np
import pytest

from meshkit import (Mesh, Sel, add_edges, add_face, add_group, add_verts, delete, faces_of, fill, grid_fill,
                     group, join, remove_group)
from conftest import approx, bad_winding_edges, boundary_edges, hand_grid


def authored_loop(coords):
    m, c = add_verts(Mesh(), coords)
    m, e = add_edges(m, c.verts, closed=True)
    return m, e.loop


def test_vert_edge_face_ngon():
    m, loop = authored_loop([(0, 0, 0), (2, 0, 0), (3, 1, 0), (1, 2, 0), (-1, 1, 0)])
    assert len(m.loose_edges) == 5 and loop.closed
    m, c = fill(m, loop)
    assert len(m.faces) == 1 and len(m.faces[0]) == 5
    assert m.loose_edges == frozenset()  # wire edges became face edges
    assert approx(m.face_normals[c.faces[0]], (0, 0, 1))  # CCW loop -> +z


def test_add_edges_pairs_and_face_validation():
    m, c = add_verts(Mesh(), [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    m, e = add_edges(m, [(0, 1), (1, 2)])
    assert list(e.edges) == [(0, 1), (1, 2)]
    m, f = add_face(m, [0, 1, 2])
    with pytest.raises(ValueError):
        add_face(m, [2, 0, 1])  # same face again
    with pytest.raises(ValueError):
        add_face(m, [0, 1, 1])
    with pytest.raises(IndexError):
        add_face(m, [0, 1, 7])


def test_grid_fill_authored_rectangle():
    # 4 edges on the long sides, 2 on the short sides; corners found by turning angle
    pts = [(i, 0, 0) for i in range(4)] + [(4, j, 0) for j in range(2)] + \
          [(4 - i, 2, 0) for i in range(4)] + [(0, 2 - j, 0) for j in range(2)]
    m, loop = authored_loop(pts)
    m, c = grid_fill(m, loop)
    assert len(c.faces) == 8 and len(c.verts) == 3
    assert all(len(f) == 4 for f in m.faces)
    assert np.allclose(m.face_normals[:, 2], 1)
    assert approx(m.face_areas.sum(), 8.0)
    assert not bad_winding_edges(m)


def test_grid_fill_circle_and_span():
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    m, loop = authored_loop(np.c_[np.cos(th), np.sin(th), 0 * th])
    m2, c = grid_fill(m, loop)
    assert len(c.faces) == 9 and len(c.verts) == 4
    m3, c3 = grid_fill(m, loop, span=2)
    assert len(c3.faces) == 2 * 4
    with pytest.raises(ValueError):
        grid_fill(authored_loop(np.c_[np.cos(th[:5]), np.sin(th[:5]), 0 * th[:5]])[0], [0, 1, 2, 3, 4])


def test_fill_and_grid_fill_on_boundary_follow_neighbors(grid3):
    m, _ = delete(grid3, [4])  # hole in the middle
    hole = [5, 6, 10, 9]  # given in the "wrong" (same as neighbors) direction
    for op in (fill, grid_fill):
        out, c = op(m, hole)
        assert not bad_winding_edges(out)
        assert approx(out.face_normals[c.faces[0]], (0, 0, 1))
        assert len(boundary_edges(out)) == 12


def test_delete_faces_groups_survive(grid3):
    m = Mesh(grid3.verts, grid3.faces, groups={"right": ("f", [2, 5, 8]), "top_row": ("v", [12, 13, 14, 15])})
    before_f = m.face_centers[[2, 5, 8]].copy()
    before_v = m.verts[[12, 13, 14, 15]].copy()
    out, c = delete(m, [0, 1, 3])
    assert len(out.faces) == 6
    assert len(out.verts) == 16 - 3  # verts 0, 1, 4 orphaned and pruned
    assert approx(out.face_centers[sorted(group(out, "right"))], before_f)
    assert approx(out.verts[sorted(group(out, "top_row"))], before_v)
    assert c.vmap[0] == -1 and c.fmap[2] == 0


def test_delete_verts_and_edges(grid3):
    out, c = delete(grid3, Sel("v", [5]))
    assert len(out.faces) == 5 and len(out.verts) == 12  # 0, 1, 4 orphaned
    out, c = delete(grid3, [(5, 6)], kind="e")
    assert len(out.faces) == 7 and len(out.verts) == 16
    out, _ = delete(grid3, [4], prune=False)
    assert len(out.verts) == 16


def test_join_and_groups():
    a = Mesh(hand_grid(1, 1).verts, hand_grid(1, 1).faces, groups={"g": ("f", [0])})
    b = Mesh(hand_grid(1, 1).verts + [0, 0, 1], hand_grid(1, 1).faces, groups={"g": ("f", [0]), "h": ("v", [0])})
    m, c = join(a, b)
    assert len(m.verts) == 8 and list(c.faces) == [1] and list(c.verts) == [4, 5, 6, 7]
    assert group(m, "g") == (0, 1) and group(m, "h") == (4,)
    m, c = add_group(m, "top", faces_of(m, Sel("v", [4, 5, 6, 7])))
    assert group(m, "top") == (1,)
    with pytest.raises(ValueError):
        add_group(m, "x", [0, 1])
    m, _ = remove_group(m, "top")
    assert "top" not in m.groups
