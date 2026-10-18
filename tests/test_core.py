import numpy as np
import pytest

from meshkit import Mesh, Sel, faces_of, group, verts_of, edges_of
from meshkit.mesh import Builder, signed_volume
from conftest import approx, hand_grid


def test_derived_edges_and_adjacency():
    m = Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)],
             [(0, 1, 2, 3), (1, 4, 5, 2)])
    assert m.edges == ((0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (2, 5), (4, 5))
    assert m.edge_faces[(1, 2)] == (0, 1)
    assert m.vert_faces[1] == (0, 1)
    assert m.vert_neighbors[2] == (1, 3, 5)


def test_loose_edges_normalized():
    m = Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0), (5, 5, 5)], [(0, 1, 2)], loose_edges=[(1, 0), (2, 3)])
    assert m.loose_edges == {(2, 3)}  # (0,1) is on a face, so not loose
    assert (2, 3) in m.edges
    with pytest.raises(ValueError):
        Mesh([(0, 0, 0)], loose_edges=[(0, 0)])


def test_immutable_and_validation():
    m = hand_grid(1, 1)
    with pytest.raises(ValueError):
        m.verts[0, 0] = 5
    with pytest.raises(TypeError):
        m.groups["x"] = None
    with pytest.raises(IndexError):
        Mesh([(0, 0, 0)], [(0, 1, 2)])
    with pytest.raises(ValueError):
        Mesh([(0, 0, 0), (1, 0, 0)], [(0, 1)])
    with pytest.raises(ValueError):
        Mesh(m.verts, m.faces, groups={"g": ("edge", [0])})
    with pytest.raises(IndexError):
        Mesh(m.verts, m.faces, groups={"g": ("f", [3])})


def test_typed_groups_and_resolution(grid3):
    m = Mesh(grid3.verts, grid3.faces, groups={"mid": ("f", [4]), "row": ("v", [0, 1, 2, 3])})
    assert group(m, "mid").kind == "f"
    assert list(verts_of(m, "mid")) == [5, 6, 9, 10]
    assert list(faces_of(m, Sel("v", [5, 6, 9, 10]))) == [4]
    assert list(faces_of(m, "row")) == []  # a row of verts encloses no face
    assert len(edges_of(m, "row")) == 3
    assert list(verts_of(m, ["mid", "row"])) == [5, 6, 9, 10, 0, 1, 2, 3]
    assert list(faces_of(m, [(5, 6), (6, 10), (9, 10), (5, 9)])) == [4]
    with pytest.raises(KeyError):
        verts_of(m, "nope")
    with pytest.raises(IndexError):
        verts_of(m, [99])


def test_sel_algebra():
    a, b = Sel("f", [1, 2, 3]), Sel("f", [3, 4])
    assert (a | b) == (1, 2, 3, 4) and (a | b).kind == "f"
    assert (a & b) == (3,) and (a - b) == (1, 2)
    assert (a + b).kind == "f"
    with pytest.raises(ValueError):
        a | Sel("v", [1])
    assert "n=3" in repr(a)
    assert Sel("e", [(3, 1)])[0] == (1, 3)


def test_geometry_caches():
    m = hand_grid(1, 1, size=2.0)
    assert approx(m.face_normals[0], (0, 0, 1))
    assert approx(m.face_areas[0], 4.0)
    assert approx(m.face_centers[0], (1, 1, 0))
    assert approx(m.vert_normals[0], (0, 0, 1))
    assert approx(m.diag, np.sqrt(8))


def test_signed_volume_unit_cube():
    V = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    assert approx(signed_volume(np.array(V, float), F), 1.0)


def test_build_compact_remaps_groups(grid3):
    m = Mesh(grid3.verts, grid3.faces, groups={"last": ("f", [8]), "corner": ("v", [15])})
    b = Builder(m)
    out, vmap, fmap = b.build_compact(drop_v=[], drop_f=[0])
    assert fmap[8] == 7 and fmap[0] == -1
    assert group(out, "last") == (7,)
    assert approx(out.face_centers[7], m.face_centers[8])
