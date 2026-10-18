import numpy as np
import pytest

from meshkit import Mesh, Sel, add_group, delete, group, join
from meshkit.mesh import signed_volume
from meshkit.ops import bridge, extrude, flip, inset, merge_by_distance, mirror, recalc_normals
from meshkit.prims import circle, cube, cylinder, grid
from meshkit.query import boundary_loops, select_by_normal
from meshkit.xform import rotate, translate
from conftest import approx, bad_winding_edges, boundary_edges, hand_grid


def closed_ok(m):
    return not boundary_edges(m) and not bad_winding_edges(m) and signed_volume(m.verts, m.faces) > 0


def vol(m):
    return signed_volume(m.verts, m.faces)


def top_face(m):
    return select_by_normal(m, "z", 1)


# ---- extrude ----------------------------------------------------------------
def test_extrude_region_cube_top_with_groups():
    c = cube(group="body")
    c, _ = add_group(c, "lid", top_face(c))
    c, _ = add_group(c, "lid_v", Sel("v", [v for v in range(8) if c.verts[v][2] > 0]))
    m, cr = extrude(c, "lid", 0.5)
    assert (len(m.verts), len(m.faces)) == (12, 10) and closed_ok(m) and approx(vol(m), 1.5)
    assert cr.top == top_face(c) and len(cr.faces) == 4 and len(cr.verts) == 4
    assert set(cr.faces) <= set(group(m, "body")) and set(cr.faces) <= set(group(m, "lid"))
    assert set(cr.verts) <= set(group(m, "lid_v"))
    m2, _ = extrude(c, "lid", -0.25)  # pocket
    assert not bad_winding_edges(m2) and approx(vol(m2), 0.75)


def test_extrude_open_grid_and_disconnected_regions():
    g = hand_grid(3, 3)
    m, cr = extrude(g, None, 1.0)
    assert len(m.faces) == 9 + 12 and not bad_winding_edges(m)
    assert approx(m.verts[sorted(set(v for f in cr.top for v in m.faces[f]))][:, 2], 1.0)
    c = cube()
    tb = select_by_normal(c, "z", 1) | select_by_normal(c, "-z", 1)
    m, _ = extrude(c, tb, 0.5)  # two regions, each along its own normal
    assert closed_ok(m) and approx(vol(m), 2.0)


def test_extrude_individual_and_direction():
    c = cube()
    two = select_by_normal(c, "z", 1) | select_by_normal(c, "x", 1)
    m, cr = extrude(c, two, 0.3, mode="individual")
    assert len(m.verts) == 16 and len(m.faces) == 14 and closed_ok(m)
    m, _ = extrude(grid(), None, 2.0, direction=(0, 0, -1))
    assert approx(m.verts[:, 2].min(), -2.0)


# ---- inset ------------------------------------------------------------------
def test_inset_single_region_individual_depth():
    m, cr = inset(hand_grid(1, 1), [0], 0.1)
    assert len(m.faces) == 5 and approx(m.face_areas[cr.inner[0]], 0.64) and not bad_winding_edges(m)
    assert approx(m.face_normals[0], (0, 0, 1))
    g = hand_grid(2, 2)
    m, cr = inset(g, None, 0.25)
    assert len(cr.verts) == 8 and approx(sum(m.face_areas[list(cr.inner)]), 1.5 ** 2)
    assert approx(m.verts[4], (1, 1, 0))  # interior vert untouched
    m, cr = inset(g, None, 0.25, mode="individual")
    assert len(cr.verts) == 16 and approx(m.face_areas[list(cr.inner)], 0.25)
    c = cube()
    m, _ = inset(c, top_face(c), 0.1, depth=-0.2)
    assert closed_ok(m) and approx(vol(m), 1 - 0.2 / 3 * (1 + 0.64 + 0.8))  # frustum-shaped pocket


def test_inset_pinched_region_raises():
    with pytest.raises(ValueError, match="touches itself"):
        inset(hand_grid(3, 3), [0, 1, 2, 3, 5, 6, 7], 0.1)  # border meets itself at vert 10


# ---- mirror -----------------------------------------------------------------
def test_mirror_weld_closes_half_box():
    c = cube(center=(0.5, 0, 0), group="half")
    c, _ = delete(c, select_by_normal(c, "-x", 1))
    m, cr = mirror(c, "x")
    assert len(m.verts) == 12 and len(m.faces) == 10 and closed_ok(m) and approx(vol(m), 2.0)
    assert len(cr.verts) == 4 and cr.vmap[0] >= 0
    assert len(group(m, "half")) == 10
    m2, _ = mirror(c, "x", suffix="_R")
    assert len(group(m2, "half")) == 5 and len(group(m2, "half_R")) == 5


def test_mirror_part_without_weld():
    g = translate(grid(), (2, 0, 0))[0]
    m, cr = mirror(g, "x", weld=False)
    assert len(m.verts) == 8 and np.allclose(m.face_normals, (0, 0, 1))
    assert approx(m.verts[4:].mean(0), (-2, 0, 0))


# ---- merge ------------------------------------------------------------------
def test_merge_by_distance_groups_survive():
    a = Mesh(hand_grid(1, 1).verts, hand_grid(1, 1).faces, groups={"left": ("f", [0])})
    b = Mesh(hand_grid(1, 1).verts + [1, 0, 0], hand_grid(1, 1).faces, groups={"right": ("f", [0]), "rv": ("v", [0, 2])})
    m, _ = join(a, b)
    centers = m.face_centers.copy()
    out, cr = merge_by_distance(m)
    assert len(out.verts) == 6 and cr.removed == 2 and cr.faces_removed == 0
    assert len(out.edges) == 7 and not bad_winding_edges(out)
    assert approx(out.face_centers[sorted(group(out, "right"))], centers[[1]])
    assert approx(out.verts[sorted(group(out, "rv"))], [(1, 0, 0), (1, 1, 0)])
    assert cr.vmap[4] == cr.vmap[1]


def test_merge_drops_collapsed_faces():
    m = Mesh([(0, 0, 0), (1, 0, 0), (1, 1e-9, 0), (0, 1, 0)], [(0, 1, 2), (0, 1, 3)])
    out, cr = merge_by_distance(m, dist=1e-6)
    assert len(out.faces) == 1 and cr.faces_removed == 1


# ---- bridge -----------------------------------------------------------------
def test_bridge_free_circles_outward_and_aligned():
    a = circle(1.0, 8)
    b = rotate(translate(circle(1.0, 8), (0, 0, 2))[0], "z", 3 * 45, pivot=(0, 0, 0))[0]
    m, _ = join(a, b)
    out, cr = bridge(m, list(range(8)), [8, 15, 14, 13, 12, 11, 10, 9], segments=2)
    assert len(cr.faces) == 16 and len(cr.verts) == 8
    radial = out.face_centers.copy()
    radial[:, 2] = 0
    assert np.all((out.face_normals * radial).sum(1) > 0.9)  # outward, no twist
    assert not bad_winding_edges(out)
    assert approx(out.face_areas.std(), 0)


def test_bridge_boundary_loops_consistent():
    t1 = cylinder(1, 1, segments=8, caps=False, center=(0, 0, 0))
    t2 = rotate(cylinder(1, 1, segments=8, caps=False, center=(0, 0, 3)), "z", 10)[0]
    m, _ = join(t1, t2)
    loops = boundary_loops(m)
    top1 = max((l for l in loops if m.verts[l[0]][2] < 1.5), key=lambda l: m.verts[l[0]][2])
    bot2 = min((l for l in loops if m.verts[l[0]][2] > 1.5), key=lambda l: m.verts[l[0]][2])
    out, _ = bridge(m, top1, bot2)
    assert not bad_winding_edges(out) and len(boundary_loops(out)) == 2
    with pytest.raises(ValueError, match="equal counts"):
        bridge(m, top1, list(bot2)[:6])


# ---- winding ----------------------------------------------------------------
def test_flip_and_recalc_normals():
    c = cube()
    m, _ = flip(c)
    assert approx(vol(m), -1)
    fixed, cr = recalc_normals(m)
    assert closed_ok(fixed) and len(cr.faces) == 6
    m, _ = flip(c, [0, 3])
    assert bad_winding_edges(m)
    fixed, cr = recalc_normals(m)
    assert closed_ok(fixed) and sorted(cr.faces) == [0, 3]
