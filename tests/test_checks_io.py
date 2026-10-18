import io
import os
import textwrap

import numpy as np
import pytest

from meshkit import (Mesh, check, contacts, cube, join, load_obj, quad_sphere, revolve, sweep, to_obj, translate)
from meshkit.contact import crossings, self_intersections
from meshkit.runner import run


def torus():
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    return revolve(np.c_[2 + 0.5 * np.cos(th), 0.5 * np.sin(th)], segments=24, closed=True)


# ---- expect_genus ----------------------------------------------------------------
def test_expect_genus():
    r = check(quad_sphere(1, 4), closed=True, expect_genus=0)
    assert r.ok and r.stats["genus"] == 0 and "genus=0" in r.head
    r = check(torus(), expect_genus=0)
    assert not r.ok and r.stats["genus"] == 1 and "genus 1, expected 0" in str(r) and "find_handles" in str(r)
    assert check(torus(), expect_genus=1).ok
    two, _ = join(quad_sphere(1, 4), translate(torus(), (6, 0, 0))[0])
    assert check(two, expect_genus=1, contacts=None).stats["genus"] == 1  # summed over pieces
    open_box = Mesh(cube().verts, cube().faces[1:])
    assert check(open_box, expect_genus=0).ok and "genus" not in check(open_box).head  # boundary loops counted
    nonman = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    r = check(nonman, expect_genus=0)
    assert r.stats["genus"] is None and "non-manifold, genus undefined" in str(r)


# ---- self-intersection ---------------------------------------------------------------
def test_intersect_flags_parts_passing_through_each_other():
    bars, _ = join(cube((3, 0.2, 0.2), group="bar"), cube((0.2, 3, 0.2), group="bar"))
    r = check(bars, intersect=True)
    assert not r.ok and "parts pass through each other (face pairs): bar.0-bar.1 (" in str(r)
    assert r.stats["intersections"]["within"] == 0 and r.stats["intersections"]["between"] > 0
    assert check(bars, intersect="self").ok  # only within-part crossings fail
    stacked, _ = join(cube(), translate(cube(), (0, 0, 1))[0])  # touching faces are not crossings
    assert check(stacked, intersect=True).ok


def test_intersect_flags_a_sweep_folding_through_itself():
    s, _ = sweep(Mesh(), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, -0.3, 0)], radius=0.3, segments=12,
                 resolution=6)
    r = check(s, intersect=True)
    assert not r.ok and "self-intersections:" in str(r) and r.stats["intersections"]["within"] > 0
    pairs = self_intersections(s)
    assert pairs == sorted(pairs) and all(a < b for a, b in pairs)
    assert not set(s.faces[pairs[0][0]]) & set(s.faces[pairs[0][1]])  # neighbours never count
    assert self_intersections(quad_sphere(1, 8)) == [] and self_intersections(torus()) == []


def test_crossings_matches_brute_force():
    m, _ = join(quad_sphere(1, 6), translate(quad_sphere(0.6, 5), (0.9, 0.3, 0.1))[0])
    E = np.array(list(m.edge_faces))
    rows, faces = crossings(m, E, tol=1e-9)
    got = set(zip(rows.tolist(), faces.tolist()))
    from meshkit.contact import _seg_tri
    T, src = m.triangles
    V = m.verts
    I, J = np.repeat(np.arange(len(E)), len(T)), np.tile(np.arange(len(T)), len(E))
    hit = _seg_tri(V[E[I, 0]], V[E[I, 1]], V[T[J, 0]], V[T[J, 1]], V[T[J, 2]], 1e-9)
    assert got == set(zip(I[hit].tolist(), src[J[hit]].tolist())) and len(got) > 0


# ---- contacts pairs + full list ----------------------------------------------------------
def row(n, size=1.0):
    parts = [translate(cube(size, group=f"c{i}"), (1.5 * i, 0, 0))[0] for i in range(n)]
    return join(*parts)[0]


def test_contacts_named_pairs_and_full_list():
    m = row(9)
    recs = contacts(m, pairs=[("c0.0", "c1.0"), ("c8.0", "c0.0")])
    assert [(r["a"], r["b"]) for r in recs] == [("c0.0", "c1.0"), ("c8.0", "c0.0")]
    assert recs[0]["state"] == "gap" and np.isclose(recs[0]["dist"], 0.5)
    by_group = contacts(m, pairs=[("c0", "c1")])  # group names switch to groups
    assert by_group[0]["a"] == "c0" and np.isclose(by_group[0]["dist"], 0.5)
    with pytest.raises(KeyError, match="parts: c0.0 c1.0"):
        contacts(m, pairs=[("c0.0", "nope")])
    text = str(check(m))
    assert "contacts(parts): 36 pairs" in text and "+" not in text.split("contacts")[1]
    assert all(f"c{i}.0-c{j}.0" in text for i in range(9) for j in range(i + 1, 9))
    r = check(m, contacts=[("c0.0", "c1.0")])
    assert "contacts(pairs): 1 pairs" in str(r) and len(r.stats["contacts"]) == 1


# ---- load_obj ---------------------------------------------------------------------
def test_load_obj_roundtrip_and_foreign_files(tmp_path):
    m, _ = join(cube(group="a"), translate(cube(group="b"), (2, 0, 0))[0])
    back = load_obj(to_obj(m, str(tmp_path / "m.obj")))
    assert np.allclose(back.verts, m.verts) and back.faces == m.faces and dict(back.groups) == dict(m.groups)
    p = tmp_path / "blender.obj"
    p.write_text(textwrap.dedent("""\
        # Blender export: Y up, o objects, v/vt/vn faces, negative indices, a wire line
        mtllib x.mtl
        o Plane
        v 0 0 0
        v 1 0 0
        v 1 0 -1
        v 0 0 -1
        vt 0 0
        vn 0 1 0
        s off
        usemtl mat
        f 1/1/1 2/1/1 3/1/1 4/1/1
        o Tri
        v 3 0 0
        v 4 0 0
        v 3 1 0
        f -3//1 -2//1 -1//1
        f 5 5 6
        l 1 3
        """))
    r = load_obj(str(p))
    assert len(r.faces) == 2 and sorted(r.groups) == ["Plane", "Tri"] and r.loose_edges == {(0, 2)}
    assert np.allclose(r.verts[2], (1, 1, 0)) and np.allclose(r.verts[6], (3, 0, 1))  # Y up -> Z up
    assert check(r).ok and np.allclose(r.face_normals[0], (0, 0, 1))
    assert np.allclose(load_obj(str(p), y_up=False).verts[2], (1, 0, -1))


def test_runner_reports_obj_files(tmp_path):
    path = to_obj(cube(), str(tmp_path / "box.obj"))
    out = io.StringIO()
    assert run(path, out=out) == 0
    text = out.getvalue()
    assert "== box  loaded in" in text and "check: OK" in text and "wrote" not in text
    assert not os.path.exists(tmp_path / "out")
