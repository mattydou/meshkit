"""Retopology: cage_from_sdf, attach_sweep, retopo_project, topo_report (and the runner printing it)."""
import io
import os
import textwrap

import numpy as np
import pytest

from meshkit import (Mesh, attach_sweep, box, capsule, cage_from_sdf, check, cone, ellipsoid, intersect, limb_sdf,
                     project_uvs, quad_sphere, retopo_project, sdf_transform, smooth_union, sphere, sweep_sdf,
                     to_mesh, topo_report, union)
from meshkit.retopo import _valence
from meshkit.runner import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(m, genus=0):
    """Closed, consistently wound, all quads (and the expected genus)."""
    c = check(m, closed=True, expect_genus=genus)
    assert c.ok, str(c)
    assert all(len(f) == 4 for f in m.faces)
    assert _valence(m).min() >= 3  # no two-edge verts anywhere
    return c


def lizard_sdf():
    """The SDF of examples/retopo_creature.py (and textured_creature.py), without running either script."""
    src = open(os.path.join(ROOT, "examples", "retopo_creature.py"), encoding="utf-8").read()
    ns = {}
    exec(compile(src[:src.index("\ncage = ")], "lizard", "exec"), ns)
    return ns["sdf"]


def test_root_is_a_quad_ellipsoid():
    s = ellipsoid((1, 0.5, 0.3), name="body")
    m = cage_from_sdf(s, density=10)
    clean(m)
    assert sorted(m.groups) == ["body"] and len(m.groups["body"].idx) == len(m.faces)
    deg = _valence(m)
    assert (deg == 3).sum() == 8 and set(deg.tolist()) == {3, 4}  # a box lattice: 8 corner poles only
    E = np.array(m.edges)
    L = np.linalg.norm(m.verts[E[:, 0]] - m.verts[E[:, 1]], axis=1)
    assert 0.06 < np.median(L) < 0.14  # about 1 / density
    r = cage_from_sdf(s, density=20)
    assert 3.2 < len(r.faces) / len(m.faces) < 4.8  # twice the density, four times the quads


def test_cage_projected_onto_the_creature_stays_clean():
    sdf = lizard_sdf()
    cage = cage_from_sdf(sdf, density=15)
    clean(cage)
    m, info = retopo_project(cage, sdf)
    clean(m)
    assert m.faces == cage.faces and len(m.triangles[0]) < 15000
    assert info.max_dist < 1e-5 * float(np.linalg.norm(sdf.hi - sdf.lo))  # verts on the surface
    d = np.abs(sdf.eval(m.face_centers))
    assert np.percentile(d, 95) < 0.25 / 15 and d.max() < 1 / 15  # face centers: chords of 1/15-long edges
    assert len(info.flipped) < 0.02 * len(m.faces)  # only where the toes fuse
    names = {"body", "head", "snout", "tail", "eye.0", "eye.1", "leg.0", "leg.3", "spike.0", "spike.5", "toe.0"}
    assert names <= set(m.groups)
    assert sum(len(g.idx) for g in m.groups.values()) == len(m.faces)  # one part per face


def test_retopo_project_keeps_groups_and_uvs_and_evens_spacing():
    s = ellipsoid((1.2, 0.6, 0.5))
    q = quad_sphere(0.8, 8, group="skin")
    q, _ = project_uvs(q, "box")
    m, info = retopo_project(q, s)
    assert m.faces == q.faces and np.array_equal(m.uvs, q.uvs, equal_nan=True)
    assert set(m.groups["skin"].idx) == set(q.groups["skin"].idx)
    assert np.abs(s.eval(m.verts)).max() < 1e-5 and info.max_dist < 1e-5 and not info.flipped
    plain, _ = retopo_project(q, s, relax=0)

    def cv(x):
        E = np.array(x.edges)
        L = np.linalg.norm(x.verts[E[:, 0]] - x.verts[E[:, 1]], axis=1)
        return L.std() / L.mean()
    assert cv(m) < cv(plain)  # tangential Taubin evens the edges out without leaving the surface


def test_attach_sweep_grows_a_clean_tube():
    body = quad_sphere(1, 6)
    path = [(0, 0, 0.5), (0, 0, 1.6), (0.5, 0, 2.2)]
    m, c = attach_sweep(body, path, radius=[0.3, 0.25, 0.1], edge=0.15, group="arm")
    clean(m)
    assert c.hole.closed and len(c.hole) % 2 == 0 and len(c.rings) > 5 and c.rings[0] == c.hole
    assert set(m.groups["arm"].idx) == set(c.faces)
    tip = m.verts[list(c.verts)][np.argmax(m.verts[list(c.verts)] @ np.array([0.6, 0, 0.8]))]
    assert np.linalg.norm(tip - (0.5, 0, 2.2)) < 0.2  # reaches the end of the path (plus its dome)
    sizes = [len(r) for r in c.rings]
    assert sizes[-1] < sizes[0] and all(n % 2 == 0 and n >= 4 for n in sizes)  # tapers by quad-only transitions
    flat, cf = attach_sweep(m, [(0, 0, 0), (1.5, 0, 0)], radius=0.2, end="flat", path="joints")
    clean(flat)
    opened, co = attach_sweep(m, [(0, 0, 0), (0, -1.5, 0)], radius=0.2, end="open")
    loops = [e for e, fs in opened.edge_faces.items() if len(fs) == 1]
    assert {v for e in loops for v in e} == set(co.end) and check(opened).ok
    tri = to_mesh(sphere(1), 24)
    t, ct = attach_sweep(tri, [(0, 0, 0), (0, 0, 1.8)], radius=0.25)
    assert check(t, closed=True, expect_genus=0).ok
    assert sum(len(f) == 3 for f in t.faces) - sum(len(f) == 3 for f in tri.faces) <= 1  # odd hole: one triangle


def test_big_taper_changes_segments_with_quads_only():
    s = smooth_union(sphere(0.6, name="body"), cone((0, 0, 0.3), (0, 0, 2.5), 0.45, 0.0, name="horn"), k=0.05)
    m = cage_from_sdf(s, density=14)
    clean(m)
    horn = m.verts[sorted({v for f in m.groups["horn"].idx for v in m.faces[f]})]
    assert horn[:, 2].max() > 2.2  # runs up to near the point
    r, info = retopo_project(m, s)
    clean(r)
    assert not info.flipped


def test_limb_joints_get_mitered_support_loops():
    leg = limb_sdf([(0, 0, 0), (0, 0.8, 0.1), (0, 0.9, -0.7)], [0.12, 0.1, 0.08], knuckle=1.2, name="leg")
    s = smooth_union(ellipsoid((0.6, 0.4, 0.4), name="body"), leg, k=0.1)
    cage = cage_from_sdf(s, density=25)
    clean(cage)
    m, info = retopo_project(cage, s)
    clean(m)
    assert not info.flipped  # the sharp knee does not fold
    rep = topo_report(m, s)
    assert rep.stats["loops"]["leg@1"] >= 3 and rep.stats["loops"]["leg|body"] >= 2
    tail = smooth_union(ellipsoid((0.6, 0.4, 0.4), name="body"),
                        sweep_sdf([(-0.4, 0, 0), (-1.2, 0, 0.2), (-1.8, 0.4, 0.3)], [0.15, 0.1, 0.05], name="tail"),
                        k=0.1)
    with_loops, without = cage_from_sdf(tail, density=25), cage_from_sdf(tail, density=25, support=0)
    assert len(without.faces) < len(with_loops.faces)  # support loops add rings either side of a sweep joint
    assert topo_report(with_loops, tail).stats["loops"]["tail@1"] > topo_report(without, tail).stats["loops"]["tail@1"]


def test_cage_variants_and_transforms():
    worm = sweep_sdf([(0, 0, 0), (1, 0.3, 0), (2, 0, 0.2), (3, 0.4, 0.1)], [0.2, 0.25, 0.2, 0.1], name="worm")
    clean(cage_from_sdf(worm, density=12))  # a lone tube is its own root, capped at both ends
    crate = smooth_union(box((1, 0.6, 0.4), round=0.05, name="torso"), capsule((0, 0, 0), (0, 0, 0.8), 0.12,
                                                                               name="neck"), k=0.05)
    m = cage_from_sdf(crate, density=15)
    clean(m)
    assert sorted(m.groups) == ["neck", "torso"]
    two = union(sphere(0.5, name="a"), sphere(0.3, center=(2, 0, 0), name="b"))
    assert sorted(cage_from_sdf(two, density=10).groups) == ["a"]
    both = cage_from_sdf(two, density=10, keep="all", group="skin")
    assert check(both, closed=True).stats["comps"] == 2 and len(both.groups["skin"].idx) == len(both.faces)
    moved = sdf_transform(smooth_union(ellipsoid((0.5, 0.3, 0.3), name="b"),
                                       capsule((0.3, 0, 0), (0.9, 0, 0.2), 0.1, name="arm"), k=0.05),
                          rotation=("z", 40), offset=(1, 2, 0), scale=1.5)
    m = cage_from_sdf(moved, density=12)
    r, info = retopo_project(m, moved)
    clean(r)
    assert sorted(r.groups) == ["arm", "b"] and info.max_dist < 1e-5 and not info.flipped
    assert np.allclose((r.verts.min(0) + r.verts.max(0)) / 2, (moved.lo + moved.hi) / 2, atol=0.1)  # posed cage


def test_cage_errors_are_clear():
    with pytest.raises(ValueError, match="can't map part .*intersect.*quad_remesh"):
        cage_from_sdf(intersect(sphere(1, name="a"), box(1.5)), density=10)
    with pytest.raises(ValueError, match="box part 'crate' unless it is the root"):
        cage_from_sdf(smooth_union(sphere(1, name="body"), box(0.5, center=(1, 0, 0), name="crate"), k=0.1),
                      density=10)
    with pytest.raises(ValueError, match="leaves 'body' and enters it again"):
        cage_from_sdf(smooth_union(sphere(1, name="body"), sweep_sdf([(0, 0.5, 0.5), (0, 0, 1.6), (0, -0.5, 0.5)],
                                                                     0.15, name="handle"), k=0.1), density=10)
    with pytest.raises(KeyError, match="no part 'zzz'"):
        cage_from_sdf(sphere(1, name="a"), root="zzz")
    with pytest.raises(ValueError, match="end must be"):
        attach_sweep(quad_sphere(), [(0, 0, 0), (0, 0, 2)], end="pointy")


def test_topo_report():
    r = topo_report(quad_sphere(1, 4))
    assert r.stats["quad_ratio"] == 1 and r.stats["valence"] == {3: 8, 4: 90} and len(r.stats["poles"]) == 8
    assert str(r).startswith("topo: OK  96 faces, quads 100%") and len(str(r).splitlines()) < 10
    tri = to_mesh(sphere(1), 16)
    t = topo_report(tri)
    assert t.stats["quad_ratio"] == 0 and 6 in t.stats["valence"] and "quads 0%" in str(t)  # regular = 6
    s = smooth_union(ellipsoid((1, 0.5, 0.4), name="body"), capsule((0.5, 0, 0), (0.5, 1.2, 0.3), 0.15, 0.08,
                                                                    name="arm"), k=0.1)
    m, _ = retopo_project(cage_from_sdf(s, density=14, group="skin"), s)
    rep = topo_report(m, s)
    text = str(rep)
    assert len(text.splitlines()) < 10 and "skin" not in rep.stats["edge_cv"]  # catch-all group skipped
    assert {"all", "body", "arm"} == set(rep.stats["edge_cv"]) and rep.stats["edge_cv"]["all"] < 0.35
    assert "arm|body" in rep.stats["loops"] and rep.stats["loops"]["arm|body"] >= 2
    joint = [p for p in rep.stats["poles"] if p["joint"]]
    assert len(joint) >= 4 and "at joints (arm|body" in text  # the hole's four corners
    assert rep.stats["max_dist"] < 1e-5 and rep.stats["folded"] == [] and "on the sdf: max dist" in text
    empty = topo_report(Mesh())
    assert empty.ok and "empty" in str(empty)


def test_runner_prints_topo_report_for_quad_meshes(tmp_path):
    p = tmp_path / "blob.py"
    p.write_text(textwrap.dedent("""
        from meshkit import *
        sdf = smooth_union(sphere(0.6, name="body"), capsule((0, 0, 0), (0, 0, 1.4), 0.2, name="neck"), k=0.1)
        mesh, _ = retopo_project(cage_from_sdf(sdf), sdf)
        check_opts = {"closed": True, "expect_genus": 0}
    """))
    out = io.StringIO()
    assert run(str(p), out=out, export=False) == 0
    text = out.getvalue()
    assert "topo: OK" in text and "quads 100%" in text and "on the sdf: max dist" in text
    lo = len(text)
    out = io.StringIO()
    assert run(str(p), out=out, export=False, res="final") == 0  # cage density follows draft / final
    assert int(out.getvalue().split("topo: OK  ")[1].split()[0]) > 3 * int(text.split("topo: OK  ")[1].split()[0])
    tri = tmp_path / "tri.py"
    tri.write_text("from meshkit import *\nmesh = to_mesh(sphere(1), 16)\n")
    out = io.StringIO()
    assert run(str(tri), out=out, export=False) == 0 and "topo:" not in out.getvalue() and lo
