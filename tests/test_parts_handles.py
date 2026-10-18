"""Part A: part names as face groups (to_mesh, decimate, Blender import), find_handles, custom render cameras,
limb_sdf / sdf_transform, and sparse vs dense meshing."""
import io
import os
import runpy
import sys
import types

import numpy as np
import pytest

from meshkit import (capsule, check, decimate, ellipsoid, find_handles, join, limb_sdf, proximity, quad_sphere,
                     render_views, revolve, sdf_transform, smooth_union, sphere, sweep_sdf, to_json, to_mesh,
                     translate, union, write_blender_script)
from meshkit.png import read_png
from meshkit.runner import run
from meshkit.spatial import _closest_on_tris

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def critter():
    return smooth_union(ellipsoid((0.8, 0.35, 0.3), name="body"), sphere(0.25, center=(0.9, 0, 0.15), name="head"),
                        capsule((0.4, 0.15, 0), (0.45, 0.25, -0.5), 0.08, name="leg"),
                        capsule((0.4, -0.15, 0), (0.45, -0.25, -0.5), 0.08, name="leg"), k=0.12)


def face_groups(m, skip=()):
    return {n: g.idx for n, g in m.groups.items() if g.kind == "f" and n not in skip}


# ---- A1 part groups ---------------------------------------------------------------
def test_to_mesh_names_part_groups_that_partition_the_faces():
    m = to_mesh(critter(), res=48, group="skin")
    parts = face_groups(m, skip=("skin",))
    assert sorted(parts) == ["body", "head", "leg.0", "leg.1"]
    assert sum(len(i) for i in parts.values()) == len(m.faces) == len(m.groups["skin"].idx)
    assert len(set().union(*parts.values())) == len(m.faces)  # each face in exactly one part
    C = m.face_centers
    head = C[sorted(parts["head"])]
    assert (np.linalg.norm(head - (0.9, 0, 0.15), axis=1) < 0.4).all()
    assert (C[sorted(parts["leg.0"])][:, 1] > 0).all() and (C[sorted(parts["leg.1"])][:, 1] < 0).all()
    assert to_mesh(critter(), res=48, parts=False).groups == {}
    assert to_mesh(union(sphere(1), sphere(0.5, center=(1, 0, 0))), res=24).groups == {}  # nothing named
    with pytest.raises(ValueError, match="also a part name"):
        to_mesh(critter(), res=24, group="head")


def test_part_groups_survive_decimate():
    m = to_mesh(critter(), res=64)
    d, _ = decimate(m, len(m.faces) // 4)
    parts = face_groups(d)
    assert sorted(parts) == ["body", "head", "leg.0", "leg.1"]
    assert sum(len(i) for i in parts.values()) == len(d.faces) and len(set().union(*parts.values())) == len(d.faces)
    for n in parts:  # groups stay where they were
        a = m.face_centers[sorted(face_groups(m)[n])].mean(0)
        b = d.face_centers[sorted(parts[n])].mean(0)
        assert np.linalg.norm(a - b) < 0.05


def test_part_groups_inside_transforms():
    pair = union(sphere(0.3, name="a"), sphere(0.3, center=(1, 0, 0), name="b"))
    moved = sdf_transform(pair, rotation=("z", 90), pivot=(0, 0, 0), offset=(0, 0, 1))
    m = to_mesh(moved, res=40, keep="all")
    g = face_groups(m)
    assert np.allclose(m.face_centers[sorted(g["a"])].mean(0), (0, 0, 1), atol=0.01)
    assert np.allclose(m.face_centers[sorted(g["b"])].mean(0), (0, 1, 1), atol=0.01)


class _Fake:
    """Just enough of bpy for the generated import script: records objects, vertex groups and attributes."""

    def __init__(self):
        fake = self

        class Objects(dict):
            def new(self, name, me):
                ob = types.SimpleNamespace(name=name, data=me, vertex_groups=VGroups())
                self[name] = ob
                return ob
        self.objects, self.collections = Objects(), []

        class VGroups(list):
            def new(self, name):
                g = types.SimpleNamespace(name=name, verts=None)
                g.add = lambda idx, w, mode: setattr(g, "verts", list(idx))
                self.append(g)
                return g

        class MeshData:
            def __init__(self, name):
                self.name, self.attrs, self.uv, self.materials = name, {}, None, []
                self.attributes = types.SimpleNamespace(new=self._attr)
                self.uv_layers = types.SimpleNamespace(new=self._uv)

            def from_pydata(self, v, e, f):
                self.verts, self.edges, self.faces = v, e, f
                self.polygons = f

            def _attr(self, name, type, domain):
                a = types.SimpleNamespace(data=types.SimpleNamespace())
                a.data.foreach_set = lambda key, vals: self.attrs.__setitem__(name, list(vals))
                return a

            def _uv(self, name):
                layer = types.SimpleNamespace(data=types.SimpleNamespace())
                layer.data.foreach_set = lambda key, vals: setattr(self, "uv", list(vals))
                return layer

            def update(self):
                pass

        class Coll:
            def __init__(self, name):
                self.name, self.items, self.kids = name, [], []
                self.objects = types.SimpleNamespace(link=self.items.append)
                self.children = types.SimpleNamespace(link=self.kids.append)

        def new_collection(name):
            c = Coll(name)
            fake.collections.append(c)
            return c
        self.scene = Coll("scene")
        self.bpy = types.SimpleNamespace(
            data=types.SimpleNamespace(meshes=types.SimpleNamespace(new=MeshData), objects=self.objects,
                                       collections=types.SimpleNamespace(new=new_collection)),
            context=types.SimpleNamespace(collection=self.scene, scene=types.SimpleNamespace(collection=self.scene)))


def run_blender_script(path, **kw):
    fake = _Fake()
    sys.modules["bpy"] = fake.bpy
    try:
        ns = runpy.run_path(path, run_name="blender_test")
        ns["build"](**kw)
    finally:
        del sys.modules["bpy"]
    return fake


def test_blender_import_makes_vertex_groups_and_split_parts(tmp_path):
    m = to_mesh(critter(), res=32, group="skin")
    j = to_json(m, str(tmp_path / "c.json"))
    s = write_blender_script(j, str(tmp_path / "c_blender.py"), "critter")
    whole = run_blender_script(s).objects["critter"]
    vg = {g.name: g.verts for g in whole.vertex_groups}
    assert sorted(vg) == ["body", "head", "leg.0", "leg.1", "skin"]
    head_verts = sorted({v for f in m.groups["head"].idx for v in m.faces[f]})
    assert vg["head"] == head_verts and len(vg["skin"]) == len(m.verts)
    assert sum(whole.data.attrs["head_faces"]) == len(m.groups["head"].idx)
    split = run_blender_script(s, split=True)
    names = sorted(split.objects)
    assert names == ["critter.body", "critter.head", "critter.leg.0", "critter.leg.1"]  # skin is the largest group
    assert sum(len(o.data.faces) for o in split.objects.values()) == len(m.faces)
    head = split.objects["critter.head"].data
    assert len(head.faces) == len(m.groups["head"].idx) and len(head.verts) == len(head_verts)
    assert split.collections[0].name == "critter" and len(split.collections[0].items) == 4


# ---- A2 find_handles ------------------------------------------------------------------
def test_find_handles_locates_each_handle():
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    tor = revolve(np.c_[2 + 0.5 * np.cos(th), 0.5 * np.sin(th)], segments=24, closed=True)
    h = find_handles(tor).stats["handles"]
    assert len(h) == 1 and 3.0 < h[0]["length"] < 3.2  # the tube's meridian, circumference ~ 2 pi 0.5
    two, _ = join(tor, translate(tor, (6, 0, 0))[0])
    xs = sorted(round(float(x["at"][0])) for x in find_handles(two).stats["handles"])
    assert len(xs) == 2 and xs[1] - xs[0] >= 5
    r = find_handles(quad_sphere(1, 4))
    assert r.ok and r.stats["handles"] == [] and "genus 0" in str(r)


def test_find_handles_names_the_parts_fused_into_a_loop():
    s = smooth_union(ellipsoid((1, 0.4, 0.4), name="body"), sphere(0.3, center=(1.3, 0, 0), name="head"),
                     sweep_sdf([(0.6, 0, 0.2), (0.3, 0, 1.0), (-0.3, 0, 1.0), (-0.6, 0, 0.2)], 0.12, name="handle"),
                     k=0.1)
    m = to_mesh(s, res=64)
    r = find_handles(m)
    (h,) = r.stats["handles"]
    assert h["parts"][0] == "handle" and set(h["ring"]) >= {"handle", "body"} and "head" not in h["ring"]
    assert h["at"][2] > 0.4 and 0.6 < h["length"] < 1.0  # around the 0.12-radius handle tube
    assert "ring closes through" in str(r)


def test_runner_prints_handles_when_genus_is_too_high(tmp_path):
    p = tmp_path / "loop.py"
    p.write_text("from meshkit import *\n"
                 "s = smooth_union(ellipsoid((1, 0.4, 0.4), name='body'), sweep_sdf([(0.6, 0, 0.2), (0.3, 0, 1.0),"
                 " (-0.3, 0, 1.0), (-0.6, 0, 0.2)], 0.12, name='handle'), k=0.1)\n"
                 "mesh = to_mesh(s)\ncheck_opts = {'expect_genus': 0}\n")
    out = io.StringIO()
    assert run(str(p), out=out, export=False) == 1
    assert "handles: OK  genus 1: 1 located" in out.getvalue() and "around handle" in out.getvalue()


# ---- A3 custom render cameras -------------------------------------------------------------
def test_render_custom_views(tmp_path):
    m = to_mesh(critter(), res=40)
    default = read_png(render_views(m, str(tmp_path / "d.png"), size=80))
    front = read_png(render_views(m, str(tmp_path / "f.png"), size=80, views=["front"]))
    assert front.shape == (80, 80, 3)
    assert np.array_equal(front[20:78, :78], default[20:78, :78])  # the default FRONT camera (label, borders aside)
    three = read_png(render_views(m, str(tmp_path / "3.png"), size=60, views=["top", (30, 20), "back"]))
    five = read_png(render_views(m, str(tmp_path / "5.png"), size=60, views=["top"] * 5))
    assert three.shape == (60, 180, 3) and five.shape == (120, 180, 3)

    def lit(img):
        return (img.mean(-1) > 90).mean()
    near = read_png(render_views(m, str(tmp_path / "n.png"), size=80, views=[dict(az=90, el=10, distance=2)]))
    far = read_png(render_views(m, str(tmp_path / "x.png"), size=80, views=[dict(az=90, el=10, distance=6)]))
    assert lit(near[16:]) > 2 * lit(far[16:]) > 0  # perspective: closer looks bigger
    head = read_png(render_views(m, str(tmp_path / "h.png"), size=80, views=[dict(az=0, el=0, target="head")], edges=False))
    assert (head[30:50, 30:50].mean(-1) > 70).all() and lit(head[16:]) > 0.3  # the head fills the center
    with pytest.raises(ValueError, match="named views"):
        render_views(m, str(tmp_path / "e.png"), views=["sideways"])
    with pytest.raises(ValueError, match="unknown camera keys"):
        render_views(m, str(tmp_path / "e.png"), views=[dict(azimut=3)])


# ---- A4 limb_sdf / sdf_transform -------------------------------------------------------------
def test_limb_sdf_segments_and_knuckles():
    pts, r = [(0, 0, 0), (1, 0, 0), (1, 1, 0)], [0.2, 0.15, 0.1]
    plain = limb_sdf(pts, r, knuckle=1.0)
    assert np.isclose(plain.eval([(0.5, 0, 0.5)])[0], 0.5 - 0.175, atol=2e-3)  # round cone, exact
    bump = limb_sdf(pts, r, knuckle=1.3, k=0.02, name="leg")
    on = bump.eval([(1, 0, 0.15 * 1.3)])[0]  # knuckle surface: 1.3 x the joint radius
    assert abs(on) < 0.006 and plain.eval([(1, 0, 0.15 * 1.3)])[0] > 0.04
    P = np.random.default_rng(0).uniform(-0.5, 1.5, (2000, 3))
    assert (bump.eval(P) <= plain.eval(P) + 1e-12).all()  # bulges only add material
    assert bump.name == "leg" and bump.kind == "limb"
    m = to_mesh(bump, res=48)
    assert check(m, closed=True, expect_genus=0, contacts=None).ok
    with pytest.raises(ValueError):
        limb_sdf([(0, 0, 0)], 0.1)
    with pytest.raises(ValueError):
        limb_sdf(pts, [0.1, 0.1])


def test_sdf_transform_matches_moved_primitives():
    P = np.random.default_rng(1).uniform(-2, 2, (3000, 3))
    a = sdf_transform(capsule((0, 0, 0), (1, 0, 0), 0.1), rotation=("z", 90), pivot=(0, 0, 0), offset=(0, 0, 1))
    assert np.allclose(a.eval(P), capsule((0, 0, 1), (0, 1, 1), 0.1).eval(P), atol=1e-12)
    b = sdf_transform(sphere(0.5, center=(1, 0, 0)), scale=2)  # about its own center by default
    assert np.allclose(b.eval(P), sphere(1.0, center=(1, 0, 0)).eval(P), atol=1e-12)
    c = sdf_transform(ellipsoid((1, 0.3, 0.2)), rotation=(0, 30, 45), offset=(0.5, 0, 0))
    m = to_mesh(c, res=40)
    assert (m.verts >= c.lo - 1e-9).all() and (m.verts <= c.hi + 1e-9).all()
    pair = sdf_transform(smooth_union(sphere(0.3, name="u"), sphere(0.3, center=(0.55, 0, 0), name="v"), k=0.1),
                         offset=(5, 0, 0))
    rec = proximity(pair).stats["pairs"][0]
    assert (rec["a"], rec["b"], rec["state"]) == ("u", "v", "joined") and rec["at"][0] > 4
    with pytest.raises(ValueError):
        sdf_transform(sphere(1), scale=0)


# ---- A6 sparse vs dense ---------------------------------------------------------------------
def surface_dist(A, B, k=12):
    """Max over A's verts of the distance to B's surface (exact against the k nearest triangle centroids)."""
    from scipy.spatial import cKDTree
    T = B.triangles[0]
    a, b, c = B.verts[T[:, 0]], B.verts[T[:, 1]], B.verts[T[:, 2]]
    _, idx = cKDTree((a + b + c) / 3).query(A.verts, k=k)
    Q = _closest_on_tris(A.verts[:, None, :], a[idx], b[idx], c[idx])
    return float(np.linalg.norm(Q - A.verts[:, None, :], axis=2).min(1).max())


def test_sparse_meshing_keeps_thin_features():
    """The example creature (claws, horns) meshed sparse and dense agree to under half a voxel both ways, so thin
    features cannot silently vanish from the sparse grid."""
    ns = runpy.run_path(os.path.join(ROOT, "examples", "sdf_creature.py"))
    sdf = ns["sdf"]
    sparse, dense = to_mesh(sdf, res="draft"), to_mesh(sdf, res="draft", sparse=False)
    voxel = float((sdf.hi - sdf.lo).max()) / 64
    assert max(surface_dist(sparse, dense), surface_dist(dense, sparse)) < voxel / 2
    for part in ("claw", "horn"):  # every thin part present in both, with the same faces
        for i in range(12 if part == "claw" else 2):
            n = f"{part}.{i}"
            assert (n in sparse.groups) == (n in dense.groups)
            if n in dense.groups:
                assert sparse.groups[n].idx == dense.groups[n].idx
