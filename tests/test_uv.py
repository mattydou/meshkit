"""B1/B2: per-corner UVs from the generators, surviving ops and subdivide; projections, pack, unwrap; UV checks and
the UV sheet."""
import numpy as np
import pytest

from meshkit import (Mesh, boundary_loops, bridge, check, cube, cylinder, decimate, delete, displace,
                     export_uv_sheet, extrude, flip, from_json, grid, inset, join, load_obj, merge_by_distance,
                     mirror, pack, project_uvs, quad_sphere, revolve, ring, rotate, scale, select_by_normal, sphere,
                     subdivide, sweep, to_json, to_mesh, to_obj, translate, unwrap, uv_islands, uv_seams)
from meshkit.png import read_png
from meshkit.uv import _signed_area, corner_tris, uv_stats


def healthy(m, islands=None):
    st = uv_stats(m)
    assert st["has"] and st["missing"] == st["flipped"] == st["zero"] == 0, st
    assert st["overlap_texels"] == st["self_overlap_texels"] == st["outside"] == 0, st
    if islands is not None:
        assert st["islands"] == islands, st
    return st


# ---- generators ---------------------------------------------------------------------------------
@pytest.mark.parametrize("make, islands", [
    (lambda: grid(2, (3, 2)), 1),
    (lambda: cube((2, 1, 0.5), divisions=2), 6),
    (lambda: quad_sphere(1, 6), 6),
    (lambda: cylinder(0.5, 2, rings=3), 3),
    (lambda: cylinder(0.5, 1, radius_top=0), 2),  # cone side unrolled flat: one island, not a fan of slivers
    (lambda: revolve([(0, 0), (0.5, 0.2), (0.4, 0.8), (0, 1)]), 1),
    (lambda: revolve([(2, 0), (2.5, 0), (2.5, 0.5), (2, 0.5)], closed=True), 1),
    (lambda: ring(), 1),
    (lambda: sweep(Mesh(), [(0, 0, 0), (1, 0, 1), (2, 0, 0)], 0.2, caps=True)[0], 3),
    (lambda: sweep(Mesh(), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 0.1, closed=True)[0], 1),
])
def test_generators_come_with_clean_uvs(make, islands):
    m = make()
    st = healthy(m, islands)
    assert m.uvs.shape == (len(m.corner_verts), 2) and (m.uvs >= 0).all() and (m.uvs <= 1).all()
    assert st["stretch_p95"] < 1.6  # world-scale UVs: little distortion


def test_layouts_follow_the_shape():
    c = cube((2, 1, 1))
    top = sorted(select_by_normal(c, "z", 1))[0]
    side = sorted(select_by_normal(c, "-y", 1))[0]
    a = np.abs(_signed_area(c.uvs, c))
    assert np.isclose(a[top] / a[side], 1.0)  # same texel density on every side (2x1 faces both)
    U = c.face_uvs(top)
    assert np.isclose(np.ptp(U[:, 0]) / np.ptp(U[:, 1]), 2.0)  # aspect kept
    s, _ = sweep(Mesh(), [(0, 0, 0), (4, 0, 0)], 0.1, segments=8)
    U = s.uvs
    assert np.ptp(U[:, 1]) > 5 * np.ptp(U[:, 0])  # u around (0.63 long), v along (4 long)
    assert len(uv_seams(quad_sphere(1, 4))) == 12 * 4  # cube-face projection: cut along the 12 cube edges


# ---- surviving ops ------------------------------------------------------------------------------------
def test_uvs_survive_transforms_and_topology_ops():
    c = cube(divisions=2)
    for m in (translate(c, (1, 2, 3))[0], rotate(c, "z", 30)[0], scale(c, 2)[0],
              displace(c, lambda P: 0.1 * P[:, 0])[0]):
        assert np.array_equal(m.uvs, c.uvs)
    top = select_by_normal(c, "z", 1)
    e, cr = extrude(c, top, 0.5)
    for f in range(len(c.faces)):  # every original face keeps its UVs (top faces moved, same UVs)
        assert np.array_equal(e.face_uvs(f), c.face_uvs(f))
    assert uv_stats(e)["zero"] == len(cr.faces)  # walls: the edge's UVs, no UV area (check() says so)
    i, ci = inset(c, top, 0.1)
    healthy(i)
    for f in ci.inner:  # inner faces shrink inside their old UV square; ring quads fill the border
        assert np.abs(_signed_area(i.uvs, i)[f]) < np.abs(_signed_area(c.uvs, c)[f])
    d, cd = delete(c, top)
    healthy(d)
    assert np.array_equal(d.face_uvs(0), c.face_uvs(cd.fmap.tolist().index(0)))
    j, _ = join(translate(c, (3, 0, 0))[0], c)
    assert uv_stats(j)["overlap_texels"] > 0 and healthy(pack(j)[0])  # join overlaps; pack fixes
    f, _ = flip(c)
    assert uv_stats(f)["flipped"] == len(c.faces)  # turned inside out: UVs read mirrored too


def test_mirror_merge_and_mixed_meshes():
    half, _ = translate(cube(), (0.6, 0, 0))
    m, _ = mirror(half, "x", weld=False)
    st = uv_stats(m)
    assert st["flipped"] == 0 and st["overlap_texels"] == 0 and st["outside"] > 0  # copies sit right of 0..1
    healthy(pack(m)[0], islands=12)
    g = grid(1, 2)
    g2, _ = join(g, translate(g, (1, 0, 0))[0])
    w, _ = merge_by_distance(g2)
    assert w.uvs is not None and len(w.faces) == 8 and np.array_equal(w.uvs[:16], g2.uvs[:16])
    plain = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    mixed, _ = join(cube(), translate(plain, (3, 0, 0))[0])
    st = uv_stats(mixed)
    assert st["missing"] == 1 and "missing UVs: 1 faces" in str(check(mixed, uvs=True, contacts=None))
    tubes, _ = join(cylinder(caps=False), translate(cylinder(caps=False), (0, 0, 2))[0])
    loops = boundary_loops(tubes)
    b, cb = bridge(tubes, loops[1], loops[2])  # faces from nothing get no UVs; check() says where
    assert uv_stats(b)["missing"] == len(cb.faces) == 16


def test_subdivide_positions_groups_and_uvs():
    c = cube(group="box")
    s, cr = subdivide(c, 2)
    assert len(s.faces) == 96 and check(s, closed=True, contacts=None).ok
    r = np.linalg.norm(s.verts, axis=1)
    assert 0.43 < r.min() and r.max() < 0.46  # rounded toward a sphere
    assert len(s.groups["box"].idx) == 96 and list(cr.verts)[0] == 8
    healthy(s, islands=6)
    lin, _ = subdivide(cube(), smooth=False)
    assert np.allclose(np.abs(lin.verts).max(1), 0.5)  # plain split stays on the cube
    q = lin.face_uvs(0)
    a = cube().face_uvs(0)
    assert np.allclose(q[0], a[0]) and np.allclose(q[1], (a[0] + a[1]) / 2) and np.allclose(q[2], a.mean(0))
    g, _ = subdivide(grid(1, 2))
    assert check(g).ok and np.allclose(g.verts[:, :2].min(0), -0.5)  # open border kept as a crease curve
    wire = Mesh([(0, 0, 0), (1, 0, 0)], loose_edges=[(0, 1)])
    w, _ = subdivide(wire)
    assert len(w.verts) == 3 and w.loose_edges == {(0, 2), (1, 2)}


def test_decimate_keeps_valid_uvs():
    m = quad_sphere(1, 16)
    d, _ = decimate(m, 800)
    st = healthy(d, islands=6)
    assert st["stretch_p95"] < 1.6 and len(d.faces) <= 800
    seam_verts = {v for e in uv_seams(m) for v in e}
    assert len(d.verts) >= len(seam_verts)  # seam verts are locked, so islands keep their outlines


def test_json_and_obj_round_trip_uvs(tmp_path):
    m, _ = join(cube(), translate(Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]), (3, 0, 0))[0])
    j = from_json(to_json(m, str(tmp_path / "m.json")))
    assert np.allclose(j.uvs, m.uvs, equal_nan=True)
    o = load_obj(to_obj(m, str(tmp_path / "m.obj")))
    assert np.allclose(o.uvs, m.uvs, atol=1e-6, equal_nan=True)  # vt per corner; the plain triangle has none
    text = open(tmp_path / "m.obj").read()
    assert "\nvt " in text and "\nf 5/9 " in text and "\nf 9 10 11\n" in text


# ---- projections, pack, unwrap ---------------------------------------------------------------------------
def test_projections():
    s = quad_sphere(1, 8)
    m, c = project_uvs(s, "box")
    healthy(m, 6)
    tube = cylinder(0.5, 2, rings=4, caps=False)
    m, c = project_uvs(tube, "cylinder")
    healthy(m, 1)
    assert c.islands == 1 and uv_stats(m)["stretch_p95"] < 1.05
    m, _ = project_uvs(s, "sphere")
    st = uv_stats(m)  # faces around the poles fold over (the projection's limit); the rest is clean
    assert st["flipped"] == st["overlap_texels"] == 0 and st["islands"] == 1
    p, _ = project_uvs(s, "planar")
    st = uv_stats(p)
    assert st["flipped"] == len(s.faces) // 2 and st["self_overlap_texels"] > 0  # the back folds onto the front
    c = cube()
    top = select_by_normal(c, "z", 1)
    m, _ = project_uvs(c, "planar", faces=top, axis="z")
    others = [f for f in range(6) if f not in top]
    assert all(np.array_equal(m.face_uvs(f), c.face_uvs(f)) for f in others)


def test_pack_uniform_density_and_margin():
    a = cube(1)
    b = translate(cube(0.5), (2, 0, 0))[0]
    j, _ = join(a, b)
    p, cr = pack(j, margin=0.02)
    healthy(p, islands=12)
    area = np.abs(_signed_area(p.uvs, p))
    assert np.isclose(area[0] / area[6], 4.0)  # UV area follows 3D area
    assert p.uvs.min() >= 0.02 - 1e-9 and p.uvs.max() <= 0.98 + 1e-9
    isl = uv_islands(p)
    lo = np.array([p.uvs[np.isin(np.repeat(np.arange(12), 4), np.flatnonzero(isl == i))].min(0) for i in range(12)])
    hi = np.array([p.uvs[np.isin(np.repeat(np.arange(12), 4), np.flatnonzero(isl == i))].max(0) for i in range(12)])
    gaps = [max((lo[x] - hi[y]).max(), (lo[y] - hi[x]).max()) for x in range(12) for y in range(x + 1, 12)]
    assert min(gaps) >= 0.02 - 1e-9  # boxes at least a margin apart
    with pytest.raises(ValueError, match="no UVs"):
        pack(to_mesh(sphere(1), 12))


def test_unwrap_xatlas_and_box():
    pytest.importorskip("xatlas")
    m = to_mesh(sphere(1), 24)
    a, ca = unwrap(m)
    b, _ = unwrap(m)
    assert np.array_equal(a.uvs, b.uvs)  # deterministic
    st = healthy(a)
    assert st["stretch_p95"] < 2 and ca.stretch_p95 < 2 and (a.uvs >= 0).all() and (a.uvs <= 1).all()
    q, _ = unwrap(cube(divisions=3))  # polygons: quads stay quads, one UV per corner
    healthy(q)
    assert len(q.faces) == 54 and q.faces == cube(divisions=3).faces
    x, cx = unwrap(m, method="box")
    healthy(x)
    assert cx.islands >= 6
    with pytest.raises(ValueError):
        unwrap(m, method="lscm")


# ---- check() and the sheet --------------------------------------------------------------------------------
def test_check_reports_uv_problems():
    r = check(quad_sphere(1, 6), uvs=True)
    assert r.ok and "uv: islands=6" in str(r) and "stretch worst=" in str(r) and r.stats["uv"]["coverage"] > 0.4
    assert "no UVs" in str(check(to_mesh(sphere(1), 12), uvs=True))
    j, _ = join(cube(), translate(cube(), (2, 0, 0))[0])
    assert "overlapping UV islands" in str(check(j, uvs=True, contacts=None)) and "(pack(m))" in str(check(j, uvs=True))
    f, _ = flip(cube(), [0])
    t = str(check(f, uvs=True, contacts=None))
    assert "flipped UV faces (mirrored in UV): 1" in t
    e, _ = extrude(cube(), select_by_normal(cube(), "z", 1), 0.5)
    assert "zero-area UV faces: 4" in str(check(e, uvs=True))
    m, _ = mirror(translate(cube(), (1, 0, 0))[0], "x", weld=False)
    assert "UVs outside 0..1: 3 faces" in str(check(m, uvs=True, contacts=None))
    p, _ = project_uvs(quad_sphere(1, 6), "planar")
    assert "folding onto themselves" in str(check(p, uvs=True))
    assert "uv:" not in str(check(cube()))  # UV checks only on request


def test_stretch_numbers():
    c = cube()
    U = c.uvs.copy()
    U[:4] = U[:4].mean(0) + (U[:4] - U[:4].mean(0)) * 2  # one face's UVs twice as big (4x area)
    st = uv_stats(Mesh(c.verts, c.faces, uvs=U))
    assert st["stretch_worst"] > 2.5 and st["stretch_p95"] > 2.5 and st["overlap_texels"] >= 0
    healthy(c)


def test_uv_sheet(tmp_path):
    j, _ = join(cube(), translate(Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]), (3, 0, 0))[0])
    j, _ = pack(project_uvs(j, "box", faces=[6])[0])
    img = read_png(export_uv_sheet(j, str(tmp_path / "uv.png"), size=256))
    assert img.shape == (256, 256, 3)
    px = img.reshape(-1, 3).astype(int)
    seam = ((px[:, 0] == 255) & (px[:, 1] == 150) & (px[:, 2] == 40)).sum()
    border = ((px[:, 0] == 110) & (px[:, 1] == 230) & (px[:, 2] == 240)).sum()
    assert seam > 100 and border > 20  # cube seams orange, the lone triangle's open border cyan
    over, _ = join(cube(), cube())
    img = read_png(export_uv_sheet(over, str(tmp_path / "o.png"), size=128)).reshape(-1, 3).astype(int)
    assert ((img[:, 0] == 220) & (img[:, 1] == 40)).sum() > 1000  # overlaps in red
    assert corner_tris(cube()).shape == (12, 3)
