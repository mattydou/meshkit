import time

import numpy as np
import pytest

import meshkit.sdf as S
from meshkit import (PRESETS, SDF, box, capsule, check, cone, displace_sdf, ellipsoid, noise, project_to_sdf,
                     proximity, smooth_union, sphere, subtract, sweep_sdf, to_mesh, union)


def forty_parts():
    parts = [ellipsoid((0.9, 0.45, 0.4), center=(0, 0, 0.75)), sphere(0.3, center=(1.0, 0, 1.05))]
    for i in range(38):
        t = i / 38 * 2 * np.pi
        parts.append(capsule((0.6 * np.cos(t), 0.35 * np.sin(t), 0.7),
                             (0.9 * np.cos(t), 0.6 * np.sin(t), 0.3 + 0.2 * np.sin(3 * t)), 0.06, 0.03))
    return smooth_union(*parts, k=0.1)


def unculled(sdf):
    return SDF(sdf.fn, sdf.lo, sdf.hi, lip=1e12)  # infinite lip: every grid point is evaluated


# ---- 1. sparse to_mesh ----------------------------------------------------------
@pytest.mark.parametrize("shape", [
    forty_parts(),
    ellipsoid((3, 0.4, 0.3), rotation=(20, 30, 40)),
    displace_sdf(sphere(1), noise(seed=2, scale=0.2, strength=0.15, octaves=5)),
    subtract(box(1.5, round=0.2), sphere(0.4, center=(0.75, 0, 0)), k=0.05),
])
def test_sparse_meshing_matches_full_grid_exactly(shape):
    a = to_mesh(shape, 48)
    evals = S._log[-1]["evals"]
    b = to_mesh(unculled(shape), 48)
    assert np.array_equal(a.verts, b.verts) and a.faces == b.faces
    assert evals < S._log[-1]["evals"]


def test_forty_primitives_at_180_under_10s():
    s = forty_parts()
    t0 = time.perf_counter()
    m = to_mesh(s, 180)
    assert time.perf_counter() - t0 < 10
    r = check(m, closed=True, expect_genus=0, contacts=None)
    assert r.ok and r.stats["comps"] == 1 and S._log[-1]["evals"] < 0.3 * S._log[-1]["dense"]


def test_union_culling_is_exact():
    parts = [sphere(0.2, center=(i * 0.3, 0, 0)) for i in range(8)] + [capsule((0, 0, 0), (2, 0, 1), 0.1)]
    P = np.random.default_rng(1).uniform(-1, 3, (8000, 3))  # > 4096: exercises block culling
    ref = parts[0].fn(P)
    for s in parts[1:]:
        ref = S._smin(ref, s.fn(P), 0.15)
    assert np.allclose(smooth_union(*parts, k=0.15).eval(P), ref, rtol=0, atol=1e-12)
    assert np.allclose(union(*parts).eval(P), np.min([s.fn(P) for s in parts], axis=0), rtol=0, atol=1e-12)
    cut = subtract(parts[0], parts[3])
    assert np.allclose(cut.eval(P), np.maximum(parts[0].fn(P), -parts[3].fn(P)), rtol=0, atol=1e-12)


# ---- 2. presets ------------------------------------------------------------------
def test_res_presets_default_draft_and_override():
    s = sphere(1)
    to_mesh(s)
    assert S._log[-1]["res"] == "draft" and S._log[-1]["cells"] == PRESETS["draft"]
    to_mesh(s, res="final")
    assert S._log[-1]["cells"] == PRESETS["final"]
    saved = S._defaults["res"]
    try:
        S._defaults["res"] = "final"  # what `run --final` does
        to_mesh(s)
        assert S._log[-1]["res"] == "final"
        to_mesh(s, 30)
        assert S._log[-1]["cells"] == 30  # an explicit number always wins
    finally:
        S._defaults["res"] = saved
    to_mesh(s, resolution=24)
    assert S._log[-1]["cells"] == 24
    with pytest.raises(ValueError, match="draft"):
        to_mesh(s, res="best")


# ---- 3. sweep_sdf ----------------------------------------------------------------
def test_sweep_sdf_curves_through_points_with_radii():
    pts = [(0, 0, 0), (1, 0, 0.5), (2, 0, 0)]
    even = sweep_sdf(pts, 0.2)
    for p in pts:  # constant radius: exact tube radius at each control point
        assert np.isclose(even((p[0], p[1] + 0.25, p[2])), 0.05, atol=1e-6)
    s = sweep_sdf(pts, [0.3, 0.2, 0.1])
    assert np.isclose(s((0, 0.35, 0)), 0.05, atol=1e-6)  # exact at the thick end
    assert 0.04 < s((1, 0.25, 0.5)) <= 0.05  # the thicker neighbour bulges past a tapered middle
    assert s((1, 0, 0.5)) < -0.19 and s((1, 0, 0)) > 0  # passes through the raised middle point, not the chord
    straight = sweep_sdf([(0, 0, 0), (0, 0, 1)], 0.25)
    assert np.allclose(straight.eval([(0.5, 0, 0.5), (0, 0, 1.5), (0, 0, 0.5)]), [0.25, 0.25, -0.25])
    assert straight.lip == 1 and np.allclose(straight.hi, (0.25, 0.25, 1.25))
    m = to_mesh(sweep_sdf([(0, 0, 0), (0.5, 0.4, 0.3), (1, 0, 0.8), (1.6, 0.2, 0.6)], [0.2, 0.12, 0.15, 0.05]), 48)
    assert check(m, closed=True, expect_genus=0).ok
    with pytest.raises(ValueError, match="radii"):
        sweep_sdf(pts, [0.1, 0.2])


def test_tapered_capsule_is_an_exact_round_cone():
    a, b, ra, rb = np.array([0, 0, 0.0]), np.array([1, 0.5, 0.2]), 0.4, 0.1
    c = capsule(a, b, ra, rb)
    P = np.random.default_rng(2).uniform(-1, 2, (3000, 3))
    t = np.linspace(0, 1, 4001)[None, :, None]
    centers, radii = a + t * (b - a), ra + t[..., 0] * (rb - ra)  # union of interpolated spheres = round cone
    brute = (np.linalg.norm(P[:, None] - centers, axis=-1) - radii).min(1)
    out = brute > 0
    assert np.allclose(c.eval(P)[out], brute[out], atol=2e-4)
    assert np.array_equal(c.eval(P) < 0, brute < 0)
    assert np.isclose(capsule((0, 0, 0), (0, 0, 0.1), 1.0, 0.2)((0, 0, 1.5)), 0.5)  # one sphere inside the other


# ---- 4. eval / normal / project ----------------------------------------------------
def test_eval_normal_and_project_to_sdf():
    s = sphere(2, center=(1, 0, 0))
    assert s.eval((1, 0, 3)).shape == (1,) and np.isclose(s.eval((1, 0, 3))[0], 1)
    assert np.allclose(s.normal([(1, 0, 5), (4, 0, 0)]), [(0, 0, 1), (1, 0, 0)], atol=1e-6)
    P = np.array([(1, 0, 5), (1.5, 0.2, 0.1), (-3, 1, 1)])
    pr = project_to_sdf(s, P)
    assert np.allclose(np.linalg.norm(pr.points - (1, 0, 0), axis=1), 2, atol=1e-6)
    assert np.allclose(pr.dist, s.eval(P)) and pr.resid.max() < 1e-6
    assert np.allclose(pr.normals, (pr.points - (1, 0, 0)) / 2, atol=1e-5)
    blob = smooth_union(ellipsoid((1, 0.5, 0.4)), capsule((0.5, 0, 0), (1.5, 0, 0.8), 0.2), k=0.2)
    pr = project_to_sdf(blob, np.random.default_rng(3).uniform(-1.5, 1.5, (200, 3)))
    assert pr.resid.max() < 1e-5 and np.abs(blob.eval(pr.points)).max() < 1e-5


# ---- 5. rotation -------------------------------------------------------------------
def test_rotation_forms_and_oriented_bounds():
    e = ellipsoid((2, 1, 0.5), rotation=("z", 90))
    assert np.allclose(e.hi, (1, 2, 0.5)) and np.isclose(e((0, 2, 0)), 0, atol=1e-12)
    R = S._rot([("x", 30), ("z", 45)])
    assert np.allclose(R, S._rot(np.array(R))) and np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(S._rot((30, 0, 45)), R)  # Euler XYZ = x first, then z
    b = box((2, 1, 1), rotation=("z", 45))
    corner = np.array([1, 0.5, 0.5]) @ S._rot(("z", 45)).T
    assert np.isclose(b(corner), 0, atol=1e-12) and np.allclose(b.hi[:2], (1.5 / np.sqrt(2),) * 2)
    c = capsule((0, 0, 0), (0, 0, 1), 0.1, rotation=("y", 90))  # b turns about a
    assert np.isclose(c((1, 0, 0)), -0.1) and np.isclose(c((0, 0, 1)), 0.9) and np.isclose(c((1, 0, 1)), 0.9)
    k = cone((0, 0, 0), (0, 0, 1), 0.5, 0.0, rotation=("x", -90))
    assert np.isclose(k((0, 1, 0)), 0, atol=1e-12) and k((0, 0.3, 0)) < 0
    m = to_mesh(ellipsoid((1, 0.3, 0.2), rotation=(0, 0, 30)), 40)
    assert check(m, closed=True).ok


# ---- 6. per-part k -----------------------------------------------------------------
def test_per_part_k_blends_sharp_or_soft_in_one_call():
    body = sphere(1)
    soft = sphere(0.4, center=(1.2, 0, 0))
    sharp = sphere(0.4, center=(-1.2, 0, 0))
    s = smooth_union(body, soft, sharp, k=[0.4, 0.4, 0.0])
    seam_soft, seam_sharp = (0.95, 0.42, 0), (-0.95, 0.42, 0)
    assert s(seam_soft) < union(body, soft)(seam_soft) - 0.02  # fillet adds material
    assert np.isclose(s(seam_sharp), union(body, sharp)(seam_sharp))  # crisp junction
    again = smooth_union(sharp, body, soft, k=[0.0, 0.4, 0.4])  # order of parts does not matter
    P = np.random.default_rng(4).uniform(-2, 2, (2000, 3))
    assert np.allclose(s.eval(P), again.eval(P))
    assert s.k == (0.4, 0.4, 0.0) and np.allclose(s.hi, (1.6 + 0.4, 1.4, 1.4))
    with pytest.raises(ValueError, match="k values"):
        smooth_union(body, soft, k=[0.1, 0.2, 0.3])


# ---- 7. proximity --------------------------------------------------------------------
def legs(gap, k=0.15):
    body = ellipsoid((0.8, 0.4, 0.3), center=(0, 0, 1), name="body")
    r = 0.1
    l1 = capsule((0.1 + gap / 2 + r, 0, 1), (0.1 + gap / 2 + r, 0, 0.1), r, name="leg")
    l2 = capsule((0.1 - gap / 2 - r, 0, 1), (0.1 - gap / 2 - r, 0, 0.1), r, name="leg")
    return smooth_union(body, l1, l2, k=k)


def test_proximity_flags_near_parts_and_allows_intended_ones():
    p = proximity(legs(0.04))
    assert not p.ok and "leg.0-leg.1 gap 0.04" in str(p) and "joined: body-leg.0 body-leg.1" in str(p)
    rec = next(r for r in p.stats["pairs"] if r["state"] == "near")
    assert np.isclose(rec["gap"], 0.04, atol=1e-3) and np.isclose(rec["k"], 0.15) and np.isclose(rec["threshold"], 0.075)
    assert proximity(legs(0.04), allow=[("leg.1", "leg.0")]).ok
    assert proximity(legs(0.2)).ok and proximity(legs(0.04), threshold=0.01).ok
    with pytest.raises(KeyError, match="parts: body leg.0 leg.1"):
        proximity(legs(0.04), allow=[("leg", "body")])


def test_proximity_covered_loose_and_tree_walk():
    pad = sphere(0.3, name="pad")
    toes = [sphere(0.08, center=(0.25, y, 0)) for y in (-0.09, 0.09)]  # 0.02 apart, but the gap is inside the pad
    eye = sphere(0.05, center=(3, 0, 0), name="eye")
    shape = displace_sdf(subtract(smooth_union(pad, *toes, eye, k=0.05), sphere(0.1, center=(-0.3, 0, 0))),
                         noise(strength=0.01))
    p = proximity(shape)
    states = {(r["a"], r["b"]): r["state"] for r in p.stats["pairs"]}
    assert p.stats["parts"] == ["pad", "sphere0", "sphere1", "eye"]  # subtracted sphere is not a part
    assert states[("sphere0", "sphere1")] == "covered" and states[("pad", "eye")] == "apart"
    assert p.ok and "loose (joined to nothing; separate pieces unless blended): eye" in str(p)


def test_genus_catches_parts_fused_into_a_loop():
    body = ellipsoid((0.8, 0.4, 0.3), center=(0, 0, 1))
    splayed = [capsule((x, 0, 1), (x * 0.1, 0, 0.1), 0.1) for x in (-0.4, 0.4)]  # legs meet at the feet
    m = to_mesh(smooth_union(body, *splayed, k=0.05))
    r = check(m, closed=True, expect_genus=0)
    assert not r.ok and r.stats["genus"] == 1 and "genus 1, expected 0" in str(r)
