"""B3-B7: masks (3D, seam-free), patterns, ramps, the layer stack, bake (maps, padding, normal map from height) and
textured renders."""
import numpy as np
import pytest

from meshkit import (Mesh, add_group, bake, color, cube, cylinder, grid, join, layers, mask_curvature, mask_distance,
                     mask_group, mask_height, mask_noise, mask_normal, quad_sphere, ramp, render_views, revolve,
                     select_by_bbox, smoothstep, sphere, spots, stripes, surface_values, sweep, to_mesh, translate,
                     uv_seams)
from meshkit.png import read_png
from meshkit.texture import Surface
from meshkit.uv import _edge_pairs


def sphere_mesh():
    return quad_sphere(1, 8)


def torus():
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return revolve(np.c_[1 + 0.4 * np.cos(th), 0.4 * np.sin(th)], segments=32, closed=True)


# ---- masks ------------------------------------------------------------------------------------------
def test_normal_height_and_noise_masks():
    m = sphere_mesh()
    up = surface_values(m, mask_normal("z", 0.0, 0.6))
    z = m.verts[:, 2]
    assert np.allclose(up[z > 0.99], 1) and np.allclose(up[z < 0], 0)
    assert 0 < up[(z > 0.2) & (z < 0.4)].mean() < 1  # smooth transition
    h = surface_values(m, mask_height(-0.5, 0.5))
    assert np.allclose(h[z <= -0.5], 0) and np.allclose(h[z >= 0.5], 1) and (np.diff(h[np.argsort(z)]) >= -1e-12).all()
    down = surface_values(m, mask_height(0.5, -0.5))
    assert np.allclose(down, 1 - h)
    n1 = surface_values(m, mask_noise(seed=3, scale=0.3))
    assert np.array_equal(n1, surface_values(m, mask_noise(seed=3, scale=0.3)))  # deterministic
    assert not np.array_equal(n1, surface_values(m, mask_noise(seed=4, scale=0.3)))
    assert 0 <= n1.min() and n1.max() <= 1 and 0.2 < n1.mean() < 0.8


def test_group_distance_and_wildcard_masks():
    a, b = cube(group="leg.0"), translate(cube(group="leg.1"), (3, 0, 0))[0]
    m, _ = join(a, b, translate(cube(group="body"), (1.5, 0, 0))[0])
    f = surface_values(m, mask_group("leg.0"), at="faces")
    assert f[:6].tolist() == [1] * 6 and f[6:].sum() == 0
    both = surface_values(m, mask_group("leg*"), at="faces")
    assert both.tolist() == [1] * 12 + [0] * 6
    assert np.array_equal(both, surface_values(m, mask_group(["leg.0", "leg.1"]), at="faces"))
    d = surface_values(m, mask_distance("leg.0", radius=2.0))
    assert np.allclose(d[:8], 1) and np.allclose(d[8:16], 0)  # leg.1 is 2.0 away at its nearest corner
    mid = d[16:]  # body verts: 0.5 and 1.5 away (x = 1 and 2)
    assert np.all((mid > 0) & (mid < 1)) and mid[m.verts[16:, 0] < 1.5].min() > mid[m.verts[16:, 0] > 1.5].max()
    g = grid(2, 20)
    g, _ = add_group(g, "left", select_by_bbox(g, hi=(0, None, None), kind="f"))
    blur = surface_values(g, mask_group("left", blur=0.4))
    x = g.verts[:, 0]
    assert np.allclose(blur[x <= -0.25], 1) and np.allclose(blur[x >= 0.25], 0)
    assert 0.3 < blur[np.isclose(x, 0)].mean() < 0.7  # half way on the border
    with pytest.raises(KeyError, match="no group matches"):
        surface_values(m, mask_group("arm*"))


def test_curvature_mask():
    from meshkit import displace
    g = grid(4, 60)
    bump = lambda P, c: np.exp(-((P[:, 0] - c) ** 2 + P[:, 1] ** 2) / 0.1)  # noqa: E731
    g, _ = displace(g, lambda P: 0.3 * bump(P, -1) - 0.3 * bump(P, 1), along="z")  # a bump and a dip
    convex = surface_values(g, mask_curvature("convex", 0.2, 1.0))
    concave = surface_values(g, mask_curvature("concave", 0.2, 1.0))
    top, pit = np.argmin(np.hypot(g.verts[:, 0] + 1, g.verts[:, 1])), np.argmin(np.hypot(g.verts[:, 0] - 1, g.verts[:, 1]))
    assert convex[top] > 0.9 and concave[top] == 0 and concave[pit] > 0.9 and convex[pit] == 0
    flat = np.hypot(g.verts[:, 0], g.verts[:, 1] - 1.5) < 0.3
    assert convex[flat].max() < 0.1 and concave[flat].max() < 0.1
    t = torus()  # outer rim curves more than the inner (mean curvature): convex reads higher outside
    r = np.hypot(t.verts[:, 0], t.verts[:, 1])
    conv = surface_values(t, mask_curvature("convex", 0.0, 1.0))
    assert conv[r > 1.3].mean() > conv[r < 0.7].mean()
    with pytest.raises(ValueError):
        mask_curvature("flat")


def test_combinators_and_labels():
    m = sphere_mesh()
    a, b = mask_normal("z"), mask_height(-0.2, 0.2)
    va, vb = surface_values(m, a), surface_values(m, b)
    assert np.allclose(surface_values(m, a * b), va * vb)
    assert np.allclose(surface_values(m, a | b), np.maximum(va, vb))
    assert np.allclose(surface_values(m, a & b), np.minimum(va, vb))
    assert np.allclose(surface_values(m, ~a), 1 - va)
    assert np.allclose(surface_values(m, 0.5 * a), 0.5 * va)
    s = surface_values(m, smoothstep(b, 0.25, 0.75))
    assert np.allclose(s[vb <= 0.25], 0) and np.allclose(s[vb >= 0.75], 1)
    assert "normal(z) * height" in repr(a * b)


# ---- patterns, ramps, colors ----------------------------------------------------------------------------
def test_stripes_along_axis_and_curve():
    c = cylinder(0.3, 4, segments=12, rings=200)
    v = surface_values(c, stripes("z", period=0.5, width=0.4, soft=0.0))
    assert abs(v.mean() - 0.4) < 0.03
    shift = surface_values(translate(c, (0, 0, 0.5))[0], stripes("z", period=0.5, width=0.4, soft=0.0))
    assert np.array_equal(v, shift)  # periodic
    from meshkit import catmull_rom
    path = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, 2, 0)]
    s, _ = sweep(Mesh(), path, 0.1, segments=8, resolution=40)
    along = surface_values(s, stripes(along=path, period=0.25, width=0.5, soft=0.0))
    C, _ = catmull_rom(path, 64)
    arc_c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    arc = arc_c[np.argmin(np.linalg.norm(s.verts[:, None] - C[None], axis=2), axis=1)]  # along the curve
    frac = (arc / 0.25) % 1
    ok = np.abs(frac - 0.5) > 0.1  # away from band edges
    assert ((along == (frac < 0.5))[ok]).mean() > 0.97  # bands follow the bend, not an axis
    straight = surface_values(c, stripes(along=[(0, 0, -2), (0, 0, 2)], period=0.5, width=0.4, soft=0.0))
    assert np.array_equal(straight, surface_values(translate(c, (0, 0, 0))[0],
                                                   stripes("z", period=0.5, width=0.4, soft=0.0, offset=4.0)))
    warped = surface_values(c, stripes("z", period=0.5, warp=0.3))
    assert not np.allclose(warped, surface_values(c, stripes("z", period=0.5)))


def test_spots_shapes():
    m = quad_sphere(1, 24)
    flat = surface_values(m, spots(0.15, size=0.4, seed=1))
    assert 0.02 < flat.mean() < 0.5 and flat.min() == 0 and flat.max() == 1
    assert np.array_equal(flat, surface_values(m, spots(0.15, size=0.4, seed=1)))
    assert not np.array_equal(flat, surface_values(m, spots(0.15, size=0.4, seed=2)))
    dome = surface_values(m, spots(0.15, size=0.4, seed=1, shape="dome"))
    assert dome.max() <= 1 and ((dome > 0) == (flat > 0)).mean() > 0.9
    cells = surface_values(m, spots(0.15, soft=0.1, shape="cells"))
    assert 0.05 < cells.mean() < 0.6
    with pytest.raises(ValueError):
        spots(shape="stars")


def test_ramp_and_colors():
    assert np.allclose(color("#ff8000"), (1, 128 / 255, 0)) and np.allclose(color((255, 0, 0)), (1, 0, 0))
    assert np.allclose(color((0.2, 0.4, 0.6)), (0.2, 0.4, 0.6))
    m = sphere_mesh()
    h = surface_values(m, mask_height(-1, 1))
    c = surface_values(m, ramp(mask_height(-1, 1), ["#000000", "#ffffff"]))
    assert np.allclose(c, np.repeat(h[:, None], 3, 1))
    c2 = surface_values(m, ramp(mask_height(-1, 1), [(0.0, "#ff0000"), (0.5, "#00ff00"), (1.0, "#0000ff")]))
    assert np.allclose(c2[np.argmax(m.verts[:, 2])], (0, 0, 1)) and np.allclose(c2[np.argmin(m.verts[:, 2])], (1, 0, 0))


# ---- layer stack ---------------------------------------------------------------------------------------------
def test_layers_blend_like_a_paint_program():
    m = sphere_mesh()
    half = mask_height(-1, 1)
    tex = layers([("#000000", None, 1), ((1.0, 1.0, 1.0), half, 0.5)],
                 roughness=[(0.8, None, 1), (0.2, mask_normal("z"), 1)],
                 height=[(spots(0.2), None, 0.01), (0.5, mask_normal("-z"), -0.002)])
    out = surface_values(m, tex)
    h = surface_values(m, half)
    assert np.allclose(out["albedo"], np.repeat((0.5 * h)[:, None], 3, 1))  # black + white x mask x opacity
    up = surface_values(m, mask_normal("z"))
    assert np.allclose(out["roughness"], 0.8 + (0.2 - 0.8) * up)
    sp, dn = surface_values(m, spots(0.2)), surface_values(m, mask_normal("-z"))
    assert np.allclose(out["height"], sp * 0.01 - 0.001 * dn)
    assert np.allclose(surface_values(m, layers("#336699"))["albedo"], color("#336699"))
    assert np.allclose(surface_values(m, layers([(0.25, 0.5, 0.75)]))["albedo"], (0.25, 0.5, 0.75))  # a bare rgb


# ---- bake ------------------------------------------------------------------------------------------------------
def test_bake_maps_padding_and_files(tmp_path):
    m = quad_sphere(1, 8)
    tex = layers([("#ff0000", None, 1), ("#0000ff", mask_normal("-z", 0.0, 0.3), 1)], roughness=0.3,
                 height=[(spots(0.3, shape="dome"), None, 0.02)])
    b = bake(m, tex, 128, out=str(tmp_path), name="ball")
    assert b.albedo.shape == (128, 128, 3) and b.roughness.shape == (128, 128) and b.normal.shape == (128, 128, 3)
    assert 0.3 < b.coverage < 0.9 and sorted(b.paths) == ["albedo", "height", "normal", "roughness"]
    assert np.array_equal(read_png(b.paths["albedo"]), b.albedo) and read_png(b.paths["height"]).dtype == np.uint16
    assert (b.roughness == 77).all()  # 0.3 everywhere, padding included
    a = b.albedo.reshape(-1, 3).astype(int)
    assert (a.sum(1) > 100).all()  # padding repeats island colors: no black gaps anywhere
    covered = np.zeros(128 * 128, dtype=bool)
    from meshkit.uv import _cover, uv_pixels
    covered[_cover(uv_pixels(m, 128), 128, 128)[0]] = True
    edge = ~covered.reshape(128, 128) & np.roll(covered.reshape(128, 128), 1, axis=1)  # just right of an island
    ys, xs = np.nonzero(edge)
    assert np.abs(b.albedo[ys, xs].astype(int) - b.albedo[ys, xs - 1].astype(int)).max() <= 2  # = its neighbor
    again = bake(m, tex, 128)
    assert np.array_equal(again.albedo, b.albedo) and np.array_equal(again.normal, b.normal)
    with pytest.raises(ValueError, match="no UVs"):
        bake(to_mesh(sphere(1), 12), tex)


def test_bake_is_seam_free():
    """Texels on both sides of every UV seam get the same color, because masks are evaluated in 3D."""
    m = quad_sphere(1, 8)
    size = 256
    tex = layers([(ramp(mask_noise(seed=1, scale=2.0, lo=-0.5, hi=0.5), ["#000000", "#ffffff"]), None, 1),
                  ("#ff0000", mask_normal("z", -0.5, 0.9), 0.7)])
    alb = bake(m, tex, size, pad=4).albedo.astype(float)
    ca, cb, cb2, ca2, f, g = _edge_pairs(m)
    U = m.uvs
    seam = np.abs(U[ca] - U[ca2]).max(1) + np.abs(U[cb] - U[cb2]).max(1) > 1e-6
    assert seam.sum() == len(uv_seams(m)) == 12 * 8  # the cube-face projection's cuts
    cs = m.corner_start

    def sample(ci, cj, face):
        mid = (U[ci] + U[cj]) / 2
        cen = np.array([U[cs[x]:cs[x + 1]].mean(0) for x in face])
        p = mid + (cen - mid) / np.linalg.norm(cen - mid, axis=1, keepdims=True) * 1.5 / size
        x = np.clip((p[:, 0] * size).astype(int), 0, size - 1)
        y = np.clip(((1 - p[:, 1]) * size).astype(int), 0, size - 1)
        return alb[y, x], p
    A, pa = sample(ca[seam], cb[seam], f[seam])
    B, pb = sample(cb2[seam], ca2[seam], g[seam])
    diff = np.abs(A - B).mean(1)
    assert diff.mean() < 6 and np.percentile(diff, 95) < 15  # continuous across seams (samples ~3 texels apart)
    assert np.abs(pa - pb).max(1).mean() > 0.1  # while the UV positions are far apart
    u_based = np.abs(pa[:, 0] - pb[:, 0]) * 255  # what a texture painted in UV space would jump by
    assert u_based.mean() > 10 * diff.mean()


def test_normal_map_follows_the_height_slope():
    g = grid(1, 4)  # UV u along +x, v along +y (tangent +x, bitangent +y)
    tex = layers("#808080", height=[(mask_height(-0.5, 0.5, axis="x"), None, 0.1)])
    b = bake(g, tex, 64)
    n = b.normal.astype(float) / 255 * 2 - 1
    c = n[32, 32]  # center: slope 0.1 x 1.5 per unit along x
    assert np.allclose(c, np.array([-0.15, 0, 1]) / np.linalg.norm([-0.15, 0, 1]), atol=0.02)
    flat = bake(g, layers("#808080"), 32).normal
    assert (flat == (128, 128, 255)).all()
    tilt_y = bake(g, layers("#808080", height=[(mask_height(-0.5, 0.5, axis="y"), None, 0.1)]), 64).normal
    assert tilt_y[32, 32, 1] < 115 and abs(int(tilt_y[32, 32, 0]) - 128) <= 2  # green channel = bitangent (+v)


def test_textured_renders(tmp_path):
    m = quad_sphere(1, 8)
    tex = layers([("#ff0000", None, 1), ("#0000ff", mask_normal("-z", 0.0, 0.3), 1)])
    img = read_png(render_views(m, str(tmp_path / "t.png"), size=100, textured=tex))
    top = img[100 + 50, 50]  # TOP panel center: the red top
    front_low = img[72, 50]  # FRONT panel, lower half: blue underside
    assert top[0] > 200 and top[2] < 50 and front_low[2] > 150 and front_low[0] < 80
    b = bake(m, tex, 128)
    via_uv = read_png(render_views(m, str(tmp_path / "b.png"), size=100, textured=b))
    assert np.abs(via_uv.astype(int) - img.astype(int)).mean() < 6  # the bake through the UVs looks the same
    chk = read_png(render_views(m, str(tmp_path / "c.png"), size=100, textured=True))
    assert len(np.unique(chk[30:70, 30:70].reshape(-1, 3), axis=0)) > 4  # checker squares
    with pytest.raises(ValueError, match="no UVs"):
        render_views(to_mesh(sphere(1), 12), str(tmp_path / "x.png"), textured=b)


def test_surface_moves_within_triangles():
    m = grid(1, 1)
    S = Surface(m, [0], [[1 / 3, 1 / 3, 1 / 3]])
    moved = S.moved(np.array([[0.1, 0.05, 0.3]]))  # the off-plane part is dropped
    assert np.allclose(moved.points - S.points, [[0.1, 0.05, 0]])
