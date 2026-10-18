import struct
import zlib

import numpy as np
import pytest

from meshkit import Mesh, cube, flip, render_views, select_by_normal


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    W, H = struct.unpack(">II", data[16:24])
    idat, pos = b"", 8
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        if data[pos + 4:pos + 8] == b"IDAT":
            idat += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(H, 1 + 3 * W)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(H, W, 3)


def is_red(px):
    return px[0] > 1.5 * px[1] and px[0] > 80


def test_contact_sheet_layout_and_content(tmp_path):
    p = render_views(cube(), str(tmp_path / "c.png"), size=120)
    img = read_png(p)
    assert img.shape == (240, 240, 3)
    for y0, x0 in ((0, 0), (0, 120), (120, 0), (120, 120)):  # every panel shows the mesh at its center
        assert img[y0 + 50:y0 + 70, x0 + 50:x0 + 70].mean() > 100
        assert img[y0 + 115, x0 + 115].mean() < 70  # corner is background
    assert not any(is_red(img[y, x]) for y in range(0, 240, 7) for x in range(0, 240, 7))
    assert img[8:22, 8:20].max() == 255  # label drawn


def test_back_faces_show_red_and_output_is_deterministic(tmp_path):
    c = cube()
    c, _ = flip(c, select_by_normal(c, "-y", 1))  # the face the FRONT view sees
    a = read_png(render_views(c, str(tmp_path / "a.png"), size=100))
    assert is_red(a[50, 50]) and not is_red(a[50, 150])
    b = read_png(render_views(c, str(tmp_path / "b.png"), size=100))
    assert np.array_equal(a, b)
    e = read_png(render_views(cube(divisions=4), str(tmp_path / "e.png"), size=100, edges=True))
    n = read_png(render_views(cube(divisions=4), str(tmp_path / "n.png"), size=100, edges=False))
    assert (e[:100, :100].mean(-1) < n[:100, :100].mean(-1) - 20).sum() > 50  # face outlines drawn
    with pytest.raises(ValueError):
        render_views(Mesh([(0, 0, 0)]), str(tmp_path / "x.png"))


def test_focus_and_zoom(tmp_path):
    from meshkit import join, translate

    def lit(img, panel=0):
        y0, x0 = (panel // 2) * 100, (panel % 2) * 100
        return (img[y0:y0 + 100, x0:x0 + 100].mean(-1) > 100).mean()
    big = cube(4)
    m, _ = join(big, translate(cube(0.2, group="tip"), (3, 0, 0))[0])  # small cube off to the side
    full = read_png(render_views(m, str(tmp_path / "full.png"), size=100))
    fit = read_png(render_views(m, str(tmp_path / "tip.png"), size=100, focus="tip"))
    cols = np.flatnonzero(fit[50, :100].mean(-1) > 100)  # the 0.2 cube spans ~42 px around the center
    assert lit(full) > 0.2 and abs(cols.min() - 29) <= 3 and abs(cols.max() - 71) <= 3 and len(cols) > 38
    z2 = read_png(render_views(m, str(tmp_path / "z2.png"), size=100, focus="tip", zoom=2))
    assert lit(z2) > lit(fit) + 0.05  # zoom 2 of the whole view: the big cube is back in frame
    top = read_png(render_views(m, str(tmp_path / "p.png"), size=100, focus=(0, 0, 2), zoom=8))
    assert lit(top, panel=2) > 0.95  # TOP panel filled by the big cube's top face
    m2, _ = join(big, translate(cube(0.2, group="tip"), (0, 3, 0))[0])  # behind the big cube in the FRONT view
    seen = read_png(render_views(m2, str(tmp_path / "b.png"), size=100, focus="tip"))
    assert (seen[45:55, 45:55].mean(-1) > 100).all() and seen[80:95, 5:15].mean() < 70  # occluder cut away
    with pytest.raises(KeyError):
        render_views(m, str(tmp_path / "x.png"), focus="nope")
