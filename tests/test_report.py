import numpy as np

from meshkit import (Mesh, add_group, check, clear_plugins, cube, delete, describe, flip, join, mirror,
                     register_check, register_scorer, run_plugins, select_by_normal, translate)
from meshkit.report import mirror_error


def test_check_clean_cube_and_open_grid():
    r = check(cube())
    assert r.ok and r.stats["chi"] == 2 and r.stats["quads"] == 6 and r.stats["comps"] == 1
    text = str(r)
    assert text.startswith("check: OK") and "\n" not in text
    c = cube()
    opened, _ = delete(c, select_by_normal(c, "z", 1))
    assert check(opened).ok and check(opened).stats["boundary"] == 4
    r = check(opened, closed=True)
    assert not r.ok and "boundary edges: 4" in str(r)


def test_check_detects_winding_inward_degenerate_duplicate_isolated_nonmanifold():
    c = cube()
    m, _ = flip(c, [0])
    r = check(m)
    assert not r.ok and r.stats["bad_winding"] == 4 and "inconsistent winding" in str(r)
    m, _ = flip(c)
    r = check(m)
    assert r.stats["inward"] == 1 and "inward normals" in str(r)
    m = Mesh([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (5, 5, 5)], [(0, 1, 2), (0, 1, 3), (1, 0, 3)])
    r = check(m)
    assert r.stats["degenerate"] == 1 and r.stats["duplicate"] == 1 and r.stats["isolated"] == 1
    fin = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    assert check(fin).stats["nonmanifold"] == 1


def test_check_counts_components_and_ngons():
    m, _ = join(cube(), translate(cube(), (3, 0, 0))[0])
    r = check(m)
    assert r.ok and r.stats["comps"] == 2 and r.stats["chi"] == 4
    ngon = Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0.5, 1.5, 0), (0, 1, 0)], [(0, 1, 2, 3, 4)])
    assert check(ngon).stats["ngons"] == 1


def test_describe_numbers_and_groups():
    c = cube(size=(2, 1, 1), center=(0, 0, 0.5), group="body")
    c, _ = add_group(c, "top", select_by_normal(c, "z", 1))
    d = describe(c)
    assert np.allclose(d.stats["dims"], (2, 1, 1)) and np.isclose(d.stats["volume"], 2.0)
    assert np.allclose(d.stats["centroid"], (0, 0, 0.5))
    assert d.stats["mirror"]["x"] == 0 and d.stats["mirror"]["z"] > 0
    assert np.allclose(d.stats["groups"]["top"]["centroid"], (0, 0, 1))
    assert np.isclose(d.stats["group_dist"][("body", "top")], 0.5)
    text = str(d)
    assert "group top[f1]" in text and "group dist: body-top=0.5" in text and len(text.splitlines()) == 5
    open_c, _ = delete(c, "top")
    assert "vol=open" in str(describe(open_c))


def test_mirror_error_detects_asymmetry():
    c = cube()
    assert mirror_error(c, 0) == 0
    m, _ = translate(c, (0.2, 0, 0))
    assert mirror_error(m, 0) > 0.1
    half = cube(center=(0.5, 0, 0))
    half, _ = delete(half, select_by_normal(half, "-x", 1))
    full, _ = mirror(half, "x")
    assert mirror_error(full, 0) == 0


def test_plugins_run_and_isolate_errors():
    clear_plugins()

    @register_check
    def tall_enough(m):
        return [] if np.ptp(m.verts[:, 2]) >= 2 else [f"height {np.ptp(m.verts[:, 2]):.2f} < 2"]

    @register_scorer
    def width(m):
        return "width", float(np.ptp(m.verts[:, 0]))

    @register_check
    def broken(m):
        raise RuntimeError("boom")

    r = run_plugins(cube())
    assert not r.ok and r.stats["scores"] == {"width": 1.0}
    text = str(r)
    assert "tall_enough: height 1.00 < 2" in text and "broken raised RuntimeError: boom" in text and "width=1" in text
    clear_plugins()
    assert run_plugins(cube()).ok


def test_describe_stays_short_with_many_groups():
    m = cube(divisions=4)
    for i in range(60):
        m, _ = add_group(m, f"g{i:02d}", [i], kind="f")
    d = describe(m)
    text = str(d)
    assert len(text.splitlines()) < 20 and "+48 more groups" in text and "group dist" not in text
    assert len(d.stats["groups"]) == 60 and len(d.stats["group_dist"]) == 60 * 59 // 2
    sub = str(describe(m, groups=["g03", "g07"]))
    assert "group g03[f1]" in sub and "group dist: g03-g07=" in sub and "g05" not in sub
    full = str(describe(m, summary=False))
    assert "bbox=" in full.splitlines()[2] and len(full.splitlines()) > 60
    import pytest
    with pytest.raises(KeyError):
        describe(m, groups=["nope"])
