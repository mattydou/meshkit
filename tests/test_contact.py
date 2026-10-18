import numpy as np

from meshkit import add_group, check, contacts, cube, join, select_by_normal, translate


def pair(offset, size_b=1.0):
    a = cube(group="a")
    b = translate(cube(size_b, group="b"), offset)[0]
    return join(a, b)[0]


def state(m, **kw):
    (r,) = contacts(m, **kw)
    return r


def test_touch_gap_penetrate():
    assert state(pair((1, 0, 0)))["state"] == "touch"
    r = state(pair((1.5, 0, 0)))
    assert r["state"] == "gap" and np.isclose(r["dist"], 0.5)
    r = state(pair((0.8, 0, 0)))
    assert r["state"] == "penetrate" and np.isclose(r["depth"], 0.2)
    r = state(pair((0, 0, 0), size_b=0.5))  # fully inside
    assert r["state"] == "penetrate" and np.isclose(r["depth"], 0.25)
    r = state(pair((10, 0, 0)))  # far: bbox distance
    assert r["state"] == "gap" and np.isclose(r["dist"], 9)


def test_crossing_bars_and_part_names():
    m, _ = join(cube((3, 0.2, 0.2), group="bar"), cube((0.2, 3, 0.2), group="bar"))
    r = state(m)
    assert r["state"] == "penetrate" and (r["a"], r["b"]) == ("bar.0", "bar.1")
    m, _ = join(cube((3, 0.2, 0.2)), cube((0.2, 3, 0.2)))
    assert (state(m)["a"], state(m)["b"]) == ("part0", "part1")


def test_groups_mode_connected_and_lists():
    c = cube()
    c, _ = add_group(c, "top", select_by_normal(c, "z", 1))
    c, _ = add_group(c, "bottom", select_by_normal(c, "-z", 1))
    c, _ = add_group(c, "side", select_by_normal(c, "x", 1))
    recs = {(r["a"], r["b"]): r for r in contacts(c, by="groups")}
    assert recs[("bottom", "top")]["state"] == "gap" and np.isclose(recs[("bottom", "top")]["dist"], 1)
    assert recs[("bottom", "side")]["state"] == "connected"
    assert len(contacts(c, by=["top", "side"])) == 1


def test_check_contact_lines_auto_and_off():
    r = check(cube())
    assert "contacts" not in r.stats and "\n" not in str(r)
    m = pair((1, 0, 0))
    r = check(m)
    text = str(r)
    assert r.ok and "contacts(parts): 1 pairs" in text and "touch 1: a.0-b.0" in text
    assert "contacts" not in check(m, contacts=None).stats
    assert "contacts(groups)" in str(check(m, contacts="groups"))
