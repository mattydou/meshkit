"""Remeshing fallbacks: isotropic_remesh (pure Python) and quad_remesh (external QuadriFlow / Instant Meshes / Blender,
faked here; the real Blender run is skipped when Blender is not installed)."""
import sys
import textwrap

import numpy as np
import pytest

from meshkit import (add_group, capsule, check, grid, isotropic_remesh, quad_remesh, smooth_union, sphere, to_mesh,
                     topo_report)
from meshkit.remesh import find_quad_remesher


def blob():
    s = smooth_union(sphere(1, name="body"), capsule((0, 0, 0), (1.6, 0, 0), 0.35, name="arm"), k=0.2)
    return s, to_mesh(s, 40)


def edge_stats(m):
    E = np.array(m.edges)
    L = np.linalg.norm(m.verts[E[:, 0]] - m.verts[E[:, 1]], axis=1)
    return L.mean(), L.std() / L.mean()


def test_isotropic_remesh_evens_out_marching_cubes():
    s, m = blob()
    m, _ = add_group(m, "tip", [int(np.argmax(m.verts[:, 0]))], kind="v")
    r, c = isotropic_remesh(m, 0.1)
    assert check(r, closed=True, expect_genus=0).ok and all(len(f) == 3 for f in r.faces)
    mean, cv = edge_stats(r)
    assert 0.085 < mean < 0.115 and cv < 0.2 and cv < 0.6 * edge_stats(m)[1]
    assert np.abs(s.eval(r.verts)).max() < 0.02  # stays on the input surface
    assert c.faces_before == len(m.faces) and c.faces_after == len(r.faces)
    assert {"body", "arm", "tip"} <= set(r.groups) and len(r.groups["tip"].idx) >= 1
    arm = r.face_centers[list(r.groups["arm"].idx)]
    assert arm[:, 0].mean() > 0.9  # face groups follow the nearest source face
    v = topo_report(r).stats["valence"]
    assert v.get(6, 0) > 0.6 * sum(v.values())  # mostly valence 6


def test_isotropic_remesh_keeps_open_borders():
    g = grid(2, 6)
    r, _ = isotropic_remesh(g, 0.15)
    assert check(r).ok
    bv = {v for e, fs in r.edge_faces.items() if len(fs) == 1 for v in e}
    P = r.verts[sorted(bv)]
    assert np.allclose(np.abs(P[:, :2]).max(1), 1, atol=1e-9)  # border verts stay on the square's edge
    assert np.allclose(r.verts[:, 2], 0)
    with pytest.raises(ValueError, match="edge_length"):
        isotropic_remesh(g, 0)


FAKE = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from meshkit import load_obj, quad_sphere, to_obj
    a = sys.argv[1:]
    if "--fail" in a:
        print("boom", file=sys.stderr)
        sys.exit(3)
    src, dst, n = a[a.index("-i") + 1], a[a.index("-o") + 1], int(a[a.index("-f") + 1])
    m = load_obj(src, y_up=False)
    lo, hi = m.verts.min(0), m.verts.max(0)
    q = quad_sphere(1.0, max(1, round((n / 6) ** 0.5)))
    from meshkit import map_verts
    q, _ = map_verts(q, lambda P: (P + 1) / 2 * (hi - lo) + lo)  # a quad blob filling the input's bounds
    to_obj(q, dst, y_up=False)
""")


def test_quad_remesh_plumbing_with_a_fake_binary(tmp_path):
    import os
    fake = tmp_path / "fake_quadriflow.py"
    fake.write_text(FAKE.format(root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    _, m = blob()
    q, c = quad_remesh(m, 600, backend="quadriflow", command=[sys.executable, str(fake)])
    assert c.backend == "quadriflow" and c.faces_after == len(q.faces) and all(len(f) == 4 for f in q.faces)
    assert 450 < len(q.faces) < 800 and {"body", "arm"} <= set(q.groups)
    with pytest.raises(RuntimeError, match="quadriflow failed .exit 3.: boom"):
        quad_remesh(m, 600, backend="quadriflow", command=[sys.executable, str(fake), "--fail"])
    with pytest.raises(ValueError, match="backend must be"):
        quad_remesh(m, 600, backend="zremesher")


def test_quad_remesh_without_a_remesher(monkeypatch):
    import meshkit.remesh as rm
    monkeypatch.setattr(rm, "find_quad_remesher", lambda backend=None: None)
    with pytest.raises(RuntimeError, match="no quad remesher found.*isotropic_remesh"):
        quad_remesh(blob()[1], 500)


@pytest.mark.skipif(find_quad_remesher("blender") is None, reason="Blender not installed")
def test_quad_remesh_with_blenders_quadriflow():
    s, m = blob()
    q, c = quad_remesh(m, 800, backend="blender")
    assert c.backend == "blender" and all(len(f) == 4 for f in q.faces)
    assert check(q, closed=True, expect_genus=0).ok and 500 < len(q.faces) < 1200
    assert np.abs(s.eval(q.verts)).max() < 0.03 and {"body", "arm"} <= set(q.groups)
