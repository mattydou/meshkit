import io
import os
import subprocess
import sys
import textwrap

import numpy as np

from meshkit import (Mesh, add_edges, add_group, cube, extrude, from_json, select_by_normal, sweep, to_json, to_obj,
                     write_blender_script)
from meshkit.runner import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample():
    m = cube(group="body")
    m, _ = add_group(m, "top", select_by_normal(m, "z", 1))
    m, _ = add_group(m, "corner", [0], kind="v")
    m, _ = add_edges(m, [(0, 6)])
    return m


def test_json_roundtrip(tmp_path):
    m = sample()
    p = to_json(m, tmp_path / "m.json")
    r = from_json(p)
    assert np.array_equal(r.verts, m.verts) and r.faces == m.faces and r.loose_edges == m.loose_edges
    assert dict(r.groups) == dict(m.groups)


def test_deterministic_output(tmp_path):
    def build():
        m, c = extrude(sample(), "top", 0.5)
        m, _ = sweep(m, [(0, 0, 1), (0.3, 0.2, 2), (0, 0.5, 3)], radius=0.1, segments=6, caps=True)
        return m
    a, b = to_json(build(), tmp_path / "a.json"), to_json(build(), tmp_path / "b.json")
    assert open(a).read() == open(b).read()


def test_obj_y_up_groups_and_lines(tmp_path):
    m = Mesh([(1, 2, 3), (0, 0, 0), (1, 0, 0), (9, 9, 9)], [(0, 1, 2)], [(2, 3)], {"tri": ("f", [0])})
    text = open(to_obj(m, tmp_path / "m.obj")).read().splitlines()
    assert "v 1.000000 3.000000 -2.000000" in text and "g tri" in text and "f 1 2 3" in text and "l 3 4" in text
    text = open(to_obj(m, tmp_path / "z.obj", y_up=False)).read()
    assert "v 1.000000 2.000000 3.000000" in text


def test_blender_script_is_valid_python(tmp_path):
    j = to_json(sample(), tmp_path / "m.json")
    s = write_blender_script(j, tmp_path / "m_blender.py", "thing")
    src = open(s).read()
    compile(src, s, "exec")
    assert "'thing'" in src and "BOOLEAN" in src and "vertex_groups" in src


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_runner_ok_and_exports(tmp_path):
    s = write(tmp_path, "box.py", """
        from meshkit import *
        mesh = cube(group="box")
        check_opts = {"closed": True}
        register_scorer(lambda m: ("height", float(m.verts[:, 2].max() - m.verts[:, 2].min())))
    """)
    out = io.StringIO()
    assert run(s, out=out) == 0
    text = out.getvalue()
    assert "check: OK" in text and "describe:" in text and "height=1" in text and "wrote" in text
    for ext in (".obj", ".json", "_blender.py"):
        assert os.path.exists(tmp_path / "out" / f"box{ext}")


def test_runner_reports_script_error_line(tmp_path):
    s = write(tmp_path, "bad.py", """
        from meshkit import *
        m = cube()
        m, c = bridge(m, [0, 1, 2, 3], [4, 5, 6])
        mesh = m
    """)
    out = io.StringIO()
    assert run(s, out=out, export=False) == 2
    text = out.getvalue()
    assert "ERROR ValueError: bridge: loops have 4 and 3 verts" in text and "at bad.py:4" in text
    assert len(text.splitlines()) <= 5


def test_runner_syntax_error_and_tuple_mistake(tmp_path):
    out = io.StringIO()
    assert run(write(tmp_path, "syn.py", "mesh = (\n"), out=out, export=False) == 2
    assert "SyntaxError" in out.getvalue()
    out = io.StringIO()
    s = write(tmp_path, "tup.py", """
        from meshkit import *
        mesh = extrude(cube(), select_by_normal(cube(), "z"), 1)
    """)
    assert run(s, out=out, export=False) == 2 and "(mesh, created) tuple" in out.getvalue()


def test_runner_failing_check_and_plugin(tmp_path):
    s = write(tmp_path, "open.py", """
        from meshkit import *
        c = cube()
        mesh, _ = delete(c, select_by_normal(c, "z"))
        check_opts = {"closed": True}
        @register_check
        def wide(m):
            return ["too narrow"]
    """)
    out = io.StringIO()
    assert run(s, out=out, export=False) == 1
    text = out.getvalue()
    assert "check: FAIL" in text and "boundary edges: 4" in text and "wide: too narrow" in text


def test_cli_module_entry(tmp_path):
    s = write(tmp_path, "cli.py", "from meshkit import *\ndef build():\n    return quad_sphere(1, 3)\n")
    r = subprocess.run([sys.executable, "-m", "meshkit", "run", s, "--no-export"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "check: OK" in r.stdout


SDF_SCRIPT = """
    from meshkit import *
    legs = [capsule((x, 0, 1), (x, 0, 0.1), 0.1, name="leg") for x in (-0.13, 0.13)]
    sdf = smooth_union(ellipsoid((0.8, 0.4, 0.3), center=(0, 0, 1), name="body"), *legs, k=0.15)
    mesh = to_mesh(sdf)
    check_opts = {"closed": True, "expect_genus": 0}
"""


def test_runner_final_flag_timing_and_proximity(tmp_path):
    s = write(tmp_path, "legs.py", SDF_SCRIPT)
    out = io.StringIO()
    assert run(s, out=out, export=False) == 1  # the legs are 0.06 apart: proximity flags them
    text = out.getvalue()
    assert "(to_mesh " in text and "res draft(64)" in text and "of grid evaluated)" in text
    assert "proximity: FAIL" in text and "leg.0-leg.1 gap 0.06" in text
    out = io.StringIO()
    run(s, out=out, export=False, res="final")
    assert "res final(180)" in out.getvalue()
    import meshkit.sdf as S
    assert S._defaults["res"] == "draft"  # restored after the run
    allowed = write(tmp_path, "ok.py", SDF_SCRIPT + "    proximity_opts = {'allow': [('leg.0', 'leg.1')]}\n")
    out = io.StringIO()
    assert run(allowed, out=out, export=False) == 0, out.getvalue()


def test_runner_render_flag(tmp_path):
    s = write(tmp_path, "box.py", """
        from meshkit import *
        mesh = join(cube(group="body"), translate(cube(0.2, group="knob"), (0, 0, 0.6))[0])[0]
    """)
    out = io.StringIO()
    assert run(s, out=out, render=True) == 0
    assert os.path.exists(tmp_path / "out" / "box.png") and "render: " in out.getvalue()
    out = io.StringIO()
    assert run(s, out=out, export=False, render="knob", zoom=3) == 0
    assert os.path.exists(tmp_path / "out" / "box_knob.png") and "focus=knob zoom=3" in out.getvalue()
    out = io.StringIO()
    assert run(s, out=out, export=False, render=[0, 0, 0.6]) == 0 and "focus=(0,0,0.6)" in out.getvalue()
    out = io.StringIO()
    assert run(s, out=out, export=False, render="nope") == 1 and "render failed: KeyError" in out.getvalue()
    out = io.StringIO()
    assert run(s, out=out, export=False) == 0 and "render" not in out.getvalue()  # never automatic


def test_cli_flags(tmp_path):
    s = write(tmp_path, "legs.py", SDF_SCRIPT)
    r = subprocess.run([sys.executable, "-m", "meshkit", "run", s, "--final", "--no-export", "--render", "0,0,0.5",
                        "--zoom", "2"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "res final(180)" in r.stdout and "focus=(0,0,0.5) zoom=2" in r.stdout
    assert os.path.exists(tmp_path / "out" / "legs_focus.png")


def test_export_scales_to_large_grouped_meshes(tmp_path):
    import time
    from meshkit import export_all, quad_sphere, select_by_normal
    m = quad_sphere(1, 60, group="skin")  # 21600 faces
    m, _ = add_group(m, "top", select_by_normal(m, "z", 60))
    t0 = time.perf_counter()
    paths = export_all(m, str(tmp_path), "big")
    assert time.perf_counter() - t0 < 3  # was quadratic in face count
    assert from_json(paths[1]).faces == m.faces


def test_runner_builds_once_when_script_sets_mesh_from_build(tmp_path):
    s = write(tmp_path, "once.py", """
        from meshkit import *
        calls = []

        def build():
            calls.append(1)
            assert len(calls) == 1, "build() ran twice"
            return cube()

        mesh = build()
    """)
    out = io.StringIO()
    assert run(s, out=out, export=False) == 0, out.getvalue()
