import io
import os
import subprocess
import sys

import pytest

from meshkit.runner import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = ["table", "limb", "building", "sdf_creature", "textured_creature", "retopo_creature"]


def needs(name):
    if name == "textured_creature":
        pytest.importorskip("xatlas")  # unwrap (the 'uv' extra)


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_passes_runner(name, tmp_path):
    needs(name)
    out = io.StringIO()
    code = run(os.path.join(ROOT, "examples", f"{name}.py"), out_dir=str(tmp_path), out=out)
    text = out.getvalue()
    assert code == 0, text
    assert "check: OK" in text and "plugins: OK" in text
    assert os.path.exists(tmp_path / f"{name}.json")


@pytest.mark.parametrize("name", ["building", "sdf_creature", "textured_creature", "retopo_creature"])
def test_deterministic_across_processes(name, tmp_path):
    needs(name)
    outs = []
    for seed in ("1", "2"):
        d = tmp_path / seed
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run([sys.executable, "-m", "meshkit", "run", os.path.join("examples", f"{name}.py"), "--out", str(d)],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        outs.append(b"".join((d / f).read_bytes() for f in sorted(os.listdir(d))  # every export, maps included
                             if not f.endswith("_blender.py")))  # (the script holds the absolute out path)
    assert outs[0] == outs[1]


def test_textured_example_meets_the_brief(tmp_path):
    """SDF creature with named parts, < 15k triangles, unwrapped, textured with group and normal masks, GLB out."""
    needs("textured_creature")
    from meshkit import load_glb
    from meshkit.gltf import glb_maps
    out = io.StringIO()
    assert run(os.path.join(ROOT, "examples", "textured_creature.py"), out_dir=str(tmp_path), out=out, res="final") == 0
    m = load_glb(str(tmp_path / "textured_creature.glb"))
    assert len(m.faces) < 15000 and m.uvs is not None and {"body", "head", "tail", "spike.0", "leg.3"} <= set(m.groups)
    maps = glb_maps(str(tmp_path / "textured_creature.glb"))
    assert maps["albedo"].shape == (1024, 1024, 3) and maps["normal"].shape == (1024, 1024, 3)
    text = out.getvalue()
    assert "uv: islands=" in text and "bake: 1024px" in text and "check: OK" in text


def test_retopo_example_meets_the_brief(tmp_path):
    """An existing SDF creature (the lizard) retopologized: all quads, closed, genus 0, < 15k triangles in final."""
    from meshkit import check, load_glb
    out = io.StringIO()
    assert run(os.path.join(ROOT, "examples", "retopo_creature.py"), out_dir=str(tmp_path), out=out, res="final") == 0
    text = out.getvalue()
    assert "topo: OK" in text and "quads 100%" in text and "loops per joint" in text
    m = load_glb(str(tmp_path / "retopo_creature.glb"))
    assert all(len(f) == 4 for f in m.faces) and len(m.triangles[0]) < 15000
    assert check(m, closed=True, expect_genus=0).ok
    assert {"body", "head", "snout", "tail", "eye.0", "spike.0", "leg.3", "toe.0", "skin"} <= set(m.groups)
