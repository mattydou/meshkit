"""B8: GLB export / import round trip, OBJ + MTL, export_all with a bake, the Blender script's material, and the
runner's texture workflow."""
import io
import json
import os
import runpy
import struct
import sys
import textwrap
import types

import numpy as np

from meshkit import (add_group, bake, cube, export_all, join, layers, load_glb, load_obj, mask_group, mask_normal,
                     quad_sphere, spots, to_glb, translate)
from meshkit.gltf import glb_maps, read_glb
from meshkit.png import read_png
from meshkit.runner import run


def textured():
    m, _ = join(cube(group="box"), translate(quad_sphere(0.5, 6, group="ball"), (2, 0, 0))[0])
    m, _ = add_group(m, "tip", [0], kind="v")
    from meshkit import pack
    m, _ = pack(m)
    tex = layers([("#c08040", None, 1), ("#204080", mask_group("ball"), 1)], roughness=[(0.7, None, 1)],
                 height=[(spots(0.2, shape="dome"), mask_normal("z"), 0.01)])
    return m, bake(m, tex, 64)


def test_glb_round_trip(tmp_path):
    m, b = textured()
    p = to_glb(m, str(tmp_path / "m.glb"), bake=b, name="thing")
    r = load_glb(p)
    assert r.faces == m.faces and dict(r.groups) == dict(m.groups)  # quads come back as quads
    assert np.allclose(r.verts, m.verts, atol=1e-6) and np.allclose(r.uvs, m.uvs, atol=1e-6)
    maps = glb_maps(p)
    assert np.array_equal(maps["albedo"], b.albedo) and np.array_equal(maps["normal"], b.normal)
    assert np.array_equal(maps["roughness"][..., 1], b.roughness) and (maps["roughness"][..., 2] == 0).all()
    plain = load_glb(to_glb(cube(), str(tmp_path / "c.glb")))  # no bake: default material, still UVs
    assert np.allclose(plain.uvs, cube().uvs, atol=1e-6)


def test_glb_structure(tmp_path):
    m, b = textured()
    p = to_glb(m, str(tmp_path / "m.glb"), bake=b)
    data = open(p, "rb").read()
    magic, version, length = struct.unpack("<III", data[:12])
    assert (magic, version, length) == (0x46546C67, 2, len(data)) and length % 4 == 0
    gl, binary = read_glb(p)
    assert gl["asset"]["version"] == "2.0" and all(v["byteOffset"] % 4 == 0 for v in gl["bufferViews"])
    prim = gl["meshes"][0]["primitives"][0]
    acc = gl["accessors"]
    pos = acc[prim["attributes"]["POSITION"]]
    assert "min" in pos and "max" in pos and np.allclose(pos["max"], [2.5, 0.5, 0.5], atol=1e-6)  # Y up: z -> y
    assert {"NORMAL", "TEXCOORD_0", "TANGENT"} <= set(prim["attributes"])
    mat = gl["materials"][0]
    assert mat["pbrMetallicRoughness"]["metallicFactor"] == 0 and mat["normalTexture"]["index"] == 2
    assert len(gl["images"]) == 3 and all(i["mimeType"] == "image/png" for i in gl["images"])
    assert sorted(gl["meshes"][0]["extras"]["meshkit"]["groups"]) == ["ball", "box", "tip"]
    from meshkit.gltf import _accessor
    t = _accessor(gl, binary, prim["attributes"]["TANGENT"])
    assert np.allclose(np.abs(t[:, 3]), 1) and np.allclose(np.linalg.norm(t[:, :3], axis=1), 1, atol=1e-5)
    uv = _accessor(gl, binary, prim["attributes"]["TEXCOORD_0"])
    assert np.isclose(uv[:, 1].min(), 1 - m.uvs[:, 1].max(), atol=1e-6)  # v flipped for glTF


def foreign_glb(path):
    """A minimal glTF from 'another tool': two meshes on translated nodes, Y up, no meshkit extras."""
    P = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=np.float32)  # quad in the XY plane
    I = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    UV = np.array([(0, 1), (1, 1), (0, 0), (1, 0)], dtype=np.float32)
    binary = P.tobytes() + I.tobytes() + UV.tobytes()
    gl = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
          "nodes": [{"mesh": 0, "translation": [0, 0, 0]}, {"mesh": 1, "translation": [5, 0, 0]}],
          "meshes": [{"name": "a", "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 2}, "indices": 1}]},
                     {"name": "b", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "buffers": [{"byteLength": len(binary)}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48,
                                                                            "byteLength": 24},
                          {"buffer": 0, "byteOffset": 72, "byteLength": 32}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0],
                         "max": [1, 1, 0]},
                        {"bufferView": 1, "componentType": 5125, "count": 6, "type": "SCALAR"},
                        {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"}]}
    js = json.dumps(gl).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)) + struct.pack("<II", len(js), 0x4E4F534A)
                 + js + struct.pack("<II", len(binary), 0x004E4942) + binary)
    return path


def test_load_foreign_glb(tmp_path):
    m = load_glb(foreign_glb(str(tmp_path / "f.glb")))
    assert len(m.verts) == 8 and len(m.faces) == 4 and sorted(m.groups) == ["a", "b"]
    assert np.allclose(m.verts[:, 1], 0) and np.isclose(m.verts[:, 0].max(), 6)  # Y up -> Z up, node moved +5 x
    assert np.allclose(m.face_normals[0], (0, -1, 0))  # glTF +Z faces -Y in Z-up
    assert np.isnan(m.uvs[6:]).all() and np.allclose(m.uvs[:6].min(0), 0)  # second mesh had no UVs
    out = io.StringIO()
    assert run(str(tmp_path / "f.glb"), out=out) == 0 and "loaded in" in out.getvalue()


def test_export_all_with_bake_writes_everything(tmp_path):
    m, b = textured()
    paths = export_all(m, str(tmp_path), "thing", bake=b)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["thing.glb", "thing.json", "thing.mtl", "thing.obj", "thing_albedo.png", "thing_blender.py",
                     "thing_normal.png", "thing_roughness.png"]
    mtl = open(tmp_path / "thing.mtl").read()
    assert "map_Kd thing_albedo.png" in mtl and "map_Pr thing_roughness.png" in mtl and "norm thing_normal.png" in mtl
    obj = open(tmp_path / "thing.obj").read()
    assert obj.startswith("# meshkit OBJ\nmtllib thing.mtl\n") and "usemtl thing" in obj
    assert np.allclose(load_obj(str(tmp_path / "thing.obj")).uvs, m.uvs, atol=1e-6)
    assert np.array_equal(read_png(str(tmp_path / "thing_normal.png")), b.normal)
    d = json.load(open(tmp_path / "thing.json"))
    assert d["material"]["albedo"] == "thing_albedo.png" and len(d["uvs"]) == len(m.faces)


class _Node:
    def __init__(self, kind):
        self.type, self.location, self.image = kind, (0, 0), None
        self.inputs = {k: types.SimpleNamespace(name=k, default_value=None)
                       for k in ("Base Color", "Roughness", "Normal", "Metallic", "Color")}
        self.outputs = {k: types.SimpleNamespace(name=k) for k in ("Color", "Normal", "BSDF")}


def fake_bpy_with_materials(record):
    nodes = []

    class Nodes(list):
        def new(self, kind):
            n = _Node(kind)
            self.append(n)
            return n

    def material(name):
        nt = types.SimpleNamespace(nodes=Nodes([_Node("BSDF_PRINCIPLED")]), links=types.SimpleNamespace(
            new=lambda a, b: record.setdefault("links", []).append((a.name, b.name))))
        nodes.append(nt)
        return types.SimpleNamespace(name=name, use_nodes=False, node_tree=nt)

    def load(path, check_existing=True):
        img = types.SimpleNamespace(path=path, colorspace_settings=types.SimpleNamespace(name="sRGB"))
        record.setdefault("images", []).append(img)
        return img
    return material, load


def test_blender_script_builds_the_material(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_parts_handles import _Fake
    m, b = textured()
    paths = export_all(m, str(tmp_path), "thing", bake=b)
    script = [p for p in paths if p.endswith("_blender.py")][0]
    fake, record = _Fake(), {}
    fake.bpy.data.materials = types.SimpleNamespace(new=fake_bpy_with_materials(record)[0])
    fake.bpy.data.images = types.SimpleNamespace(load=fake_bpy_with_materials(record)[1])
    sys.modules["bpy"] = fake.bpy
    try:
        runpy.run_path(script, run_name="blender_test")["build"]()
    finally:
        del sys.modules["bpy"]
    ob = fake.objects["thing"]
    assert len(ob.data.uv) == 2 * len(m.corner_verts) and np.allclose(ob.data.uv, m.uvs.reshape(-1))
    assert ob.data.materials[0].name == "thing"
    assert sorted(record["links"]) == [("Color", "Base Color"), ("Color", "Color"), ("Color", "Roughness"),
                                       ("Normal", "Normal")]
    spaces = {os.path.basename(i.path): i.colorspace_settings.name for i in record["images"]}
    assert spaces == {"thing_albedo.png": "sRGB", "thing_roughness.png": "Non-Color", "thing_normal.png": "Non-Color"}


def test_runner_bakes_and_exports_textured_models(tmp_path):
    p = tmp_path / "ball.py"
    p.write_text(textwrap.dedent("""
        from meshkit import *
        mesh = quad_sphere(1, 6, group="skin")
        texture = layers([("#884422", None, 1), ("#eeddaa", mask_normal("-z"), 1)], roughness=0.5,
                         height=[(spots(0.2, shape="dome"), None, 0.01)])
        bake_opts = {"pad": 4}
    """))
    out = io.StringIO()
    assert run(str(p), out=out, render=True) == 0
    text = out.getvalue()
    assert "uv: islands=6" in text and "bake: 512px albedo roughness normal height" in text
    for f in ("ball.glb", "ball_albedo.png", "ball_normal.png", "ball_roughness.png", "ball_height.png", "ball.mtl",
              "ball_textured.png"):
        assert os.path.exists(tmp_path / "out" / f), f
    out = io.StringIO()
    assert run(str(p), out=out, res="final", export=True) == 0 and "bake: 1024px" in out.getvalue()
    bad = tmp_path / "bad.py"
    bad.write_text("from meshkit import *\nmesh = to_mesh(sphere(1), 12)\ntexture = layers('#ffffff')\n")
    out = io.StringIO()
    assert run(str(bad), out=out) == 1
    assert "no UVs" in out.getvalue() and "bake failed: ValueError: bake: mesh has no UVs" in out.getvalue()
