"""GLB (binary glTF 2.0) export and import. One mesh node, Y up (x, z, -y), fan-triangulated; vertices split where UVs
differ. A bake becomes a PBR material: baseColor, metallicRoughness (G = roughness, B = metal 0) and a tangent-space
normal map (with TANGENT attributes matching the bake). Groups, polygon sizes and original vert ids ride along in
mesh extras / a _MESHKIT_VERT attribute, so load_glb gives back the same Mesh."""
import json
import struct

import numpy as np

from .mesh import Mesh
from .png import png_bytes, read_png

_FLOAT, _UINT = 5126, 5125
_ARRAY, _ELEMENTS = 34962, 34963
_TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8, 5122: np.int16, 5120: np.int8}


def _y_up(P):
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def _z_up(P):
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)


def _rle(sizes):
    out = []
    for s in sizes:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return out


def to_glb(mesh, path, bake=None, name="mesh"):
    """Write a GLB: positions, smooth normals, UVs (TEXCOORD_0, v flipped for glTF), triangles; with a bake, a PBR
    material using its albedo, roughness and normal maps (embedded PNGs) plus TANGENTs; groups in the mesh extras.
    Returns path."""
    T, src = mesh.triangles
    if not len(T):
        raise ValueError("to_glb: mesh has no faces")
    has_uv = mesh.uvs is not None
    if has_uv:
        from .texture import _wedge_ids
        from .uv import corner_tris
        wid = _wedge_ids(mesh)  # one glTF vertex per (vert, UV)
        CT = corner_tris(mesh)
        _, first = np.unique(wid, return_index=True)
        vert_of, idx = mesh.corner_verts[first], wid[CT]
        uv = np.where(np.isnan(mesh.uvs[first]), 0.0, mesh.uvs[first])
    else:
        vert_of, idx = np.arange(len(mesh.verts)), T
    P = _y_up(mesh.verts[vert_of]).astype(np.float32)
    N = _y_up(mesh.vert_normals[vert_of]).astype(np.float32)
    blobs, views, accessors = [], [], []

    def add(arr, comp, kind, target=None, minmax=False):
        a = np.ascontiguousarray(arr)
        off = sum(len(b) for b in blobs)
        data = a.tobytes()
        blobs.append(data + b"\0" * (-len(data) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(a), "type": kind}
        if minmax:
            acc["min"], acc["max"] = a.min(0).tolist(), a.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attrs = {"POSITION": add(P, _FLOAT, "VEC3", _ARRAY, True), "NORMAL": add(N, _FLOAT, "VEC3", _ARRAY)}
    if has_uv:
        attrs["TEXCOORD_0"] = add(np.c_[uv[:, 0], 1 - uv[:, 1]].astype(np.float32), _FLOAT, "VEC2", _ARRAY)
    attrs["_MESHKIT_VERT"] = add(vert_of.astype(np.float32), _FLOAT, "SCALAR", _ARRAY)
    gl = {"asset": {"version": "2.0", "generator": "meshkit"}, "scene": 0, "scenes": [{"nodes": [0]}],
          "nodes": [{"name": name, "mesh": 0}]}
    prim = {"attributes": attrs, "indices": add(idx.reshape(-1).astype(np.uint32), _UINT, "SCALAR", _ELEMENTS),
            "mode": 4, "material": 0}
    mat = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0], "metallicFactor": 0.0,
                                                  "roughnessFactor": 0.6}}
    if bake is not None:
        if not has_uv:
            raise ValueError("to_glb: a bake needs a mesh with UVs")
        from .texture import corner_tangents
        tg = corner_tangents(mesh)[first]
        attrs["TANGENT"] = add(np.c_[_y_up(tg[:, :3]), tg[:, 3]].astype(np.float32), _FLOAT, "VEC4", _ARRAY)
        images = []

        def image(arr):
            data = png_bytes(arr)
            blobs.append(data + b"\0" * (-len(data) % 4))
            views.append({"buffer": 0, "byteOffset": sum(len(b) for b in blobs[:-1]), "byteLength": len(data)})
            images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})
            return len(images) - 1
        orm = np.stack([np.full_like(bake.roughness, 255), bake.roughness, np.zeros_like(bake.roughness)], -1)
        tex = [image(bake.albedo), image(orm), image(bake.normal)]
        gl["images"] = images
        gl["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gl["textures"] = [{"source": i, "sampler": 0} for i in tex]
        mat["pbrMetallicRoughness"] = {"baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1},
                                       "metallicFactor": 0.0, "roughnessFactor": 1.0}
        mat["normalTexture"] = {"index": 2, "scale": 1.0}
    sizes = [len(f) for f in mesh.faces]
    extras = {"groups": {n: {"kind": "vert" if g.kind == "v" else "face", "indices": sorted(g.idx)}
                         for n, g in sorted(mesh.groups.items())}}
    if any(s != 3 for s in sizes):
        extras["face_sizes"] = _rle(sizes)
    if mesh.uvs is not None and np.isnan(mesh.uvs).any():
        face_of = np.repeat(np.arange(len(mesh.faces)), np.diff(mesh.corner_start))
        extras["faces_without_uvs"] = np.unique(face_of[np.isnan(mesh.uvs).any(1)]).tolist()
    gl["meshes"] = [{"name": name, "primitives": [prim], "extras": {"meshkit": extras}}]
    gl["materials"] = [mat]
    gl["accessors"], gl["bufferViews"] = accessors, views
    binary = b"".join(blobs)
    gl["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gl, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    out = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary))
           + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary)
    with open(path, "wb") as fh:
        fh.write(out)
    return path


def read_glb(path):
    """(json dict, binary chunk bytes) of a GLB file."""
    data = open(path, "rb").read()
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path}: not a glTF 2.0 binary")
    pos, gl, binary = 12, None, b""
    while pos < length:
        n, kind = struct.unpack("<II", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if kind == 0x4E4F534A:
            gl = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            binary = chunk
        pos += 8 + n
    return gl, binary


def _accessor(gl, binary, i):
    a = gl["accessors"][i]
    k = _TYPES[a["type"]]
    dt = np.dtype(_DTYPES[a["componentType"]]).newbyteorder("<")
    v = gl["bufferViews"][a["bufferView"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", dt.itemsize * k)
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * k, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * k), (stride, 1))
    out = np.ascontiguousarray(rows).view(dt).reshape(a["count"], k).astype(np.float64)
    if a.get("normalized"):
        out /= np.iinfo(_DTYPES[a["componentType"]]).max
    return out if k > 1 else out[:, 0]


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T
    M = np.eye(4)
    x, y, z, w = n.get("rotation", (0, 0, 0, 1))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = R * np.array(n.get("scale", (1, 1, 1)))
    M[:3, 3] = n.get("translation", (0, 0, 0))
    return M


def load_glb(path, images=False):
    """Read a GLB: meshkit's own files come back exactly (polygons, verts, UVs, groups); other files are welded by
    position, with a face group per mesh when there are several, node transforms applied, converted to Z up.
    images=True returns (mesh, {'albedo' | 'roughness' (metallicRoughness PNG) | 'normal': png bytes})."""
    gl, binary = read_glb(path)
    parents = {c: i for i, n in enumerate(gl.get("nodes", [])) for c in n.get("children", [])}

    def world(i):
        M = _node_matrix(gl["nodes"][i])
        return world(parents[i]) @ M if i in parents else M
    V, F, U, groups, own = [], [], [], {}, None
    nodes = [(i, n) for i, n in enumerate(gl.get("nodes", [])) if "mesh" in n]
    for ni, node in nodes:
        mesh_def = gl["meshes"][node["mesh"]]
        M = world(ni)
        ext = mesh_def.get("extras", {}).get("meshkit")
        for prim in mesh_def["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            P = _accessor(gl, binary, at["POSITION"])
            P = P @ M[:3, :3].T + M[:3, 3]
            idx = _accessor(gl, binary, prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim \
                else np.arange(len(P)).reshape(-1, 3)
            uv = _accessor(gl, binary, at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
            if ext is not None and "_MESHKIT_VERT" in at and len(nodes) == 1:
                own = (P, idx, uv, np.rint(_accessor(gl, binary, at["_MESHKIT_VERT"])).astype(np.int64), ext)
                continue
            base = sum(len(p) for p in V)
            f0 = len(F)
            V.append(P)
            F += (idx + base).tolist()
            U.append(uv if uv is not None else np.full((len(P), 2), np.nan))
            gname = mesh_def.get("name") or f"mesh{node['mesh']}"
            groups.setdefault(gname, []).extend(range(f0, len(F)))
    if own is not None:
        P, idx, uv, vid, ext = own
        n = int(vid.max()) + 1
        verts = np.zeros((n, 3))
        verts[vid] = _z_up(P)
        sizes = [s for s, c in ext.get("face_sizes", [[3, len(idx)]]) for _ in range(c)]
        faces, cuv, t = [], [], 0
        for s in sizes:
            tri = idx[t:t + s - 2]
            corners = [tri[0][0], tri[0][1]] + [r[2] for r in tri]
            faces.append([int(vid[c]) for c in corners])
            if uv is not None:
                cuv.append([(x, 1 - y) for x, y in uv[corners]])
            t += s - 2
        if uv is not None:
            for f in ext.get("faces_without_uvs", []):
                cuv[f] = None
        g = {k: (v["kind"], v["indices"]) for k, v in ext.get("groups", {}).items()}
        mesh = Mesh(verts, faces, groups=g, uvs=cuv if uv is not None else None)
    else:
        P = _z_up(np.vstack(V)) if V else np.zeros((0, 3))
        UV = np.vstack(U) if U else np.zeros((0, 2))
        _, first, weld = np.unique(P, axis=0, return_index=True, return_inverse=True)
        weld = weld.reshape(-1)
        order = np.argsort(first)
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        faces = [[int(remap[weld[v]]) for v in f] for f in F]
        cuv = [[(x, 1 - y) for x, y in UV[f]] for f in F]
        has = not np.isnan(UV).all()
        g = {k: ("f", v) for k, v in groups.items()} if len(groups) > 1 else {}
        mesh = Mesh(P[first[order]], faces, groups=g, uvs=cuv if has else None)
    if not images:
        return mesh
    out = {}
    mats = gl.get("materials", [])
    if mats and gl.get("images"):
        m0 = mats[0]
        pbr = m0.get("pbrMetallicRoughness", {})
        for key, ref in (("albedo", pbr.get("baseColorTexture")), ("roughness", pbr.get("metallicRoughnessTexture")),
                         ("normal", m0.get("normalTexture"))):
            if ref is not None:
                im = gl["images"][gl["textures"][ref["index"]]["source"]]
                v = gl["bufferViews"][im["bufferView"]]
                out[key] = binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
    return mesh, out


def glb_maps(path):
    """{'albedo' | 'roughness' | 'normal': decoded image array} embedded in a GLB's first material."""
    _, imgs = load_glb(path, images=True)
    return {k: read_png(v) for k, v in imgs.items()}
