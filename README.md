# meshkit

Code-first polygon mesh kernel for AI agents: build rough meshes as pure data (Python + numpy), verify them with
math, unwrap and texture them procedurally, iterate in seconds, then export to Blender or any DCC (GLB, OBJ + MTL, JSON
+ import script) for detailed work. API: [CHEATSHEET.md](CHEATSHEET.md).

## Setup (project venv)

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"      # Windows; on macOS/Linux: .venv/bin/python -m pip ...
```

Everything installs into `.venv`, nothing global. The core needs only numpy (`pip install -e .`). Extras: `sdf` =
scikit-image (+ scipy) for `to_mesh`; `uv` = xatlas for `unwrap` (`unwrap(m, method="box")` needs nothing); `test` =
pytest; `dev` = all of them. Run commands with the venv's Python (or activate it):

```
.venv\Scripts\python -m meshkit run examples/limb.py            # check + describe + plugins, exports to examples/out/
.venv\Scripts\python -m meshkit run examples/sdf_creature.py --render head    # + PNG zoomed on the head part
.venv\Scripts\python -m meshkit run examples/textured_creature.py --final --render   # unwrap, bake 1024 px, GLB
.venv\Scripts\python -m meshkit run some_export.obj                          # score a mesh from another tool (.obj .json .glb)
.venv\Scripts\python -m pytest                                               # tests
```

Iterate in draft, export in final: `to_mesh` uses the `draft` preset (64 cells) and bakes 512 px unless the runner gets
`--final` (180 cells, 1024 px). Textures are layer stacks of masks built from anatomy (normals, curvature, part groups)
and evaluated in 3D, so UV seams never show; `bake` turns them into albedo, roughness and normal maps.

In Blender: run `<name>_blender.py` (Text Editor > Run Script) for vertex groups per part (Edit Mode: Vertex Groups >
Select), UVs and the baked material; set `SPLIT_PARTS = True` to get one object per part. Or import the `.glb`.

Layout: `meshkit/mesh.py` (Mesh with per-corner UVs, Sel, selection resolution) · `author.py` (verts/edges/faces,
fill, delete, join, groups) · `query.py` · `spatial.py` (closest point, raycast, inside) · `prims.py` (with UVs) ·
`xform.py` · `field.py` (seeded noise) · `ops.py` (extrude ... subdivide) · `sweep.py` · `sdf.py` (fields, sparse
meshing, part groups, proximity) · `decimate.py` · `handles.py` (find_handles) · `uv.py` (islands, projections, pack,
unwrap, UV stats and sheet) · `texture.py` (masks, patterns, layers, bake) · `report.py` (check, describe, plugins) ·
`contact.py` · `render.py` (PNG sheets, custom cameras, textured) · `png.py` · `export.py` (OBJ/MTL/JSON, Blender
script) · `gltf.py` (GLB in/out) · `runner.py`.
