# meshkit cheatsheet
`from meshkit import *` · Z up · CCW winding = front · run: `python -m meshkit run model.py|.obj|.json|.glb [--final] [--render [FOCUS]] [--zoom X] [--out DIR] [--no-export]`
Script: `mesh = ...` or `def build()`; `check_opts = {"closed": True, "expect_genus": 0, "uvs": True}`; `sdf = shape` adds proximity (`proximity_opts`);
`texture = layers(...)` adds UV checks, bakes maps (`bake_opts`) and exports them (GLB / OBJ+MTL / Blender script). Exit 0 ok / 1 fail / 2 error.
**Primitives/SDF meshing return a Mesh. Every op returns `(mesh, created)`; meshes are immutable.** `created`: `.verts .faces .edges` (Sel) + extras.
**sel** = None (all) · group name(s) · Sel · int · list of ints (op's own kind) · list of (a,b) edges. Sels convert (faces→verts; verts→faces inside).
New faces/verts inherit their source's groups and UVs. Tolerances are fractions of `mesh.diag`. Axes `'x' '-z'`, 3-vectors or scalars.
**Tip: iterate in draft, export in final** (`to_mesh` draft 64 cells / `--final` 180; bakes 512 / 1024). Render only at milestones.
**Tip: build masks from anatomy (normals, curvature, groups), not from UV coordinates.**

## Core
`Mesh(verts, faces=(), loose_edges=(), groups={name: ('v'|'f', idx)}, uvs=None)` · derived: `edges edge_faces vert_faces vert_neighbors face_normals face_areas face_centers vert_normals triangles diag`
`m.uvs` (C,2) per face corner in face order, NaN = missing · `m.corner_start` (face f = rows cs[f]:cs[f+1]) · `m.corner_verts` · `m.face_uvs(f)`
`Sel(kind 'v'|'e'|'f', idx, closed=False)`; `| & - +` · `verts_of/faces_of/edges_of(m, sel)` · `group(m, name)` → Sel · `resolve(m, sel, kind)` · `Created` `Group` `edge_key` `vec(d)` `unit(d)`

## Authoring & queries
`add_verts(m, coords)` · `add_edges(m, pairs | path, closed=False)` · `add_face(m, verts)` · `fill(m, loop)` · `grid_fill(m, loop, span=None, offset=None)` · `delete(m, sel, kind='f')` → vmap, fmap
`join(m, *others)` (groups merge, UVs kept: `pack` after) · `add_group(m, name, sel, kind=None)` · `remove_group(m, name)` · loops = vert Sel/list, edges or group
`boundary_loops` `edge_loop(m,(a,b))` `edge_ring` · `select_by_bbox(m, lo, hi, kind='v')` · `select_by_normal(m, dir, angle=30)` · `select_by_plane(m, point, normal, side)` · `nearest_verts` `bbox` `centroid`
`closest_point(m, P, faces=None)` · `raycast(m, origins, dirs)` · `inside(m, P)` · `winding_number` · `self_intersections(m)` → [(fa, fb)]

## Primitives (outward normals, `group=`; UVs: planar sides / cube-face / around x along, caps own islands)
`grid(size=1, divisions=1, axis='z')` · `cube(size=1, divisions=1)` · `quad_sphere(radius=1, subdivisions=4)` · `cylinder(radius=.5, depth=1, segments=16, rings=1, caps=True, radius_top=None)`
`circle(radius=.5, segments=16, fill=False)` wire · `ring(outer=1, inner=.5, depth=0)` · `revolve([(r,h),...], segments=16, closed=False, caps=False)` lathe, r=0 → pole

## Transforms & ops → `(mesh, created)`
`translate(m, offset, sel)` · `rotate(m, axis, deg, sel, pivot)` · `scale(m, f, sel, pivot)` · `map_verts(m, fn, sel)` · `displace(m, field, sel, along='normal'|axis|None)`
`noise(seed=0, scale=1, strength=1, octaves=4)` fbm f(P) → ±strength; **scale = feature size** in world units (bigger = broader blobs), not a frequency
`extrude(m, faces, dist, mode='region'|'individual')` → top · `inset(m, faces, thickness, depth=0)` → inner · `bridge(m, loop_a, loop_b, segments=1)` · `flip` · `recalc_normals`
`mirror(m, axis='x', weld=True, suffix=None)` (UV copies go right of 0..1) · `merge_by_distance(m, dist=None)` · `decimate(m, target_faces)` (groups follow, UV seams locked)
`subdivide(m, levels=1, smooth=True)` Catmull-Clark (or plain split): UVs interpolate, groups follow, borders are creases
`sweep(m, points, radius=.1, profile=None, segments=12, resolution=8, taper=1, caps=False, closed=False, path='smooth'|'joints')` → start, end, rings · `catmull_rom(points)`

## SDF (negative inside; `name=` names a part) → `to_mesh(sdf, res=None|'draft'|'final'|cells, bounds=None, keep='largest', group=None, parts=True, sparse=True)`
to_mesh makes one face group per named part (each face → nearest part); `sparse=False` = full-grid reference; `PRESETS`. `sphere(r, center)` `ellipsoid(radii, center, rotation)`
`box(size, center, round=0, rotation)` · `capsule(a, b, radius, radius_b=None, rotation)` · `cone(a, b, radius_a, radius_b=0, rotation)` · rotation = (axis, deg) | [(axis, deg), ...] | (rx,ry,rz) | 3x3
`sweep_sdf(points, radii, resolution=8)` curved tube · `limb_sdf(points, radii, knuckle=1.1, k=None)` straight segments + knuckle bulges · `sdf_transform(s, rotation, offset, pivot=None, scale=1)`
`union(*s)` · `smooth_union(*s, k=.1)` k per part ok · `subtract(a, b, k=0)` · `intersect(a, b)` · `displace_sdf(s, field)` · `SDF` `.eval(P)` `.normal(P)` · `project_to_sdf(s, P)` → points, normals
`proximity(s, threshold=None, allow=[(a, b)])` parts closer than max(k/2, final voxel) in empty space would fuse

## UVs
`unwrap(m, method='xatlas'|'box', margin=.01, max_stretch=2)` (xatlas = `[uv]` extra) · `pack(m, margin=.01, rotate=True)` re-pack islands at one texel density
`project_uvs(m, mode='box'|'planar'|'cylinder'|'sphere', faces=None, axis='z')` · `uv_islands(m)` → id per face · `uv_seams(m)` · `export_uv_sheet(m, path, size=1024)` seams orange, borders cyan, flips magenta, overlaps red

## Texture (fields on the 3D surface: seam-free on any UVs)
masks: `mask_normal(dir, lo=0, hi=.6)` · `mask_height(lo, hi, axis='z')` · `mask_group(name|'claw*'|[names], blur=0)` · `mask_curvature('convex'|'concave', lo=.2, hi=1)` · `mask_distance(group, radius)` · `mask_noise(seed, scale=.2)`
combine: `a * b` · `a | b` max · `a & b` min · `~a` invert · `a.smoothstep(lo, hi)` or `smoothstep(a, lo, hi)` · numbers work as masks · `stripes(axis | along=points, period, width=.5, warp=0)` bands
`spots(scale=.1, size=.35, seed, shape='flat'|'dome'|'cells')` cellular · `ramp(field, ['#hex', ...] | [(t, color), ...])` · `color(c)` · `surface_values(m, field_or_texture, at='verts'|'faces')`
`layers([(color | ramp | field, mask, opacity), ...], roughness=.6 | [(value, mask, opacity)], height=[(field, mask, depth)])` painted bottom to top
`bake(m, texture, size=None, out=None, name='texture', pad=16)` → Bake: albedo, roughness, normal (tangent space, from height), height PNGs, `coverage`, `paths`

## Verify
`check(m, closed=False, contacts='auto', expect_genus=None, intersect=False, uvs=False)` → Report (.ok .failures .stats): non-manifold, winding, inward, degenerate, duplicate, isolated
  (+ boundary if closed, genus ≠ expect, crossings if intersect; uvs=True: missing / flipped / zero-area / overlapping / outside 0..1, stretch p95 + worst by area, coverage)
`find_handles(m)` each handle: where, around which parts, which parts close the ring (runner prints it when genus > expected) · `contacts(m, by='parts', pairs=None)`
`describe(m, groups=None)` · `mirror_error(m, 'x')` · `register_check(fn)` fn(m) → [str] · `register_scorer(fn)` fn(m) → (name, float) · `run_plugins(m)` · `clear_plugins()`

## Export, import & render
`export_all(m, dir, name, bake=None)` → .obj .json _blender.py .glb (+ maps, .mtl) · `to_glb(m, path, bake=None)` PBR material, groups in extras · `load_glb(path)` · `load_obj(path)` (vt, g/o)
`to_obj(m, path, y_up=True, material=None)` · `to_json` / `from_json` · `write_blender_script` (vertex group per group: select a part in Edit Mode; `SPLIT_PARTS = True` = object per part)
`render_views(m, path, size=400, focus=None, zoom=None, views=None, textured=None)` PNG, back faces red. views = ['front' '3/4' ..., (az, el[, dist]), dict(az, el, distance, target, zoom, label)]:
  az 0 = front (looking +Y), 90 = from +X; distance → perspective. textured = a texture / bake (unlit albedo, for checking placement) or True (UV checker)
