"""meshkit: a code-first polygon mesh kernel for AI agents (pure Python + numpy).

Build meshes as data, verify them with check()/describe(), unwrap and texture them procedurally, iterate, then export
to Blender or any DCC (GLB, OBJ + MTL, JSON + import script).
Primitives return a Mesh; every op returns (new_mesh, created). See CHEATSHEET.md."""
import sys as _sys

from .mesh import Mesh, Sel, Created, Group, edge_key, vec, unit, resolve, verts_of, faces_of, edges_of, group
from .author import (add_verts, add_edges, add_face, fill, grid_fill, delete, join, add_group, remove_group)
from .query import (boundary_loops, edge_loop, edge_ring, select_by_bbox, select_by_normal, select_by_plane,
                    nearest_verts, bbox, centroid)
from .prims import grid, cube, quad_sphere, cylinder, circle, revolve, ring
from .xform import translate, rotate, scale, map_verts, displace
from .field import noise
from .spatial import closest_point, raycast, inside, winding_number
from .contact import contacts, self_intersections
from .sdf import (SDF, PRESETS, sphere, ellipsoid, capsule, box, cone, sweep_sdf, limb_sdf, sdf_transform, union,
                  smooth_union, subtract, intersect, displace_sdf, project_to_sdf, proximity, to_mesh)
from .handles import find_handles
from .decimate import decimate
from .render import render_views
from .ops import extrude, inset, mirror, merge_by_distance, bridge, flip, recalc_normals, subdivide
from .uv import uv_islands, uv_seams, project_uvs, pack, unwrap, export_uv_sheet
from .texture import (mask_normal, mask_height, mask_group, mask_curvature, mask_distance, mask_noise, smoothstep,
                      stripes, spots, ramp, color, layers, bake, surface_values)
from .gltf import to_glb, load_glb
from .sweep import sweep, catmull_rom
from .report import check, describe, mirror_error, register_check, register_scorer, clear_plugins, run_plugins
from .export import to_obj, to_json, from_json, load_obj, write_blender_script, export_all

__all__ = [n for n, v in list(globals().items()) if not n.startswith("_") and not isinstance(v, type(_sys))]
