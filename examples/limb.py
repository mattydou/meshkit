"""Organic: an ellipsoid body patch with a limb swept along a curve and bridged into a hole, then mirrored across x
with a welded seam. Shows queries feeding ops, sweep + bridge, delete, mirror and Sel algebra."""
import numpy as np

from meshkit import *

SEG = 16  # verts around the limb; must match the hole's boundary loop

body = scale(quad_sphere(1.0, 8, group="body"), (1.2, 0.8, 0.9))[0]
m, _ = delete(body, select_by_plane(body, (0, 0, 0), "x", side="-", kind="f"))  # keep the +x half

hole_faces = select_by_bbox(m, lo=(0.5, -0.3, -0.3), hi=(None, 0.3, 0.3), kind="f")
assert len(hole_faces) == 16, f"expected a 4x4 hole, got {len(hole_faces)} faces"
m, _ = delete(m, hole_faces)
hole = next(l for l in boundary_loops(m) if len(l) == SEG)

c = centroid(m, hole)
r = float(np.linalg.norm(m.verts[list(hole)] - c, axis=1).mean())
path = [c + (0.15, 0, 0), c + (0.55, 0, -0.15), c + (0.85, 0.05, -0.65), c + (0.95, 0.1, -1.3)]
m, limb = sweep(m, path, radius=0.8 * r, segments=SEG, resolution=6, taper=0.55, caps="end", group="limb")
m, shoulder = bridge(m, hole, limb.start, segments=2)
m, _ = add_group(m, "shoulder", shoulder.faces)

m, _ = mirror(m, "x")  # weld seam verts on x=0; groups now cover both sides
neg = select_by_plane(m, (0, 0, 0), "x", side="-", kind="f")
m, _ = add_group(m, "limb_nx", faces_of(m, "limb") & neg)
m, _ = add_group(m, "limb_px", faces_of(m, "limb") - neg)
mesh = m
check_opts = {"closed": True}


@register_check
def symmetric(mesh):
    e = mirror_error(mesh, "x")
    return [] if e < 1e-9 else [f"x mirror error {e:.2e} (fraction of diagonal)"]


@register_scorer
def reach(mesh):
    lo, hi = bbox(mesh, "limb_px")
    return "limb_reach_x", float(hi[0])

