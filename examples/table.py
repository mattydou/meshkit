"""Table from primitives (furniture/props): slab top, tapered legs and apron rails; one leg and two rails are
built, then mirrored into place using the indices each step returned."""
from meshkit import *

W, D, H, T = 1.6, 0.9, 0.75, 0.04  # width, depth, height, top thickness
LEG_IN, RAIL_H = 0.09, 0.08        # leg inset from the edges, apron rail height
lx, ly = W / 2 - LEG_IN, D / 2 - LEG_IN

m = cube((W, D, T), center=(0, 0, H - T / 2), group="top")
m, leg = join(m, cylinder(0.025, H - T, segments=12, radius_top=0.035, center=(lx, ly, (H - T) / 2), group="legs"))
m, rail_x = join(m, cube((2 * lx, 0.02, RAIL_H), center=(0, ly, H - T - RAIL_H / 2), group="apron"))
m, rail_y = join(m, cube((0.02, 2 * ly, RAIL_H), center=(lx, 0, H - T - RAIL_H / 2), group="apron"))

m, _ = mirror(m, "x", sel=leg.faces | rail_y.faces, weld=False)  # right side -> left side
m, _ = mirror(m, "y", sel="legs", weld=False)                    # front legs -> back legs
m, _ = mirror(m, "y", sel=rail_x.faces, weld=False)              # front rail -> back rail
mesh = m
check_opts = {"closed": True}


@register_check
def legs_reach_floor_and_top(mesh):
    lo, hi = bbox(mesh, "legs")
    out = []
    if abs(lo[2]) > 1e-6:
        out.append(f"legs start at z={lo[2]:.4f}, not on the floor")
    if abs(hi[2] - (H - T)) > 1e-6:
        out.append(f"legs end at z={hi[2]:.4f}; top underside is at {H - T:.4f}")
    if len(faces_of(mesh, "legs")) != 4 * 14:
        out.append(f"expected 4 legs (56 faces), got {len(faces_of(mesh, 'legs'))} faces")
    return out


@register_scorer
def leg_spread(mesh):
    lo, hi = bbox(mesh, "legs")
    return "leg_spread_x", float(hi[0] - lo[0])
