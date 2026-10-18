"""Organic form from ~40 SDF primitives: body, head, snout, turned ears and horns, curved legs (sweep_sdf), foot
pads, toes and sharp claws (per-part k), a curling tail, eye sockets snapped onto the surface (project_to_sdf) and
carved, seeded noise for lumpy skin. Sparse marching cubes in draft while iterating; `run --final` for export.
One closed piece, genus 0, no parts fused by accident (proximity report via `sdf`)."""
from meshkit import *

parts, ks = [], []


def add(sdf, k):
    parts.append(sdf)
    ks.append(k)


add(ellipsoid((0.9, 0.45, 0.4), center=(0, 0, 0.75), rotation=("y", -6), name="body"), 0.18)
add(sphere(0.3, center=(1.0, 0, 1.05), name="head"), 0.18)
add(capsule((1.1, 0, 1.0), (1.42, 0, 0.92), 0.16, 0.1, name="snout"), 0.18)
for s in (-1, 1):
    add(ellipsoid((0.05, 0.1, 0.15), center=(0.9, 0.21 * s, 1.3), rotation=[("x", -30 * s), ("y", -15)], name="ear"),
        0.06)
    add(cone((1.05, 0.12 * s, 1.27), (1.05, 0.12 * s, 1.47), 0.05, 0.0, rotation=[("y", -35), ("x", -20 * s)],
             name="horn"), 0.03)
for x in (-0.55, 0.55):
    for y in (-0.22, 0.22):
        fx, fy = x * 1.08, y * 1.15
        bend = 0.07 if x > 0 else -0.07  # knees bend forward, elbows back
        add(sweep_sdf([(x, y, 0.62), (x + bend, y * 1.08, 0.36), (fx, fy, 0.1)], [0.17, 0.12, 0.1], name="leg"), 0.18)
        add(ellipsoid((0.14, 0.11, 0.07), center=(fx + 0.04, fy, 0.07), name="pad"), 0.08)
        for t in (-1, 0, 1):
            toe = (fx + 0.15, fy + 0.1 * t, 0.055)
            add(sphere(0.055, center=toe, name="toe"), 0.04)
            add(cone(toe, (toe[0] + 0.12, toe[1] + 0.02 * t, 0.015), 0.03, 0.0, name="claw"), 0.01)
add(sweep_sdf([(-0.8, 0, 0.8), (-1.2, 0, 0.92), (-1.45, 0, 1.2), (-1.38, 0, 1.45)], [0.12, 0.08, 0.05, 0.03],
              name="tail"), 0.18)
mass = smooth_union(*parts, k=ks)

eye_at = project_to_sdf(mass, [(1.3, 0.14, 1.2), (1.3, -0.14, 1.2)]).points  # sockets sit on the actual surface
eyes = union(*[sphere(0.065, center=p) for p in eye_at])
sdf = displace_sdf(subtract(mass, eyes, k=0.03), noise(seed=11, scale=0.35, strength=0.02, octaves=3))

m = to_mesh(sdf, group="skin")  # part names become face groups; res follows the runner (draft 64, --final 180)
m, _ = decimate(m, max(5000, len(m.faces) // 3))
m, _ = translate(m, (0, 0, -m.verts[:, 2].min()))  # stand on z = 0
m, _ = add_group(m, "feet", select_by_bbox(m, hi=(None, None, 0.05), kind="f"))
mesh = m
check_opts = {"closed": True, "expect_genus": 0}


@register_check
def one_piece_on_four_feet(mesh):
    out = []
    if check(mesh, contacts=None).stats["comps"] != 1:
        out.append("expected a single connected piece")
    feet = mesh.face_centers[list(faces_of(mesh, "feet"))]
    quadrants = {(x > 0, y > 0) for x, y, _ in feet}
    if len(quadrants) != 4:
        out.append(f"feet touch the ground in {len(quadrants)} of 4 quadrants")
    return out


register_scorer(lambda mesh: ("height", float(bbox(mesh)[1][2])))
register_scorer(lambda mesh: ("length", float(bbox(mesh)[1][0] - bbox(mesh)[0][0])))
