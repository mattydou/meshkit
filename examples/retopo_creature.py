"""Retopology of the textured lizard (same SDF as textured_creature.py): instead of marching cubes + decimate,
cage_from_sdf builds all-quad topology from the named parts (a quad ellipsoid body; head, snout and eyes lathed out
of their parents; jointed legs with mitered knee loops, toes, spikes and the tail as tubes) and retopo_project snaps
it onto the surface. Closed, genus 0, all quads, under 15k triangles in final; the runner adds topo_report.
SDF finds the form, cage_from_sdf + retopo_project makes the final topology."""
from meshkit import *

parts, ks = [], []


def add(sdf, k):
    parts.append(sdf)
    ks.append(k)


body = ellipsoid((0.75, 0.32, 0.22), center=(0, 0, 0.42), name="body")
add(body, 0.15)
add(ellipsoid((0.28, 0.21, 0.15), center=(0.85, 0, 0.5), rotation=("y", -8), name="head"), 0.12)
add(ellipsoid((0.17, 0.13, 0.07), center=(1.08, 0, 0.46), name="snout"), 0.08)
for s in (-1, 1):
    add(sphere(0.065, center=(0.93, 0.14 * s, 0.6), name="eye"), 0.03)
    for x in (-0.42, 0.42):
        bend = 0.06 if x > 0 else -0.06  # elbows back, knees forward
        hip, knee, foot = (x, 0.2 * s, 0.38), (x + bend, 0.46 * s, 0.4), (x + 1.5 * bend, 0.56 * s, 0.05)
        add(limb_sdf([hip, knee, foot], [0.09, 0.065, 0.05], knuckle=1.25, name="leg"), 0.1)
        for t in (-1, 0, 1):
            tip = (foot[0] + 0.15 * (0.4 * t + (1 if x > 0 else -0.3)), foot[1] + s * 0.06 + 0.07 * t, 0.025)
            add(capsule(foot, tip, 0.028, 0.016, name="toe"), 0.03)
TAIL = [(-0.6, 0, 0.4), (-1.1, 0, 0.3), (-1.55, 0.18, 0.2), (-1.9, 0.42, 0.14)]
add(sweep_sdf(TAIL, [0.17, 0.11, 0.06, 0.025], name="tail"), 0.15)
for i, base in enumerate(project_to_sdf(body, [(0.45 - 0.17 * i, 0, 0.9) for i in range(6)]).points):
    add(cone(base - (0, 0, 0.03), base + (-0.04, 0, 0.11 - 0.012 * i), 0.045, 0.0, name="spike"), 0.04)
sdf = smooth_union(*parts, k=ks)  # also gives the runner's proximity report and topo_report's knee joints

cage = cage_from_sdf(sdf, group="skin")  # density follows the runner: draft ~15, --final ~30 edges per unit
mesh, snap = retopo_project(cage, sdf)  # project, Taubin-relax tangentially, re-project (x3), untangle folds
check_opts = {"closed": True, "expect_genus": 0}


@register_check
def clean_quads_on_the_surface(mesh):
    out = []
    if any(len(f) != 4 for f in mesh.faces):
        out.append("expected all quads")
    if len(mesh.triangles[0]) >= 15000:
        out.append(f"{len(mesh.triangles[0])} triangles (want < 15000)")
    if snap.max_dist > 1e-4:
        out.append(f"verts up to {snap.max_dist:.2g} off the SDF surface")
    if len(snap.flipped) > 0.02 * len(mesh.faces):  # a few fold where the three toes fuse into a webbed foot
        out.append(f"{len(snap.flipped)} faces folded over")
    return out


@register_check
def parts_kept(mesh):
    missing = [n for n in ("body", "head", "snout", "tail", "eye.0", "eye.1", "spike.0", "leg.3", "toe.0")
               if n not in mesh.groups]
    return [f"part groups missing: {' '.join(missing)}"] if missing else []


register_scorer(lambda mesh: ("triangles", len(mesh.triangles[0])))
register_scorer(lambda mesh: ("folded", len(snap.flipped)))
