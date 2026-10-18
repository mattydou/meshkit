"""Textured SDF lizard: named parts (body, head, jointed legs via limb_sdf, toes, eyes, dorsal spikes, curling tail)
become face groups; decimated under 15k triangles, unwrapped (xatlas), painted with masks built from anatomy
(groups, normals, curvature) and exported as GLB with baked albedo / roughness / normal maps.
`run --final` meshes at 180 cells and bakes 1024 px; `--render` adds an unlit textured sheet."""
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
sdf = smooth_union(*parts, k=ks)

m = to_mesh(sdf, group="skin")  # one face group per part (body, head, leg.0, toe.5, spike.2, ...)
m, _ = decimate(m, min(14000, len(m.faces)))  # under 15k triangles; part groups follow
m, _ = translate(m, (0, 0, -m.verts[:, 2].min()))
m, _ = unwrap(m)  # xatlas charts, packed with a margin
mesh = m
check_opts = {"closed": True, "expect_genus": 0}  # the runner adds UV checks because `texture` is set

# ---- texture: masks from anatomy, never from UV coordinates ----
back, belly = mask_normal("z", 0.0, 0.6), mask_normal("-z", 0.15, 0.7)
keratin = mask_group(["spike*", "toe*"])
texture = layers([
    ("#5d6b2f", None, 1),                                                         # olive base
    (ramp(mask_noise(seed=4, scale=0.25), ["#4b5724", "#76843d"]), None, 0.8),   # mottling
    ("#2c3417", spots(scale=0.12, size=0.38, seed=7) * back, 0.85),              # dark dorsal spots
    ("#e3d59c", belly, 1),                                                        # pale belly
    ("#22270f", stripes(along=TAIL, period=0.14, width=0.35) * mask_group("tail", blur=0.1) * ~belly, 0.9),
    ("#1c1813", keratin, 1),                                                      # dark spikes and toes
    ("#f0b430", mask_group("eye*"), 1),                                           # yellow eyes
    ("#1f230e", mask_curvature("concave", 0.4, 1.0), 0.35),                      # grime in the creases
], roughness=[(0.65, None, 1), (0.45, belly, 1), (0.3, keratin, 1), (0.15, mask_group("eye*"), 1)],
    height=[(spots(scale=0.05, size=0.45, seed=3, shape="dome"), back * ~keratin, 0.004)])  # small warts


@register_check
def under_15k_triangles(mesh):
    return [] if len(mesh.faces) < 15000 else [f"{len(mesh.faces)} triangles (want < 15000)"]


@register_check
def parts_kept(mesh):
    missing = [n for n in ("body", "head", "tail", "eye.0", "eye.1", "spike.0", "leg.3") if n not in mesh.groups]
    return [f"part groups missing: {' '.join(missing)}"] if missing else []


register_scorer(lambda mesh: ("triangles", len(mesh.faces)))
