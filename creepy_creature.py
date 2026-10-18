"""Creepy eight-legged creature blockout (meshkit SDF). Faces -Y, Z up, ground at z=0."""
import numpy as np
from meshkit import *

rng = np.random.default_rng(7)
Z = np.array([0.0, 0.0, 1.0])
RES = None  # follows the runner: draft while iterating, 180 cells with --final


def v(*a):
    return np.array(a, dtype=float)


# ---------------- body: big forms ----------------
ceph_c = v(0, -0.25, 0.80)
ceph = ellipsoid((0.56, 0.64, 0.42), ceph_c, rotation=("x", 8), name="ceph")
thorax_hump = ellipsoid((0.34, 0.34, 0.22), ceph_c + v(0, -0.22, 0.30), name="hump")  # domed carapace, narrower than the leg ring
head = ellipsoid((0.44, 0.40, 0.38), v(0, -0.95, 1.00), rotation=("x", -10), name="head")
abd_c = v(0.03, 1.12, 1.16)
abdomen = ellipsoid((0.88, 1.08, 0.80), abd_c, rotation=[("x", -18), ("y", 4)], name="abdomen")
waist = capsule(v(0, 0.22, 0.90), v(0, 0.45, 1.0), 0.19, radius_b=0.27, name="waist")

# medium forms: tumorous lumps on the abdomen, a dorsal ridge of knobs
lumps = [
    sphere(0.36, abd_c + v(0.32, 0.25, 0.46), name="lumpA"),
    sphere(0.28, abd_c + v(-0.30, 0.0, 0.55), name="lumpB"),
    sphere(0.24, abd_c + v(-0.30, 0.70, 0.30), name="lumpC"),
    sphere(0.20, abd_c + v(0.55, -0.35, 0.05), name="lumpD"),
    ellipsoid((0.42, 0.36, 0.40), abd_c + v(-0.72, 0.30, -0.05), rotation=("z", 25), name="growth"),
    sphere(0.16, abd_c + v(0.10, 1.05, -0.15), name="spinnerets"),
]
ridge = [sphere(0.13 - 0.02 * i, abd_c + v(0.02 * i, -0.55 + 0.42 * i, 0.74 - 0.12 * i * i), name=f"knob{i}")
         for i in range(4)]
abd = smooth_union(abdomen, *lumps, *ridge, k=[0.3] + [0.22] * len(lumps) + [0.12] * len(ridge))
abd = displace_sdf(abd, noise(seed=3, scale=0.3, strength=0.11))

# fangs (chelicerae) hanging under the eyes, curling inward
fangs = []
for s in (1, -1):
    root = v(s * 0.15, -1.0, 0.72)
    fangs.append(sweep_sdf([root, root + v(s * 0.03, -0.12, -0.18), root + v(-s * 0.04, -0.10, -0.38)],
                           [0.12, 0.08, 0.02]))

front = smooth_union(ceph, thorax_hump, head, *fangs, k=0.14)
front = displace_sdf(front, noise(seed=11, scale=0.5, strength=0.04))

# ---------------- eyes: small forms seated on a raised ocular mound ----------------
mound = ellipsoid((0.16, 0.10, 0.10), v(0, -1.02, 1.24), rotation=("x", -25), name="mound")
front = smooth_union(front, mound, k=0.18)
body = smooth_union(front, waist, abd, k=0.22)

head_c = v(0, -0.98, 1.04)
eye_specs = [  # (azimuth deg from -Y toward +X, elevation deg, radius) - deliberately irregular
    (16, 4, 0.185), (-17, 2, 0.17),          # big main pair
    (45, -8, 0.10), (-43, -10, 0.09),      # outer medium pair
    (6, 38, 0.07), (-8, 36, 0.065),        # upper pair
    (40, 30, 0.05), (-38, 28, 0.045),       # small rear-lateral pair
]
dirs = np.array([[np.sin(np.radians(a)) * np.cos(np.radians(e)), -np.cos(np.radians(a)) * np.cos(np.radians(e)),
                  np.sin(np.radians(e))] for a, e, _ in eye_specs])
surf, nrm, _, _ = project_to_sdf(front, head_c + dirs * 0.6)
eyes = [sphere(r, surf[i] - nrm[i] * r * 0.15, name=f"eye{i}") for i, (_, _, r) in enumerate(eye_specs)]


# ---------------- legs ----------------
def leg(side, idx, raise_amt=0.0, reach=1.0, knee=0.0):
    """side +1 = right (+x), -1 = left. idx 0 = front. raise_amt lifts the leg into a threat pose."""
    az0 = [40, 70, 104, 140][idx] + rng.uniform(-7, 7)
    bend = [-22, -8, 10, 20][idx] + rng.uniform(-5, 5)   # lower leg sweeps forward (front) / back (rear)
    def d_at(t):
        a = np.radians(az0 + bend * t)
        return v(side * np.sin(a), -np.cos(a), 0)
    L = [1.25, 1.05, 0.95, 1.2][idx] * rng.uniform(0.93, 1.07) * reach
    d = d_at(0)
    hip = ceph_c + v(d[0] * 0.48, d[1] * 0.45 + 0.08, -0.08)
    knee_h = 1.0 + rng.uniform(-0.1, 0.12) + knee
    pts = [
        hip,
        hip + d_at(0.0) * 0.26 * L + Z * 0.16,                 # coxa
        hip + d_at(0.2) * 0.85 * L + Z * knee_h,               # femur up to a high knee
        hip + d_at(0.45) * 1.18 * L + Z * (knee_h + 0.02),     # short patella
        hip + d_at(0.8) * 1.85 * L + Z * 0.0,                  # tibia down
        hip + d_at(1.0) * 2.20 * L + Z * -0.62,                # metatarsus
        hip + d_at(1.1) * 2.36 * L + Z * -0.70,                # tarsus flattens onto the ground
    ]
    pts[-2][2] = max(pts[-2][2], 0.14)
    pts[-1][2] = 0.04
    if raise_amt:  # menacing raised foreleg: high knee, tibia reaching forward, tip hooked down
        fwd = unit(d_at(0) + v(0, -0.8, 0))
        pts[2] = hip + d * 0.6 * L + Z * 1.35
        pts[3] = pts[2] + unit(fwd + Z * 0.35) * 0.3 * L
        pts[4] = pts[3] + unit(fwd + Z * 0.1) * 0.7 * L
        pts[5] = pts[4] + unit(fwd * 0.6 - Z) * 0.38 * L
        pts[6] = pts[5] + unit(-fwd * 0.3 - Z) * 0.16 * L
    radii = np.array([0.17, 0.155, 0.115, 0.095, 0.06, 0.04, 0.03]) * rng.uniform(0.9, 1.1)
    parts = []
    for i in range(len(pts) - 1):
        parts.append(capsule(pts[i], pts[i + 1], radii[i], radius_b=radii[i + 1] * 0.9))
        if i >= 1:  # subtle knuckles at the joints
            parts.append(sphere(radii[i] * 1.07, pts[i]))
    # femur muscle bulge, slightly off-centre toward the top
    a, b = pts[1], pts[2]
    parts.append(ellipsoid((radii[1] * 1.2, radii[1] * 1.2, radii[1] * 1.2), a + (b - a) * 0.4 + Z * 0.03))
    tip, prev = pts[-1], pts[-2]
    fwd = unit(tip - prev)
    down = -Z if not raise_amt else unit(np.cross(fwd, np.cross(-Z, fwd)) + 1e-9)
    claw = sweep_sdf([tip, tip + fwd * 0.08, tip + fwd * 0.13 + down * 0.06], [0.034, 0.022, 0.006])
    parts.append(claw)
    return smooth_union(*parts, k=0.06)


legs = []
for s in (1, -1):
    for i in range(4):
        pose = {(-1, 0): dict(raise_amt=1.0),             # threat pose
                (1, 3): dict(reach=1.12, knee=0.12),       # rear leg stretched back, pushing off
                (-1, 1): dict(reach=0.88, knee=0.18),      # middle leg tucked in, knee high
                (1, 1): dict(reach=1.04, knee=-0.06)}.get((s, i), {})
        legs.append(leg(s, i, **pose))

# pedipalps: short feelers beside the fangs
palps = []
for s in (1, -1):
    r0 = v(s * 0.3, -0.85, 0.72)
    palps.append(sweep_sdf([r0, r0 + v(s * 0.15, -0.25, 0.12), r0 + v(s * 0.18, -0.48, -0.1), r0 + v(s * 0.12, -0.55, -0.35)],
                           [0.075, 0.06, 0.05, 0.03]))

creature = smooth_union(body, *legs, *palps, k=0.12)
creature = smooth_union(creature, *eyes, k=0.015)

sdf = creature
check_opts = {"closed": True, "expect_genus": 0}
# legs, palps and eyes are meant to grow out of the head/body
proximity_opts = {"allow": [("head", "capsule1"), ("eye4", "eye6"), ("eye5", "eye7"), ("capsule0", "sweep10"), ("sphere0", "sweep10"), ("mound", "eye7"), ("mound", "eye6")]}


def build():
    return to_mesh(creature, res=RES, group="creature")
