"""Creepy 8-legged creature blockout (meshkit). Front faces -Y, Z up."""
import math
import os
import random

import numpy as np
from meshkit import *

RES = int(os.environ.get("RES", 180))
PASS = os.environ.get("PASS", "0")

CEPH_C = np.array([0.0, -0.3, 0.62])
CEPH_R = np.array([0.52, 0.66, 0.36])


def V(*a):
    return np.array(a, dtype=float)


def chain(pts, radii, knuckle=1.0):
    """Capsules through pts with per-point radii; optional knuckle spheres at inner joints."""
    out = []
    for a, b, ra, rb in zip(pts[:-1], pts[1:], radii[:-1], radii[1:]):
        out.append(capsule(a, b, ra, rb))
    if knuckle > 1.0:
        for p, r in zip(pts[1:-1], radii[1:-1]):
            out.append(sphere(r * knuckle, p))
    return out


def leg(side, i, rng):
    """One spider leg: coxa, femur (swollen), patella, tibia, metatarsus, tarsus, claw."""
    hip_y = [-0.62, -0.42, -0.2, 0.02][i] + rng.uniform(-0.03, 0.03)
    hip = V(side * [0.34, 0.4, 0.42, 0.4][i], hip_y, 0.55)
    ang = math.radians([-62, -24, 8, 32][i] + rng.uniform(-9, 9))
    out = V(side * math.cos(ang), math.sin(ang), 0.0)
    L = [1.2, 1.0, 0.92, 1.12][i] * rng.uniform(0.92, 1.08)
    kz = rng.uniform(0.85, 1.1)  # knee height variation
    # gesture: front pair reaches up/forward (one higher), left 2nd leg mid-step
    lift = {(1, 0): 0.95, (-1, 0): 0.6, (-1, 1): 0.25}.get((side, i), 0.0)

    bow = [-0.65, -0.3, 0.35, 0.85][i] + rng.uniform(-0.08, 0.08)

    def P(d, z):
        return hip + out * d * L + V(0, bow * (d / 2.2) ** 2, z)

    coxa = P(0.2, 0.08)
    fem_mid = P(0.42, 0.75 * kz)
    knee = P(0.72, 1.25 * kz + lift * 0.25)
    patella = P(0.9, 1.25 * kz + lift * 0.35)
    ankle = P(1.45, 0.6 + lift * 0.85)
    meta = P(1.95, -0.15 + lift * 0.95)
    foot = P(2.2, -0.52 + lift * 0.9)
    r = 0.13 * rng.uniform(0.9, 1.1) * (1.1 if i in (0, 3) else 1.0)
    pts = [hip, coxa, fem_mid, knee, patella, ankle, meta, foot]
    radii = [r * 1.2, r * 1.0, r * 1.15, r * 0.85, r * 0.8, r * 0.55, r * 0.4, r * 0.28]
    parts = chain(pts, radii)
    # knuckles only at real joints, varied
    for p, rr, kk in [(knee, radii[3], 1.22), (patella, radii[4], 1.12),
                      (ankle, radii[5], 1.18), (meta, radii[6], 1.1)]:
        parts.append(sphere(rr * kk * rng.uniform(0.95, 1.05), p))
    # claw: hooked, two cones curving down and inward
    d = foot - meta
    d = d / np.linalg.norm(d)
    c1 = foot + d * 0.09 + out * 0.03
    c2 = c1 + V(0, 0, -0.12) - out * 0.06
    parts.append(cone(foot, c1, r * 0.34, r * 0.2))
    parts.append(cone(c1, c2, r * 0.2, 0.0))
    return parts


def on_ceph(x, z, inset):
    """Point on the front of the cephalothorax at (x, z), pushed inward by inset."""
    c, R = CEPH_C, CEPH_R
    t = 1 - (x / R[0]) ** 2 - ((z - c[2]) / R[2]) ** 2
    y = c[1] - R[1] * math.sqrt(max(t, 0.02))
    return V(x, y + inset, z)


def build():
    rng = random.Random(7)
    ceph = ellipsoid(CEPH_R, CEPH_C)
    ocular = ellipsoid((0.3, 0.28, 0.2), (0, -0.78, 0.86))  # raised eye mound
    pedicel = capsule((0, 0.25, 0.62), (0, 0.55, 0.78), 0.16)
    # abdomen: overlapping lobes, rear raised, slightly asymmetric
    lobes = [
        ellipsoid((0.72, 0.8, 0.62), (0, 1.15, 1.0)),
        ellipsoid((0.64, 0.66, 0.62), (0.06, 1.72, 1.3)),
        sphere(0.4, (0.3, 1.4, 1.48)),
        sphere(0.34, (-0.28, 1.58, 1.55)),
        sphere(0.25, (0.0, 2.15, 1.25)),
    ]
    abdomen = smooth_union(*lobes, k=0.25)
    # tubercles: two dorsal rows snapped onto the lobe surface by raycasting a coarse mesh
    coarse = to_mesh(abdomen, resolution=40)
    orgs, rads = [], []
    for j in range(6):
        for sx in (1, -1):
            orgs.append((sx * (0.18 + 0.1 * rng.random() + 0.04 * j), 0.95 + j * 0.24 + rng.uniform(-0.05, 0.05), 4.0))
            rads.append(0.06 + 0.05 * rng.random() * (1.2 - j * 0.12))
    hit, pts, _, _ = raycast(coarse, np.array(orgs), np.tile((0, 0, -1.0), (len(orgs), 1)))
    tub = [sphere(r_, p - V(0, 0, r_ * 0.35)) for h, p, r_ in zip(hit, pts, rads) if h]
    abdomen = smooth_union(abdomen, *tub, k=0.05)
    # spinnerets at the rear tip
    spin = [cone((sx * 0.07, 2.3, 1.05), (sx * 0.1, 2.52, 0.92), 0.08, 0.025) for sx in (1, -1)]
    abdomen = smooth_union(abdomen, *spin, k=0.06)
    abdomen = displace_sdf(abdomen, noise(seed=3, scale=1.5, strength=0.08))
    body = smooth_union(ceph, ocular, pedicel, abdomen, k=0.14)
    body = subtract(body, sphere(0.09, (0, -0.2, 1.04)), k=0.06)  # fovea dent
    body = displace_sdf(body, noise(seed=11, scale=5.0, strength=0.015))

    legs = []
    for side in (1, -1):
        for i in range(4):
            legs += leg(side, i, rng)
        # pedipalp: short feeler leg in front
        b = V(side * 0.16, -0.85, 0.5)
        palp = [b, b + V(side * 0.12, -0.25, 0.12), b + V(side * 0.2, -0.5, -0.05),
                b + V(side * 0.18, -0.62, -0.25)]
        legs += chain(palp, [0.07, 0.065, 0.05, 0.035], knuckle=1.15)
        # chelicera + fang
        cb = V(side * 0.1, -0.9, 0.48)
        ct = cb + V(side * 0.02, -0.12, -0.28)
        legs.append(capsule(cb, ct, 0.11, 0.08))
        legs.append(cone(ct, ct + V(-side * 0.07, -0.02, -0.14), 0.04, 0.0))

    s = smooth_union(body, *legs, k=0.07)

    # 8 eyes: big anterior medians, smaller laterals, posterior row on top
    eyes = []
    for x, z, rr in [(0.09, 0.82, 0.092), (0.2, 0.78, 0.05), (0.06, 0.96, 0.045),
                     (0.2, 0.93, 0.04)]:
        for sx in (1, -1):
            jitter = rng.uniform(0.92, 1.08)
            r_ = rr * jitter
            base = on_ceph_mound(sx * x, z, r_ * 0.35)
            eyes.append(sphere(r_, base))
    s = union(s, *eyes)
    return to_mesh(s, resolution=RES)


def on_ceph_mound(x, z, inset):
    """Point on the front of the ocular mound ellipsoid."""
    c, R = V(0, -0.78, 0.86), V(0.3, 0.28, 0.2)
    t = 1 - (x / R[0]) ** 2 - ((z - c[2]) / R[2]) ** 2
    y = c[1] - R[1] * math.sqrt(max(t, 0.05))
    return V(x, y + inset, z)


mesh = build()
if os.environ.get("RENDER"):
    render_views(mesh, f"out_crawler/pass{PASS}.png", size=500)
