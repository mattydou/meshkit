from meshkit import *
import numpy as np

# Front = +x, Z up. Body hovers ~1 unit above the ground on 8 legs.
BODY_Z = 1.0


def build():
    # Lumpy body: abdomen + cephalothorax, smooth-unioned and noise-displaced.
    body_sdf = smooth_union(
        ellipsoid((1.1, 0.85, 0.6), (-0.6, 0, BODY_Z + 0.05)),
        ellipsoid((0.75, 0.65, 0.48), (0.55, 0, BODY_Z)),
        k=0.25,
    )
    body_sdf = displace_sdf(body_sdf, noise(seed=3, scale=1.8, strength=0.18))
    m = to_mesh(body_sdf, resolution=72, group="body")

    # 8 legs: 4 per side, jointed (hip -> knee -> ankle -> foot), each with a claw.
    hip_x = [0.85, 0.45, 0.0, -0.45]
    splay = [35, 12, -12, -35]  # degrees of fore/aft fan-out
    for side, s in (("L", 1), ("R", -1)):
        for i, (hx, a) in enumerate(zip(hip_x, splay)):
            t = np.radians(a)
            d = np.array([np.sin(t), s * np.cos(t), 0.0])  # outward direction in XY
            hip = np.array([hx, s * 0.35, BODY_Z])
            knee = hip + d * 1.1 + [0, 0, 0.75]
            ankle = hip + d * 2.0 + [0, 0, 0.1]
            foot = hip + d * 2.35 + [0, 0, -BODY_Z + 0.18]
            name = f"leg_{side}{i + 1}"
            leg, c = sweep(m, [hip, knee, ankle, foot], radius=0.09, taper=0.55,
                           segments=10, path="joints", joint_radius=0.06,
                           caps=True, group=name)
            m = leg
            # Claw: small cone pointing down and slightly outward from the foot.
            tip = foot + d * 0.12 + [0, 0, -0.17]
            claw = to_mesh(cone(foot + [0, 0, 0.03], tip, 0.06, 0.0),
                           resolution=20, group=f"claw_{side}{i + 1}")
            m, _ = join(m, claw)

    # Several eyes clustered on the front of the head.
    eye_specs = [  # (y, z, radius)
        (0.16, BODY_Z + 0.28, 0.11), (-0.16, BODY_Z + 0.28, 0.11),  # main pair
        (0.36, BODY_Z + 0.20, 0.07), (-0.36, BODY_Z + 0.20, 0.07),
        (0.08, BODY_Z + 0.45, 0.06), (-0.08, BODY_Z + 0.45, 0.06),
    ]
    for j, (y, z, r) in enumerate(eye_specs):
        # Push each eye forward until it sits on the head surface.
        hit, pts, _, _ = raycast(m, [[3.0, y, z]], [[-1, 0, 0]])
        cx = pts[0][0] if hit[0] else 1.2
        eye = quad_sphere(r, subdivisions=3, center=(cx - r * 0.35, y, z),
                          group=f"eye_{j + 1}")
        m, _ = join(m, eye)
    return m


check_opts = {"closed": True, "contacts": "groups"}
