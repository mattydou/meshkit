from meshkit import *
import numpy as np

check_opts = {"closed": True}


def smooth(m, iterations=4, lam=0.5, mu=-0.53, sel=None):
    """Taubin smoothing built on vert_neighbors (no built-in smooth op)."""
    nbrs = [list(n) for n in m.vert_neighbors]
    for i in range(iterations):
        for f in (lam, mu):
            V = np.asarray(m.verts, float)
            avg = np.array([V[n].mean(0) if len(n) else V[k] for k, n in enumerate(nbrs)])
            new = V + f * (avg - V)
            m, _ = map_verts(m, lambda P, new=new: new, None)
    return m


def build():
    m = quad_sphere(radius=1, subdivisions=8, group="body")
    m, _ = scale(m, (1.35, 1.0, 0.75))

    for s in (1, -1):
        for i, x0 in enumerate((0.75, 0.25, -0.25, -0.75)):
            target = np.array([x0 * 1.35, s * 1.2, -0.1])
            body_f = set(resolve(m, "body", "f"))
            cand = [k for k in resolve(m, "body", "v")
                    if len(m.vert_faces[k]) == 4 and set(m.vert_faces[k]) <= body_f]
            V = np.asarray(m.verts)
            v = cand[int(np.argmin(np.linalg.norm(V[cand] - target, axis=1)))]
            faces = list(m.vert_faces[v])
            m, _ = delete(m, Sel("f", faces))
            loop = boundary_loops(m)[0]
            c = centroid(m, loop)
            n = unit(c * np.array([1 / 1.35**2, 1, 1 / 0.75**2]))
            fwd = x0 * 0.9 + (0.25 if x0 > 0 else -0.25)
            pts = [
                c + n * 0.08,
                c + n * 0.25,
                np.array([c[0] + fwd * 0.6, s * 1.9, 0.75]),
                np.array([c[0] + fwd * 1.0, s * 2.7, -0.3]),
                np.array([c[0] + fwd * 1.1, s * 2.9, -1.05]),
                np.array([c[0] + fwd * 1.1, s * 2.72, -1.32]),
            ]
            m, leg = sweep(m, pts, radius=0.16, segments=8, path="joints",
                           joint_radius=0.12, taper=[1, 0.95, 0.75, 0.55, 0.38, 0.06],
                           caps=False, group=f"leg{s}_{i}")
            m, _ = bridge(m, loop, leg.start)
            m, _ = grid_fill(m, leg.end)

    m, _ = displace(m, noise(seed=3, scale=1.8, strength=0.18), sel="body")

    # eyes: bulges on the front (+x); pick all faces up front so indices stay valid
    eye_yz = [(0.14, 0.28), (-0.14, 0.28), (0.4, 0.18), (-0.4, 0.18), (0.28, 0.48), (-0.28, 0.48)]
    body_f = resolve(m, "body", "f")
    C = m.face_centers[body_f]
    eye_f = []
    for y, z in eye_yz:
        d = np.linalg.norm((C - [1.4, y, z]) * [0.3, 1, 1], axis=1)
        eye_f.append(int(body_f[int(np.argmin(d))]))
    assert len(set(eye_f)) == len(eye_f), eye_f
    m, r = inset(m, Sel("f", eye_f), 0.025, mode="individual")
    m, e = extrude(m, r.inner, 0.08, mode="individual")
    m, _ = add_group(m, "eyes", e.top)

    m = smooth(m, iterations=3)
    return m
