"""Sweep: a profile carried along a centripetal Catmull-Rom curve with parallel-transport frames (no twist, no flips)."""
import numpy as np

from .mesh import Builder, Created, Sel, unit
from .prims import cap_uv, finish_uvs, loft


def _cr_segment(P0, P1, P2, P3, s, alpha=0.5):
    """Centripetal Catmull-Rom between P1 and P2 at s in [0, 1] (Barry-Goldman)."""
    def knot(t, a, b):
        return t + max(float(np.linalg.norm(b - a)) ** alpha, 1e-9)
    t0 = 0.0
    t1 = knot(t0, P0, P1)
    t2 = knot(t1, P1, P2)
    t3 = knot(t2, P2, P3)
    t = (t1 + s * (t2 - t1))[:, None]
    A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


def catmull_rom(points, resolution=8, closed=False):
    """Smooth curve through points: resolution samples per span, evenly spaced by arc length.
    Returns (samples (M,3), u (M,) = position in control-point index units)."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(P)
    if n < 2:
        raise ValueError("a curve needs >= 2 points")
    if closed:
        ext = np.vstack([P[-1], P, P[0], P[1]])
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    spans = n if closed else n - 1
    dense_n = 32 * resolution
    s = np.linspace(0, 1, dense_n, endpoint=False)
    D, U = [], []
    for i in range(spans):
        D.append(_cr_segment(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], s))
        U.append(i + s)
    D.append((P[0] if closed else P[-1])[None])
    U.append([float(spans)])
    D, U = np.vstack(D), np.concatenate(U)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))])
    m = spans * resolution
    targets = np.linspace(0, L[-1], m + 1)
    if closed:
        targets = targets[:-1]
    C = np.column_stack([np.interp(targets, L, D[:, k]) for k in range(3)])
    return C, np.interp(targets, L, U)


def transport_frames(C, up=None, closed=False, T=None):
    """Tangents T (given, or by central differences) and reference vectors R (perpendicular to T, starting near
    `up`, default +z) by double-reflection parallel transport. Closed curves spread the end mismatch evenly.
    Returns (T, R)."""
    n = len(C)
    if T is not None:
        T = np.array(T, dtype=float)
    elif closed:
        T = np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)
    else:
        T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-12)[:, None]
    ref = unit(up) if up is not None else np.array([0.0, 0.0, 1.0])
    if abs(ref @ T[0]) > 0.99:
        ref = np.array([1.0, 0.0, 0.0]) if abs(T[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    R = np.zeros_like(C)
    R[0] = unit(ref - (ref @ T[0]) * T[0])
    steps = n if closed else n - 1
    r = R[0]
    for i in range(steps):
        j = (i + 1) % n
        v1 = C[j] - C[i]
        c1 = v1 @ v1
        if c1 < 1e-24:
            r_next = r
        else:
            rL = r - (2 / c1) * (v1 @ r) * v1
            tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
            v2 = T[j] - tL
            c2 = v2 @ v2
            r_next = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-24 else rL
        r_next = unit(r_next - (r_next @ T[j]) * T[j])
        if j == 0:
            # closed: angle between transported and original frame at the start, spread along the curve
            ang = np.arctan2(np.cross(r_next, R[0]) @ T[0], r_next @ R[0])
            for k in range(1, n):
                a = ang * k / n
                t = T[k]
                R[k] = R[k] * np.cos(a) + np.cross(t, R[k]) * np.sin(a) + t * (t @ R[k]) * (1 - np.cos(a))
        else:
            R[j] = r = r_next
    return T, R


def _slerp(a, b, t):
    """Unit vectors from unit a to unit b at parameters t."""
    ang = np.arccos(np.clip(a @ b, -1, 1))
    if ang < 1e-9:
        return np.tile(a, (len(t), 1))
    return (np.sin((1 - t) * ang)[:, None] * a + np.sin(t * ang)[:, None] * b) / np.sin(ang)


def joint_path(points, resolution=4, closed=False, joint_radius=0.0):
    """Polyline through joints: straight segments with `resolution` samples each. joint_radius > 0 rounds every joint
    with a circular arc (clamped to half the shorter segment); otherwise joints are mitered.
    Returns (samples C, tangents T, u in control-point units, miter) with miter[i] = (bend dir, stretch) or None."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(P)
    if n < 2:
        raise ValueError("a joint chain needs >= 2 points")
    nseg = n if closed else n - 1
    D, L = [], []
    for i in range(nseg):
        v = P[(i + 1) % n] - P[i]
        if np.linalg.norm(v) < 1e-12:
            raise ValueError(f"joints {i} and {(i + 1) % n} coincide")
        L.append(float(np.linalg.norm(v)))
        D.append(v / L[-1])
    trim, kind, phis = np.zeros(n), [None] * n, np.zeros(n)
    for j in range(n):
        if not closed and j in (0, n - 1):
            continue
        d_in, d_out = D[j - 1], D[j % nseg]
        phis[j] = phi = float(np.arccos(np.clip(d_in @ d_out, -1, 1)))
        if phi < 1e-6:
            continue
        if joint_radius > 0:
            trim[j] = min(joint_radius * np.tan(phi / 2), 0.5 * L[j - 1], 0.5 * L[j % nseg])
            kind[j] = "arc"
        elif phi > np.radians(170):
            raise ValueError(f"joint {j} turns back {np.degrees(phi):.0f} degrees; use joint_radius or more joints")
        else:
            kind[j] = "miter"
    C, T, U, M = [], [], [], []

    def add(p, t, u, m=None):
        C.append(p)
        T.append(t)
        U.append(u)
        M.append(m)

    for i in range(nseg):
        d, nxt = D[i], (i + 1) % n
        S, E = P[i] + trim[i] * d, P[nxt] - trim[nxt] * d
        k0 = 0
        if kind[i] == "miter":
            d_in = D[i - 1]
            add(P[i], d_in + d, i, (unit(d - d_in), min(1 / np.cos(phis[i] / 2), 10.0)))
            k0 = 1
        elif kind[i] == "arc":
            d_in, phi, t = D[i - 1], phis[i], trim[i]
            r = t / np.tan(phi / 2)
            O = P[i] + unit(d - d_in) * r / np.cos(phi / 2)
            A = P[i] - t * d_in
            na = max(2, int(np.ceil(phi / (np.pi / 12))))
            tau = np.arange(na) / na
            for p, tan_, s in zip(O + r * _slerp(unit(A - O), unit(S - O), tau), _slerp(d_in, d, tau), tau):
                add(p, tan_, i + (s - 1) * t / L[i - 1] + s * t / L[i])
        for k in range(k0, resolution):
            f = k / resolution
            add(S + (E - S) * f, d, i + (trim[i] + (L[i] - trim[i] - trim[nxt]) * f) / L[i])
    if not closed:
        add(P[-1], D[-1], n - 1)
    return np.array(C), np.array(T), np.array(U), M


def _taper(taper, t, u, n_points, closed=False):
    if callable(taper):
        return np.array([float(taper(x)) for x in t])
    if np.ndim(taper) == 0:
        return 1 + (float(taper) - 1) * t
    vals = np.asarray(taper, dtype=float)
    if len(vals) != n_points:
        raise ValueError(f"taper list needs one value per control point ({n_points})")
    if closed:
        return np.interp(np.mod(u, n_points), np.arange(n_points + 1), np.append(vals, vals[0]))
    return np.interp(u, np.arange(n_points), vals)


def sweep(mesh, points, radius=0.1, profile=None, segments=12, resolution=8, taper=1.0, caps=False,
          closed=False, up=None, group=None, path="smooth", joint_radius=0.0):
    """Sweep a profile along points; returns the mesh with the tube added.
    path: 'smooth' = Catmull-Rom curve through the points; 'joints' = straight segments between them (a jointed
    chain), joints mitered or rounded with arcs of joint_radius. resolution = rings per span/segment.
    profile: None = circle (segments verts), or closed [(x, y), ...] scaled by radius; y follows `up`, x = up x tangent.
    taper: end scale (linear from 1), one scale per control point, or f(t) with t in [0, 1] by arc length.
    caps: False | True | 'start' | 'end' (n-gons; a ring scaled to 0 becomes a point).
    UVs: u around (profile length), v along (arc length), caps their own islands, packed into 0..1.
    Created: verts, faces, start, end (vert loops at the ends), rings (all loops), joints (ring nearest each point)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if path == "smooth":
        C, u = catmull_rom(pts, resolution, closed)
        T0, miter = None, [None] * len(C)
    elif path == "joints":
        C, T0, u, miter = joint_path(pts, resolution, closed, joint_radius)
    else:
        raise ValueError("path must be 'smooth' or 'joints'")
    T, Y = transport_frames(C, up, closed, T=T0)
    X = np.cross(Y, T)
    if profile is None:
        th = 2 * np.pi * np.arange(segments) / segments
        prof = np.c_[np.cos(th), np.sin(th)]
    else:
        prof = np.asarray(profile, dtype=float).reshape(-1, 2)
        x, y = prof[:, 0], prof[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
            prof = prof[::-1]
    t = np.linspace(0, 1, len(C))
    s = _taper(taper, t, u, len(pts), closed) * radius
    rings = []
    for i in range(len(C)):
        if abs(s[i]) * np.abs(prof).max() < 1e-12:
            rings.append(C[i].copy())
            continue
        base = s[i] * (prof[:, :1] * X[i] + prof[:, 1:] * Y[i])
        if miter[i] is not None:  # exact miter: stretch the section along the bend direction
            w, f = miter[i]
            base += (f - 1) * (base @ w)[:, None] * w
        rings.append(C[i] + base)
    b = Builder(mesh)
    v0, f0 = len(b.V), len(b.F)
    perim = np.linalg.norm(np.diff(np.vstack([prof, prof[:1]]), axis=0), axis=1)
    us = np.r_[0, np.cumsum(perim)] * float(np.abs(s).mean())  # u around (profile length), v along (arc length)
    vs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([C, C[:1]]) if closed else C, axis=0), axis=1))]
    ids = loft(b, rings, closed=closed, uv=(us, vs))
    isl = [0] * (len(b.F) - f0)
    if not closed:
        if caps in (True, "start") and len(ids[0]) > 2:
            b.face(ids[0][::-1], uv=cap_uv(b, ids[0][::-1], -T[0]))
            isl.append(1)
        if caps in (True, "end") and len(ids[-1]) > 2:
            b.face(ids[-1], uv=cap_uv(b, ids[-1], T[-1]))
            isl.append(2)
    finish_uvs(b, f0, isl)
    faces = range(f0, len(b.F))
    if group:
        b.add_to_group(group, "f", faces)
    loops = [Sel("v", r, closed=len(r) > 1) for r in ids]
    n = len(pts)
    gap = [np.abs(u - c) if not closed else np.minimum(np.abs(np.mod(u, n) - c), n - np.abs(np.mod(u, n) - c))
           for c in range(n)]
    joints = [loops[int(np.argmin(g))] for g in gap]
    return b.build(), Created(verts=range(v0, len(b.V)), faces=faces, start=loops[0], end=loops[-1], rings=loops,
                              joints=joints)
