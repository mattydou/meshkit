"""Surface queries against a mesh's (fan-triangulated) faces: closest_point, raycast, inside.
Brute force in numpy, chunked to bound memory; fine for pre-DCC sizes (tens of thousands of faces)."""
from collections import namedtuple

import numpy as np

from .mesh import resolve, unit

Closest = namedtuple("Closest", "points faces dists")
Hits = namedtuple("Hits", "hit points faces dists")


def _points(p):
    P = np.asarray(p, dtype=float)
    return np.full((1, 3), float(P)) if P.ndim == 0 else P.reshape(-1, 3)


def _tris(mesh, faces):
    T, src = mesh.triangles
    if faces is not None:
        keep = np.isin(src, resolve(mesh, faces, "f"))
        T, src = T[keep], src[keep]
    if len(T) == 0:
        raise ValueError("no faces to query")
    V = mesh.verts
    return V[T[:, 0]], V[T[:, 1]], V[T[:, 2]], src


def _rows(n_tris, budget=4e5):
    return max(1, int(budget // max(n_tris, 1)))


def _dot(a, b):
    return (a * b).sum(-1)


def _closest_on_tris(P, A, B, C):
    """Closest points on each triangle for each point (Ericson). P (r,1,3), A/B/C (T,3) -> (r,T,3)."""
    ab, ac = B - A, C - A
    ap, bp, cp = P - A, P - B, P - C
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.nan_to_num(d1 / (d1 - d3))
        t_ac = np.nan_to_num(d2 / (d2 - d6))
        t_bc = np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        den = va + vb + vc
        v = np.nan_to_num(vb / den)
        w = np.nan_to_num(vc / den)
    conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6),
             (vc <= 0) & (d1 >= 0) & (d3 <= 0), (vb <= 0) & (d2 >= 0) & (d6 <= 0),
             (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
    out = A + ab * v[..., None] + ac * w[..., None]
    choices = [np.broadcast_to(A, out.shape), np.broadcast_to(B, out.shape), np.broadcast_to(C, out.shape),
               A + ab * t_ab[..., None], A + ac * t_ac[..., None], B + (C - B) * t_bc[..., None]]
    for cond, choice in zip(conds[::-1], choices[::-1]):  # first matching region wins
        out = np.where(cond[..., None], choice, out)
    return out


def closest_point(mesh, points, faces=None):
    """Nearest surface point for each query point (optionally only against `faces`).
    Returns Closest(points (k,3), faces (k,), dists (k,))."""
    P = _points(points)
    A, B, C, src = _tris(mesh, faces)
    pts, fi, dist = np.empty_like(P), np.empty(len(P), dtype=np.int64), np.empty(len(P))
    r = _rows(len(A))
    for i in range(0, len(P), r):
        q = P[i:i + r, None, :]
        X = _closest_on_tris(q, A, B, C)
        d2 = ((X - q) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        k = np.arange(len(j))
        pts[i:i + r], fi[i:i + r], dist[i:i + r] = X[k, j], src[j], np.sqrt(d2[k, j])
    return Closest(pts, fi, dist)


def raycast(mesh, origins, dirs, max_dist=np.inf, faces=None):
    """First hit of each ray (two-sided; origins/dirs broadcast; dirs may be axes like '-z'). Distances along unit dirs.
    Returns Hits(hit bool, points (nan = miss), faces (-1 = miss), dists (inf = miss))."""
    D = unit(dirs)[None] if isinstance(dirs, str) else np.array([unit(d) for d in _points(dirs)])
    O, D = np.broadcast_arrays(_points(origins), D)
    A, B, C, src = _tris(mesh, faces)
    e1, e2 = B - A, C - A
    eps = 1e-12 * mesh.diag
    best = np.full(len(O), np.inf)
    face = np.full(len(O), -1, dtype=np.int64)
    r = _rows(len(A))
    for i in range(0, len(O), r):
        o, d = O[i:i + r, None, :], D[i:i + r, None, :]
        pv = np.cross(d, e2)
        det = _dot(e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - A
        u = _dot(tv, pv) * inv
        qv = np.cross(tv, e1)
        v = _dot(d, qv) * inv
        t = _dot(e2, qv) * inv
        hit = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > eps) & (t <= max_dist)
        t = np.where(hit, t, np.inf)
        j = np.argmin(t, axis=1)
        tj = t[np.arange(len(j)), j]
        best[i:i + r] = tj
        face[i:i + r] = np.where(np.isfinite(tj), src[j], -1)
    hit = np.isfinite(best)
    pts = np.where(hit[:, None], O + D * np.where(hit, best, 0)[:, None], np.nan)
    return Hits(hit, pts, face, best)


def winding_number(mesh, points, faces=None):
    """Generalized winding number of the surface around each point (1 inside a closed outward mesh, 0 outside)."""
    P = _points(points)
    A, B, C, _ = _tris(mesh, faces)
    out = np.empty(len(P))
    r = _rows(len(A))
    for i in range(0, len(P), r):
        q = P[i:i + r, None, :]
        a, b, c = A - q, B - q, C - q
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = _dot(a, np.cross(b, c))
        den = la * lb * lc + _dot(a, b) * lc + _dot(b, c) * la + _dot(c, a) * lb
        out[i:i + r] = np.arctan2(num, den).sum(axis=1) / (2 * np.pi)
    return out


def inside(mesh, points, faces=None):
    """True where a point is inside the (closed) surface, by winding number > 0.5."""
    return winding_number(mesh, points, faces) > 0.5
