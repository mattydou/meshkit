"""Retopology: clean quad topology for SDF forms.

cage_from_sdf(sdf) builds an all-quad cage from the named parts. The root mass (the largest sphere / ellipsoid / box)
becomes a quad ellipsoid or box lattice; every other part grows out of its parent through a hole cut where it leaves
it (the same step as attach_sweep): tubes follow capsule / cone / sweep_sdf / limb_sdf paths with the parts' radii,
rounder masses are lathed along the direction they leave their parent, and free ends close with quad grid caps.
Rings sit one edge apart (density = edges per unit length) with support loops at joints, and the segment count
changes through quad-only transitions as a part tapers. retopo_project(cage, sdf) snaps the cage onto the surface
(Newton projection with tangential Taubin relaxation in between), and topo_report(m) sums up the result: quad ratio,
valence, poles (flagged at joints), edge-length uniformity per group and loops per joint."""
import numpy as np

from . import sdf as _sdf
from .mesh import Builder, Created, Mesh, Sel, edge_key, resolve, unit
from .prims import _box_surface
from .report import Report, fmt, fvec
from .sdf import SDF, _chain, _cr_scalar, _min_sum, _pair_k, _parts, _transformed, project_to_sdf
from .sweep import catmull_rom, transport_frames

CAGE_PRESETS = {"draft": 50, "final": 100}  # cage edges along the longest side of the SDF's bounds
_CAP = np.pi / 4  # rounded ends: rings up to 45 degrees from the tip, a quad grid cap beyond
_MAPS = "sphere, ellipsoid, box, capsule, cone, sweep_sdf and limb_sdf"


# ---- parts -> pieces ------------------------------------------------------------------
class _Piece:
    """One mappable primitive of a named part, in world space. kind 'blob' (c, A: unit sphere -> ellipsoid),
    'box' (c, A: unit cube -> box) or 'tube' (centerline C, radii r, arc length S, joint arc lengths, end types)."""

    def __init__(self, name, kind, sdf, path, **geo):
        self.name, self.kind, self.sdf, self.path = name, kind, sdf, path
        self.joints, self.knuckle = [], 1.0
        self.__dict__.update(geo)

    @property
    def size(self):
        """Typical thickness."""
        return float(np.mean(self.r)) if self.kind == "tube" else float(np.linalg.norm(self.A, axis=0).min())

    @property
    def volume(self):
        if self.kind == "tube":
            rm = (self.r[1:] + self.r[:-1]) / 2
            return float(np.pi * (rm ** 2 * np.diff(self.S)).sum())
        det = abs(float(np.linalg.det(self.A)))
        return 4 / 3 * np.pi * det if self.kind == "blob" else 8 * det

    def core(self, p):
        """The point of the piece's core (center or centerline) nearest p."""
        if self.kind != "tube":
            return self.c
        return self.C[int(np.argmin(np.linalg.norm(self.C - p, axis=1)))]


def _xf_points(P, chain):
    for R, pivot, offset, scale in chain:
        P = (P - pivot) * scale @ (np.eye(3) if R is None else R).T + pivot + offset
    return P


def _xf_linear(A, chain):
    for R, _, _, scale in chain:
        A = scale * ((np.eye(3) if R is None else R) @ A)
    return A


def _arc(C):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]


def _pieces(name, s, path, chain=()):
    """The primitives making up part `name` (through transforms, unions, subtract / displace first operands)."""
    if getattr(s, "src", None) is not None:
        return _pieces(name, s.src, path, (s.xf,) + tuple(chain))
    if s.kind in ("union", "smooth_union"):
        return [p for kid in s.kids for p in _pieces(name, kid, path, chain)]
    if s.kind in ("subtract", "displace"):  # carvings and displacement are left to retopo_project
        return _pieces(name, s.kids[0], path, chain)
    q = s.params
    if q is None:
        raise ValueError(f"cage_from_sdf: can't map part {name!r} (kind {s.kind}); it maps {_MAPS} parts. For other "
                         "shapes mesh with to_mesh() and use quad_remesh() or isotropic_remesh()")
    world = s
    for xf in chain:
        world = _transformed(world, *xf)
    scale = float(np.prod([xf[3] for xf in chain])) if chain else 1.0
    if s.kind in ("sphere", "ellipsoid", "box"):
        A = (np.eye(3) * q["radius"] if s.kind == "sphere" else q["rotation"] @ np.diag(q["radii"])
             if s.kind == "ellipsoid" else q["rotation"] @ np.diag(q["size"] / 2))
        return [_Piece(name, "box" if s.kind == "box" else "blob", world, path,
                       c=_xf_points(q["center"][None], chain)[0], A=_xf_linear(A, chain))]
    joints, knuckle = [], 1.0
    if s.kind in ("capsule", "cone"):
        C, r = q["points"], q["radii"]
        ends = ("round", "round") if s.kind == "capsule" else ("flat", "flat")
    elif s.kind == "sweep":
        C, r = q["curve"], q["curve_radii"]
        joints = [int(np.argmin(np.abs(q["u"] - j))) for j in range(1, len(q["points"]) - 1)]
        ends = ("round", "round")
    else:  # limb: straight segments, knuckle bulges at inner joints
        P, rr = q["points"], q["radii"]
        t = np.linspace(0, 1, 9)[1:]
        C, r = [P[:1]], [rr[:1]]
        for i in range(len(P) - 1):
            C.append(P[i] + (P[i + 1] - P[i]) * t[:, None])
            r.append(rr[i] + (rr[i + 1] - rr[i]) * t)
            if i < len(P) - 2:
                joints.append(8 * (i + 1))
        C, r, knuckle, ends = np.concatenate(C), np.concatenate(r), q["knuckle"], ("round", "round")
    C = _xf_points(np.asarray(C, dtype=float), chain)
    keep = np.r_[True, np.linalg.norm(np.diff(C, axis=0), axis=1) > 1e-12]
    idx = np.cumsum(keep) - 1
    C, r, S = C[keep], np.asarray(r, dtype=float)[keep] * scale, None
    S = _arc(C)
    return [_Piece(name, "tube", world, path, C=C, r=r, S=S, joints=[float(S[idx[j]]) for j in joints],
                   ends=ends, knuckle=knuckle)]


def _reversed(p):
    """The same tube walked from its other end."""
    L = p.S[-1]
    return _Piece(p.name, "tube", p.sdf, p.path, C=p.C[::-1].copy(), r=p.r[::-1].copy(), S=L - p.S[::-1],
                  joints=sorted(L - s for s in p.joints), ends=p.ends[::-1], knuckle=p.knuckle)


# ---- which part grows out of which ------------------------------------------------------
def _connect(pieces, e):
    """{(i, j): (gap, k)} for pieces that touch or blend (gap < max(k/2, e/4)), both orders."""
    out = {}
    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            k = _pair_k(a.path, b.path)
            thr = max(k / 2, 0.25 * e)
            lo, hi = np.maximum(a.sdf.lo, b.sdf.lo) - thr, np.minimum(a.sdf.hi, b.sdf.hi) + thr
            if (lo <= hi).all():
                g, _ = _min_sum(a.sdf, b.sdf, lo, hi)
                if g < thr:
                    out[i, j] = out[j, i] = (g, k)
    return out


def _tree(pieces, gap, root, placed):
    """Parents by breadth-first search from root: first through solid overlaps (deeper than 15% of the thinner
    piece), then through any contact; each piece hangs off the frontier piece it overlaps most."""
    parent, order = {root: None}, [root]
    for solid in (True, False):
        frontier = list(order)
        while frontier:
            found = {}
            for i in range(len(pieces)):
                if i in parent or i in placed:
                    continue
                c = [(gap[i, j][0], j) for j in frontier if (i, j) in gap
                     and (not solid or gap[i, j][0] < -0.15 * min(pieces[i].size, pieces[j].size))]
                if c:
                    found[i] = min(c)[1]
            parent.update(found)
            order += sorted(found)
            frontier = sorted(found)
    return parent, order


# ---- rings ----------------------------------------------------------------------------------
def _perimeter(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def _stations(c, T, a, b=None, X=None, Y=None):
    c, T = np.asarray(c, dtype=float).reshape(-1, 3), np.asarray(T, dtype=float).reshape(-1, 3)
    if X is None:
        T, Y = transport_frames(c, T=T)
        X = np.cross(Y, T)
    a = np.asarray(a, dtype=float).reshape(-1)
    return dict(c=c, T=T, X=np.broadcast_to(X, c.shape).copy(), Y=np.broadcast_to(Y, c.shape).copy(), a=a,
                b=a.copy() if b is None else np.asarray(b, dtype=float).reshape(-1), w=np.zeros_like(c),
                f=np.ones(len(c)))  # w, f: miter (stretch by f along w) for rings in a bend's bisector plane


def _interp(S, X, s):
    s = np.asarray(s, dtype=float)
    return np.stack([np.interp(s, S, X[:, i]) for i in range(X.shape[1])], -1) if X.ndim == 2 else np.interp(s, S, X)


def _tangents(C):
    D = np.diff(C, axis=0)
    D /= np.linalg.norm(D, axis=1)[:, None]
    T = np.vstack([D[:1], D[:-1] + D[1:], D[-1:]]) if len(D) > 1 else np.vstack([D, D])
    return T / np.linalg.norm(T, axis=1)[:, None]


def _sphere_lift(E, r, T):
    def lift(p):
        x = p - E
        lat = x - (x @ T) * T
        return E + lat + T * np.sqrt(max(r * r - lat @ lat, 0.0))
    return lift


def _tube_stations(p, s0, e, support, both=False, clear=None):
    """Rings along tube piece p from arc length s0 to its far end (and past a round end, up to 45 degrees from the
    tip), with support loops around inner joints; none before arc length `clear` (where the tube leaves fused
    siblings). both=True also rounds the start (a free-standing tube).
    Returns (stations, cap) with cap = ('round', lift) | ('flat', None)."""
    S, C, r = p.S, p.C, p.r
    L = S[-1]
    n = max(1, int(round((L - s0) / e)))
    pos = np.linspace(s0, L, n + 1) if both else np.linspace(s0 + 0.5 * e, L, n) if L - s0 > 0.5 * e else np.r_[L]
    if clear is not None and clear > s0:
        pos = np.r_[clear, pos[pos > clear + 0.5 * e]] if clear < L else pos[-1:]
    bends = []  # (s, bend direction, miter stretch, half-width kept clear of other rings)
    for sj in p.joints:
        if not s0 + e < sj < L - 0.5 * e:
            continue
        h = 1e-4 * L
        tin = unit(_interp(S, C, sj) - _interp(S, C, sj - h))
        tout = unit(_interp(S, C, sj + h) - _interp(S, C, sj))
        phi = float(np.arccos(np.clip(tin @ tout, -1, 1)))
        rk = np.interp(sj, S, r) * max(p.knuckle, 1.0)
        clear = max(0.75 * e, 1.15 * rk * np.tan(min(phi, 2.6) / 2)) if phi > 0.2 else 0.75 * e
        d = max(support * e, clear) if phi > 0.2 else support * e
        pos = np.r_[pos[np.abs(pos - sj) > clear], sj, [sj - d, sj + d] if d > 0 else []]
        if phi > 0.2:  # a sharp bend: the joint ring lies in the bisector plane, its neighbours square to the segments
            bends.append((sj, unit(tout - tin), min(1 / np.cos(phi / 2), 3.0), 1.5 * clear))
    pos = np.unique(pos)
    rad = _interp(S, r, pos)
    for sj in p.joints:  # knuckle bulge: a sphere of knuckle x the joint radius
        rj = p.knuckle * np.interp(sj, S, r)
        rad = np.maximum(rad, np.sqrt(np.maximum(rj * rj - (pos - sj) ** 2, 0)))
    keep = rad > 0.2 * e  # a pointed end stops short of the tip
    pos, rad = pos[keep], rad[keep]
    Tc = _tangents(C)
    cen, tan = _interp(S, C, pos), _interp(S, Tc, pos)
    seg = np.clip(np.searchsorted(S, pos, side="right") - 1, 0, len(S) - 2)
    D = C[seg + 1] - C[seg]
    for sj, _, _, zone in bends:
        near = (np.abs(pos - sj) < zone) & (np.abs(pos - sj) > 1e-9 * L)
        tan[near] = D[near]
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    parts = []

    def dome(E, r_end, T):
        th = np.arange(1, int(_CAP / max(e / r_end, 1e-9) + 1e-9) + 1) * e / r_end if r_end > 0 else np.zeros(0)
        th = th[th <= _CAP + 1e-9]
        return E + np.sin(th)[:, None] * r_end * T, np.cos(th) * r_end

    if both and p.ends[0] == "round":
        c0, r0 = dome(C[0], r[0], -Tc[0])
        parts.append((c0[::-1], np.tile(Tc[0], (len(c0), 1)), r0[::-1]))
    parts.append((cen, tan, rad))
    cap = ("flat", None)
    if p.ends[1] == "round" and r[-1] > 0:
        c1, r1 = dome(C[-1], r[-1], Tc[-1])
        parts.append((c1, np.tile(Tc[-1], (len(c1), 1)), r1))
        cap = ("round", _sphere_lift(C[-1], r[-1], Tc[-1]))
    cen, tan, rad = (np.concatenate([q[i] for q in parts]) for i in range(3))
    if not len(cen):
        return None, cap
    st = _stations(cen, tan, rad)
    for sj, w, f, _ in bends:
        i = int(np.argmin(np.linalg.norm(cen - _interp(S, C, sj), axis=1)))
        st["w"][i], st["f"][i] = w, f
    return st, cap


def _ellipse_frame(A, d):
    """Semi-axes (a1, a2) and directions (u1, u2) of the section of ellipsoid A (unit sphere -> it) through its
    center, perpendicular to d; and h, half the chord along d."""
    Ai = np.linalg.inv(A)
    M = Ai.T @ Ai
    e1 = unit(np.cross(d, (1, 0, 0) if abs(d[0]) < 0.9 else (0, 1, 0)))
    e2 = np.cross(d, e1)
    E = np.stack([e1, e2], 1)
    lam, Q = np.linalg.eigh(E.T @ M @ E)
    u = E @ Q
    u1, u2 = u[:, 0], u[:, 1] if np.cross(u[:, 0], u[:, 1]) @ d > 0 else -u[:, 1]
    return 1 / np.sqrt(lam), (u1, u2), 1 / np.sqrt(d @ M @ d)


def _blob_stations(p, parent, e):
    """A mass growing out of its parent, lathed along the direction from the parent's core to its center: rings
    from where that axis leaves the parent to 45 degrees from the tip. Returns (stations, cap, exit point) or None
    when the axis never leaves the parent."""
    q = parent.core(p.c)
    d = p.c - q
    if np.linalg.norm(d) < 1e-9 * max(p.size, 1e-12):
        d = p.A[:, int(np.argmax(np.linalg.norm(p.A, axis=0)))]
    d = unit(d)
    (a1, a2), (u1, u2), h = _ellipse_frame(p.A, d)
    s = np.linspace(-h, h, 129)
    inside = parent.sdf.fn(p.c + s[:, None] * d) <= 0
    if inside[-1]:
        return None
    k = int(np.flatnonzero(inside)[-1]) if inside.any() else -1
    if k < 0:
        s_exit = -h
    else:
        f0, f1 = parent.sdf.fn(p.c + s[[k, k + 1], None] * d)
        s_exit = s[k] + (s[k + 1] - s[k]) * f0 / (f0 - f1) if f0 != f1 else s[k]
    th0 = np.arcsin(np.clip(max(s_exit, -h * np.sin(_CAP)) / h, -1, 1))
    th = np.linspace(th0, max(th0, _CAP), 400)
    speed = np.hypot(h * np.cos(th), 0.5 * (a1 + a2) * np.sin(th))
    arc = np.r_[0, np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(th))]
    marks = np.arange(0.5 * e, arc[-1] + 1e-9, e) if arc[-1] >= 0.5 * e else np.zeros(0)
    tk = np.interp(marks, arc, th)
    st = None
    if len(tk):
        st = _stations(p.c + (h * np.sin(tk))[:, None] * d, np.tile(d, (len(tk), 1)), a1 * np.cos(tk),
                       a2 * np.cos(tk), X=u1, Y=u2)

    def lift(x):
        y = x - p.c
        l1, l2 = y @ u1, y @ u2
        rho = (l1 / a1) ** 2 + (l2 / a2) ** 2
        return p.c + l1 * u1 + l2 * u2 + d * h * np.sqrt(max(1 - rho, 0.0))
    return st, ("round", lift), p.c + s_exit * d


def _ring_points(st, i, phi):
    base = st["a"][i] * np.cos(phi)[:, None] * st["X"][i] + st["b"][i] * np.sin(phi)[:, None] * st["Y"][i]
    if st["f"][i] != 1:
        w = st["w"][i]
        base = base + (st["f"][i] - 1) * (base @ w)[:, None] * w
    return st["c"][i] + base


def _desired(st, i, e):
    return max(4, 2 * int(round(_perimeter(st["a"][i], st["b"][i]) / (2 * e))))


def _next_count(M, D):
    """Segments for the next ring: unchanged unless the wanted count D is 1.5x off; at most 3x per band."""
    if D <= M / 1.5:
        n = max(D, 2 * int(np.ceil(M / 6)))
    elif D >= 1.5 * M:
        n = min(D, 3 * M)
    else:
        return M
    return max(4, n - n % 2)


def _uniform(phi):
    j = np.arange(len(phi)) * 2 * np.pi / len(phi)
    return np.mean(phi - j) + j


def _turn(A, phi, off):
    """Ring A started at index off (angles kept increasing)."""
    return list(A[off:]) + list(A[:off]), np.r_[phi[off:], phi[:off] + 2 * np.pi]


def _band(b, A, phiA, st, i, n, out, weak=()):
    """Ring for station i with n verts, joined to ring A (angles phiA, increasing) by quads; a segment change
    uses 3-to-1 (or 1-to-3) quad patterns spread around the ring, placed so that no `weak` vert of A (one with no
    edge leading away from the ring) is left with two edges. Returns (ring, angles)."""
    M = len(A)
    V = b.V

    def P(v):
        return np.asarray(V[v])

    def face(*vs):
        out.append(b.face(vs))

    if M % 2:  # odd hole (triangle mesh): one triangle evens it out
        ext = np.r_[phiA, phiA[0] + 2 * np.pi]
        phiB = np.r_[phiA[0], (ext[0] + ext[1]) / 2, phiA[1:]]
        phiB = 0.5 * phiB + 0.5 * _uniform(phiB)
        B = [b.vert(p) for p in _ring_points(st, i, phiB)]
        face(A[0], A[1], B[2], B[1])
        face(A[0], B[1], B[0])
        for j in range(1, M):
            face(A[j], A[(j + 1) % M], B[(j + 2) % (M + 1)], B[j + 1])
        return B, phiB
    if n == M:
        phiB = 0.5 * phiA + 0.5 * _uniform(phiA)
        B = [b.vert(p) for p in _ring_points(st, i, phiB)]
        for j in range(M):
            face(A[j], A[(j + 1) % M], B[(j + 1) % M], B[j])
        return B, phiB
    if n < M:
        k = (M - n) // 2
        starts = {(p * M) // k for p in range(k)}
        if weak:
            off = next((o for o in range(M) if not any(A[(s + 2 + o) % M] in weak for s in starts)), 0)
            A, phiA = _turn(A, phiA, off)
        drop = {(s + 1) % M for s in starts} | {(s + 2) % M for s in starts}
        keep = [j for j in range(M) if j not in drop]
        phiB = phiA[keep]
        phiB = 0.5 * phiB + 0.5 * _uniform(phiB)
        B = [b.vert(p) for p in _ring_points(st, i, phiB)]
        under = dict(zip(keep, B))
        skip = set()
        for j in range(M):
            if j in skip:
                continue
            j1, j2, j3 = (j + 1) % M, (j + 2) % M, (j + 3) % M
            if j in starts:  # t0 t1 t2 t3 above -> b0 b1: three quads around a center vert
                c = b.vert(0.25 * (P(A[j1]) + P(A[j2]) + P(under[j]) + P(under[j3])))
                face(A[j], A[j1], c, under[j])
                face(A[j1], A[j2], A[j3], c)
                face(c, A[j3], under[j3], under[j])
                skip |= {j1, j2}
            else:
                face(A[j], A[j1], under[j1], under[j])
        return B, phiB
    k = (n - M) // 2
    starts = {(p * M) // k for p in range(k)}
    nxt = np.r_[phiA[1:], phiA[0] + 2 * np.pi]
    phiB, kind = [], []
    for j in range(M):
        phiB.append(phiA[j])
        kind.append((0, j))
        if j in starts:
            phiB += [phiA[j] + (nxt[j] - phiA[j]) / 3, phiA[j] + 2 * (nxt[j] - phiA[j]) / 3]
            kind += [(1, j), (1, j)]
    phiB = np.array(phiB)
    phiB = 0.5 * phiB + 0.5 * _uniform(phiB)
    B = [b.vert(p) for p in _ring_points(st, i, phiB)]
    under, extra = {}, {}
    for v, (x, j) in zip(B, kind):
        if x:
            extra.setdefault(j, []).append(v)
        else:
            under[j] = v
    for j in range(M):
        j1 = (j + 1) % M
        if j in starts:  # b0 b1 below -> t0 t1 t2 t3
            t1, t2 = extra[j]
            c = b.vert(0.25 * (P(A[j]) + P(A[j1]) + P(t1) + P(t2)))
            face(A[j], c, t1, under[j])
            face(c, under[j1], t2, t1)
            face(A[j], A[j1], under[j1], c)
        else:
            face(A[j], A[j1], under[j1], under[j])
    return B, phiB


def _cap(b, loop, lift, out, weak=()):
    """Close an even loop with an a x c grid of quads (Coons patch), interior verts lifted by lift(p); the four
    corners avoid `weak` verts."""
    M = len(loop)
    a = M // 4
    c = M // 2 - a
    if weak:
        off = next((o for o in range(M) if not any(loop[(k + o) % M] in weak for k in (0, a, a + c, 2 * a + c))), 0)
        loop = list(loop[off:]) + list(loop[:off])
    G = np.zeros((a + 1, c + 1), dtype=np.int64)
    for i in range(a + 1):
        G[i, 0], G[i, c] = loop[i], loop[2 * a + c - i]
    for j in range(c + 1):
        G[a, j] = loop[a + j]
        if j:
            G[0, j] = loop[M - j]
    V = b.V

    def P(v):
        return np.asarray(V[v])

    p00, pa0, p0c, pac = P(G[0, 0]), P(G[a, 0]), P(G[0, c]), P(G[a, c])
    for i in range(1, a):
        for j in range(1, c):
            u, v = i / a, j / c
            p = ((1 - v) * P(G[i, 0]) + v * P(G[i, c]) + (1 - u) * P(G[0, j]) + u * P(G[a, j])
                 - ((1 - u) * (1 - v) * p00 + u * (1 - v) * pa0 + (1 - u) * v * p0c + u * v * pac))
            G[i, j] = b.vert(lift(p) if lift else p)
    for i in range(a):
        for j in range(c):
            out.append(b.face((G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1])))


# ---- holes --------------------------------------------------------------------------------
def _pairs(f):
    return zip(f, f[1:] + f[:1])


def _boundary(mesh, region):
    """(loop, None) when region is a disk with one simple boundary (loop in the region faces' winding), else
    (None, (problem, vert))."""
    ef, F = mesh.edge_faces, mesh.faces
    out = {}
    for f in region:
        for a, b in _pairs(F[f]):
            if not any(g in region for g in ef[edge_key(a, b)] if g != f):
                if a in out:
                    return None, ("pinch", a)
                out[a] = b
    if not out:
        return None, ("closed", None)
    start = next(iter(out))
    loop, v = [start], out[start]
    while v != start and v in out and len(loop) <= len(out):
        loop.append(v)
        v = out[v]
    if v != start or len(loop) != len(out):
        return None, ("loops", None)
    verts = {v for f in region for v in F[f]}
    edges = {edge_key(a, b) for f in region for a, b in _pairs(F[f])}
    if len(verts) - len(edges) + len(region) != 1:
        return None, ("genus", None)
    return loop, None


def _components(mesh, faces):
    ef, F = mesh.edge_faces, mesh.faces
    faces, seen, comps = set(faces), set(), []
    for f0 in faces:
        if f0 in seen:
            continue
        comp, stack = [], [f0]
        seen.add(f0)
        while stack:
            f = stack.pop()
            comp.append(f)
            for a, b in _pairs(F[f]):
                for g in ef[edge_key(a, b)]:
                    if g in faces and g not in seen:
                        seen.add(g)
                        stack.append(g)
        comps.append(comp)
    return comps


def _square(mesh, allowed, seed, O, side):
    """A side x side block of faces around seed (grown ring by ring through shared verts, like select-more; even
    sides start from the seed face's vert nearest O), or None if it leaves the allowed faces or is no disk."""
    F, vf = mesh.faces, mesh.vert_faces
    if side % 2:
        region = {seed}
    else:
        v = min(F[seed], key=lambda v: float(np.linalg.norm(mesh.verts[v] - O)))
        region = set(vf[v])
    for _ in range((side - 1) // 2):
        grow = {g for f in region for v in F[f] for g in vf[v]}
        if not grow <= allowed:
            return None
        region = grow
    if not region <= allowed:
        return None
    loop, _ = _boundary(mesh, region)
    return None if loop is None else (region, loop)


def _hole(mesh, allowed, in_print, seed_point, what, side=None):
    """Faces to cut where a part leaves: a square block of `side` faces around the allowed face nearest
    seed_point when it fits (4 corners, segment count 4 x side), else the footprint component around that face
    grown into a disk. Returns (region, boundary loop)."""
    if not len(allowed):
        raise ValueError(f"cage_from_sdf: no faces left to grow {what} from")
    seed = int(allowed[np.argmin(np.linalg.norm(mesh.face_centers[allowed] - seed_point, axis=1))])
    if side:
        got = _square(mesh, set(allowed.tolist()), seed, seed_point, side)
        if got is not None:
            return got
    cand = set(allowed[in_print].tolist()) | {seed}
    region = set(next(c for c in _components(mesh, cand) if seed in c))
    ok = set(allowed.tolist())
    for _ in range(12):
        loop, why = _boundary(mesh, region)
        if loop is not None and len(loop) >= 4:
            return region, loop
        problem, v = why or ("small", None)
        if problem == "pinch":
            grow = set(mesh.vert_faces[v]) & ok
        else:  # enclosed islands of allowed faces join the hole
            rest = sorted(_components(mesh, set(range(len(mesh.faces))) - region), key=len)
            grow = {f for c in rest[:-1] if set(c) <= ok for f in c}
        if not grow - region:
            break
        region |= grow
    order = allowed[np.argsort(np.linalg.norm(mesh.face_centers[allowed] - seed_point, axis=1))]
    for f in order[:16]:  # crowded: a single face near the exit
        loop, _ = _boundary(mesh, {int(f)})
        if loop is not None:
            return {int(f)}, loop
    raise ValueError(f"cage_from_sdf: can't cut a clean hole for {what} near {fvec(seed_point)} (parts too close "
                     "together for this density; raise density or move the part)")


def _loop_angles(V, loop, O, st, i=0):
    """Angles of the hole loop around station i's axis through O: arc-length spaced (monotonic), rotated to fit.
    Flips the station frames' Y when the loop runs clockwise, so ring angles always increase along the loop."""
    P = np.asarray([V[v] for v in loop]) - O
    x, y = P @ st["X"][i], P @ st["Y"][i]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        st["Y"] = -st["Y"]
        y = -y
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    base = 2 * np.pi * np.r_[0, np.cumsum(seg)[:-1]] / max(seg.sum(), 1e-300)
    off = np.angle(np.exp(1j * (np.arctan2(y, x) - base)).mean())
    return base + off


def _grow(mesh, label, allowed, footprint, margin, O, stations, cap, what, e, tag):
    """Cut a hole in mesh (allowed faces, footprint sdf < margin, around O), grow rings through stations from it
    and close the end (cap = (kind, lift); kind 'open' leaves it open). Returns (mesh, label, info)."""
    allowed = np.asarray(allowed, dtype=np.int64)
    in_print = footprint.fn(mesh.face_centers[allowed]) < margin if len(allowed) else np.zeros(0, dtype=bool)
    want = _desired(stations, 0, e) if stations is not None else 4 * max(1, int(round(np.sqrt(in_print.sum()))))
    region, loop = _hole(mesh, allowed, in_print, O, what, side=max(1, int(round(want / 4))))
    vf = mesh.vert_faces
    weak = {v for v in loop if len(set(vf[v]) - region) == 1}  # no edge leads away from the hole
    b = Builder(mesh)
    new = []
    ring = loop
    rings = [list(loop)]
    if stations is not None:
        phi = _loop_angles(b.V, loop, O, stations)
        for i in range(len(stations["c"])):
            n = _next_count(len(ring), _desired(stations, i, e))
            ring, phi = _band(b, ring, phi, stations, i, n, new, weak if i == 0 else ())
            rings.append(list(ring))
    if cap[0] != "open":
        _cap(b, ring, cap[1], new, weak if stations is None else ())
    used = {v for f in range(len(b.F)) if f not in region for v in b.F[f]}
    drop = {v for f in region for v in mesh.faces[f]} - used
    out, vmap, fmap = b.build_compact(drop, region)
    lab = None
    if label is not None:
        lab = np.full(len(out.faces), tag, dtype=np.int64)
        old = fmap[:len(label)]
        lab[old[old >= 0]] = label[old >= 0]
    info = dict(faces=sorted(int(fmap[f]) for f in new), rings=[[int(vmap[v]) for v in r] for r in rings],
                verts=sorted(set(range(len(mesh.verts) - len(drop), len(out.verts)))))
    return out, lab, info


# ---- cages ------------------------------------------------------------------------------------
def _root_cage(p, e, support):
    """Quad ellipsoid / box lattice for a blob or box, or a capped tube."""
    if p.kind != "tube":
        L = np.linalg.norm(p.A, axis=0)
        n = [max(1, int(round((np.pi / 2 if p.kind == "blob" else 2) * x / e))) for x in L]
        P, F, _ = _box_surface(tuple(n))
        U = 2 * P / np.array(n) - 1
        if p.kind == "blob":
            U = np.tan(U * np.pi / 4)
            U /= np.linalg.norm(U, axis=1)[:, None]
        return Mesh(p.c + U @ p.A.T, F)
    st, cap = _tube_stations(p, 0.0, e, support, both=True)
    b = Builder(Mesh())
    new = []
    n0 = _desired(st, 0, e)
    phi = 2 * np.pi * np.arange(n0) / n0
    ring = first = [b.vert(x) for x in _ring_points(st, 0, phi)]
    for i in range(1, len(st["c"])):
        ring, phi = _band(b, ring, phi, st, i, _next_count(len(ring), _desired(st, i, e)), new)
    _cap(b, ring, cap[1], new)
    lift0 = _sphere_lift(p.C[0], p.r[0], -_tangents(p.C)[0]) if p.ends[0] == "round" else None
    _cap(b, first[::-1], lift0, new)
    return b.build()


def _density(sdf, density):
    if density is not None:
        return float(density)
    res = _sdf._defaults["res"]
    return CAGE_PRESETS.get(res, CAGE_PRESETS["draft"]) / float((sdf.hi - sdf.lo).max())


def cage_from_sdf(sdf, density=None, root=None, support=0.5, margin=None, keep="largest", group=None):
    """All-quad cage from the named parts of sdf, for retopo_project. The root (largest sphere / ellipsoid / box,
    or the part named root=) becomes a quad ellipsoid / box lattice; each other part grows out of the part it
    overlaps most (breadth-first from the root) through a hole cut in that parent's faces where it leaves it:
    capsule / cone / sweep_sdf / limb_sdf parts as tubes along their paths with their radii, spheres and
    ellipsoids lathed along the direction they leave the parent; free ends get quad grid caps (domes on round
    ends). density = edges per unit length (default CAGE_PRESETS 'draft' 50 / 'final' 100 edges along the
    longest side, following the runner); rings sit 1/density apart with support loops (support x an edge
    either side) at limb and sweep joints, and segment counts change with quad-only transitions as parts taper.
    Holes cover the parent faces within margin (default k/2 + edge/4) of the child. One face group per part
    (to_mesh's names); group= adds one group of all faces. keep='largest' drops parts not connected to the root;
    'all' starts a new cage for each loose cluster. Raises ValueError for parts it can't map (raw SDFs,
    intersections, boxes that are not the root, tubes looping back into their parent)."""
    pieces = [p for name, s, path in _parts(sdf) for p in _pieces(name, s, path)]
    e = 1.0 / _density(sdf, density)
    gap = _connect(pieces, e)
    if root is not None:
        cand = [i for i, p in enumerate(pieces) if p.name == root]
        if not cand:
            raise KeyError(f"cage_from_sdf: no part {root!r}; parts: {' '.join(sorted({p.name for p in pieces}))}")
    else:
        cand = [i for i, p in enumerate(pieces) if p.kind != "tube"] or list(range(len(pieces)))
    r0 = max(cand, key=lambda i: pieces[i].volume)
    trees, placed = [], set()
    while True:
        parent, order = _tree(pieces, gap, r0, placed)
        trees.append((parent, order))
        placed |= set(order)
        rest = [i for i in range(len(pieces)) if i not in placed]
        if keep != "all" or not rest:
            break
        r0 = max(rest, key=lambda i: (pieces[i].kind != "tube", pieces[i].volume))
    mesh, label = Mesh(), np.zeros(0, dtype=np.int64)
    for parent, order in trees:
        rc = _root_cage(pieces[order[0]], e, support)
        n0 = len(mesh.verts)
        mesh = Mesh(np.vstack([mesh.verts, rc.verts]), list(mesh.faces) + [tuple(v + n0 for v in f) for f in rc.faces])
        label = np.r_[label, np.full(len(rc.faces), order[0])]
        for i in order[1:]:
            mesh, label = _attach_piece(mesh, label, pieces, i, parent, gap, e, support, margin)
    groups = {}
    for i, p in enumerate(pieces):
        f = np.flatnonzero(label == i)
        if len(f):
            groups.setdefault(p.name, set()).update(f.tolist())
    if group:
        if group in groups:
            raise ValueError(f"cage_from_sdf: group={group!r} is also a part name; pick another")
        groups[group] = range(len(mesh.faces))
    return Mesh(mesh.verts, mesh.faces, groups={n: ("f", f) for n, f in groups.items()})


def _attach_piece(mesh, label, pieces, i, parent, gap, e, support, margin):
    j = parent[i]
    while j is not None and not (label == j).any():  # a parent hidden inside its own parent
        j = parent[j]
    p, q = pieces[i], pieces[j]
    k = gap.get((i, parent[i]), (0, 0))[1]
    mg = 0.5 * k + 0.25 * e if margin is None else float(margin)
    if p.kind == "box":
        raise ValueError(f"cage_from_sdf: can't map box part {p.name!r} unless it is the root (root={p.name!r}); "
                         "use quad_remesh() / isotropic_remesh() on to_mesh() instead")
    if p.kind == "tube":
        if q.sdf.fn(p.C[-1:])[0] < q.sdf.fn(p.C[:1])[0]:
            p = _reversed(p)
        f = q.sdf.fn(p.C)
        out = np.flatnonzero(f >= 0)
        if not len(out):
            return mesh, label  # hidden inside its parent
        a = int(out[0])
        s0 = 0.0 if a == 0 else p.S[a - 1] + (p.S[a] - p.S[a - 1]) * f[a - 1] / (f[a - 1] - f[a])
        if (f[a:] < 0).any():
            raise ValueError(f"cage_from_sdf: part {p.name!r} leaves {q.name!r} and enters it again (a loop); "
                             "split it into two parts or use quad_remesh()")
        sib = [pieces[x] for x, y in parent.items() if y == parent[i] and x != i]
        clear = None
        if sib:  # siblings fused with this tube near its root (toes on one foot): rings start where it is free
            g = np.min([s.sdf.fn(p.C) for s in sib], axis=0)
            free = np.flatnonzero((g >= 0) & (p.S >= s0))
            clear = float(p.S[free[0]]) if len(free) else p.S[-1]
        st, cap = _tube_stations(p, s0, e, support, clear=clear)
        O = _interp(p.S, p.C, s0)
    else:
        got = _blob_stations(p, q, e)
        if got is None:
            return mesh, label
        st, cap, O = got
    mesh, label, _ = _grow(mesh, label, np.flatnonzero(label == j), p.sdf, mg, O, st, cap,
                           f"part {p.name!r}", e, i)
    return mesh, label


def attach_sweep(mesh, points, radius=0.1, sel=None, edge=None, end="round", path="smooth", resolution=8,
                 support=0.5, margin=None, group=None):
    """Grow a quad tube out of mesh along points (the first point inside or on the surface): cuts a hole where the
    path leaves the surface (only in faces of sel; default all), bridges it to the first ring with quads, spaces
    rings one edge apart (edge: default the mesh's mean edge length) with support loops (support x an edge) at the
    inner points, changes the segment count with quad-only transitions as the radius (scalar or one per point)
    tapers, and closes the far end: 'round' (dome of quads), 'flat' or 'open'. path 'smooth' (Catmull-Rom) or
    'joints' (straight segments). The hole covers faces within margin (default edge/2) of the tube.
    Created: verts, faces, hole (the cut loop), rings (vert loops, hole first), end (last ring)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.asarray(radius, dtype=float)
    r = np.full(len(pts), float(r)) if r.ndim == 0 else r.reshape(-1)
    if len(pts) < 2 or len(r) != len(pts):
        raise ValueError("attach_sweep: need >= 2 points and one radius or one per point")
    if end not in ("round", "flat", "open"):
        raise ValueError("attach_sweep: end must be 'round', 'flat' or 'open'")
    if path == "smooth":
        C, u = catmull_rom(pts, resolution)
        rr = np.clip(_cr_scalar(r, u), r.min(), r.max())
        jidx = [int(np.argmin(np.abs(u - j))) for j in range(1, len(pts) - 1)]
    elif path == "joints":
        t = np.linspace(0, 1, resolution + 1)[1:]
        C = np.vstack([pts[:1]] + [pts[i] + (pts[i + 1] - pts[i]) * t[:, None] for i in range(len(pts) - 1)])
        rr = np.r_[r[:1], np.concatenate([r[i] + (r[i + 1] - r[i]) * t for i in range(len(pts) - 1)])]
        jidx = [resolution * i for i in range(1, len(pts) - 1)]
    else:
        raise ValueError("path must be 'smooth' or 'joints'")
    S = _arc(C)
    A, B, R1, R2 = C[:-1], C[1:], rr[:-1], rr[1:]
    tube = SDF(lambda Q: _chain(Q, A, B, R1, R2), (C - rr[:, None]).min(0), (C + rr[:, None]).max(0))
    p = _Piece(None, "tube", tube, None, C=C, r=rr, S=S, joints=[float(S[j]) for j in jidx],
               ends=("round", "flat" if end == "open" else end))
    if edge is None:
        E = np.array(mesh.edges)
        edge = float(np.linalg.norm(mesh.verts[E[:, 0]] - mesh.verts[E[:, 1]], axis=1).mean())
    from .spatial import winding_number
    f = 0.5 - winding_number(mesh, C)
    a = int(np.argmax(f >= 0)) if (f >= 0).any() else len(C) - 1
    s0 = 0.0 if a == 0 else S[a - 1] + (S[a] - S[a - 1]) * f[a - 1] / (f[a - 1] - f[a])
    st, cap = _tube_stations(p, s0, edge, support)
    if end == "open":
        cap = ("open", None)
    allowed = np.arange(len(mesh.faces)) if sel is None else np.array(sorted(resolve(mesh, sel, "f")))
    out, _, info = _grow(mesh, None, allowed, tube, 0.5 * edge if margin is None else margin,
                         _interp(S, C, s0), st, cap, "the sweep", edge, 0)
    if group:
        b = Builder(out)
        b.add_to_group(group, "f", info["faces"])
        out = b.build()
    rings = [Sel("v", rg, closed=True) for rg in info["rings"]]
    return out, Created(verts=info["verts"], faces=info["faces"], hole=rings[0], rings=rings, end=rings[-1])


# ---- projection ----------------------------------------------------------------------------
def _neighbors(mesh):
    """Uniform-Laplacian neighbors as (rows, cols): boundary verts only see boundary neighbors."""
    ef = mesh.edge_faces
    bverts = {v for e, fs in ef.items() if len(fs) == 1 for v in e}
    rows, cols = [], []
    for (a, b), fs in ef.items():
        both = a in bverts and b in bverts
        if a not in bverts or both and len(fs) == 1:
            rows.append(a)
            cols.append(b)
        if b not in bverts or both and len(fs) == 1:
            rows.append(b)
            cols.append(a)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def _vert_normals(V, T):
    """Area-weighted vertex normals from triangles T."""
    N = np.zeros_like(V)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], n)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    return np.divide(N, L, out=np.zeros_like(N), where=L > 0)


def _taubin(V, mesh, nbr, lam, steps):
    rows, cols = nbr
    T = mesh.triangles[0]
    n = len(V)
    deg = np.bincount(rows, minlength=n).astype(float)
    move = deg > 0
    mu = 1 / (0.1 - 1 / lam) if lam > 0 else 0.0
    for _ in range(steps):
        for w in (lam, mu):
            S = np.zeros_like(V)
            np.add.at(S, rows, V[cols])
            D = np.zeros_like(V)
            D[move] = S[move] / deg[move, None] - V[move]
            N = _vert_normals(V, T)
            D -= (D * N).sum(1, keepdims=True) * N  # tangential only
            V = V + w * D
    return V


def _flipped(V, mesh, sdf):
    """Faces whose normal points against the SDF gradient at their center."""
    if not len(mesh.faces):
        return np.zeros(0, dtype=np.int64)
    T, src = mesh.triangles
    N = np.zeros((len(mesh.faces), 3))
    np.add.at(N, src, np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]))
    cs = mesh.corner_start
    ctr = np.add.reduceat(V[mesh.corner_verts], cs[:-1], axis=0) / np.diff(cs)[:, None]
    return np.flatnonzero((N * sdf.normal(ctr)).sum(1) < 0)


def _untangle(V, mesh, sdf, rounds):
    """Smooth the neighbourhood (one ring) of faces folded over by projection and re-project; rounds that do not
    reduce the folds are undone."""
    nb = mesh.vert_neighbors
    bad = _flipped(V, mesh, sdf)
    for _ in range(rounds):
        if not len(bad):
            break
        vs = {v for f in bad for v in mesh.faces[f]}
        vs = np.array(sorted(vs | {u for v in vs for u in nb[v]}))
        W = V.copy()
        for _ in range(3):
            avg = np.array([W[list(nb[v])].mean(0) for v in vs])
            W[vs] += 0.5 * (avg - W[vs])
        W[vs] = project_to_sdf(sdf, W[vs]).points
        now = _flipped(W, mesh, sdf)
        if len(now) >= len(bad):
            break
        V, bad = W, now
    return V


def retopo_project(cage, sdf, iterations=3, relax=0.5, steps=4, untangle=8):
    """Snap a cage onto the SDF's zero surface: project the verts (Newton steps along the gradient), relax them
    tangentially (steps Taubin passes, lambda = relax) and re-project, `iterations` times; then up to `untangle`
    rounds smooth and re-project around faces that projection folded over (overlapping parts). Topology, groups
    and UVs are kept. Created: verts (all), resid (|distance| per vert after), max_dist, flipped (faces still
    facing against the SDF gradient)."""
    V = np.array(cage.verts, dtype=float)
    nbr = _neighbors(cage)
    for _ in range(int(iterations)):
        V = project_to_sdf(sdf, V).points
        if relax > 0 and steps > 0:
            V = _taubin(V, cage, nbr, float(relax), int(steps))
    V = project_to_sdf(sdf, V).points
    V = _untangle(V, cage, sdf, int(untangle))
    pr = project_to_sdf(sdf, V)
    out = Mesh(pr.points, cage.faces, cage.loose_edges, {k: (g.kind, g.idx) for k, g in cage.groups.items()},
               cage.uvs)
    return out, Created(verts=range(len(V)), resid=pr.resid, max_dist=float(pr.resid.max()) if len(V) else 0.0,
                        flipped=_flipped(pr.points, out, sdf).tolist())


# ---- report -----------------------------------------------------------------------------------
def _valence(mesh):
    deg = np.zeros(len(mesh.verts), dtype=np.int64)
    for a, b in mesh.edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def _part_groups(mesh):
    """Face groups that look like parts (containers such as an all-faces 'skin' group are skipped: a group mostly
    covered by smaller groups), and a part index per face (-1 = none)."""
    names, lab = [], np.full(len(mesh.faces), -1, dtype=np.int64)
    for n in sorted((n for n, g in mesh.groups.items() if g.kind == "f" and g.idx), key=lambda n: len(mesh.groups[n].idx)):
        idx = np.array(sorted(mesh.groups[n].idx))
        if (lab[idx] >= 0).mean() >= 0.5:
            continue
        lab[idx[lab[idx] < 0]] = len(names)
        names.append(n)
    return names, lab


def _loops(mesh, J, t, w, verts, start=None):
    """Edge loops within w of a joint at J (axis t): rings reached breadth-first from the start loop (a seam; then
    counted on the +t side only) or from the verts nearest the joint's plane (counted on both sides), while a ring's
    mean offset along t stays within w."""
    V, nb = mesh.verts, mesh.vert_neighbors
    ok = set(verts.tolist())
    two = start is None
    if two:
        s = (V[verts] - J) @ t
        lat = np.linalg.norm(V[verts] - J - s[:, None] * t, axis=1)
        nbrs = [(v, u) for v in verts[:2000] for u in nb[v] if u in ok]
        h = float(np.median([np.linalg.norm(V[v] - V[u]) for v, u in nbrs])) if nbrs else w
        start = verts[(np.abs(s) < 0.5 * h) & (lat < 3 * w)]
    level, seen = set(int(v) for v in start), set(int(v) for v in start)
    if not level:
        return 0
    count, sides = 1, {1: True, -1: two}
    while level and any(sides.values()):
        level = {u for v in level for u in nb[v] if u in ok and u not in seen}
        seen |= level
        if not level:
            break
        off = (V[sorted(level)] - J) @ t
        for sg in (1, -1):
            part = off[off * sg > 0]
            if sides[sg] and len(part) and abs(part.mean()) <= w:
                count += 1
            elif sides[sg] and (len(part) or not two):
                sides[sg] = False
    return count


def _joints(mesh, sdf, names, lab):
    """[(label, J, t, w, part verts, start verts or None)] joints to count loops at: seams between part groups
    (child side), plus inner joints of limb_sdf / sweep_sdf parts when sdf is given."""
    out = []
    ef, F = mesh.edge_faces, mesh.faces
    seams = {}
    for (a, b), fs in ef.items():
        if len(fs) == 2 and lab[fs[0]] != lab[fs[1]] and min(lab[fs[0]], lab[fs[1]]) >= 0:
            seams.setdefault(tuple(sorted((lab[fs[0]], lab[fs[1]]))), set()).update((a, b))
    area = np.bincount(lab[lab >= 0], weights=mesh.face_areas[lab >= 0], minlength=len(names))

    def verts_of(g):
        fs = np.flatnonzero(lab == g)
        return np.unique(np.concatenate([F[f] for f in fs])) if len(fs) else np.zeros(0, dtype=np.int64)

    for (g1, g2), vs in sorted(seams.items()):
        if len(vs) < 4:
            continue
        child = g1 if area[g1] < area[g2] else g2
        P = mesh.verts[sorted(vs)]
        J = P.mean(0)
        t = np.linalg.svd(P - J)[2][2]
        cv = verts_of(child)
        if (mesh.verts[cv].mean(0) - J) @ t < 0:
            t = -t
        w = float(np.linalg.norm(P - J, axis=1).mean())
        out.append((f"{names[g1]}|{names[g2]}", J, t, w, cv, np.array(sorted(vs))))
    if sdf is not None:
        for name, s, path in _parts(sdf):
            if name not in names:
                continue
            try:
                ps = _pieces(name, s, path)
            except ValueError:
                continue
            pv = verts_of(names.index(name))
            for p in ps:
                if p.kind != "tube" or not len(pv):
                    continue
                T = _tangents(p.C)
                for n, sj in enumerate(p.joints):
                    J, t = _interp(p.S, p.C, sj), unit(_interp(p.S, T, sj))
                    out.append((f"{name}@{n + 1}", J, t, float(np.interp(sj, p.S, p.r)) * p.knuckle, pv, None))
    return out


def topo_report(mesh, sdf=None):
    """Quad-topology summary (Report, under 10 lines): face types and quad ratio, valence histogram, poles (verts
    off valence 4, or 6 on triangle meshes) and how many sit at joints (touching two part groups, or near a limb /
    sweep joint of sdf), edge-length uniformity (cv = std / mean) overall and per part group, and edge loops
    crossed within one radius of each joint (seams between part groups; inner limb / sweep joints with sdf); with
    sdf also the largest vert distance from its surface and faces folded against its gradient.
    stats: quads, tris, ngons, quad_ratio, valence {n: count}, poles [dict(vert, valence, pos, joint)],
    edge_cv {group: cv}, loops {joint: count} (+ max_dist, folded with sdf)."""
    F = mesh.faces
    if not len(F):
        return Report("topo", True, [], {}, "empty mesh")
    sizes = np.fromiter((len(f) for f in F), dtype=np.int64, count=len(F))
    nq, nt, nn = int((sizes == 4).sum()), int((sizes == 3).sum()), int((sizes > 4).sum())
    ratio = nq / len(F)
    regular = 4 if ratio >= 0.5 else 6
    deg = _valence(mesh)
    ef = mesh.edge_faces
    bverts = {v for e, fs in ef.items() if len(fs) == 1 for v in e}
    used = np.zeros(len(mesh.verts), dtype=bool)
    used[mesh.corner_verts] = True
    inner = np.flatnonzero(used & ~np.isin(np.arange(len(deg)), list(bverts)))
    hist = {}
    for v in deg[inner]:
        k = int(min(v, regular + 2))
        hist[k] = hist.get(k, 0) + 1
    names, lab = _part_groups(mesh)
    vlab = [set() for _ in range(len(mesh.verts))]
    for f, g in zip(F, lab):
        if g >= 0:
            for v in f:
                vlab[v].add(int(g))
    at_joint = np.array([len(s) > 1 for s in vlab])
    nbrs = mesh.vert_neighbors
    ring1 = at_joint.copy()
    for v in np.flatnonzero(at_joint):
        ring1[list(nbrs[v])] = True
    joints = _joints(mesh, sdf, names, lab)
    for name, J, t, w, _, _ in joints:
        if "@" in name:
            ring1 |= np.linalg.norm(mesh.verts - J, axis=1) < 1.5 * w
    poles = [dict(vert=int(v), valence=int(deg[v]), pos=mesh.verts[v].tolist(), joint=bool(ring1[v]))
             for v in inner if deg[v] != regular]
    E = np.array(mesh.edges)
    elen = np.linalg.norm(mesh.verts[E[:, 0]] - mesh.verts[E[:, 1]], axis=1)
    cv = {"all": float(elen.std() / elen.mean())}
    for i, n in enumerate(names):
        fs = np.flatnonzero(lab == i)
        ek = np.array(sorted({edge_key(a, b) for f in fs for a, b in _pairs(F[f])}))
        ln = np.linalg.norm(mesh.verts[ek[:, 0]] - mesh.verts[ek[:, 1]], axis=1)
        cv[n] = float(ln.std() / ln.mean())
    loops = {name: _loops(mesh, J, t, w, pv, st) for name, J, t, w, pv, st in joints}
    vh = " ".join(f"{k}{'+' if k == regular + 2 else ''}:{hist[k]}" for k in sorted(hist))
    pct = "100%" if nq == len(F) else "0%" if not nq else f"{min(ratio, 0.999):.1%}"
    head = (f"{len(F)} faces, quads {pct} (tris {nt}, ngons {nn}) | valence {vh}"
            + (f" | boundary verts {len(bverts)}" if bverts else ""))
    lines = []
    jp = [p for p in poles if p["joint"]]
    by = {}
    for p in jp:
        key = "|".join(names[g] for g in sorted(vlab[p["vert"]])) or "joint"
        by[key] = by.get(key, 0) + 1
    top = sorted(by.items(), key=lambda kv: -kv[1])[:4]
    lines.append(f"poles {len(poles)}: {len(jp)} at joints" + (" (" + " ".join(f"{k} {n}" for k, n in top)
                                                                + (" ..." if len(by) > 4 else "") + ")" if top else "")
                 + (f" e.g. {fvec(jp[0]['pos'])}" if jp else ""))
    worst = sorted(((n, c) for n, c in cv.items() if n != "all"), key=lambda kv: -kv[1])[:3]
    lines.append(f"edge cv (std/mean, < 0.35 even): all {cv['all']:.2f}"
                 + (" | worst " + ", ".join(f"{n} {c:.2f}{'!' if c > 0.35 else ''}" for n, c in worst)
                    if worst else ""))
    if loops:
        items = sorted(loops.items(), key=lambda kv: kv[1])
        low = [f"{n} {c}" for n, c in items if c < 3]
        lines.append(f"loops per joint (3+ to bend): min {items[0][1]} median {int(np.median([c for _, c in items]))}"
                     f" over {len(items)} joints" + (" | under 3: " + ", ".join(low[:5])
                                                     + (" ..." if len(low) > 5 else "") if low else ""))
    stats = dict(quads=nq, tris=nt, ngons=nn, quad_ratio=ratio, valence=hist, poles=poles, edge_cv=cv, loops=loops)
    if sdf is not None:
        dist = float(np.abs(sdf.eval(mesh.verts)).max())
        folded = _flipped(np.asarray(mesh.verts, dtype=float), mesh, sdf)
        by = {}
        for f in folded:
            key = names[lab[f]] if lab[f] >= 0 else "-"
            by[key] = by.get(key, 0) + 1
        top = sorted(by.items(), key=lambda kv: -kv[1])[:4]
        lines.append(f"on the sdf: max dist {fmt(dist)} | folded {len(folded)} faces"
                     + (" (" + " ".join(f"{k} {n}" for k, n in top) + (" ..." if len(by) > 4 else "") + ")"
                        if top else ""))
        stats.update(max_dist=dist, folded=folded.tolist())
    return Report("topo", True, [], stats, head, lines)
