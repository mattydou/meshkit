"""Signed distance fields (negative inside) for organic masses: oriented primitives, swept tubes and jointed limbs,
whole-SDF transforms, CSG and smooth blends (per-part k), field displacement, direct queries (eval, normal,
project_to_sdf), a parts-proximity report, and to_mesh(): sparse marching cubes (needs scikit-image) giving one closed,
outward shell whose faces are grouped by part name.
Each SDF carries bounds (lo, hi) of its surface, a Lipschitz estimate (lip) so to_mesh can skip empty space, and
its build tree (kind, kids, name) for proximity()."""
import time
import warnings
from collections import namedtuple

import numpy as np

from .mesh import Mesh, vec
from .xform import rotation_matrix

PRESETS = {"draft": 64, "final": 180}  # to_mesh cells along the longest side: iterate in draft, export in final
_defaults = {"res": "draft"}  # the runner's --final switches this for one run
_log = []  # one dict per to_mesh call (seconds, res, cells, evals, faces); the runner prints and clears it

Projected = namedtuple("Projected", "points normals dist resid")


def _pts(p):
    P = np.asarray(p, dtype=float)
    return np.full((1, 3), float(P)) if P.ndim == 0 else P.reshape(-1, 3)


def _norm(X):
    return np.sqrt((X * X).sum(-1))


class SDF:
    """Callable f(P (k,3) or (3,)) -> distances (negative inside) with .lo/.hi bounds of the zero surface.
    .lip bounds |gradient| near the surface (1 = exact distance), .lbf says values are >= lbf x the distance to
    the bounds box (lets unions skip far parts), and .kind/.kids/.name/.k record the build tree."""

    def __init__(self, fn, lo, hi, lip=1.0, kind="sdf", kids=(), name=None, lbf=1.0, k=None, params=None):
        self.fn, self.lo, self.hi = fn, vec(lo), vec(hi)
        self.lip, self.lbf = float(lip), float(lbf)
        self.kind, self.kids, self.name, self.k = kind, tuple(kids), name, k
        self.params = params  # primitives: their defining numbers (cage_from_sdf builds topology from them)

    def __call__(self, P):
        P = np.asarray(P, dtype=float)
        out = self.fn(P.reshape(-1, 3))
        return out if P.ndim > 1 else float(out[0])

    def eval(self, points):
        """Signed distances at points ((k,3) or (3,)) -> (k,) array."""
        return self.fn(_pts(points))

    def normal(self, points):
        """Outward unit normals (normalized central-difference gradient) at points -> (k,3)."""
        g = _grad(self, _pts(points))
        L = np.linalg.norm(g, axis=1, keepdims=True)
        return np.divide(g, L, out=np.zeros_like(g), where=L > 0)

    def __repr__(self):
        nm = f"{self.name!r} " if self.name else ""
        return (f"SDF({nm}{self.kind} bounds={np.round(self.lo, 4).tolist()}..{np.round(self.hi, 4).tolist()} "
                f"lip={self.lip:.3g})")


def _grad(sdf, P, eps=None):
    eps = 1e-5 * max(float(np.linalg.norm(sdf.hi - sdf.lo)), 1e-9) if eps is None else eps
    g = np.empty_like(P)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        g[:, i] = (sdf.fn(P + d) - sdf.fn(P - d)) / (2 * eps)
    return g


def _rot(rotation):
    """3x3 matrix from None | (axis, degrees) | [(axis, degrees), ...] applied in order | (rx, ry, rz) Euler degrees
    (X, then Y, then Z, like Blender's XYZ) | a 3x3 matrix."""
    if rotation is None:
        return None
    r = rotation
    if (isinstance(r, (tuple, list)) and len(r) == 2 and np.ndim(r[1]) == 0
            and (isinstance(r[0], str) or np.size(r[0]) == 3)):
        return rotation_matrix(r[0], r[1])
    try:
        M = np.asarray(r, dtype=float)
    except (TypeError, ValueError):
        M = None
    if M is not None and M.shape == (3, 3):
        return M
    if M is not None and M.shape == (3,):
        return rotation_matrix("z", M[2]) @ rotation_matrix("y", M[1]) @ rotation_matrix("x", M[0])
    R = np.eye(3)
    for axis, deg in r:
        R = rotation_matrix(axis, deg) @ R
    return R


def _local(P, c, R):
    """World points -> the primitive's frame (rotation R about c)."""
    return P - c if R is None else (P - c) @ R


# ---- primitives -----------------------------------------------------------------
def sphere(radius=1.0, center=0, name=None):
    c = vec(center)
    return SDF(lambda P: _norm(P - c) - radius, c - radius, c + radius, kind="sphere", name=name,
               params=dict(center=c, radius=float(radius)))


def ellipsoid(radii=(1, 1, 1), center=0, rotation=None, name=None):
    """Ellipsoid with semi-axes radii, turned by rotation about its center (approximate distance, exact surface)."""
    r, c, R = vec(radii), vec(center), _rot(rotation)
    if r.min() <= 0:
        raise ValueError("ellipsoid radii must be > 0")

    def f(P):
        Q = _local(P, c, R)
        k0, k1 = _norm(Q / r), _norm(Q / (r * r))
        return np.where(k1 > 1e-12, k0 * (k0 - 1) / np.maximum(k1, 1e-12), -r.min())
    ext = r if R is None else np.sqrt(((R * r) ** 2).sum(1))
    ratio = r.max() / r.min()  # the approximation's gradient and underestimate grow with elongation (measured)
    return SDF(f, c - ext, c + ext, lip=1 + 0.35 * (ratio - 1), lbf=1 / ratio, kind="ellipsoid", name=name,
               params=dict(center=c, radii=r, rotation=np.eye(3) if R is None else R))


def _round_cones(P, A, B, R1, R2):
    """Exact distance from P (k,3) to round cones (hulls of spheres (A, R1) and (B, R2)), A/B (S,3) -> (k,S)."""
    BA = B - A
    l2 = np.maximum((BA * BA).sum(1), 1e-30)
    rr = R1 - R2
    a2 = l2 - rr * rr
    PA = P[:, None, :] - A[None]
    y = (PA * BA).sum(-1)
    z = y - l2
    X = PA * l2[:, None] - BA * y[..., None]
    x2 = (X * X).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        k = np.sign(rr) * rr * rr * x2
        d_b = np.sqrt(x2 + z * z * l2) / l2 - R2
        d_a = np.sqrt(x2 + y * y * l2) / l2 - R1
        d_s = (np.sqrt(np.maximum(x2 * a2 / l2, 0)) + y * rr) / l2 - R1
        out = np.where(np.sign(z) * a2 * z * z * l2 > k, d_b, np.where(np.sign(y) * a2 * y * y * l2 < k, d_a, d_s))
    inner = a2 <= 0  # one sphere contains the other
    if inner.any():
        out[:, inner] = np.minimum(_norm(PA) - R1, _norm(P[:, None, :] - B[None]) - R2)[:, inner]
    return out


def _chain(P, A, B, R1, R2):
    """Min over round-cone segments, chunked to bound memory."""
    out = np.empty(len(P))
    r = max(1, int(2e5 // max(len(A), 1)))
    for i in range(0, len(P), r):
        out[i:i + r] = _round_cones(P[i:i + r], A, B, R1, R2).min(axis=1)
    return out


def capsule(a, b, radius=0.25, radius_b=None, rotation=None, name=None):
    """Rounded segment a -> b; radius_b tapers towards b (a round cone; exact distance). rotation turns b about a."""
    a, b = vec(a), vec(b)
    if rotation is not None:
        b = a + _rot(rotation) @ (b - a)
    ra = float(radius)
    rb = ra if radius_b is None else float(radius_b)
    A, B, R1, R2 = a[None], b[None], np.array([ra]), np.array([rb])
    return SDF(lambda P: _round_cones(P, A, B, R1, R2)[:, 0], np.minimum(a - ra, b - rb),
               np.maximum(a + ra, b + rb), kind="capsule", name=name, params=dict(points=np.array([a, b]),
                                                                                  radii=np.array([ra, rb])))


def box(size=1.0, center=0, round=0.0, rotation=None, name=None):
    """Box of full size (scalar or xyz), edges rounded by `round`, turned by rotation about its center."""
    half, c, R = vec(size) / 2, vec(center), _rot(rotation)

    def f(P):
        q = np.abs(_local(P, c, R)) - (half - round)
        return _norm(np.maximum(q, 0)) + np.minimum(q.max(axis=1), 0) - round
    ext = half if R is None else np.abs(R) @ half
    return SDF(f, c - ext, c + ext, kind="box", name=name,
               params=dict(center=c, size=2 * half, rotation=np.eye(3) if R is None else R, round=float(round)))


def cone(a, b, radius_a=0.5, radius_b=0.0, rotation=None, name=None):
    """Capped cone from a (radius_a) to b (radius_b); exact distance. rotation turns b about a."""
    a, b = vec(a), vec(b)
    if rotation is not None:
        b = a + _rot(rotation) @ (b - a)
    ra, rb = float(radius_a), float(radius_b)
    ba = b - a
    baba = float(ba @ ba)

    def f(P):
        pa = P - a
        papa = (pa * pa).sum(1)
        paba = pa @ ba / baba
        x = np.sqrt(np.maximum(papa - paba * paba * baba, 0))
        cax = np.maximum(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = (rb - ra) ** 2 + baba
        t = np.clip(((rb - ra) * (x - ra) + paba * baba) / k, 0, 1)
        cbx, cby = x - ra - t * (rb - ra), paba - t
        s = np.where((cbx < 0) & (cay < 0), -1.0, 1.0)
        return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))
    R = max(ra, rb)
    return SDF(f, np.minimum(a, b) - R, np.maximum(a, b) + R, kind="cone", name=name,
               params=dict(points=np.array([a, b]), radii=np.array([ra, rb])))


def _cr_scalar(v, u):
    """Uniform Catmull-Rom interpolation of per-point values v at fractional indices u."""
    n = len(v)
    if n == 1:
        return np.full(len(u), v[0])
    ext = np.concatenate([[2 * v[0] - v[1]], v, [2 * v[-1] - v[-2]]])
    i = np.clip(np.floor(u).astype(int), 0, n - 2)
    s = u - i
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    return 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                  + (3 * p1 - p0 - 3 * p2 + p3) * s ** 3)


def sweep_sdf(points, radii=0.1, resolution=8, name=None):
    """Tube along a centripetal Catmull-Rom curve through points with a per-point radius (scalar or one per point,
    interpolated smoothly): a chain of exact round cones, so limbs curve between joints. Exact distance."""
    from .sweep import catmull_rom
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float)
    r = np.full(len(P), float(r)) if r.ndim == 0 else r.reshape(-1)
    if len(r) != len(P):
        raise ValueError(f"sweep_sdf: {len(r)} radii for {len(P)} points")
    if r.min() < 0:
        raise ValueError("sweep_sdf: radii must be >= 0")
    C, u = catmull_rom(P, resolution)
    rs = np.clip(_cr_scalar(r, u), r.min(), r.max())
    A, B, R1, R2 = C[:-1], C[1:], rs[:-1], rs[1:]
    return SDF(lambda Q: _chain(Q, A, B, R1, R2), (C - rs[:, None]).min(0), (C + rs[:, None]).max(0),
               kind="sweep", name=name, params=dict(points=P, radii=r, curve=C, curve_radii=rs, u=u))


def limb_sdf(points, radii=0.1, knuckle=1.1, k=None, name=None):
    """Jointed limb: straight round-cone segments between joints (one radius per joint, or one for all), with a
    knuckle bulge at each inner joint (sphere of knuckle x the joint radius) blended in with k (default 0.3 x the
    mean radius). For insect legs and fingers; sweep_sdf is the smooth-curve version."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(P) < 2:
        raise ValueError("limb_sdf needs >= 2 points")
    r = np.asarray(radii, dtype=float)
    r = np.full(len(P), float(r)) if r.ndim == 0 else r.reshape(-1)
    if len(r) != len(P):
        raise ValueError(f"limb_sdf: {len(r)} radii for {len(P)} points")
    if r.min() <= 0:
        raise ValueError("limb_sdf: radii must be > 0")
    k = 0.3 * float(r.mean()) if k is None else float(k)
    A, B, R1, R2 = P[:-1], P[1:], r[:-1], r[1:]
    J, RJ = P[1:-1], r[1:-1] * knuckle
    bulge = len(J) and knuckle != 1

    def f(Q):
        d = _chain(Q, A, B, R1, R2)
        if bulge:
            for j in range(len(J)):
                d = _smin(d, _norm(Q - J[j]) - RJ[j], k)
        return d
    rr = np.maximum(r, np.r_[0, RJ, 0]) if bulge else r
    pad = max(k, 0) / 4  # the smooth min lowers values by at most k/4
    return SDF(f, (P - rr[:, None]).min(0) - pad, (P + rr[:, None]).max(0) + pad, kind="limb", name=name,
               params=dict(points=P, radii=r, knuckle=float(knuckle) if bulge else 1.0))


def _transformed(s, R, pivot, offset, scale, name=None, kind="transform", kids=None):
    """s moved: world = R (local - pivot) scale + pivot + offset."""
    Rt = np.eye(3) if R is None else R

    def f(P):
        return s.fn(((P - pivot - offset) @ Rt) / scale + pivot) * scale
    C = np.array([[(s.lo, s.hi)[i][0], (s.lo, s.hi)[j][1], (s.lo, s.hi)[l][2]] for i in (0, 1) for j in (0, 1)
                  for l in (0, 1)])
    W = (C - pivot) * scale @ Rt.T + pivot + offset
    out = SDF(f, W.min(0), W.max(0), lip=s.lip, lbf=s.lbf, kind=kind, kids=(s,) if kids is None else kids,
              name=name, k=s.k)
    out.xf, out.src = (R, pivot, offset, scale), s
    return out


def sdf_transform(s, rotation=None, offset=0, pivot=None, scale=1.0, name=None):
    """Move or pose a whole SDF: rotate (same forms as primitives) and uniformly scale about pivot (default: the
    center of its bounds), then translate by offset. Parts inside keep their names for proximity and to_mesh."""
    if float(scale) <= 0:
        raise ValueError("sdf_transform: scale must be > 0")
    pivot = (s.lo + s.hi) / 2 if pivot is None else vec(pivot)
    return _transformed(s, _rot(rotation), pivot, vec(offset), float(scale), name=name)


# ---- combination ----------------------------------------------------------------
def _lb(s, P):
    """Lower bound on s at P from its bounds box: lbf x distance outside the box, -inf inside it (s may be < 0)."""
    q = np.maximum(s.lo - P, P - s.hi)
    return np.where((q > 0).any(1), s.lbf * _norm(np.maximum(q, 0)), -np.inf)


def _smin(a, b, k):
    """Polynomial smooth min; exactly min(a, b) where |a - b| >= k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - 0.25 * k * h * h


def _fold(d, s, P, k):
    """Blend s into running distances d; s is evaluated only where its bounds say it can change the result."""
    idx = np.flatnonzero(_lb(s, P) < d + max(k, 0))
    if len(idx) == len(d):
        return _smin(d, s.fn(P), k)
    if len(idx):
        d[idx] = _smin(d[idx], s.fn(P[idx]), k)
    return d


def _fold_all(S, K, P):
    """Fold parts S (blend sizes K) in order over points P. With many parts and points, points are bucketed into
    blocks and a part is only tested on blocks its bounds can reach (d only decreases, so the first part's block
    maxima bound every later step); results are the same as folding everywhere."""
    d = S[0].fn(P).copy()
    if len(S) < 8 or len(P) < 4096:
        for s, k in zip(S[1:], K[1:]):
            d = _fold(d, s, P, k)
        return d
    lo = P.min(0)
    cs = max(float((P.max(0) - lo).max()) / 16, 1e-12)
    _, bucket = np.unique(np.floor((P - lo) / cs).astype(np.int64) @ np.array([1 << 40, 1 << 20, 1]),
                          return_inverse=True)
    bucket = bucket.reshape(-1)
    order = np.argsort(bucket, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(bucket[order]) != 0])
    bmin, bmax = np.minimum.reduceat(P[order], starts), np.maximum.reduceat(P[order], starts)
    dmax = np.maximum.reduceat(d[order], starts)
    LO, HI = np.array([s.lo for s in S[1:]]), np.array([s.hi for s in S[1:]])
    gap = np.maximum(np.maximum(LO[None] - bmax[:, None], bmin[:, None] - HI[None]), 0)
    gap = _norm(gap)  # (blocks, parts) box-to-box distance
    lbf, ks = np.array([s.lbf for s in S[1:]]), np.maximum(np.array(K[1:]), 0)
    rel = (gap <= 0) | (lbf[None] * gap < dmax[:, None] + ks[None])
    for j, (s, k) in enumerate(zip(S[1:], K[1:])):
        idx = np.flatnonzero(rel[bucket, j])
        if len(idx) == len(P):
            d = _fold(d, s, P, k)
        elif len(idx):
            d[idx] = _fold(d[idx], s, P[idx], k)
    return d


def _parts_arg(sdfs):
    sdfs = tuple(sdfs[0]) if len(sdfs) == 1 and isinstance(sdfs[0], (list, tuple)) else tuple(sdfs)
    if not sdfs:
        raise ValueError("need at least one SDF")
    return sdfs


def _span(sdfs, pad=0.0):
    return np.min([s.lo for s in sdfs], axis=0) - pad, np.max([s.hi for s in sdfs], axis=0) + pad


def union(*sdfs, name=None):
    sdfs = _parts_arg(sdfs)

    zeros = [0.0] * len(sdfs)
    return SDF(lambda P: _fold_all(sdfs, zeros, P), *_span(sdfs), lip=max(s.lip for s in sdfs),
               lbf=min(s.lbf for s in sdfs), kind="union",
               kids=sdfs, name=name)


def smooth_union(*sdfs, k=0.1, name=None):
    """Union with fillets of size ~k blending the pieces into one mass. k = one number, or one per part: parts
    blend in softest first, each with its own k, so a junction takes the sharper part's k."""
    sdfs = _parts_arg(sdfs)
    ks = np.asarray(k, dtype=float)
    ks = np.full(len(sdfs), float(ks)) if ks.ndim == 0 else ks.reshape(-1)
    if len(ks) != len(sdfs):
        raise ValueError(f"smooth_union: {len(ks)} k values for {len(sdfs)} parts")
    if ks.max() <= 0:
        return union(*sdfs, name=name)
    order = sorted(range(len(sdfs)), key=lambda i: -ks[i])
    S, K = [sdfs[i] for i in order], [float(ks[i]) for i in order]

    return SDF(lambda P: _fold_all(S, K, P), *_span(sdfs, float(ks.max())), lip=max(s.lip for s in sdfs), lbf=min(s.lbf for s in sdfs),
               kind="smooth_union", kids=sdfs, name=name, k=tuple(float(x) for x in ks))


def subtract(a, b, k=0.0, name=None):
    """a minus b; k > 0 rounds the cut edge."""
    def f(P):
        d = a.fn(P).copy()
        idx = np.flatnonzero(_lb(b, P) < max(k, 0) - d)  # elsewhere b cannot cut
        if len(idx):
            x = b.fn(P[idx])
            d[idx] = -_smin(-d[idx], x, k) if k > 0 else np.maximum(d[idx], -x)
        return d
    return SDF(f, a.lo, a.hi, lip=max(a.lip, b.lip), lbf=a.lbf, kind="subtract", kids=(a, b), name=name)


def intersect(a, b, name=None):
    return SDF(lambda P: np.maximum(a.fn(P), b.fn(P)), np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi),
               lip=max(a.lip, b.lip), lbf=0.0, kind="intersect", kids=(a, b), name=name)


def _estimate_lip(field, lo, hi):
    """Max |gradient| of a field over its box by central differences at fixed sample points, with headroom."""
    P = np.random.default_rng(0).uniform(lo, hi, (4096, 3))
    eps = 1e-4 * max(float(np.linalg.norm(hi - lo)), 1e-9)
    g = np.stack([(field(P + d) - field(P - d)) / (2 * eps) for d in np.eye(3) * eps], axis=1)
    return 1.5 * float(np.linalg.norm(g, axis=1).max())


def displace_sdf(sdf, field, amplitude=None, name=None):
    """Push the surface outward by field(P) (e.g. noise()); bounds grow by amplitude (default field.amplitude)."""
    amp = getattr(field, "amplitude", None) if amplitude is None else amplitude
    if amp is None:
        raise ValueError("displace_sdf: pass amplitude= (max |field|) for a field without .amplitude")
    lo, hi = sdf.lo - amp, sdf.hi + amp
    flip = getattr(field, "lip", None)
    flip = _estimate_lip(field, lo, hi) if flip is None else flip
    return SDF(lambda P: sdf.fn(P) - field(P), lo, hi, lip=sdf.lip + flip, lbf=sdf.lbf, kind="displace",
               kids=(sdf,), name=name)


# ---- direct queries -----------------------------------------------------------------
def project_to_sdf(sdf, points, iters=16, tol=None):
    """Snap points onto the zero surface by Newton steps along the gradient (no meshing).
    Returns Projected(points (k,3), normals (outward unit), dist (signed distance before), resid (|dist| after))."""
    P = _pts(points).copy()
    d0 = sdf.fn(P)
    d = d0.copy()
    tol = 1e-7 * max(float(np.linalg.norm(sdf.hi - sdf.lo)), 1e-9) if tol is None else tol
    for _ in range(iters):
        act = np.flatnonzero(np.abs(d) > tol)
        if not len(act):
            break
        g = _grad(sdf, P[act])
        P[act] -= (d[act] / np.maximum((g * g).sum(1), 0.25))[:, None] * g  # no leaps where the gradient fades
        d[act] = sdf.fn(P[act])
    return Projected(P, sdf.normal(P), d0, np.abs(d))


# ---- parts proximity ------------------------------------------------------------------
def _parts(sdf):
    """Material parts: named SDFs, else leaf primitives, reached through unions, transforms, displacement and
    subtract's first operand. [(name, sdf, path)] with path = ((node, kid index), ...) from the root; parts under a
    transform come back moved."""
    found, counts = [], {}

    def walk(s, path, wrap):
        if s.name is None and s.kind in ("union", "smooth_union"):
            for i, c in enumerate(s.kids):
                walk(c, path + ((s, i),), wrap)
        elif s.name is None and s.kind in ("subtract", "displace"):
            walk(s.kids[0], path + ((s, 0),), wrap)
        elif s.name is None and s.kind == "transform":  # parts inside are evaluated where the transform puts them
            walk(s.kids[0], path + ((s, 0),), lambda x, t=s: wrap(_transformed(x, *t.xf, name=x.name, kind=x.kind,
                                                                                kids=x.kids)))
        else:
            base = s.name or s.kind
            counts[base] = counts.get(base, 0) + 1
            found.append([base, wrap(s), path])
    walk(sdf, (), lambda x: x)
    seen = {}
    for p in found:  # repeated names get .0 .1 ...; unnamed primitives are numbered by kind
        base = p[0]
        if counts[base] > 1 or p[1].name is None:
            seen[base] = seen.get(base, -1) + 1
            p[0] = f"{base}{'.' if p[1].name else ''}{seen[base]}"
    return [tuple(p) for p in found]


def _pair_k(pa, pb):
    """Blend k between two parts: at the node where their paths split (0 for a plain union)."""
    for (na, ia), (nb, ib) in zip(pa, pb):
        if na is not nb:
            break
        if ia != ib:
            return min(na.k[ia], na.k[ib]) if na.kind == "smooth_union" else 0.0
    return 0.0


def _min_sum(a, b, lo, hi, n=12, rounds=5, keep=4):
    """(min, argmin) over P in [lo, hi] of a(P) + b(P): a grid, then repeated zooms around the best samples."""
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g = (hi - lo) / (n - 1)
    v = a.fn(P) + b.fn(P)
    j = int(np.argmin(v))
    best, at = float(v[j]), P[j]
    off = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3, indexing="ij"), -1).reshape(-1, 3)
    for _ in range(rounds):
        top = P[np.argsort(v, kind="stable")[:keep]]
        P = (top[:, None, :] + off[None] * g).reshape(-1, 3)
        g = g / 2
        v = a.fn(P) + b.fn(P)
        j = int(np.argmin(v))
        if v[j] < best:
            best, at = float(v[j]), P[j]
    return best, at


def proximity(sdf, threshold=None, allow=()):
    """Gap between each pair of parts (named SDFs, else primitives): min over P of a(P) + b(P), the surface gap for
    exact distances (<= 0: overlapping = joined). Fails on 'near' pairs (0 < gap < threshold) whose gap lies in
    empty space, where fusing would add an unplanned bridge or handle; near pairs whose closest point is already
    within another part ('covered', e.g. toes on one pad) and allow=[(a, b), ...] pairs are fine. threshold
    defaults per pair to max(blend k / 2, one 'final' voxel): smooth unions bridge gaps under k/2, meshing those
    under a voxel. Report; stats['pairs'] = [dict(a, b, gap, k, threshold, state, at)] with state joined | near |
    covered | allowed | apart."""
    from .report import Report, fmt, fvec
    parts = _parts(sdf)
    voxel = float((sdf.hi - sdf.lo).max()) / PRESETS["final"]
    allowed = {frozenset(p) for p in allow}
    names = {p[0] for p in parts}
    for p in allowed:
        for n in p:
            if n not in names:
                raise KeyError(f"proximity: no part {n!r}; parts: {' '.join(sorted(names))}")
    recs = []
    for i, (na, a, pa) in enumerate(parts):
        for nb, b, pb in parts[i + 1:]:
            k = _pair_k(pa, pb)
            thr = max(k / 2, voxel) if threshold is None else float(threshold)
            lo, hi = np.maximum(a.lo, b.lo) - thr, np.minimum(a.hi, b.hi) + thr
            gap, at = _min_sum(a, b, lo, hi) if (lo <= hi).all() else (float(np.inf), None)
            state = "joined" if gap <= 0 else "apart" if gap >= thr else (
                "allowed" if frozenset((na, nb)) in allowed else "near")
            if state == "near" and min(float(c.fn(at[None])[0]) for nc, c, _ in parts if nc not in (na, nb)) < 0:
                state = "covered"
            recs.append(dict(a=na, b=nb, gap=gap, k=k, threshold=thr, state=state, at=at))
    count = {s: sum(r["state"] == s for r in recs) for s in ("joined", "near", "covered", "allowed", "apart")}
    near = sorted((r for r in recs if r["state"] == "near"), key=lambda r: r["gap"])
    fails = [f"near (may fuse; allow=[(a, b)] if meant to merge): {r['a']}-{r['b']} gap {fmt(r['gap'])} < "
             f"{fmt(r['threshold'])} at {fvec(r['at'])}" for r in near]
    lines = []
    joined = [f"{r['a']}-{r['b']}" for r in recs if r["state"] in ("joined", "allowed")]
    for j in range(0, len(joined), 8):
        lines.append(("joined: " if j == 0 else "  ") + " ".join(joined[j:j + 8]))
    linked = {n for r in recs if r["state"] in ("joined", "allowed") for n in (r["a"], r["b"])}
    loose = [p[0] for p in parts if p[0] not in linked]
    if loose and len(parts) > 1:
        lines.append("loose (joined to nothing; separate pieces unless blended): " + " ".join(loose))
    thr_txt = fmt(threshold) if threshold is not None else f"max(k/2, {fmt(voxel)})"
    head = (f"{len(parts)} parts, {len(recs)} pairs | joined {count['joined']} near {count['near']} covered "
            f"{count['covered']} allowed {count['allowed']} apart {count['apart']} | threshold {thr_txt}")
    return Report("proximity", not fails, fails, {"pairs": recs, "parts": [p[0] for p in parts]}, head, lines)


# ---- meshing ----------------------------------------------------------------------
_OFF8 = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_OFF27 = np.array([(i, j, k) for i in (0, 1, 2) for j in (0, 1, 2) for k in (0, 1, 2)])


def _cells(res):
    res = _defaults["res"] if res is None else res
    if isinstance(res, str):
        if res not in PRESETS:
            raise ValueError(f"res must be one of {sorted(PRESETS)} or a number of cells, got {res!r}")
        return res, PRESETS[res]
    if int(res) < 4:
        raise ValueError("to_mesh: need >= 4 cells along the longest side")
    return str(int(res)), int(res)


def _sample(sdf, lo, hi, cells, top=8, dense=False):
    """Sparse sampling on a grid of `cells` along the longest side: coarse blocks first, refined only where the
    surface may pass (|d| <= lip x half-diagonal). Returns (volume, kept-cell corner mask, origin, h, evals).
    dense=True evaluates every grid point instead (mask None)."""
    h = float((hi - lo).max()) / cells
    n = np.ceil((hi - lo) / h).astype(int) + 4  # 2 cells of padding per side
    N = -(-n // top) * top
    origin = (lo + hi) / 2 - h * N / 2  # centered, so symmetric shapes mesh symmetrically
    if dense:
        idx = np.indices(tuple(N + 1)).reshape(3, -1).T
        vol = np.concatenate([sdf.fn(origin + h * idx[i:i + 200000]) for i in range(0, len(idx), 200000)])
        return vol.reshape(tuple(N + 1)), None, origin, h, len(idx)
    vol = np.full(tuple(N + 1), np.nan)
    evals = 0

    def evaluate(flat):
        nonlocal evals
        if len(flat):
            idx = np.stack(np.unravel_index(flat, vol.shape), 1)
            vol.flat[flat] = sdf.fn(origin + h * idx)
            evals += len(flat)

    S = top
    cell = np.indices(tuple(N // S)).reshape(3, -1).T
    evaluate(np.ravel_multi_index(tuple((np.indices(tuple(N // S + 1)).reshape(3, -1) * S)), vol.shape))
    sign = np.zeros(tuple(N // S), dtype=np.int8)  # per cell at the current level: +1/-1 skipped, 0 kept
    lip = max(sdf.lip, 1.0)
    while True:
        cor = (cell[:, None, :] + _OFF8) * S
        V = vol[cor[..., 0], cor[..., 1], cor[..., 2]]
        pos, neg = (V > 0).all(1), (V < 0).all(1)
        skip = (pos | neg) & (np.abs(V).min(1) > lip * 0.8661 * S * h)
        sign[tuple(cell[skip].T)] = np.where(pos[skip], 1, -1)
        kept = cell[~skip]
        if S == 1:
            break
        S //= 2
        sign = sign.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        flat = np.unique(np.ravel_multi_index(tuple(((kept[:, None, :] * 2 + _OFF27) * S).reshape(-1, 3).T),
                                              vol.shape))
        evaluate(flat[np.isnan(vol.flat[flat])])
        cell = (kept[:, None, :] * 2 + _OFF8).reshape(-1, 3)
    fill = np.pad(sign, ((0, 1), (0, 1), (0, 1)), mode="edge").astype(float)  # skipped cells: one sign throughout
    fill *= h
    np.copyto(vol, fill, where=np.isnan(vol))
    mask = np.zeros(vol.shape, dtype=bool)
    for o in _OFF8:
        c = kept + o
        mask[c[:, 0], c[:, 1], c[:, 2]] = True
    return vol, mask, origin, h, evals


def _part_groups(sdf, P):
    """{part name: face ids}: each point (face center) goes to the part with the smallest distance (first on ties).
    Empty when no part is named; parts that win no face get no group."""
    parts = _parts(sdf)
    if not any(s.name for _, s, _ in parts):
        return {}
    best, who = np.full(len(P), np.inf), np.zeros(len(P), dtype=np.int64)
    for j, (_, s, _) in enumerate(parts):
        idx = np.flatnonzero(_lb(s, P) < best)  # the bounds box rules a part out where it cannot be nearest
        if len(idx):
            d = s.fn(P[idx])
            win = d < best[idx]
            best[idx[win]], who[idx[win]] = d[win], j
    out = {}
    for j, (n, _, _) in enumerate(parts):
        idx = np.flatnonzero(who == j)
        if len(idx):
            out[n] = idx
    return out


def to_mesh(sdf, res=None, bounds=None, keep="largest", group=None, resolution=None, parts=True, sparse=True):
    """Mesh the zero surface by sparse marching cubes. res = 'draft' (64 cells along the longest side), 'final'
    (180) or a number of cells; None follows the runner ('draft', or 'final' under `run --final`); resolution= is
    an alias. Returns one closed, outward-facing triangle shell (keep='largest' drops smaller pieces; 'all' keeps
    them). parts=True names a face group after each SDF part (when any part is named; unnamed ones use proximity's
    names), each face going to the part nearest its center; group= adds one group of all faces. sparse=False
    evaluates every grid point (a slow reference for checking the sparse result). Raises if the surface reaches
    the sampling bounds (pass bounds=(lo, hi))."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from skimage.measure import marching_cubes
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_mesh needs scikit-image: pip install -e .[sdf]") from e
    t0 = time.perf_counter()
    label, cells = _cells(resolution if resolution is not None else res)
    lo, hi = (vec(bounds[0]), vec(bounds[1])) if bounds is not None else (sdf.lo, sdf.hi)
    if sparse:
        vol, mask, origin, h, evals = _sample(sdf, lo, hi, cells)
    else:
        vol, mask, origin, h, evals = _sample(sdf, lo, hi, cells, dense=True)
    border = np.concatenate([vol[[0, -1]].ravel(), vol[:, [0, -1]].ravel(), vol[:, :, [0, -1]].ravel()])
    if (border <= 0).any():
        raise ValueError("to_mesh: the surface reaches the sampling bounds; pass bounds=(lo, hi)")
    if not (vol < 0).any():
        raise ValueError("to_mesh: the field has no inside (no negative values) within the bounds")
    tiny = 0.01 * h  # values near 0 at grid points make sliver triangles; moves the surface <= 1% of a voxel
    small = np.abs(vol) < tiny
    vol[small] = np.where(vol[small] < 0, -tiny, tiny)
    with warnings.catch_warnings():  # scikit-image internals warn under newer NumPy; keep agent output clean
        warnings.simplefilter("ignore", DeprecationWarning)
        V, F, _, _ = marching_cubes(vol, level=0.0, spacing=(h, h, h), mask=mask)
    V, F = V + origin, F.astype(np.int64)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    if keep == "largest" and len(F):
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
        _, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V),) * 2),
                                      directed=False)
        F = F[lab[F[:, 0]] == np.argmax(np.bincount(lab[F[:, 0]]))]
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->", A, np.cross(B, C)) < 0:
        F = F[:, ::-1].copy()
    groups = _part_groups(sdf, V[F].mean(1)) if parts else {}
    if group:
        if group in groups:
            raise ValueError(f"to_mesh: group={group!r} is also a part name; pick another")
        groups[group] = range(len(F))
    m = Mesh(V, F, groups={n: ("f", idx) for n, idx in groups.items()})
    _log.append(dict(seconds=time.perf_counter() - t0, res=label, cells=cells, evals=evals, dense=vol.size,
                     faces=len(F)))
    return m
