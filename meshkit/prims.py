"""Primitives. Each returns a new Mesh (combine with join); outward normals; group= puts every face in a face group.
Z is up. axis= sets the primitive's main axis ('x', '-y', a vector...). Every face gets UVs: planar per side (grid,
cube), cube-face projection (quad_sphere), around x along (cylinder, revolve, sweep), caps as their own islands;
world-scale (uniform texel density), packed into 0..1 with a small margin."""
import numpy as np

from .mesh import Builder, Mesh, unit, vec
from .uv import layout_islands, planar_coords

UV_MARGIN = 0.01  # between generated UV islands (UV units)


def basis(axis):
    """Right-handed (U, V, W) with W along axis; principal axes map cyclically (z -> x,y,z; x -> y,z,x; y -> z,x,y)."""
    W = unit(axis)
    for i in range(3):
        if abs(W[i]) == 1.0:
            U, V = np.eye(3)[(i + 1) % 3], np.eye(3)[(i + 2) % 3]
            return U, (V if W[i] > 0 else -V), W
    helper = np.array([0.0, 0.0, 1.0]) if abs(W[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    U = np.cross(helper, W)
    U /= np.linalg.norm(U)
    return U, np.cross(W, U), W


def _pair(x):
    return (x, x) if np.ndim(x) == 0 else tuple(x)


def _triple(x):
    return (x, x, x) if np.ndim(x) == 0 else tuple(x)


def _finish(b, group):
    if group:
        if b.F:
            b.add_to_group(group, "f", range(len(b.F)))
        else:
            b.add_to_group(group, "v", range(len(b.V)))
    return b.build()


def loft(b, rings, closed=False, uv=None):
    """Add rings (each an (s,3) array, or a single (3,) point = pole) to builder b and join consecutive rings with
    quads (tris at poles). Ring order follows +angle around the travel direction for outward normals.
    uv = raw corner UVs: (us (s+1,) around, including the wrap; vs (one per ring, +1 more if closed) along), poles
    at the middle of their edge's u; or a function (ring index, column 0..s or None for a pole) -> (u, v).
    Returns per-ring vertex index lists."""
    ids = [[b.vert(r)] if np.ndim(r) == 1 else [b.vert(p) for p in r] for r in rings]
    pairs = list(zip(ids, ids[1:])) + ([(ids[-1], ids[0])] if closed else [])
    for k, (A, B) in enumerate(pairs):
        s = max(len(A), len(B))
        if len(A) == len(B) == 1:
            raise ValueError("loft: two consecutive rings collapsed to points")
        if 1 < min(len(A), len(B)) and len(A) != len(B):
            raise ValueError("loft: rings must have equal vertex counts")
        fn = uv if uv is None or callable(uv) else _separable(*uv)
        ka, kb = k, k + 1
        for j in range(s):
            j1 = (j + 1) % s
            t = None
            if len(A) == 1:
                f = (A[0], B[j1], B[j])
                if fn is not None:
                    t = (_pole(fn, ka, j), fn(kb, j + 1), fn(kb, j))
            elif len(B) == 1:
                f = (A[j], A[j1], B[0])
                if fn is not None:
                    t = (fn(ka, j), fn(ka, j + 1), _pole(fn, kb, j))
            else:
                f = (A[j], A[j1], B[j1], B[j])
                if fn is not None:
                    t = (fn(ka, j), fn(ka, j + 1), fn(kb, j + 1), fn(kb, j))
            b.face(f, uv=t)
    return ids


def _separable(us, vs):
    """UV function for loft from u per column and v per ring (a pole sits mid-way along its edge)."""
    def fn(i, j, mid=None):
        return ((us[mid] + us[mid + 1]) / 2 if j is None else us[j], vs[i])
    return fn


def _pole(fn, i, j):
    try:
        return fn(i, None, mid=j)
    except TypeError:
        return fn(i, None)


def ring_uv(segments, radius, profile_v):
    """Raw loft UVs: u around (arc length at radius), v = profile_v (one per ring)."""
    return 2 * np.pi * radius * np.arange(segments + 1) / segments, np.asarray(profile_v, dtype=float)


def cap_uv(b, ids, normal):
    """Planar raw UVs for an n-gon cap through verts ids, seen from its outward normal."""
    return planar_coords(np.array([b.V[i] for i in ids]), normal)


def finish_uvs(b, f0, islands, margin=UV_MARGIN):
    """Pack the raw UVs of faces f0.. (one island id per face) into the unit square."""
    raw = b.UV[f0:]
    if raw and all(u is not None for u in raw):
        b.UV[f0:] = [tuple(map(tuple, u)) for u in layout_islands([np.array(u) for u in raw], islands, margin)]


def grid(size=1.0, divisions=1, center=(0, 0, 0), axis="z", group=None):
    """Flat quad grid, size (sx, sy) or scalar, divisions (nx, ny) or int, facing axis."""
    (sx, sy), (nx, ny) = _pair(size), _pair(divisions)
    U, V, _ = basis(axis)
    c = vec(center)
    b = Builder(Mesh())
    for j in range(ny + 1):
        for i in range(nx + 1):
            b.vert(c + (i / nx - 0.5) * sx * U + (j / ny - 0.5) * sy * V)
    for j in range(ny):
        for i in range(nx):
            r0, r1 = j * (nx + 1), (j + 1) * (nx + 1)
            q = (r0 + i, r0 + i + 1, r1 + i + 1, r1 + i)
            b.face(q, uv=[((k % (nx + 1)) / nx * sx, (k // (nx + 1)) / ny * sy) for k in q])
    finish_uvs(b, 0, [0] * len(b.F))
    return _finish(b, group)


def _box_surface(n):
    """Integer lattice points, outward quads and each quad's side (0..5 = +x -x +y -y +z -z) on the surface of an
    n[0] x n[1] x n[2] box."""
    index, P, F, S = {}, [], [], []

    def vid(p):
        if p not in index:
            index[p] = len(P)
            P.append(p)
        return index[p]

    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for side in (0, n[a]):
            for j in range(n[c]):
                for i in range(n[b]):
                    q = []
                    for ii, jj in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0, 0, 0]
                        p[a], p[b], p[c] = side, ii, jj
                        q.append(vid(tuple(p)))
                    F.append(tuple(q) if side else tuple(q[::-1]))
                    S.append(2 * a + (0 if side else 1))
    return np.array(P, dtype=float), F, S


_SIDES = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=float)


def _side_uvs(Q, F, S):
    """Raw per-face UVs: each box side planar-projected from outside, from per-vert coordinates Q."""
    return [planar_coords(Q[list(f)], _SIDES[s]) for f, s in zip(F, S)]


def cube(size=1.0, center=(0, 0, 0), divisions=1, group=None):
    """Box of quads; size and divisions are scalars or (x, y, z)."""
    n = tuple(int(d) for d in _triple(divisions))
    P, F, S = _box_surface(n)
    V = (P / np.array(n) - 0.5) * np.array(_triple(size), float) + vec(center)
    uv = layout_islands(_side_uvs(V, F, S), S, UV_MARGIN)
    return _finish(Builder(Mesh(V, F, uvs=uv)), group)


def quad_sphere(radius=1.0, subdivisions=4, center=(0, 0, 0), group=None):
    """Cube subdivided (subdivisions per edge) and projected to a sphere, equal-angle spacing; all quads."""
    n = int(subdivisions)
    P, F, S = _box_surface((n, n, n))
    A = (2 * P / n - 1) * np.pi / 4  # equal-angle lattice coordinates
    T = np.tan(A)
    V = T / np.linalg.norm(T, axis=1)[:, None] * radius + vec(center)
    uv = layout_islands(_side_uvs(A * radius, F, S), S, UV_MARGIN)  # cube-face projection, arc-length units
    return _finish(Builder(Mesh(V, F, uvs=uv)), group)


def _circle(segments, U, V):
    th = 2 * np.pi * np.arange(segments) / segments
    return np.cos(th)[:, None] * U + np.sin(th)[:, None] * V


def _caps(b, ids, W, islands):
    for k, ring_ids, nrm in ((1, ids[0][::-1], -W), (2, ids[-1], W)):
        if len(ring_ids) > 2:
            b.face(ring_ids, uv=cap_uv(b, ring_ids, nrm))
            islands.append(k)


def cylinder(radius=0.5, depth=1.0, segments=16, rings=1, caps=True, radius_top=None, axis="z",
             center=(0, 0, 0), group=None):
    """Cylinder along axis, centered; rings = height divisions; caps = n-gons; radius_top for cones/tapers
    (0 makes a pointed tip)."""
    if segments < 3:
        raise ValueError("cylinder needs segments >= 3")
    rt = radius if radius_top is None else radius_top
    U, V, W = basis(axis)
    c, circ = vec(center), _circle(segments, U, V)
    pts = []
    for i in range(rings + 1):
        t = i / rings
        r, h = radius + (rt - radius) * t, (t - 0.5) * depth
        pts.append(c + h * W if r <= 1e-12 else c + h * W + r * circ)
    b = Builder(Mesh())
    slant = np.hypot(depth, rt - radius)
    if abs(radius - rt) > 1e-9 * max(radius, rt):  # cone / frustum: unrolled flat (developable, no stretch)
        k = slant / abs(radius - rt)
        rad = [radius + (rt - radius) * i / rings for i in range(rings + 1)]
        sgn = 1.0 if rt < radius else -1.0  # keep counter-clockwise UVs either way

        def fn(i, j, mid=None):
            if j is None:
                return (0.0, 0.0)
            psi = 2 * np.pi * j / segments / k
            return (sgn * rad[i] * k * np.sin(psi), -rad[i] * k * np.cos(psi))
        ids = loft(b, pts, uv=fn)
    else:
        ids = loft(b, pts, uv=ring_uv(segments, radius, slant * np.arange(rings + 1) / rings))
    isl = [0] * len(b.F)
    if caps:
        _caps(b, ids, W, isl)
    finish_uvs(b, 0, isl)
    return _finish(b, group)


def circle(radius=0.5, segments=16, axis="z", center=(0, 0, 0), fill=False, group=None):
    """Ring of verts joined by loose edges (fill=True makes one n-gon facing axis). Group is a vert group if unfilled."""
    U, V, W = basis(axis)
    b = Builder(Mesh())
    ids = [b.vert(p) for p in vec(center) + radius * _circle(segments, U, V)]
    if fill:
        b.face(ids, uv=cap_uv(b, ids, W))
        finish_uvs(b, 0, [0])
    else:
        b.L |= {tuple(sorted((ids[i], ids[(i + 1) % segments]))) for i in range(segments)}
    return _finish(b, group)


def revolve(profile, segments=16, axis="z", center=(0, 0, 0), closed=False, caps=False, group=None):
    """Lathe: revolve a 2D profile [(r, h), ...] around axis. List an open profile bottom -> top along the outside
    for outward normals; r = 0 points become poles. closed=True joins last to first (e.g. a torus);
    caps=True fills open end rings with n-gons."""
    P = np.asarray(profile, dtype=float).reshape(-1, 2)
    if closed:
        r, h = P[:, 0], P[:, 1]
        if np.sum(r * np.roll(h, -1) - np.roll(r, -1) * h) < 0:
            P = P[::-1]
    U, V, W = basis(axis)
    c, circ = vec(center), _circle(segments, U, V)
    pts = [c + h * W if r <= 1e-12 else c + h * W + r * circ for r, h in P]
    b = Builder(Mesh())
    steps = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1)
    rim = P[:, 0] > 1e-12
    r_mean = float(P[rim, 0].mean()) if rim.any() else 1.0
    ids = loft(b, pts, closed=closed, uv=ring_uv(segments, r_mean, np.r_[0, np.cumsum(steps)]))
    isl = [0] * len(b.F)
    if caps and not closed:
        _caps(b, ids, W, isl)
    finish_uvs(b, 0, isl)
    return _finish(b, group)


def ring(outer=1.0, inner=0.5, segments=16, depth=0.0, axis="z", center=(0, 0, 0), group=None):
    """Flat annulus facing axis (depth=0) or a closed thick ring/washer of the given depth."""
    if depth:
        d = depth / 2
        prof = [(outer, -d), (outer, d), (inner, d), (inner, -d)]
        return revolve(prof, segments, axis, center, closed=True, group=group)
    m = revolve([(outer, 0.0), (inner, 0.0)], segments, axis, center, group=group)
    _, _, W = basis(axis)
    flat = layout_islands([planar_coords(m.verts[list(f)], W) for f in m.faces], [0] * len(m.faces), UV_MARGIN)
    return Mesh(m.verts, m.faces, m.loose_edges, m.groups, flat)  # a flat ring maps best planar
