"""decimate: quadric-error edge collapse (Garland-Heckbert) down to a target face count; output is triangles.
Collapses keep the topology (link condition), never fold faces over, never create duplicates, and leave open
borders and non-manifold edges in place (their verts are locked). UVs: verts on UV seams are locked too, a collapse
interpolates the UV along the edge, and collapses that would flip a UV triangle are refused, so islands stay valid.
Deterministic (ties break by vert index)."""
import heapq

import numpy as np

from .mesh import Created, Mesh, edge_key


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tri_normal(p0, p1, p2):
    return _cross(_sub(p1, p0), _sub(p2, p0))


def _plane_quadric(p0, p1, p2):
    n = _tri_normal(p0, p1, p2)
    L = _dot(n, n) ** 0.5
    if L == 0:
        return (0.0,) * 10
    a, b, c = n[0] / L, n[1] / L, n[2] / L
    d = -(a * p0[0] + b * p0[1] + c * p0[2])
    w = L / 2  # area weight
    return (w * a * a, w * a * b, w * a * c, w * a * d, w * b * b, w * b * c, w * b * d, w * c * c, w * c * d, w * d * d)


def _qadd(q, r):
    return tuple(x + y for x, y in zip(q, r))


def _qerr(q, p):
    a2, ab, ac, ad, b2, bc, bd, c2, cd, d2 = q
    x, y, z = p
    return (a2 * x * x + b2 * y * y + c2 * z * z + 2 * (ab * x * y + ac * x * z + bc * y * z)
            + 2 * (ad * x + bd * y + cd * z) + d2)


def _qsolve(q):
    """Position minimizing the quadric, or None when the 3x3 system is near-singular (flat or ridge regions)."""
    a2, ab, ac, ad, b2, bc, bd, c2, cd, _ = q
    det = a2 * (b2 * c2 - bc * bc) - ab * (ab * c2 - bc * ac) + ac * (ab * bc - b2 * ac)
    if abs(det) <= 1e-6 * (a2 + b2 + c2) ** 3 or det == 0:
        return None
    r0, r1, r2 = -ad, -bd, -cd
    x = (r0 * (b2 * c2 - bc * bc) - ab * (r1 * c2 - bc * r2) + ac * (r1 * bc - b2 * r2)) / det
    y = (a2 * (r1 * c2 - bc * r2) - r0 * (ab * c2 - bc * ac) + ac * (ab * r2 - r1 * ac)) / det
    z = (a2 * (b2 * r2 - r1 * bc) - ab * (ab * r2 - r1 * ac) + r0 * (ab * bc - b2 * ac)) / det
    return (x, y, z)


def _uv_area(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _wedges(T, TUV, nv, tol=1e-9):
    """Verts whose corners carry more than one UV (seams; or a mix of UVs and none): (wedge set, per-vert UV)."""
    C, U = T.reshape(-1), TUV.reshape(-1, 2)
    nan = np.isnan(U).any(1)
    W = np.where(nan[:, None], 1e30, U)
    o = np.argsort(C, kind="stable")
    c = C[o]
    start = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    spread = np.maximum.reduceat(W[o], start) - np.minimum.reduceat(W[o], start)
    wedge = set(c[start][spread.max(1) > tol].tolist())
    uvv = [None] * nv
    for v, uv in zip(c[start].tolist(), U[o][start].tolist()):
        uvv[v] = tuple(uv)
    return wedge, uvv


def decimate(mesh, target_faces):
    """Collapse edges (cheapest quadric error first) until at most target_faces triangles remain (or no valid
    collapse is left). Face groups follow surviving triangles; vert groups follow merged verts; UVs follow (seam verts
    stay put). Created: vmap (old -> new vert), faces_before, faces_after."""
    T, src = mesh.triangles
    V = [tuple(p) for p in mesh.verts.tolist()]
    tris = [list(t) for t in T.tolist()]
    nv, nt = len(V), len(tris)
    alive_t, alive_v = [True] * nt, [True] * nv
    vt = [set() for _ in range(nv)]
    Q = [(0.0,) * 10 for _ in range(nv)]
    ecount = {}
    for ti, t in enumerate(tris):
        q = _plane_quadric(V[t[0]], V[t[1]], V[t[2]])
        for i in range(3):
            vt[t[i]].add(ti)
            Q[t[i]] = _qadd(Q[t[i]], q)
            k = edge_key(t[i], t[(i + 1) % 3])
            ecount[k] = ecount.get(k, 0) + 1
    locked = {v for e, c in ecount.items() if c != 2 for v in e}
    tuv, uvv = None, None
    if mesh.uvs is not None:
        from .uv import corner_tris
        TUV = mesh.uvs[corner_tris(mesh)]
        wedge, uvv = _wedges(T, TUV, nv)
        locked |= wedge
        tuv = [[tuple(c) for c in t] for t in TUV.tolist()]
    stamp, rep = [0] * nv, list(range(nv))
    min_n = 2e-10 * mesh.diag ** 2  # |cross| below this = degenerate (matches check())

    def candidate(a, b):
        q = _qadd(Q[a], Q[b])
        if a in locked:
            return _qerr(q, V[a]), V[a]
        if b in locked:
            return _qerr(q, V[b]), V[b]
        mid = tuple((x + y) / 2 for x, y in zip(V[a], V[b]))
        x = _qsolve(q)
        span = _dot(_sub(V[a], V[b]), _sub(V[a], V[b]))
        if x is not None and _dot(_sub(x, mid), _sub(x, mid)) <= 4 * span:
            return _qerr(q, x), x
        return min((_qerr(q, p), p) for p in (V[a], V[b], mid))

    def entry(a, b):
        a, b = min(a, b), max(a, b)
        if a in locked and b in locked:
            return None
        cost, x = candidate(a, b)
        return (cost, a, b, stamp[a], stamp[b], x)

    heap = [e for e in (entry(a, b) for a, b in sorted(ecount)) if e]
    heapq.heapify(heap)

    def ring(v):
        return {w for t in vt[v] for w in tris[t]} - {v}

    def new_uvs(u, v, x, shared):
        """{(tri, corner): uv} for corners of u/v in the triangles that survive the collapse."""
        s0 = tris[min(shared)]
        if u in locked:  # x = V[u]: v's fan takes u's UV on that side
            keep, take, uv = u, v, tuv[min(shared)][s0.index(u)]
        elif v in locked:  # x = V[v]: u's fan takes v's UV on that side
            keep, take, uv = v, u, tuv[min(shared)][s0.index(v)]
        else:
            d = _sub(V[v], V[u])
            t = min(max(_dot(_sub(x, V[u]), d) / max(_dot(d, d), 1e-300), 0.0), 1.0)
            uv = tuple((1 - t) * a + t * b for a, b in zip(uvv[u], uvv[v]))
            keep, take = None, None
        out = {}
        for t in (vt[u] | vt[v]) - shared:
            for k, i in enumerate(tris[t]):
                if i == take or (keep is None and i in (u, v)):
                    out[t, k] = uv
        return out, uv

    def try_collapse(u, v, x):
        shared = vt[u] & vt[v]
        if not shared or nf - len(shared) < 4:
            return 0
        opp = {w for t in shared for w in tris[t]} - {u, v}
        if ring(u) & ring(v) != opp:
            return 0  # link condition: collapsing would pinch the surface
        seen = set()
        moved = {}
        if tuv is not None:
            moved, uv_new = new_uvs(u, v, x, shared)
        for t in (vt[u] | vt[v]) - shared:
            tri = tris[t]
            old = [V[i] for i in tri]
            new = [x if i in (u, v) else V[i] for i in tri]
            n0, n1 = _tri_normal(*old), _tri_normal(*new)
            l1 = _dot(n1, n1) ** 0.5
            if l1 <= min_n or _dot(n0, n1) < 0.2 * l1 * _dot(n0, n0) ** 0.5:
                return 0  # degenerate or folded over
            key = frozenset(u if i == v else i for i in tri)
            if key in seen:
                return 0
            seen.add(key)
            if tuv is not None:
                a0 = _uv_area(*tuv[t])
                a1 = _uv_area(*[moved.get((t, k), tuv[t][k]) for k in range(3)])
                if a0 == a0 and (a0 > 0) != (a1 > 0) or (a0 != 0 and a1 == 0):
                    return 0  # the UV triangle would flip or collapse
        if tuv is not None:
            for (t, k), uv in moved.items():
                tuv[t][k] = uv
            if u not in locked and v not in locked:
                uvv[u] = uv_new
        V[u] = tuple(x)
        for t in shared:
            alive_t[t] = False
            for w in tris[t]:
                vt[w].discard(t)
        for t in vt[v]:
            tris[t] = [u if i == v else i for i in tris[t]]
            vt[u].add(t)
        vt[v], alive_v[v], rep[v] = set(), False, u
        Q[u] = _qadd(Q[u], Q[v])
        stamp[u] += 1
        if v in locked:
            locked.add(u)
        return len(shared)

    nf = nt
    while nf > target_faces and heap:
        cost, u, v, su, sv, x = heapq.heappop(heap)
        if not (alive_v[u] and alive_v[v]) or stamp[u] != su or stamp[v] != sv:
            continue
        removed = try_collapse(u, v, x)
        if removed:
            nf -= removed
            for w in sorted(ring(u)):
                e = entry(u, w)
                if e:
                    heapq.heappush(heap, e)

    def find(v):
        while rep[v] != v:
            rep[v] = rep[rep[v]]
            v = rep[v]
        return v

    used = sorted({v for t, ok in zip(tris, alive_t) if ok for v in t})
    new_index = {v: i for i, v in enumerate(used)}
    faces = [tuple(new_index[v] for v in t) for t, ok in zip(tris, alive_t) if ok]
    tsrc = [int(s) for s, ok in zip(src, alive_t) if ok]
    vmap = np.array([new_index.get(find(v), -1) for v in range(nv)], dtype=np.int64)
    groups = {}
    for name, g in mesh.groups.items():
        if g.kind == "f":
            groups[name] = ("f", [i for i, s in enumerate(tsrc) if s in g.idx])
        else:
            groups[name] = ("v", {int(vmap[v]) for v in g.idx if vmap[v] >= 0})
    uvs = [tuv[t] for t, ok in zip(range(nt), alive_t) if ok] if tuv is not None else None
    out = Mesh([V[v] for v in used], faces, groups=groups, uvs=uvs)
    return out, Created(vmap=vmap, faces_before=len(mesh.faces), faces_after=len(faces))
