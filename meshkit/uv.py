"""UVs: per face corner (mesh.uvs, (C,2), face order; NaN = missing), set by the generators, kept by ops.
Islands and seams, projections (planar / box / cylinder / sphere), pack (shelf packer, uniform texel density),
unwrap (xatlas, optional extra; or 'box'), statistics for check() and a PNG sheet of the layout.
UV origin is bottom-left (Blender / OBJ); image row 0 is v = 1."""
import numpy as np

from .mesh import Created, Mesh, resolve, unit

_NAN2 = (np.nan, np.nan)


# ---- corners and islands ----------------------------------------------------------------
def corner_tris(mesh):
    """(T,3) corner indices (rows of mesh.uvs) of mesh.triangles' fan triangles."""
    cs = mesh.corner_start
    sizes = np.diff(cs)
    n = np.maximum(sizes - 2, 0)
    f = np.repeat(np.arange(len(sizes)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    first = cs[f]
    return np.stack([first, first + k + 1, first + k + 2], 1)


def _half_edges(mesh):
    """Per corner: (face, vert a, next vert b, corner of a, corner of b)."""
    cs, cv = mesh.corner_start, mesh.corner_verts
    face = np.repeat(np.arange(len(mesh.faces)), np.diff(cs))
    nxt = np.arange(len(cv)) + 1
    last = cs[1:] - 1
    nxt[last] = cs[:-1]
    return face, cv, cv[nxt], np.arange(len(cv)), nxt


def _edge_pairs(mesh):
    """Manifold interior edges as corner pairs: (corner a in f, corner b in f, corner b' in g, corner a' in g, f, g)
    where face f runs a->b and face g runs b->a."""
    face, a, b, ca, cb = _half_edges(mesh)
    key = np.minimum(a, b) * max(len(mesh.verts), 1) + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    k = key[o]
    start = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    cnt = np.diff(np.r_[start, len(k)])
    two = start[cnt == 2]
    i, j = o[two], o[two + 1]
    rev = (a[i] == b[j]) & (b[i] == a[j])  # consistently wound pair
    i, j = i[rev], j[rev]
    return ca[i], cb[i], ca[j], cb[j], face[i], face[j]


def _same(U, x, y, tol):
    d = np.abs(U[x] - U[y]).max(1)
    return d <= tol


def _seam_mask(mesh, tol=1e-6):
    """(pairs, seam bool) for the manifold edges of _edge_pairs: seam = the two faces' UVs differ there."""
    pa = _edge_pairs(mesh)
    ca, cb, cb2, ca2, f, g = pa
    U = mesh.uvs
    seam = ~(_same(U, ca, ca2, tol) & _same(U, cb, cb2, tol))
    return pa, seam


def _components(n, A, B):
    """Connected component label per node (lowest node id first-seen order) for undirected edges A-B."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        _, lab = connected_components(coo_matrix((np.ones(len(A)), (A, B)), shape=(n, n)), directed=False)
    except ImportError:
        parent = list(range(n))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for x, y in zip(A.tolist(), B.tolist()):
            rx, ry = root(x), root(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
        lab = np.array([root(x) for x in range(n)])
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))  # relabel by first face index
    return order[inv.reshape(-1)]


def uv_islands(mesh, tol=1e-6):
    """(F,) island id per face: faces joined across edges whose UVs match on both sides (ids by first face)."""
    if mesh.uvs is None:
        raise ValueError("mesh has no UVs")
    (ca, cb, cb2, ca2, f, g), seam = _seam_mask(mesh, tol)
    return _components(len(mesh.faces), f[~seam], g[~seam])


def uv_seams(mesh, tol=1e-6):
    """Sorted (a, b) edges where the UVs of the two faces differ (cuts between islands; open borders excluded)."""
    (ca, cb, cb2, ca2, f, g), seam = _seam_mask(mesh, tol)
    cv = mesh.corner_verts
    a, b = cv[ca[seam]], cv[cb[seam]]
    return sorted(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))


def _signed_area(U, mesh):
    """Signed UV area per face (shoelace over corners; positive = counter-clockwise)."""
    cs = mesh.corner_start
    face, _, _, ca, cb = _half_edges(mesh)
    cross = U[ca, 0] * U[cb, 1] - U[cb, 0] * U[ca, 1]
    return np.bincount(face, weights=cross, minlength=len(mesh.faces)) / 2 if len(cs) > 1 else np.zeros(0)


# ---- layout -----------------------------------------------------------------------------
def _shelf(W, H, limit, gap):
    """Positions (x, y) of boxes W x H (already sorted) packed in shelves no wider than limit, gap between them."""
    pos, x, y, row, used = np.zeros((len(W), 2)), 0.0, 0.0, 0.0, 0.0
    for i, (w, h) in enumerate(zip(W, H)):
        if x > 0 and x + w + gap > limit:
            y += row + gap
            x, row = 0.0, 0.0
        pos[i] = (x, y)
        x += w + gap
        row = max(row, h)
        used = max(used, x - gap)
    return pos, used, y + row


def _layout(lo, hi, margin, rotate):
    """Pack island boxes (lo, hi corners, (n,2) in any common unit) into the unit square with `margin` between and
    around islands. Returns (scale, offsets (n,2), rotated (n,) bool): an island point p maps to
    (local(p) * scale + offset) with local = p - lo, or (p_y - lo_y, hi_x - p_x) when rotated (a quarter turn)."""
    size = np.maximum(hi - lo, 1e-12)
    rot = (size[:, 1] > size[:, 0] * 1.0001) if rotate else np.zeros(len(size), dtype=bool)
    dims = np.where(rot[:, None], size[:, ::-1], size)
    order = np.lexsort((np.arange(len(dims)), -dims[:, 0], -dims[:, 1]))  # tallest first
    W, H = dims[order, 0], dims[order, 1]
    area = float((W * H).sum())
    best = None
    for f in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        s = 1.0 / max(np.sqrt(area) * 1.15, W.max(), H.max())
        for _ in range(8):  # margin is in output units, so the gap depends on the scale: iterate
            gap = margin / s
            limit = max(f * np.sqrt(area + gap * gap * len(W)), W.max())
            pos, used_w, used_h = _shelf(W, H, limit, gap)
            s_new = (1.0 - 2 * margin) / max(used_w, used_h)  # margin on the outside too
            if abs(s_new - s) < 1e-9 * s:
                break
            s = s_new
        if best is None or s > best[0] * (1 + 1e-9):
            best = (s, pos.copy())
    s, pos = best
    off = np.zeros((len(dims), 2))
    off[order] = pos * s + margin
    return s, off, rot


def _hull(P):
    """Convex hull (monotone chain) of 2D points, counter-clockwise."""
    P = np.unique(P, axis=0)
    if len(P) < 3:
        return P

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(tuple(p))
        return out
    lower, upper = half(P), half(P[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _best_angle(P):
    """Rotation (radians) giving the smallest axis-aligned box of P (rotating calipers over hull edges)."""
    H = _hull(P)
    if len(H) < 3:
        return 0.0
    E = np.roll(H, -1, axis=0) - H
    ang = np.unique(np.round(np.mod(np.arctan2(E[:, 1], E[:, 0]), np.pi / 2), 12))
    best, best_a = np.inf, 0.0
    for a in ang:
        c, s_ = np.cos(-a), np.sin(-a)
        R = H @ np.array([[c, s_], [-s_, c]])
        area = np.prod(R.max(0) - R.min(0))
        if area < best * (1 - 1e-9):
            best, best_a = area, a
    return float(best_a)


def _turn(U, isl, n):
    """Rotate each island about its centroid to its minimum-area bounding box (orientation is free: texturing is
    3D)."""
    out = U.copy()
    for i in range(n):
        m = isl == i
        if m.sum() < 3:
            continue
        a = _best_angle(U[m])
        if a:
            c, s_ = np.cos(-a), np.sin(-a)
            ctr = U[m].mean(0)
            out[m] = (U[m] - ctr) @ np.array([[c, s_], [-s_, c]]) + ctr
    return out


def _apply_layout(P, lo, hi, s, off, rot):
    """Map island points P (k,2) with their island's lo/hi/off/rot rows (k,...) into the packed square."""
    loc = P - lo
    loc = np.where(rot[:, None], np.stack([P[:, 1] - lo[:, 1], hi[:, 0] - P[:, 0]], 1), loc)
    return loc * s + off


def layout_islands(face_uvs, islands, margin=0.01, rotate=False):
    """Normalize raw per-face UVs (list of (n,2) arrays in world-like units, islands = id per face) into the unit
    square: islands keep their relative scale and shape and are shelf-packed with `margin`. Returns the new list."""
    if not face_uvs:
        return []
    U = np.concatenate([np.asarray(u, dtype=float).reshape(-1, 2) for u in face_uvs])
    isl = np.repeat(np.asarray(islands), [len(u) for u in face_uvs])
    ids, inv = np.unique(isl, return_inverse=True)
    lo = np.full((len(ids), 2), np.inf)
    hi = np.full((len(ids), 2), -np.inf)
    np.minimum.at(lo, inv, U)
    np.maximum.at(hi, inv, U)
    s, off, rot = _layout(lo, hi, margin, rotate)
    out = _apply_layout(U, lo[inv], hi[inv], s, off[inv], rot[inv])
    sizes = np.cumsum([len(u) for u in face_uvs])[:-1]
    return np.split(out, sizes)


def _with_uvs(mesh, U):
    return Mesh(mesh.verts, mesh.faces, mesh.loose_edges, mesh.groups, U)


def _planar_basis(n):
    """(right, up) for a planar projection seen from direction n: up = +Z unless n is (near) vertical, then +Y."""
    n = unit(n)
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    up = unit(up - (up @ n) * n)
    return np.cross(up, n), up


def planar_coords(P, n):
    """(k,2) planar UV coordinates of points P seen from direction n (counter-clockwise faces stay counter-clockwise)."""
    r, u = _planar_basis(n)
    return np.stack([P @ r, P @ u], 1)


# ---- projections, pack, unwrap ------------------------------------------------------------------
_BOX = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=float)


def project_uvs(mesh, mode="box", faces=None, axis="z", margin=0.01):
    """Project UVs for faces (default all) and pack those faces' islands into the unit square (other faces keep
    theirs; call pack(m) afterwards if both are to share the texture). mode: 'planar' (seen along axis), 'box'
    (each face from the nearest of 6 axis directions), 'cylinder' (around axis, through the selection's center) or
    'sphere' (longitude / latitude about the center); cylinder and sphere suit tubes and balls, but faces around the
    axis (caps, poles) fold over: use 'box' or unwrap for those. World-scale coordinates (uniform texel density).
    Created: faces, islands (count)."""
    F = resolve(mesh, faces, "f")
    if not F:
        raise ValueError("project_uvs: no faces selected")
    if mode not in ("planar", "box", "cylinder", "sphere"):
        raise ValueError("project_uvs: mode must be 'planar', 'box', 'cylinder' or 'sphere'")
    V = mesh.verts
    vs = sorted({v for f in F for v in mesh.faces[f]})
    c = (V[vs].min(0) + V[vs].max(0)) / 2
    raw, key = [], []
    if mode == "box":
        side = np.argmax(mesh.face_normals[F] @ _BOX.T, axis=1)
    for i, f in enumerate(F):
        P = V[list(mesh.faces[f])] - c
        if mode == "planar":
            raw.append(planar_coords(P, axis))
            key.append(0)
        elif mode == "box":
            raw.append(planar_coords(P, _BOX[side[i]]))
            key.append(int(side[i]))
        else:
            W = unit(axis)
            r_, u_ = _planar_basis(W)
            x, y, h = P @ r_, P @ u_, P @ W
            ang = np.arctan2(y, x)
            if ang.max() - ang.min() > np.pi:  # the face straddles the wrap: keep it in one piece
                ang = np.where(ang < 0, ang + 2 * np.pi, ang)
            if mode == "cylinder":
                rad = float(np.hypot((V[vs] - c) @ r_, (V[vs] - c) @ u_).mean()) or 1.0
                raw.append(np.stack([ang * rad, h], 1))
            else:
                rad = float(np.linalg.norm(V[vs] - c, axis=1).mean()) or 1.0
                raw.append(np.stack([ang * rad, np.arctan2(h, np.hypot(x, y)) * rad], 1))
            key.append(0)
    tmp = [None] * len(mesh.faces)
    for f, r in zip(F, raw):
        tmp[f] = r
    sub = Mesh(V, [mesh.faces[f] for f in F], uvs=[tmp[f] for f in F])
    isl = uv_islands(sub)
    if mode == "box":
        isl = isl * 6 + np.array(key)  # never join islands seen from different sides
    if mode in ("cylinder", "sphere"):  # a projection that faces away flips; mirror those faces' islands
        area = _signed_area(sub.uvs, sub)
        flip = np.zeros(isl.max() + 1)
        np.add.at(flip, isl, area)
        raw = [np.c_[-r[:, 0], r[:, 1]] if flip[isl[i]] < 0 else r for i, r in enumerate(raw)]
    out = layout_islands(raw, isl, margin)
    U = [None] * len(mesh.faces) if mesh.uvs is None else [mesh.face_uvs(f) for f in range(len(mesh.faces))]
    for f, u in zip(F, out):
        U[f] = u
    return _with_uvs(mesh, U), Created(faces=F, islands=len(np.unique(isl)))


def pack(mesh, margin=0.01, rotate=True):
    """Re-pack all UV islands into the unit square: each island scaled to the same texel density (UV area
    proportional to 3D area); rotate=True turns each to its tightest bounding box (and landscape) so shelves fill well;
    `margin` (UV units, ~ texels / size) between islands. Fixes overlaps after join/mirror. Raises if some faces have no UVs.
    Created: islands (count), scale (UV units per world unit)."""
    if mesh.uvs is None:
        raise ValueError("pack: mesh has no UVs; unwrap(m) or project_uvs(m) first")
    missing = np.isnan(mesh.uvs).any(1)
    if missing.any():
        n = len(np.unique(np.repeat(np.arange(len(mesh.faces)), np.diff(mesh.corner_start))[missing]))
        raise ValueError(f"pack: {n} faces have no UVs; give them some with project_uvs(m, faces=...) first")
    isl = uv_islands(mesh)
    A3 = np.bincount(isl, weights=mesh.face_areas, minlength=isl.max() + 1)
    Auv = np.bincount(isl, weights=np.abs(_signed_area(mesh.uvs, mesh)), minlength=isl.max() + 1)
    k = np.sqrt(A3 / np.maximum(Auv, 1e-30))
    k = np.where(Auv > 1e-30, k, 1.0)
    face_of = np.repeat(np.arange(len(mesh.faces)), np.diff(mesh.corner_start))
    ci = isl[face_of]
    U = mesh.uvs * k[ci][:, None]
    if rotate:
        U = _turn(U, ci, len(k))
    lo = np.full((len(k), 2), np.inf)
    hi = np.full((len(k), 2), -np.inf)
    np.minimum.at(lo, ci, U)
    np.maximum.at(hi, ci, U)
    s, off, rot = _layout(lo, hi, margin, rotate)
    out = _apply_layout(U, lo[ci], hi[ci], s, off[ci], rot[ci])
    return _with_uvs(mesh, out), Created(islands=len(k), scale=float(s))


def _stretch(mesh, U, ok=None):
    """(per-face stretch max(r, 1/r), area-weighted percentile fn) for faces with UVs and area."""
    sa = _signed_area(U, mesh)
    A3 = mesh.face_areas
    good = (np.abs(sa) > 1e-12 * max(float(np.abs(sa).sum()), 1e-30)) & (A3 > 0)
    if ok is not None:
        good &= ok
    r = (np.abs(sa[good]) / max(np.abs(sa[good]).sum(), 1e-30)) / (A3[good] / max(A3[good].sum(), 1e-30))
    s = np.maximum(r, 1 / np.maximum(r, 1e-30))
    o = np.argsort(s)
    w = np.cumsum(A3[good][o]) / max(A3[good].sum(), 1e-300)  # by surface area: slivers can't dominate

    def pct(q):
        return float(s[o][min(np.searchsorted(w, q), len(o) - 1)]) if len(o) else 1.0
    return s, pct


def _xatlas(mesh, margin, max_cost):
    import xatlas
    T, src = mesh.triangles
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.verts.astype(np.float32), T.astype(np.uint32))
    po = xatlas.PackOptions()
    po.resolution, po.padding, po.bilinear = 1024, max(1, int(round(margin * 1024))), True
    po.rotate_charts = False  # xatlas turns charts by transposing them, which mirrors them
    co = xatlas.ChartOptions()
    co.fix_winding = True  # counter-clockwise UVs, like the faces
    co.max_cost = max_cost  # lower than xatlas' default 2: no big conformal charts over spikes and horns
    atlas.generate(co, po)
    w, h = atlas.width, atlas.height
    vmap, idx, uv = atlas[0]
    if len(idx) != len(T) or not (vmap[idx] == T).all():
        raise RuntimeError("unwrap: xatlas changed the triangle order")
    uv = uv.astype(float) * np.array([w, h]) / max(w, h)  # xatlas normalizes each axis; keep the aspect
    tri_uv = uv[idx]  # (T,3,2)
    U = np.full((int(mesh.corner_start[-1]), 2), np.nan)
    C = corner_tris(mesh)
    bad = np.zeros(len(mesh.faces), dtype=bool)
    for k in range(3):
        cur = U[C[:, k]]
        clash = ~np.isnan(cur[:, 0]) & (np.abs(cur - tri_uv[:, k]).max(1) > 1e-7)
        bad[src[clash]] = True
        U[C[:, k]] = tri_uv[:, k]
    m = _with_uvs(mesh, U)
    isl = uv_islands(m)
    face_of = np.repeat(np.arange(len(mesh.faces)), np.diff(mesh.corner_start))
    area = np.bincount(isl, weights=_signed_area(U, m), minlength=isl.max() + 1)
    mirrored = (area < 0).any()
    if mirrored:  # a chart still came back mirrored: mirror it back in place, then re-pack (it may now overlap)
        ci = isl[face_of]
        lo = np.full(len(area), np.inf)
        hi = np.full(len(area), -np.inf)
        np.minimum.at(lo, ci, U[:, 0])
        np.maximum.at(hi, ci, U[:, 0])
        neg = area[ci] < 0
        U[neg, 0] = lo[ci][neg] + hi[ci][neg] - U[neg, 0]
        m = _with_uvs(mesh, U)
    if bad.any():  # polygons split between charts: own planar island each
        m, _ = project_uvs(m, "box", faces=list(np.flatnonzero(bad)), margin=margin)
    if mirrored or bad.any():
        m, _ = pack(m, margin)
    return m


def unwrap(mesh, method="xatlas", margin=0.01, max_stretch=2.0):
    """New UVs for the whole mesh. method='xatlas' (optional extra: pip install -e .[uv]): charts cut along
    creases and high-distortion lines, parameterized and packed with margin; deterministic. Charts are kept small
    enough that the area-weighted p95 stretch stays under max_stretch (retried with smaller charts if not).
    method='box': box projection + pack (no dependency; more, smaller islands). Faces are fan-triangulated for
    xatlas; a polygon whose triangles land in different charts gets its own planar island.
    Created: islands (count), stretch_p95."""
    if not mesh.faces:
        raise ValueError("unwrap: mesh has no faces")
    if method == "box":
        m, c = project_uvs(mesh, "box", margin=margin)
        return m, Created(islands=c.islands, stretch_p95=_stretch(m, m.uvs)[1](0.95))
    if method != "xatlas":
        raise ValueError("unwrap: method must be 'xatlas' or 'box'")
    import importlib.util
    if importlib.util.find_spec("xatlas") is None:
        raise ImportError("unwrap needs xatlas: pip install -e .[uv] (or unwrap(m, method='box'))")
    best = None
    for cost in (0.5, 0.25, 0.12):
        m = _xatlas(mesh, margin, cost)
        p95 = _stretch(m, m.uvs)[1](0.95)
        if best is None or p95 < best[1]:
            best = (m, p95)
        if p95 <= max_stretch:
            break
    m, p95 = best
    return m, Created(islands=int(uv_islands(m).max()) + 1, stretch_p95=p95)


# ---- statistics, coverage raster, sheet ------------------------------------------------------------
def _cover(P, W, H, strict=False, budget=4_000_000):
    """All (pixel, triangle, barycentrics) with the pixel center inside triangle P (T,3,2) (pixel units, any
    orientation). Returns (pix (k,), tri (k,), bary (k,3))."""
    xs, ys = P[..., 0], P[..., 1]
    i0 = np.maximum(np.ceil(xs.min(1) - 0.5), 0).astype(np.int64)
    i1 = np.minimum(np.floor(xs.max(1) - 0.5), W - 1).astype(np.int64)
    j0 = np.maximum(np.ceil(ys.min(1) - 0.5), 0).astype(np.int64)
    j1 = np.minimum(np.floor(ys.max(1) - 0.5), H - 1).astype(np.int64)
    area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0])
    ok = (i0 <= i1) & (j0 <= j1) & (np.abs(area) > 1e-12) & np.isfinite(area)
    span = np.maximum(i1 - i0, j1 - j0) + 1
    eps = 1e-6 if strict else -1e-9
    pix, tri, bary = [], [], []

    def test(t, px, py):
        fx, fy = px + 0.5, py + 0.5
        X0, X1, X2 = xs[t, 0, None], xs[t, 1, None], xs[t, 2, None]
        Y0, Y1, Y2 = ys[t, 0, None], ys[t, 1, None], ys[t, 2, None]
        a = area[t, None]
        w0 = ((X2 - X1) * (fy - Y1) - (Y2 - Y1) * (fx - X1)) / a
        w1 = ((X0 - X2) * (fy - Y2) - (Y0 - Y2) * (fx - X2)) / a
        w2 = 1 - w0 - w1
        m = (px <= i1[t, None]) & (py <= j1[t, None]) & (w0 > eps) & (w1 > eps) & (w2 > eps)
        pix.append((py * W + px)[m])
        tri.append(np.broadcast_to(t[:, None], m.shape)[m])
        bary.append(np.stack([w0[m], w1[m], w2[m]], 1))

    lo = 0
    for c in (1, 2, 4, 8, 16, 32, 64):
        ts = np.flatnonzero(ok & (span > lo) & (span <= c))
        lo = c
        oy, ox = np.divmod(np.arange(c * c), c)
        step = max(1, budget // (c * c))
        for s in range(0, len(ts), step):
            t = ts[s:s + step]
            test(t, i0[t, None] + ox, j0[t, None] + oy)
    for t in np.flatnonzero(ok & (span > 64)):
        py, px = np.mgrid[j0[t]:j1[t] + 1, i0[t]:i1[t] + 1]
        test(np.array([t]), px.reshape(1, -1), py.reshape(1, -1))
    if not pix:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    return np.concatenate(pix), np.concatenate(tri), np.concatenate(bary)


def uv_pixels(mesh, size):
    """(T,3,2) UV triangle corners in pixel units of a size x size image (row 0 = v 1)."""
    UV = mesh.uvs[corner_tris(mesh)]
    return np.stack([UV[..., 0] * size, (1 - UV[..., 1]) * size], -1)


def uv_stats(mesh, res=512, tol=1e-6):
    """UV health numbers: missing / flipped / zero-area faces, islands, seams, outside 0..1, overlap (texels at res
    covered by two islands, and the island pairs; self-overlap = an island folding onto itself), stretch (per face
    max(r, 1/r) with r = its share of UV area over its share of 3D area; p95 and worst = 99.9th percentile weighted by
    surface area, so sliver triangles don't dominate; stretch_max = the single worst face), coverage (texels inside
    UV islands)."""
    if mesh.uvs is None:
        return dict(has=False)
    U, F = mesh.uvs, len(mesh.faces)
    face_of = np.repeat(np.arange(F), np.diff(mesh.corner_start))
    missing = np.zeros(F, dtype=bool)
    missing[face_of[np.isnan(U).any(1)]] = True
    ok = ~missing
    Uz = np.where(np.isnan(U), 0, U)
    sa = _signed_area(Uz, mesh)
    tiny = 1e-12 * max(float(np.abs(sa[ok]).sum()) if ok.any() else 0, 1e-30)
    zero = ok & (np.abs(sa) <= tiny)
    flipped = ok & (sa < -tiny)
    sub = Mesh(mesh.verts, mesh.faces, uvs=Uz)
    isl = uv_islands(sub, tol)
    isl = np.where(ok, isl, -1)
    seams = len(uv_seams(sub, tol))
    s, pct = _stretch(mesh, Uz, ok & ~zero)
    outside = np.zeros(F, dtype=bool)
    outside[face_of[((Uz < -1e-9) | (Uz > 1 + 1e-9)).any(1)]] = True
    outside &= ok
    T, src = mesh.triangles
    keep = np.flatnonzero(ok[src] & ~zero[src])
    pix, tri, _ = _cover(uv_pixels(sub, res)[keep], res, res, strict=True)
    pairs, overlap, self_overlap, covered = {}, 0, 0, 0
    if len(pix):
        fid = src[keep][tri]
        u = np.unique(pix * F + fid)  # (texel, face): a face's own fan triangles never count twice
        p, f = u // F, u % F
        multi = np.flatnonzero(np.r_[p[1:] == p[:-1], False] | np.r_[False, p[1:] == p[:-1]])
        start = np.flatnonzero(np.r_[True, p[multi][1:] != p[multi][:-1]]) if len(multi) else multi
        for a, b in zip(start, np.r_[start[1:], len(multi)]):
            ids = sorted(set(isl[f[multi[a:b]]].tolist()))
            if len(ids) == 1:
                self_overlap += 1
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs[(ids[x], ids[y])] = pairs.get((ids[x], ids[y]), 0) + 1
        overlap = len(start) - self_overlap
        covered = len(np.unique(pix))
    return dict(has=True, missing=int(missing.sum()), flipped=int(flipped.sum()), zero=int(zero.sum()),
                islands=int(len(np.unique(isl[ok]))), seams=seams, outside=int(outside.sum()),
                overlap_texels=overlap, overlap_pairs=pairs, self_overlap_texels=self_overlap, res=res,
                coverage=covered / res ** 2,
                stretch_worst=pct(0.999), stretch_p95=pct(0.95), stretch_max=float(s.max()) if len(s) else 1.0,
                flipped_faces=np.flatnonzero(flipped)[:20].tolist(), missing_faces=np.flatnonzero(missing)[:20].tolist())


def _segments(img, A, B, color):
    """Draw pixel-space segments A->B (k,2) into img."""
    if not len(A):
        return
    n = np.maximum(np.ceil(np.abs(B - A).max(1)).astype(np.int64), 1) + 1
    t = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = t / np.repeat(n - 1, n)
    P = np.repeat(A, n, 0) + (np.repeat(B - A, n, 0)) * t[:, None]
    x = np.clip(np.floor(P[:, 0]).astype(np.int64), 0, img.shape[1] - 1)
    y = np.clip(np.floor(P[:, 1]).astype(np.int64), 0, img.shape[0] - 1)
    img[y, x] = color


def export_uv_sheet(mesh, path, size=1024):
    """Write a PNG of the UV layout: islands filled (a muted color each), face edges faint; island outlines are
    seams in orange (cuts between islands) and open mesh borders in cyan; flipped faces magenta, overlapping texels
    red, 0.1 grid. Returns path."""
    from .png import write_png
    if mesh.uvs is None:
        raise ValueError("export_uv_sheet: mesh has no UVs")
    img = np.zeros((size, size, 3))
    img[:] = (34, 36, 40)
    g = np.arange(1, 10) * size // 10
    img[g, :] = (52, 54, 60)
    img[:, g] = (52, 54, 60)
    Uz = np.where(np.isnan(mesh.uvs), 0, mesh.uvs)
    sub = Mesh(mesh.verts, mesh.faces, uvs=Uz)
    isl = uv_islands(sub)
    sa = _signed_area(Uz, mesh)
    T, src = mesh.triangles
    pix, tri, _ = _cover(uv_pixels(sub, size), size, size)
    hue = (isl[src[tri]] * 0.618034) % 1.0
    col = 70 + 40 * np.stack([np.cos(2 * np.pi * (hue + k / 3)) for k in range(3)], 1)
    col = np.where((sa[src[tri]] < 0)[:, None], (190, 40, 190), col)
    img.reshape(-1, 3)[pix] = col
    u, cnt = np.unique(pix * (isl.max() + 2) + isl[src[tri]], return_counts=True)
    p = u // (isl.max() + 2)
    over = p[np.r_[p[1:] == p[:-1], False]]
    img.reshape(-1, 3)[over] = (220, 40, 40)
    face, a, b, ca, cb = _half_edges(mesh)
    px = np.stack([Uz[:, 0] * size, (1 - Uz[:, 1]) * size], 1)
    _segments(img, px[ca], px[cb], (95, 100, 110))
    (pa, pb, pb2, pa2, f, gg), seam = _seam_mask(sub)
    interior = set(zip(pa.tolist(), pb.tolist())) | set(zip(pb2.tolist(), pa2.tolist()))
    border = np.array([(x, y) not in interior for x, y in zip(ca.tolist(), cb.tolist())], dtype=bool)
    _segments(img, px[ca[border]], px[cb[border]], (110, 230, 240))
    _segments(img, px[pa[seam]], px[pb[seam]], (255, 150, 40))
    _segments(img, px[pb2[seam]], px[pa2[seam]], (255, 150, 40))
    return write_png(path, np.clip(img, 0, 255).astype(np.uint8))
