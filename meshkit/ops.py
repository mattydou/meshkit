"""Mesh ops: extrude, inset, mirror, merge_by_distance, bridge, flip, recalc_normals.
New faces/verts inherit the groups of the face/vert they were made from. Corner UVs follow: kept faces keep theirs,
faces made from a face take its UVs (inset interpolates inside the face; extrude walls get the edge's UVs, zero UV
area), mirrored copies are reflected beside the original; faces from nothing (bridge) have none."""
import numpy as np

from .loops import as_loop, loop_dir, reverse_loop
from .mesh import Builder, Created, Mesh, Sel, edge_key, face_edges, resolve, signed_volume, unit


def _regions(mesh, faces):
    """Split faces into edge-connected regions (lists in input order)."""
    parent = {f: f for f in faces}

    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    fs = set(faces)
    for f in faces:
        for a, b in face_edges(mesh.faces[f]):
            for g in mesh.edge_faces.get(edge_key(a, b), ()):
                if g in fs:
                    ra, rb = root(f), root(g)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    out = {}
    for f in faces:
        out.setdefault(root(f), []).append(f)
    return list(out.values())


def _border(mesh, region):
    """Edges used by exactly one face of the region."""
    cnt = {}
    for f in region:
        for a, b in face_edges(mesh.faces[f]):
            k = edge_key(a, b)
            cnt[k] = cnt.get(k, 0) + 1
    return {k for k, c in cnt.items() if c == 1}


def _uv_jacobian(P, U):
    """(3,2) map from 3D offsets in a face's plane to UV offsets (least squares over its corners)."""
    J, *_ = np.linalg.lstsq(P - P.mean(0), U - U.mean(0), rcond=None)
    return J


def _region_normal(mesh, region):
    n = (mesh.face_normals[region] * mesh.face_areas[region][:, None]).sum(axis=0)
    L = np.linalg.norm(n)
    if L < 1e-12:
        raise ValueError("region has no average normal (closed or cancelling); pass direction=")
    return n / L


def _select_regions(mesh, faces, mode, op):
    F = resolve(mesh, faces, "f")
    if not F:
        raise ValueError(f"{op}: no faces selected")
    if mode == "individual":
        return F, [[f] for f in F]
    if mode == "region":
        return F, _regions(mesh, F)
    raise ValueError(f"{op}: mode must be 'region' or 'individual'")


def extrude(mesh, faces, distance, mode="region", direction=None):
    """Extrude faces by distance along each connected region's average normal (or each face's normal with
    mode='individual', or `direction`). Side walls are added on region borders.
    Created: verts (new), faces (side walls), top (the moved faces; same indices as before)."""
    F, regions = _select_regions(mesh, faces, mode, "extrude")
    b = Builder(mesh)
    new_v, sides = [], []
    for R in regions:
        d = (unit(direction) if direction is not None else _region_normal(mesh, R)) * distance
        Rs, border = set(R), _border(mesh, R)
        border_v = {v for e in border for v in e}
        verts = sorted({v for f in R for v in mesh.faces[f]})
        new_of = {}
        for v in verts:
            shared = any(g not in Rs for g in mesh.vert_faces[v]) or any(v in e for e in mesh.loose_edges)
            if v in border_v or shared:
                new_of[v] = b.vert(mesh.verts[v] + d, src=v)
                new_v.append(new_of[v])
            else:
                b.V[v] = list(mesh.verts[v] + d)
        for f in R:
            q = mesh.faces[f]
            b.F[f] = tuple(new_of.get(v, v) for v in q)
            for a, c in face_edges(q):
                if edge_key(a, c) in border:
                    sides.append(b.face((a, c, new_of[c], new_of[a]), src=f))
    return b.build(), Created(verts=new_v, faces=sides, top=Sel("f", F))


def inset(mesh, faces, thickness, depth=0.0, mode="region"):
    """Inset faces: shrink each region's border inward by thickness (constant width, mitered corners) adding a ring
    of quads; depth then moves the inner faces along the region normal (negative = into the surface).
    Created: verts (inner border verts), faces (ring quads), inner (the inset faces; same indices)."""
    F, regions = _select_regions(mesh, faces, mode, "inset")
    b = Builder(mesh)
    new_v, ring = [], []
    N = mesh.face_normals
    for R in regions:
        border = _border(mesh, R)
        out_e, in_e = {}, {}  # border vert -> (next vert, face), (prev vert, face) in face winding
        for f in R:
            for a, c in face_edges(mesh.faces[f]):
                if edge_key(a, c) in border:
                    if a in out_e or c in in_e:
                        raise ValueError(f"inset: region border touches itself at vert {a if a in out_e else c}; "
                                         "use mode='individual' or split the selection")
                    out_e[a], in_e[c] = (c, f), (a, f)
        lift = _region_normal(mesh, R) * depth if depth else np.zeros(3)
        new_of = {}
        for v in sorted(out_e):
            p = mesh.verts[v]
            (w, fo), (u, fi) = out_e[v], in_e[v]
            d1, d2 = unit(p - mesh.verts[u]), unit(mesh.verts[w] - p)
            i1, i2 = np.cross(N[fi], d1), np.cross(N[fo], d2)
            i1, i2 = i1 / max(np.linalg.norm(i1), 1e-12), i2 / max(np.linalg.norm(i2), 1e-12)
            m = (i1 + i2) / max(1.0 + float(i1 @ i2), 0.1)
            new_of[v] = b.vert(p + thickness * m + lift, src=v)
            new_v.append(new_of[v])
        if depth:
            interior = {v for f in R for v in mesh.faces[f]} - set(new_of)
            for v in interior:
                b.V[v] = list(mesh.verts[v] + lift)
        for f in R:
            q = mesh.faces[f]
            b.F[f] = tuple(new_of.get(v, v) for v in q)
            uv = None
            if b.UV[f] is not None:  # inner corners: the face's own UV mapping at their new spots
                U = np.array(b.UV[f])
                J = _uv_jacobian(mesh.verts[list(q)], U)
                uv = {v: U[i] + (np.array(b.V[new_of[v]]) - mesh.verts[v]) @ J if v in new_of else U[i]
                      for i, v in enumerate(q)}
                b.UV[f] = tuple(tuple(uv[v]) for v in q)
            for a, c in face_edges(q):
                if edge_key(a, c) in border:
                    ring.append(b.face((a, c, new_of[c], new_of[a]), src=f,
                                       uv=None if uv is None else (U[q.index(a)], U[q.index(c)], uv[c], uv[a])))
    return b.build(), Created(verts=new_v, faces=ring, inner=Sel("f", F))


def mirror(mesh, axis="x", sel=None, weld=True, tol=None, suffix=None):
    """Mirror faces (default: the whole mesh, loose edges too) across the plane through the origin normal to axis.
    weld: verts within tol (default 1e-5 x diagonal) of the plane snap onto it and are shared by both halves.
    suffix: mirrored elements go to '<group><suffix>' instead of the original groups. UVs of the copies are mirrored
    to the right of the originals (no overlap, not flipped; outside 0..1 until pack(m)).
    Created: verts, faces (mirrored copies), vmap (vert -> its mirror; -1 = not mirrored)."""
    if axis not in ("x", "y", "z"):
        raise ValueError("mirror axis must be 'x', 'y' or 'z'")
    ax = "xyz".index(axis)
    F = resolve(mesh, sel, "f")
    verts = range(len(mesh.verts)) if sel is None else sorted({v for f in F for v in mesh.faces[f]})
    tol = 1e-5 * mesh.diag if tol is None else tol
    b = Builder(mesh)
    src = None if suffix else True
    vmap = np.full(len(mesh.verts), -1, dtype=np.int64)
    new_v = []
    for v in verts:
        p = mesh.verts[v].copy()
        if weld and abs(p[ax]) <= tol:
            b.V[v][ax] = 0.0
            vmap[v] = v
        else:
            p[ax] = -p[ax]
            vmap[v] = b.vert(p, src=v if src else None)
            new_v.append(int(vmap[v]))
    fmap, new_f = {}, []
    U = [b.UV[f] for f in F if b.UV[f] is not None]
    edge = max((u for uv in U for u, _ in uv if u == u), default=0.0)  # copies go right of the originals
    for f in F:
        q = mesh.faces[f]
        mq = tuple(int(vmap[v]) for v in reversed(q))
        if set(mq) == set(q):
            continue  # face lies on the mirror plane
        uv = None if b.UV[f] is None else [(2 * edge - u, v) for u, v in reversed(b.UV[f])]
        fmap[f] = b.face(mq, src=f if src else None, uv=uv)
        new_f.append(fmap[f])
    if sel is None:
        for a, c in mesh.loose_edges:
            k = edge_key(vmap[a], vmap[c])
            if k[0] != k[1]:
                b.L.add(k)
    if suffix:
        for name, g in mesh.groups.items():
            m = ({int(vmap[i]) for i in g.idx if vmap[i] >= 0} if g.kind == "v"
                 else {fmap[i] for i in g.idx if i in fmap})
            if m:
                b.add_to_group(name + suffix, g.kind, m)
    return b.build(), Created(verts=new_v, faces=new_f, vmap=vmap)


def merge_by_distance(mesh, dist=None, sel=None):
    """Weld verts within dist (default 1e-5 x diagonal) of a lower-index vert, optionally only among sel verts.
    Collapsed and duplicate faces are dropped. Renumbers (groups follow).
    Created: vmap (old -> new), removed (verts merged away), faces_removed."""
    n = len(mesh.verts)
    dist = 1e-5 * mesh.diag if dist is None else float(dist)
    idx = sorted(resolve(mesh, sel, "v"))
    P = mesh.verts
    target = np.arange(n)
    cells = {}
    for v in idx:
        cells.setdefault(tuple(np.floor(P[v] / dist).astype(np.int64)), []).append(v)
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for v in idx:
        if target[v] != v:
            continue
        c = np.floor(P[v] / dist).astype(np.int64)
        for o in offs:
            for w in cells.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), ()):
                if w > v and target[w] == w and np.linalg.norm(P[w] - P[v]) <= dist:
                    target[w] = v
    b = Builder(mesh)
    for g in b.G.values():
        if g[0] == "v":
            g[1] = {int(target[i]) for i in g[1]}
    drop_f, seen = set(), set()
    for fi, f in enumerate(mesh.faces):
        q = [int(target[v]) for v in f]
        keep = [i for i, v in enumerate(q) if v != q[i - 1]] if len(set(q)) > 1 else [0]
        q = [q[i] for i in keep]
        key = frozenset(q)
        if len(q) < 3 or len(key) < len(q) or key in seen:
            drop_f.add(fi)
        else:
            seen.add(key)
            b.F[fi] = tuple(q)
            if b.UV[fi] is not None:
                b.UV[fi] = tuple(b.UV[fi][i] for i in keep)
    b.L = {edge_key(target[a], target[c]) for a, c in mesh.loose_edges if target[a] != target[c]}
    drop_v = {v for v in range(n) if target[v] != v}
    m, vmap, _ = b.build_compact(drop_v, drop_f)
    vmap = vmap[target]
    return m, Created(vmap=vmap, removed=len(drop_v), faces_removed=len(drop_f))


def _align(P, A, B, closed, allow_reverse):
    """B reordered (cyclic offset and, if allowed, direction) to best match A's directions from its centroid."""
    def dirs(L):
        D = P[L] - P[L].mean(axis=0)
        return D / np.maximum(np.linalg.norm(D, axis=1), 1e-12)[:, None]
    dA, best = dirs(A), None
    for cand in [B] + ([reverse_loop(B, closed)] if allow_reverse else []):
        dB = dirs(cand)
        for o in range(len(B)) if closed else [0]:
            cost = float(((dA - np.roll(dB, -o, axis=0)) ** 2).sum())
            if best is None or cost < best[0] - 1e-9:
                best = (cost, cand[o:] + cand[:o])
    return best[1]


def bridge(mesh, loop_a, loop_b, segments=1):
    """Join two edge loops with equal vertex counts by a band of quads (segments rows). Loops: vert Sel/list, edges or
    a group. Winding follows faces next to either loop; the pairing (cyclic offset) is chosen geometrically.
    Created: verts (intermediate rows), faces."""
    A, ca = as_loop(mesh, loop_a)
    B, cb = as_loop(mesh, loop_b)
    if len(A) != len(B):
        raise ValueError(f"bridge: loops have {len(A)} and {len(B)} verts; equal counts required "
                         "(bridging mismatched loops is not supported yet)")
    if ca != cb:
        raise ValueError("bridge: one loop is closed and the other open")
    if set(A) & set(B):
        raise ValueError("bridge: loops share verts")
    da, db = loop_dir(mesh, A, ca), loop_dir(mesh, B, ca)
    if da > 0:
        A = reverse_loop(A, ca)
    if db < 0:
        B = reverse_loop(B, ca)
    P = mesh.verts
    B = _align(P, A, B, ca, allow_reverse=(db == 0))
    if da == 0 and db == 0 and ca:
        # free loops: face the band away from the axis through the loop centroids
        a0, axis = P[A].mean(0), P[B].mean(0) - P[A].mean(0)
        n_out = 0.0
        for i in range(len(A)):
            j = (i + 1) % len(A)
            nrm = np.cross(P[A[j]] - P[A[i]], P[B[i]] - P[A[i]])
            c = (P[A[i]] + P[A[j]] + P[B[i]] + P[B[j]]) / 4 - a0
            radial = c - axis * (c @ axis) / max(axis @ axis, 1e-24)
            n_out += float(nrm @ radial)
        if n_out < 0:
            A, B = reverse_loop(A, ca), reverse_loop(B, ca)
    b = Builder(mesh)
    rows, new_v = [A], []
    for k in range(1, segments):
        t = k / segments
        row = [b.vert((1 - t) * P[x] + t * P[y]) for x, y in zip(A, B)]
        rows.append(row)
        new_v += row
    rows.append(B)
    n = len(A)
    faces = [b.face((R0[i], R0[(i + 1) % n], R1[(i + 1) % n], R1[i]))
             for R0, R1 in zip(rows, rows[1:]) for i in range(n if ca else n - 1)]
    return b.build(), Created(verts=new_v, faces=faces)


def flip(mesh, faces=None):
    """Reverse the winding of faces (default all). Created: faces (flipped)."""
    F = resolve(mesh, faces, "f")
    b = Builder(mesh)
    for f in F:
        q = mesh.faces[f]
        b.F[f] = (q[0],) + q[:0:-1]
        if b.UV[f] is not None:  # corners keep their UVs, so a flipped face reads mirrored in UV space too
            b.UV[f] = (b.UV[f][0],) + b.UV[f][:0:-1]
    return b.build(), Created(faces=F)


def recalc_normals(mesh):
    """Make winding consistent within each connected surface, then point closed surfaces outward
    (positive signed volume). Created: faces (flipped)."""
    nf = len(mesh.faces)
    flipped = [False] * nf
    seen = [False] * nf
    for seed in range(nf):
        if seen[seed]:
            continue
        seen[seed], comp, queue = True, [seed], [seed]
        while queue:
            f = queue.pop()
            q = mesh.faces[f][::-1] if flipped[f] else mesh.faces[f]
            for a, c in face_edges(q):
                fs = mesh.edge_faces.get(edge_key(a, c), ())
                if len(fs) != 2:
                    continue
                g = fs[0] if fs[1] == f else fs[1]
                if seen[g]:
                    continue
                gq = mesh.faces[g]
                k = gq.index(a)
                flipped[g] = gq[(k + 1) % len(gq)] == c  # same direction -> flip g
                seen[g] = True
                comp.append(g)
                queue.append(g)
        closed = all(len(mesh.edge_faces[edge_key(a, c)]) == 2 for f in comp for a, c in face_edges(mesh.faces[f]))
        if closed:
            faces = [mesh.faces[f][::-1] if flipped[f] else mesh.faces[f] for f in comp]
            if signed_volume(mesh.verts, faces) < 0:
                for f in comp:
                    flipped[f] = not flipped[f]
    return flip(mesh, [f for f in range(nf) if flipped[f]])


def _subdivide_once(mesh, smooth):
    V, F, n = mesh.verts, mesh.faces, len(mesh.verts)
    keys = sorted(mesh.edge_faces)
    E = np.array(keys, dtype=np.int64).reshape(-1, 2)
    ne, nf = len(E), len(F)
    cnt = np.array([len(mesh.edge_faces[k]) for k in keys], dtype=np.int64)
    fp = mesh.face_centers
    mid = (V[E[:, 0]] + V[E[:, 1]]) / 2
    ep = mid.copy()
    vp = V.copy()
    if smooth and ne:
        inner = np.flatnonzero(cnt == 2)
        f2 = np.array([mesh.edge_faces[keys[i]] for i in inner], dtype=np.int64).reshape(-1, 2)
        ep[inner] = (V[E[inner, 0]] + V[E[inner, 1]] + fp[f2[:, 0]] + fp[f2[:, 1]]) / 4
        deg, crease, nfv = np.zeros(n), np.zeros(n), np.zeros(n)
        rsum, qsum, bsum = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
        for k in (0, 1):
            np.add.at(deg, E[:, k], 1)
            np.add.at(rsum, E[:, k], mid)
        edge = cnt != 2  # open borders and non-manifold edges are creases
        for k in (0, 1):
            np.add.at(crease, E[edge, k], 1)
        border = cnt == 1
        np.add.at(bsum, E[border, 0], V[E[border, 1]])
        np.add.at(bsum, E[border, 1], V[E[border, 0]])
        nb = np.zeros(n)
        for k in (0, 1):
            np.add.at(nb, E[border, k], 1)
        for fi, f in enumerate(F):
            u = list(dict.fromkeys(f))
            qsum[u] += fp[fi]
            nfv[u] += 1
        fixed = np.zeros(n, dtype=bool)
        fixed[[v for e in mesh.loose_edges for v in e]] = True
        smooth_v = (crease == 0) & (nfv > 0) & ~fixed
        m = nfv[smooth_v][:, None]
        vp[smooth_v] = (qsum[smooth_v] / m + 2 * rsum[smooth_v] / deg[smooth_v][:, None]
                        + (m - 3) * V[smooth_v]) / m
        curve = (crease == 2) & (nb == 2) & ~fixed
        vp[curve] = (6 * V[curve] + bsum[curve]) / 8
    cs, cv = mesh.corner_start, mesh.corner_verts
    nxt = np.arange(len(cv)) + 1
    nxt[cs[1:] - 1] = cs[:-1]
    prv = np.arange(len(cv)) - 1
    prv[cs[:-1]] = cs[1:] - 1
    ekey = E[:, 0] * n + E[:, 1]
    a, b = cv, cv[nxt]
    e_next = np.searchsorted(ekey, np.minimum(a, b) * n + np.maximum(a, b))  # edge from each corner to the next
    e_prev = e_next[prv]
    face_of = np.repeat(np.arange(nf), np.diff(cs))
    quads = np.stack([cv, n + e_next, n + ne + face_of, n + e_prev], 1)
    loose = sorted(mesh.loose_edges)
    L = []
    lv = []
    for k, (x, y) in enumerate(loose):
        w = n + ne + nf + k
        lv.append((V[x] + V[y]) / 2)
        L += [(x, w), (w, y)]
    verts = np.vstack([vp, ep, fp] + ([np.array(lv)] if lv else []))
    uvs = None
    if mesh.uvs is not None:
        U = mesh.uvs
        fu = np.zeros((nf, 2))
        np.add.at(fu, face_of, U)
        fu /= np.diff(cs)[:, None]
        uvs = np.stack([U, (U + U[nxt]) / 2, fu[face_of], (U[prv] + U) / 2], 1).reshape(-1, 2)
    groups = {}
    for name, g in mesh.groups.items():
        if g.kind == "f":
            groups[name] = ("f", [c for f in g.idx for c in range(cs[f], cs[f + 1])])
        else:
            s = g.idx
            groups[name] = ("v", list(s) + [n + i for i, (x, y) in enumerate(keys) if x in s and y in s]
                            + [n + ne + fi for fi, f in enumerate(F) if all(v in s for v in f)]
                            + [n + ne + nf + k for k, (x, y) in enumerate(loose) if x in s and y in s])
    return Mesh(verts, quads, L, groups, uvs)


def subdivide(mesh, levels=1, smooth=True):
    """Catmull-Clark (smooth=True) or a plain split (False): every n-gon becomes n quads through its center and edge
    midpoints. Open borders and non-manifold edges act as creases (border curve rule; corners stay). Original verts
    keep their indices; loose edges are split in two. UVs interpolate linearly per face (corners keep theirs); face
    groups go to the child quads, vert groups to the original verts plus new verts made only from group verts.
    Created: verts (new), faces (all)."""
    m = mesh
    for _ in range(int(levels)):
        m = _subdivide_once(m, smooth)
    return m, Created(verts=range(len(mesh.verts), len(m.verts)), faces=range(len(m.faces)))
