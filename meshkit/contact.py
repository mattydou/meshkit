"""Contacts between parts (connected components) or groups: touch, penetrate (with depth) or gap (with distance),
and face crossings (an edge passing through a face) for self-intersection checks. Samples are verts + face
centers; penetration = samples inside the other closed piece or edges crossing its faces."""
import numpy as np

from .mesh import edge_key, face_edges
from .spatial import closest_point, inside

_E = np.empty(0, dtype=np.int64)


def _units(mesh, by):
    """[(name, faces)] for parts or groups."""
    if by == "parts":
        from .report import components
        roots, _, _ = components(mesh)
        comps = {}
        for fi, f in enumerate(mesh.faces):
            comps.setdefault(roots[f[0]], []).append(fi)
        fgroups = sorted((n, g.idx) for n, g in mesh.groups.items() if g.kind == "f")
        out, used = [], {}
        for k, faces in enumerate(comps.values()):
            best, best_n = None, 0  # dominant face group (first by name on ties) names the part
            for n, idx in fgroups:
                c = sum(f in idx for f in faces)
                if c > best_n:
                    best, best_n = n, c
            if best and best_n * 2 >= len(faces):
                used[best] = used.get(best, -1) + 1
                out.append((f"{best}.{used[best]}", faces))
            else:
                out.append((f"part{k}", faces))
        return out
    names = sorted(n for n, g in mesh.groups.items() if g.kind == "f") if by == "groups" else list(by)
    out = []
    for n in names:
        if n not in mesh.groups:
            raise KeyError(f"no group {n!r}; groups: {sorted(mesh.groups)}")
        g = mesh.groups[n]
        faces = sorted(g.idx) if g.kind == "f" else [fi for fi, f in enumerate(mesh.faces) if set(f) <= g.idx]
        if faces:
            out.append((n, faces))
    return out


def _info(mesh, faces):
    verts = sorted({v for f in faces for v in mesh.faces[f]})
    fs = set(faces)
    edges = sorted({edge_key(a, b) for f in faces for a, b in face_edges(mesh.faces[f]) if a != b})
    closed = all(len([g for g in mesh.edge_faces[e] if g in fs]) == 2 for e in edges)
    P = np.vstack([mesh.verts[verts], mesh.face_centers[faces]])
    return dict(faces=faces, verts=set(verts), edges=np.array(edges).reshape(-1, 2), closed=closed, samples=P,
                lo=mesh.verts[verts].min(0), hi=mesh.verts[verts].max(0))


# ---- face crossings -------------------------------------------------------------
def _grid_keys(lo, hi, s, origin, dims):
    """(cell key, item id) for every grid cell each box overlaps."""
    i0 = np.floor((lo - origin) / s).astype(np.int64)
    d = np.floor((hi - origin) / s).astype(np.int64) - i0 + 1
    cnt = d.prod(1)
    ids = np.repeat(np.arange(len(lo)), cnt)
    loc = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    dd = d[ids]
    c = i0[ids] + np.stack([loc // (dd[:, 1] * dd[:, 2]), (loc // dd[:, 2]) % dd[:, 1], loc % dd[:, 2]], 1)
    return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2], ids


def _seg_tri(a, b, A, B, C, tol):
    """Segment a->b passes through triangle ABC (interior or edges), excluding hits within tol of its ends."""
    d, e1, e2 = b - a, B - A, C - A
    pv = np.cross(d, e2)
    det = (e1 * pv).sum(1)
    L = np.linalg.norm(d, axis=1)
    ok = np.abs(det) > 1e-12 * L * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = a - A
    qv = np.cross(tv, e1)
    u, v, t = (tv * pv).sum(1) * inv, (d * qv).sum(1) * inv, (e2 * qv).sum(1) * inv
    tt = np.maximum(tol / np.maximum(L, 1e-300), 1e-9)
    return ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > tt) & (t < 1 - tt)


def crossings(mesh, edges, faces=None, tol=0.0, skip_shared=False):
    """Edges (k,2 vert pairs) passing through faces (all, or the given face ids), not just touching them (hits
    within tol of an edge end are ignored). skip_shared drops faces that share a vert with the edge.
    Returns (edge rows, face ids) arrays, unique pairs. Uniform-grid broadphase + segment/triangle tests."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    T, src = mesh.triangles
    if faces is not None:
        keep = np.isin(src, np.asarray(list(faces), dtype=np.int64))
        T, src = T[keep], src[keep]
    if not len(edges) or not len(T):
        return _E, _E
    V = mesh.verts
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    tlo, thi = np.minimum(np.minimum(A, B), C) - tol, np.maximum(np.maximum(A, B), C) + tol
    a, b = V[edges[:, 0]], V[edges[:, 1]]
    slo, shi = np.minimum(a, b) - tol, np.maximum(a, b) + tol
    origin = np.minimum(tlo.min(0), slo.min(0))
    top = np.maximum(thi.max(0), shi.max(0))
    s = max(float(np.median(np.concatenate([thi - tlo, shi - slo]).max(1))), 1e-12 * mesh.diag, 1e-300)
    s = max(s, float((top - origin).max()) / 2e5)  # cells per axis stay small enough for int64 keys
    dims = np.floor((top - origin) / s).astype(np.int64) + 1
    tk, ti = _grid_keys(tlo, thi, s, origin, dims)
    o = np.argsort(tk, kind="stable")
    tk, ti = tk[o], ti[o]
    rows, hits = [], []
    step = 20000
    for c0 in range(0, len(edges), step):
        sk, si = _grid_keys(slo[c0:c0 + step], shi[c0:c0 + step], s, origin, dims)
        st = np.searchsorted(tk, sk, "left")
        n = np.searchsorted(tk, sk, "right") - st
        I = np.repeat(si, n) + c0
        J = ti[np.repeat(st, n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))]
        u = np.unique(I * len(T) + J)
        I, J = u // len(T), u % len(T)
        ok = (slo[I] <= thi[J]).all(1) & (tlo[J] <= shi[I]).all(1)
        if skip_shared:
            ok &= ~(T[J] == edges[I, :1]).any(1) & ~(T[J] == edges[I, 1:]).any(1)
        I, J = I[ok], J[ok]
        hit = _seg_tri(a[I], b[I], A[J], B[J], C[J], tol)
        rows.append(I[hit])
        hits.append(src[J[hit]])
    u = np.unique(np.concatenate(rows) * len(mesh.faces) + np.concatenate(hits))
    return u // len(mesh.faces), u % len(mesh.faces)


def self_intersections(mesh, tol=None):
    """Face pairs (a < b) passing through each other: an edge of one crosses the other's interior (faces sharing a
    vert are skipped, so neighbours never count). tol (default 1e-6 x diagonal) ignores end touches. Sorted list."""
    if not mesh.faces:
        return []
    tol = 1e-6 * mesh.diag if tol is None else tol
    keys = list(mesh.edge_faces)
    rows, hit = crossings(mesh, np.array(keys), tol=tol, skip_shared=True)
    pairs = set()
    for r, g in zip(rows.tolist(), hit.tolist()):
        for f in mesh.edge_faces[keys[r]]:
            if f != g:
                pairs.add((min(f, g), max(f, g)))
    return sorted(pairs)


def _crosses(mesh, edges, faces, tol):
    """True if any edge passes through the faces (not just ending on them)."""
    return len(crossings(mesh, edges, faces, tol)[0]) > 0


# ---- contacts -------------------------------------------------------------------
def contacts(mesh, by="parts", tol=None, far=0.1, pairs=None):
    """Contact state for each pair of parts (by='parts'), all face groups (by='groups') or the named groups (a list).
    pairs=[(a, b), ...] checks only those pairs (names of parts, or of groups: all-group names switch by to groups).
    Returns [dict(a, b, state, dist, depth)], state in touch | penetrate | gap | connected (groups sharing verts).
    tol defaults to 1e-4 x diagonal; pairs whose bboxes are > far x diagonal apart are gaps by bbox distance."""
    tol = 1e-4 * mesh.diag if tol is None else tol
    if pairs is not None:
        pairs = [tuple(p) for p in pairs]
        names = {n for p in pairs for n in p}
        if by == "parts" and names and all(n in mesh.groups for n in names):
            by = sorted(names)
    units = dict(_units(mesh, by))
    if pairs is None:
        order = list(units)
        pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    else:
        for n in sorted({n for p in pairs for n in p}):
            if n not in units:
                kind = "part" if by == "parts" else "group"
                raise KeyError(f"no {kind} {n!r}; {kind}s: {' '.join(units)}")
    info = {}

    def get(n):
        if n not in info:
            info[n] = _info(mesh, units[n])
        return info[n]
    out = []
    for na, nb in pairs:
        A, B = get(na), get(nb)
        rec = dict(a=na, b=nb, state="gap", dist=0.0, depth=0.0)
        out.append(rec)
        if A["verts"] & B["verts"]:
            rec["state"] = "connected"
            continue
        box = float(np.linalg.norm(np.maximum(0, np.maximum(A["lo"] - B["hi"], B["lo"] - A["hi"]))))
        if box > far * mesh.diag:
            rec["dist"] = box
            continue
        da = closest_point(mesh, A["samples"], faces=B["faces"]).dists
        db = closest_point(mesh, B["samples"], faces=A["faces"]).dists
        depth = 0.0
        if B["closed"]:
            ins = inside(mesh, A["samples"], faces=B["faces"])
            depth = max(depth, float(da[ins].max()) if ins.any() else 0.0)
        if A["closed"]:
            ins = inside(mesh, B["samples"], faces=A["faces"])
            depth = max(depth, float(db[ins].max()) if ins.any() else 0.0)
        crossing = (depth <= tol and (_crosses(mesh, A["edges"], B["faces"], tol)
                                      or _crosses(mesh, B["edges"], A["faces"], tol)))
        dist = float(min(da.min(), db.min()))
        if depth > tol or crossing:
            rec.update(state="penetrate", depth=depth if depth > tol else 0.0)  # 0 = edges cross, no sample inside
        elif dist <= tol:
            rec["state"] = "touch"
        else:
            rec["dist"] = dist
    return out


def contact_lines(recs, label, per_line=8):
    """A count line, then every pair grouped by state (deepest penetrations and nearest gaps first), wrapped
    per_line pairs per line."""
    from .report import fmt
    lines = [f"contacts({label}): {len(recs)} pairs"]
    for state, key in (("penetrate", lambda r: -r["depth"]), ("touch", None), ("gap", lambda r: r["dist"]),
                       ("connected", None)):
        rs = [r for r in recs if r["state"] == state]
        if key:
            rs.sort(key=key)
        if not rs:
            continue
        items = []
        for r in rs:
            pair = f"{r['a']}-{r['b']}"
            if state == "penetrate":
                pair += f" {fmt(r['depth'])}" if r["depth"] > 0 else " (edges cross)"
            elif state == "gap":
                pair += f" {fmt(r['dist'])}"
            items.append(pair)
        title = {"penetrate": "penetrate (depth)", "gap": "gap (dist)"}.get(state, state)
        for j in range(0, len(items), per_line):
            lines.append((f"  {title} {len(rs)}: " if j == 0 else "    ") + ", ".join(items[j:j + per_line]))
    return lines
