"""find_handles: where the extra handles are when check() reports a higher genus than expected. Each handle is
located by the shortest non-separating loop around it (tree-cotree loops from several roots, trimmed to cycles);
the surface is then cut along that loop and capped, and the search repeats for the next handle."""
import heapq

import numpy as np

from .mesh import Mesh, edge_key, face_edges


def _dijkstra(n, E, L, roots):
    """(dist, pred) arrays (len(roots), n) of shortest paths over undirected edges E with lengths L."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        G = coo_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        d, p = dijkstra(G, indices=roots, return_predecessors=True)
        return d, p
    except ImportError:
        pass
    adj = [[] for _ in range(n)]
    for (a, b), w in zip(E.tolist(), L.tolist()):
        adj[a].append((b, w))
        adj[b].append((a, w))
    D, P = np.full((len(roots), n), np.inf), np.full((len(roots), n), -9999, dtype=np.int64)
    for k, r in enumerate(roots):
        dist, pred = D[k], P[k]
        dist[r] = 0.0
        heap = [(0.0, r)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            for v, w in adj[u]:
                if du + w < dist[v]:
                    dist[v], pred[v] = du + w, u
                    heapq.heappush(heap, (du + w, v))
    return D, P


def _max_spanning_forest(n, A, B, W):
    """Bool mask of the edges (A[i], B[i]) with weight W[i] in a maximum spanning forest (scipy, else Kruskal)."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        order = np.lexsort((-W, hi, lo))  # one edge per face pair (the heaviest), so coo entries never add up
        first = order[np.r_[True, (np.diff(lo[order]) != 0) | (np.diff(hi[order]) != 0)]]
        cost = (W.max() + 1.0) - W[first]
        T = minimum_spanning_tree(coo_matrix((cost, (lo[first], hi[first])), shape=(n, n))).tocoo()
        chosen = np.minimum(T.row, T.col).astype(np.int64) * n + np.maximum(T.row, T.col)
        keep = np.zeros(len(A), dtype=bool)
        keep[first] = np.isin(lo[first].astype(np.int64) * n + hi[first], chosen)
        return keep
    except ImportError:
        pass
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    keep = np.zeros(len(A), dtype=bool)
    for i in np.argsort(-W, kind="stable").tolist():
        ra, rb = root(int(A[i])), root(int(B[i]))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            keep[i] = True
    return keep


def _shortest_loop(mesh, roots):
    """Shortest non-separating vertex cycle found from the given roots, as (length, [verts]) or None (genus 0)."""
    keys = list(mesh.edge_faces)
    E = np.array(keys, dtype=np.int64).reshape(-1, 2)
    L = np.linalg.norm(mesh.verts[E[:, 0]] - mesh.verts[E[:, 1]], axis=1)
    L = np.maximum(L, 1e-12 * mesh.diag)
    EF = np.array([mesh.edge_faces[k] for k in keys], dtype=np.int64)
    n = len(mesh.verts)
    ekey = E[:, 0] * n + E[:, 1]  # edge_faces keys are (low, high)
    order = np.argsort(ekey)
    D, P = _dijkstra(n, E, L, roots)
    best = None
    for dist, pred in zip(D, P):
        tree = np.zeros(len(E), dtype=bool)
        child = np.flatnonzero(pred >= 0)
        tk = np.minimum(child, pred[child]) * n + np.maximum(child, pred[child])
        tree[order[np.searchsorted(ekey[order], tk)]] = True
        rest = np.flatnonzero(~tree & np.isfinite(dist[E[:, 0]]))
        sigma = dist[E[rest, 0]] + dist[E[rest, 1]] + L[rest]
        cotree = _max_spanning_forest(len(mesh.faces), EF[rest, 0], EF[rest, 1], sigma)
        for i in rest[~cotree].tolist():  # 2g leftover edges, each closing a non-separating loop
            u, w = E[i]
            up, x = {}, int(u)
            while x >= 0:
                up[x] = len(up)
                x = int(pred[x])
            path_w, x = [], int(w)
            while x not in up:
                path_w.append(x)
                x = int(pred[x])
            lca = x
            length = dist[u] + dist[w] + L[i] - 2 * dist[lca]
            if best is None or length < best[0] - 1e-12:
                path_u, x = [], int(u)
                while x != lca:
                    path_u.append(x)
                    x = int(pred[x])
                best = (float(length), [lca] + path_u[::-1] + path_w)  # lca -> ... -> u, w -> ... back to lca
    return best


def _cut_and_cap(mesh, loop):
    """Cut the surface along a closed vertex loop (verts on its right side get copies) and cap both sides with
    n-gons. Returns the new mesh; copies are appended after the existing verts in loop order."""
    k, n = len(loop), len(mesh.verts)
    faces = [list(f) for f in mesh.faces]
    copy = {v: n + i for i, v in enumerate(loop)}
    for i, v in enumerate(loop):
        p, nx = loop[i - 1], loop[(i + 1) % k]
        f = next(g for g in mesh.edge_faces[edge_key(v, nx)]
                 if any(a == v and b == nx for a, b in face_edges(mesh.faces[g])))
        left = set()
        for _ in range(len(mesh.vert_faces[v]) + 1):
            left.add(f)
            q = mesh.faces[f][mesh.faces[f].index(v) - 1]
            if q == p:
                break
            f = next(g for g in mesh.edge_faces[edge_key(q, v)] if g != f)
        else:
            raise ValueError("loop is not a simple cycle on a manifold surface")
        for g in mesh.vert_faces[v]:
            if g not in left:
                faces[g] = [copy[v] if x == v else x for x in faces[g]]
    faces.append(loop[::-1])
    faces.append([copy[v] for v in loop])
    return Mesh(np.vstack([mesh.verts, mesh.verts[loop]]), faces)


def _groups_near(mesh, verts, fgroups):
    """{face group: count of its faces touching verts}."""
    faces = {f for v in verts for f in mesh.vert_faces[v]}
    out = {}
    for name, idx in fgroups:
        c = len(faces & idx)
        if c:
            out[name] = c
    return out


def _rank(counts):
    return sorted(counts, key=lambda n: (-counts[n], n))


def _ring_parts(mesh, cut, loop, vmap, n_before, fgroups):
    """Face groups along the shortest path from the loop's left side round to its right side on the cut surface:
    the parts that close the ring (e.g. a leg fused back into the body)."""
    E = np.array(list(cut.edge_faces), dtype=np.int64).reshape(-1, 2)
    L = np.maximum(np.linalg.norm(cut.verts[E[:, 0]] - cut.verts[E[:, 1]], axis=1), 1e-12 * cut.diag)
    picks = sorted({int(i * len(loop) / 4) for i in range(4)})
    D, P = _dijkstra(len(cut.verts), E, L, [loop[i] for i in picks])
    k = int(np.argmin([D[j, n_before + i] for j, i in enumerate(picks)]))
    x, path = n_before + picks[k], []
    while x >= 0:
        path.append(x)
        x = int(P[k, x])
    vmap = np.r_[vmap, vmap[loop]]
    verts = sorted({int(vmap[v]) for v in path if vmap[v] < len(mesh.verts)})
    return _rank(_groups_near(mesh, verts, fgroups))


def _genus(mesh):
    return (2 - (len(mesh.verts) - len(mesh.edges) + len(mesh.faces))) // 2


def _component(mesh, fis):
    vs = sorted({v for f in fis for v in mesh.faces[f]})
    vmap = {v: i for i, v in enumerate(vs)}
    return Mesh(mesh.verts[vs], [[vmap[v] for v in mesh.faces[f]] for f in fis]), np.array(vs)


def _roots(P, k):
    """k well-spread verts (farthest-point sampling from the lowest-x vert)."""
    k = min(k, len(P))
    out = [int(np.lexsort(P.T[::-1])[0])]
    d = np.linalg.norm(P - P[out[0]], axis=1)
    for _ in range(k - 1):
        out.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[out[-1]], axis=1))
    return out


def find_handles(mesh, roots=24, max_handles=8):
    """Locate each handle (genus) of a closed manifold mesh: per handle, the shortest loop around it found from
    `roots` sample verts, its center ('at'), length and the face groups (parts) it runs through. Open borders are
    capped first. 'ring' lists the groups along the shortest way round the handle (the parts fused into a loop).
    Report; stats['handles'] = [dict(at, length, verts, parts, ring)] (verts index the input mesh)."""
    from .query import boundary_loops
    from .report import Report, check, components, fmt, fvec
    c = check(mesh, contacts=None)
    if c.stats["nonmanifold"] or c.stats["wire"] or c.stats["isolated"]:
        return Report("handles", False, ["non-manifold, wire edges or isolated verts: genus undefined"], {}, "")
    work = mesh
    if c.stats["boundary"]:
        from .author import fill
        for loop in boundary_loops(mesh):
            work, _ = fill(work, loop)
    roots_of, _, _ = components(work)
    by_comp = {}
    for fi, f in enumerate(work.faces):
        by_comp.setdefault(roots_of[f[0]], []).append(fi)
    fgroups = [(name, g.idx) for name, g in sorted(mesh.groups.items()) if g.kind == "f"]
    handles, total = [], 0
    for fis in by_comp.values():
        part, vmap = _component(work, fis)
        g = _genus(part)
        total += g
        for _ in range(g):
            if len(handles) >= max_handles:
                break
            found = _shortest_loop(part, _roots(part.verts, roots))
            if found is None:
                break
            length, loop = found
            orig = vmap[[v for v in loop]]
            orig = np.where(orig < len(mesh.verts), orig, -1)
            vs = [int(v) for v in orig if v >= 0]
            near = _groups_near(mesh, vs, fgroups)
            h = dict(at=part.verts[loop].mean(0), length=length, verts=vs, parts=_rank(near), ring=[])
            handles.append(h)
            try:
                cut = _cut_and_cap(part, loop)
            except (ValueError, StopIteration):
                break
            h["ring"] = _ring_parts(mesh, cut, loop, vmap, len(part.verts), fgroups)
            if _genus(cut) != _genus(part) - 1:
                break
            vmap = np.r_[vmap, vmap[loop]]
            part = cut
    lines = []
    def names(ns, k=4):
        return " ".join(ns[:k]) + (" ..." if len(ns) > k else "") if ns else "-"
    for i, h in enumerate(handles):
        lines.append(f"handle {i}: at {fvec(h['at'])} around {names(h['parts'], 3)} (loop {fmt(h['length'])})"
                     + (f"; ring closes through {names(h['ring'], 6)}" if h["ring"] else ""))
    head = f"genus {total}: {len(handles)} located" + (f" (first {max_handles})" if total > max_handles else "")
    return Report("handles", True, [], {"handles": handles, "genus": total}, head, lines)
