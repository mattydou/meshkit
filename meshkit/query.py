"""Queries: return typed selections (Sel), never mutate. Every op accepts these results directly."""
import numpy as np

from .loops import loop_dir, reverse_loop
from .mesh import Sel, edge_key, resolve, unit, vec


def _one_edge(mesh, edge):
    e = edge_key(*(edge[0] if isinstance(edge, Sel) and edge.kind == "e" else edge))
    if e not in mesh.edge_faces and e not in mesh.loose_edges:
        raise IndexError(f"edge {e} is not in the mesh")
    return e


def boundary_loops(mesh):
    """Closed vertex loops along boundary edges (edges with one face), oriented like their faces, sorted by first vert."""
    adj = {}
    for (a, b), fs in mesh.edge_faces.items():
        if len(fs) == 1:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    used, loops = set(), []
    for s in sorted(adj):
        for t in sorted(adj[s]):
            if edge_key(s, t) in used:
                continue
            loop, cur, closed = [s], t, False
            used.add(edge_key(s, t))
            while True:
                if cur == s:
                    closed = True
                    break
                loop.append(cur)
                nxt = [w for w in sorted(adj[cur]) if edge_key(cur, w) not in used]
                if not nxt:
                    break
                used.add(edge_key(cur, nxt[0]))
                cur = nxt[0]
            if loop_dir(mesh, loop, closed) < 0:
                loop = reverse_loop(loop, closed)
            loops.append(Sel("v", loop, closed=closed))
    return loops


def _quad_step(mesh, u, v):
    nb, fs = mesh.vert_neighbors[v], mesh.vert_faces[v]
    if len(nb) != 4 or len(fs) != 4 or any(len(mesh.faces[f]) != 4 for f in fs):
        return None
    fu = set(mesh.edge_faces.get(edge_key(u, v), ()))
    c = [w for w in nb if w != u and not fu & set(mesh.edge_faces.get(edge_key(v, w), ()))]
    return c[0] if len(c) == 1 else None


def _boundary_step(mesh, u, v):
    c = [w for w in mesh.vert_neighbors[v] if w != u and len(mesh.edge_faces.get(edge_key(v, w), ())) == 1]
    return c[0] if len(c) == 1 else None


def _wire_step(mesh, u, v):
    c = [w for w in mesh.vert_neighbors[v] if w != u and edge_key(v, w) in mesh.loose_edges]
    return c[0] if len(c) == 1 else None


def _walk(mesh, start, prev, cur, step, seen):
    path = [cur]
    while True:
        nxt = step(mesh, prev, cur)
        if nxt == start:
            return path, True
        if nxt is None or nxt in seen:
            return path, False
        seen.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt


def edge_loop(mesh, edge):
    """Vertex loop through edge: straight across valence-4 quad verts, along the boundary for boundary edges,
    along the wire for loose edges. Returns a vert Sel with .closed."""
    a, b = _one_edge(mesh, edge)
    nf = len(mesh.edge_faces.get((a, b), ()))
    step = _wire_step if nf == 0 else _boundary_step if nf == 1 else _quad_step
    seen = {a, b}
    fwd, closed = _walk(mesh, a, a, b, step, seen)
    if closed:
        return Sel("v", [a] + fwd, closed=True)
    back, _ = _walk(mesh, b, b, a, step, seen)
    return Sel("v", back[::-1] + fwd, closed=False)


def edge_ring(mesh, edge):
    """Edges opposite each other across a strip of quads, starting from edge. Returns an edge Sel with .closed."""
    e0 = _one_edge(mesh, edge)
    seen = {e0}

    def walk(f):
        out, e = [], e0
        while f is not None and len(mesh.faces[f]) == 4:
            q = mesh.faces[f]
            k = next(i for i in range(4) if edge_key(q[i], q[(i + 1) % 4]) == e)
            opp = edge_key(q[(k + 2) % 4], q[(k + 3) % 4])
            if opp == e0:
                return out, True
            if opp in seen:
                break
            seen.add(opp)
            out.append(opp)
            nxt = [g for g in mesh.edge_faces[opp] if g != f]
            f, e = (nxt[0] if len(nxt) == 1 else None), opp
        return out, False

    fs = mesh.edge_faces.get(e0, ())
    fwd, closed = walk(fs[0]) if fs else ([], False)
    if closed:
        return Sel("e", [e0] + fwd, closed=True)
    back = walk(fs[1])[0] if len(fs) == 2 else []
    return Sel("e", back[::-1] + [e0] + fwd)


def _points(mesh, kind):
    if kind not in ("v", "f"):
        raise ValueError("kind must be 'v' (verts) or 'f' (faces, by center)")
    return mesh.verts if kind == "v" else mesh.face_centers


def select_by_bbox(mesh, lo=None, hi=None, kind="v"):
    """Verts (or faces by center, kind='f') inside the box lo..hi (inclusive); scalars broadcast; None or a None
    entry = unbounded."""
    P = _points(mesh, kind)
    def bound(b, inf):
        b = (None,) * 3 if b is None else ((b,) * 3 if np.ndim(b) == 0 else tuple(b))
        return np.array([inf if x is None else x for x in b], float)
    lo, hi = bound(lo, -np.inf), bound(hi, np.inf)
    eps = 1e-9 * mesh.diag
    return Sel(kind, np.nonzero(np.all((P >= lo - eps) & (P <= hi + eps), axis=1))[0])


def select_by_normal(mesh, direction, angle=30.0):
    """Faces whose normal is within angle (degrees) of direction ('z', '-x' or a vector)."""
    d = unit(direction)
    dots = mesh.face_normals @ d
    return Sel("f", np.nonzero(dots >= np.cos(np.radians(angle)) - 1e-12)[0])


def select_by_plane(mesh, point, normal, side="+", kind="v", tol=None):
    """Verts (or faces by center) on side '+' (along normal), '-' or 'on' of the plane through point.
    tol defaults to 1e-6 x bbox diagonal."""
    d = (_points(mesh, kind) - vec(point)) @ unit(normal)
    tol = 1e-6 * mesh.diag if tol is None else tol
    masks = {"+": d > tol, "-": d < -tol, "on": np.abs(d) <= tol}
    if side not in masks:
        raise ValueError("side must be '+', '-' or 'on'")
    return Sel(kind, np.nonzero(masks[side])[0])


def nearest_verts(mesh, point, k=1, radius=None):
    """The k verts nearest point (closest first, ties by index); radius limits the distance."""
    d = np.linalg.norm(mesh.verts - vec(point), axis=1)
    order = np.argsort(d, kind="stable")[:k]
    if radius is not None:
        order = order[d[order] <= radius]
    return Sel("v", order)


def bbox(mesh, sel=None):
    """(lo, hi) corners of the verts named by sel (default all)."""
    P = mesh.verts[resolve(mesh, sel, "v")]
    if len(P) == 0:
        raise ValueError("bbox of an empty selection")
    return P.min(axis=0), P.max(axis=0)


def centroid(mesh, sel=None):
    """Mean position of the verts named by sel (default all)."""
    P = mesh.verts[resolve(mesh, sel, "v")]
    if len(P) == 0:
        raise ValueError("centroid of an empty selection")
    return P.mean(axis=0)
