"""Internal helpers for ordered vertex loops (used by fill, grid_fill, bridge and the loop queries)."""
import numpy as np

from .mesh import Sel, edge_key, resolve


def chain_edges(edges):
    """Order undirected edges into one vertex path. Returns (verts, closed); raises unless they form one chain."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj:
        raise ValueError("empty loop")
    ends = sorted(v for v, n in adj.items() if len(n) == 1)
    if any(len(n) > 2 for n in adj.values()) or len(ends) not in (0, 2):
        raise ValueError("edges do not form a single loop or path (a vert has more than 2 loop edges)")
    closed = not ends
    start = min(adj) if closed else ends[0]
    path, seen, cur = [start], {start}, start
    while len(path) < len(adj):
        nxt = [w for w in sorted(adj[cur]) if w not in seen]
        if not nxt:
            break
        cur = nxt[0]
        path.append(cur)
        seen.add(cur)
    if len(path) != len(adj):
        raise ValueError("edges form more than one loop")
    return path, closed


def as_loop(mesh, sel):
    """Ordered vertex loop from a vert Sel (order and .closed kept), a plain list of verts (closed),
    an edge Sel / list of (a, b) pairs, or a group name. Returns (verts, closed)."""
    if isinstance(sel, Sel) and sel.kind == "v":
        verts, closed = list(sel), sel.closed
    elif isinstance(sel, (Sel, str)):
        verts, closed = chain_edges(resolve(mesh, sel, "e"))
    else:
        items = list(sel)
        if items and isinstance(items[0], (int, np.integer)):
            verts, closed = [int(v) for v in items], True
        else:
            verts, closed = chain_edges(resolve(mesh, items, "e"))
    if len(set(verts)) != len(verts):
        raise ValueError("loop repeats a vertex")
    resolve(mesh, verts, "v")  # range check
    return verts, closed


def loop_edges(loop, closed):
    n = len(loop)
    return [(loop[i], loop[(i + 1) % n]) for i in range(n if closed else n - 1)]


def loop_dir(mesh, loop, closed=True):
    """+1 if faces next to the loop traverse its edges in loop order, -1 if opposite, 0 if no adjacent faces."""
    votes = 0
    for a, b in loop_edges(loop, closed):
        for fi in mesh.edge_faces.get(edge_key(a, b), ()):
            f = mesh.faces[fi]
            k = f.index(a)
            votes += 1 if f[(k + 1) % len(f)] == b else -1
    return (votes > 0) - (votes < 0)


def reverse_loop(loop, closed=True):
    """Reverse direction; a closed loop keeps its first vertex."""
    return [loop[0]] + loop[:0:-1] if closed else loop[::-1]
