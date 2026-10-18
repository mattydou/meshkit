"""Low-level authoring: verts -> edges -> faces, loop fills, deletion, joining meshes and group edits."""
import numpy as np

from .loops import as_loop, loop_dir, reverse_loop
from .mesh import _KIND_ALIASES, Builder, Created, Sel, edge_key, resolve


def add_verts(mesh, coords):
    """Append verts. Created: verts."""
    b = Builder(mesh)
    new = [b.vert(c) for c in np.asarray(coords, dtype=float).reshape(-1, 3)]
    return b.build(), Created(verts=new)


def add_edges(mesh, edges, closed=False):
    """Add loose (wire) edges from (a, b) pairs, or from a vertex path [v0, v1, ...] (closed=True joins the ends).
    Created: edges, plus loop (vert Sel) for a path."""
    items = list(edges)
    path = bool(items) and isinstance(items[0], (int, np.integer))
    if path:
        vs = [int(v) for v in items]
        pairs = list(zip(vs, vs[1:])) + ([(vs[-1], vs[0])] if closed and len(vs) > 2 else [])
    else:
        pairs = [tuple(e) for e in items]
    resolve(mesh, sorted({v for e in pairs for v in e}), "v")  # range check
    b = Builder(mesh)
    keys = []
    for a, c in pairs:
        if a == c:
            raise ValueError(f"edge ({a}, {c}) connects a vert to itself")
        k = edge_key(a, c)
        b.L.add(k)
        keys.append(k)
    extra = {"loop": Sel("v", vs, closed=closed)} if path else {}
    return b.build(), Created(edges=keys, **extra)


def _no_duplicate(mesh, verts):
    s = set(verts)
    for fi in mesh.vert_faces[verts[0]]:
        if set(mesh.faces[fi]) == s:
            raise ValueError(f"a face on verts {sorted(s)} already exists (face {fi})")


def add_face(mesh, verts):
    """Add one face through verts in the given winding order (counter-clockwise seen from the front). Created: faces."""
    vs = [int(v) for v in verts]
    if len(vs) < 3 or len(set(vs)) != len(vs):
        raise ValueError(f"add_face needs >= 3 distinct verts, got {vs}")
    resolve(mesh, vs, "v")
    _no_duplicate(mesh, vs)
    b = Builder(mesh)
    f = b.face(vs)
    return b.build(), Created(faces=[f])


def fill(mesh, loop):
    """One n-gon from a closed loop (vert Sel/list, edges or group). Winding follows adjacent faces,
    else the loop order. Created: faces."""
    L, _ = as_loop(mesh, loop)
    if len(L) < 3:
        raise ValueError("fill needs a loop of >= 3 verts")
    if loop_dir(mesh, L) > 0:
        L = reverse_loop(L)
    _no_duplicate(mesh, L)
    b = Builder(mesh)
    f = b.face(L)
    return b.build(), Created(faces=[f])


def _grid_sides(P, span, offset):
    """(first corner position, edges on sides 0/2, edges on sides 1/3) for grid_fill."""
    n = len(P)
    if span is not None:
        a = int(span)
        if a < 1 or n // 2 - a < 1:
            raise ValueError(f"grid_fill span must be 1..{n // 2 - 1} for a {n}-vert loop")
        return (0 if offset is None else int(offset) % n), a, n // 2 - a
    d_in = P - np.roll(P, 1, axis=0)
    d_out = np.roll(P, -1, axis=0) - P
    d_in /= np.maximum(np.linalg.norm(d_in, axis=1), 1e-300)[:, None]
    d_out /= np.maximum(np.linalg.norm(d_out, axis=1), 1e-300)[:, None]
    turn = np.arccos(np.clip((d_in * d_out).sum(axis=1), -1, 1))
    order = sorted(range(n), key=lambda i: (-round(float(turn[i]), 6), i))
    if offset is None:
        c = sorted(order[:4])
        s = [c[1] - c[0], c[2] - c[1], c[3] - c[2], c[0] + n - c[3]]
        if s[0] == s[2] and s[1] == s[3]:
            return c[0], s[0], s[1]
    a = max(1, n // 4)
    return (order[0] if offset is None else int(offset) % n), a, n // 2 - a


def grid_fill(mesh, loop, span=None, offset=None):
    """Fill a closed loop with an even vert count by a grid of quads (Coons patch). Corners = the 4 sharpest turns
    when they give equal opposite sides, else span (edges on the first side, default n//4) from loop position
    offset. Winding follows adjacent faces, else the loop order. Created: verts (interior), faces."""
    L, _ = as_loop(mesh, loop)
    n = len(L)
    if n < 4 or n % 2:
        raise ValueError(f"grid_fill needs an even loop of >= 4 verts, got {n}; use fill() for an n-gon")
    if loop_dir(mesh, L) > 0:
        L = reverse_loop(L)
    o, a, b_ = _grid_sides(mesh.verts[L], span, offset)
    L = L[o:] + L[:o]
    k1, k3 = a, 2 * a + b_
    grid = {}
    for i in range(a + 1):
        grid[i, 0] = L[i]
        grid[i, b_] = L[(k3 - i) % n]
    for j in range(b_ + 1):
        grid[a, j] = L[k1 + j]
        grid[0, j] = L[(n - j) % n]
    V = mesh.verts
    X = {k: V[v] for k, v in grid.items()}
    b = Builder(mesh)
    new = []
    for j in range(1, b_):
        for i in range(1, a):
            u, v = i / a, j / b_
            p = ((1 - v) * X[i, 0] + v * X[i, b_] + (1 - u) * X[0, j] + u * X[a, j]
                 - ((1 - u) * (1 - v) * X[0, 0] + u * (1 - v) * X[a, 0] + (1 - u) * v * X[0, b_] + u * v * X[a, b_]))
            grid[i, j] = b.vert(p)
            new.append(grid[i, j])
    faces = [b.face((grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]))
             for j in range(b_) for i in range(a)]
    return b.build(), Created(verts=new, faces=faces)


def _kind_of(mesh, sel, default):
    if isinstance(sel, Sel):
        return sel.kind
    if isinstance(sel, str) and sel in mesh.groups:
        return mesh.groups[sel].kind
    return default


def delete(mesh, sel, kind="f", prune=True):
    """Delete verts, edges or faces (kind comes from a Sel/group, else `kind`). Faces using deleted verts/edges go too;
    verts left unused by the deletion are pruned. Renumbers (groups follow). Created: vmap, fmap (old -> new, -1 gone)."""
    k = _kind_of(mesh, sel, kind)
    idx = resolve(mesh, sel, k)
    b = Builder(mesh)
    drop_v, touched = set(), set()
    if k == "v":
        drop_v = set(idx)
        drop_f = {f for v in idx for f in mesh.vert_faces[v]}
        dead_l = {e for e in mesh.loose_edges if e[0] in drop_v or e[1] in drop_v}
    elif k == "e":
        es = set(idx)
        drop_f = {f for e in es for f in mesh.edge_faces.get(e, ())}
        dead_l = es & mesh.loose_edges
        touched = {v for e in es for v in e}
    else:
        drop_f, dead_l = set(idx), set()
    touched |= {v for f in drop_f for v in mesh.faces[f]} | {v for e in dead_l for v in e}
    b.L -= dead_l
    if prune:
        used = {v for fi, f in enumerate(mesh.faces) if fi not in drop_f for v in f}
        used |= {v for e in b.L for v in e}
        drop_v |= {v for v in touched if v not in used}
    m, vmap, fmap = b.build_compact(drop_v, drop_f)
    return m, Created(vmap=vmap, fmap=fmap)


def join(mesh, *others):
    """Append other meshes (indices offset); same-name groups merge. Created: verts, faces, edges of the appended part."""
    b = Builder(mesh)
    v0, f0 = len(b.V), len(b.F)
    for o in others:
        ov, of = len(b.V), len(b.F)
        b.V.extend(o.verts.tolist())
        b.F.extend(tuple(v + ov for v in f) for f in o.faces)
        b.UV.extend(Builder(o).UV)  # UVs come along as they are (overlapping islands: pack(m) to share a texture)
        b.L |= {(x + ov, y + ov) for x, y in o.loose_edges}
        for name, g in o.groups.items():
            b.add_to_group(name, g.kind, (i + (ov if g.kind == "v" else of) for i in g.idx))
    m = b.build()
    return m, Created(verts=range(v0, len(b.V)), faces=range(f0, len(b.F)),
                      edges=[e for e in m.edges if e[0] >= v0])


def add_group(mesh, name, sel, kind=None):
    """Create or replace a group. Kind comes from a Sel/group (edges become verts); plain lists need kind='v'|'f'.
    Created: group (the Sel stored)."""
    if kind is None:
        if isinstance(sel, Sel):
            kind = "v" if sel.kind == "e" else sel.kind
        elif isinstance(sel, str):
            kind = _kind_of(mesh, sel, None)
            if kind is None:
                raise KeyError(f"no group {sel!r}; groups: {sorted(mesh.groups)}")
        if kind is None:
            raise ValueError("add_group: pass kind='v' or 'f' for a plain index list")
    if kind not in _KIND_ALIASES:
        raise ValueError(f"kind must be 'v' or 'f', got {kind!r}")
    kind = _KIND_ALIASES[kind]
    idx = resolve(mesh, sel, kind)
    b = Builder(mesh)
    b.G[str(name)] = [kind, set(idx)]
    return b.build(), Created(group=Sel(kind, sorted(idx)))


def remove_group(mesh, name):
    """Drop a group (geometry untouched)."""
    if name not in mesh.groups:
        raise KeyError(f"no group {name!r}; groups: {sorted(mesh.groups)}")
    b = Builder(mesh)
    del b.G[name]
    return b.build(), Created()
