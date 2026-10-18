"""Validation (check), shape summary (describe) and plugin checks/scorers. All output is terse: key numbers and
failures only, never mesh dumps."""
import numpy as np

from .contact import _units, contact_lines, contacts as find_contacts, self_intersections
from .mesh import edge_key, face_edges, signed_volume


def fmt(x):
    """Compact number: 4 significant digits, tiny values -> 0."""
    x = float(x)
    return "0" if abs(x) < 1e-9 else f"{x:.4g}"


def fvec(v):
    return "(" + ",".join(fmt(x) for x in v) + ")"


def _eg(items, n=3):
    items = list(items)
    s = " ".join(str(i).replace(" ", "") for i in items[:n])
    return s + (" ..." if len(items) > n else "")


class Report:
    """Result of check/describe/run_plugins: .ok, .failures (list of str), .stats (dict); str() is the terse text."""

    def __init__(self, title, ok, failures, stats, head="", lines=()):
        self.title, self.ok, self.failures, self.stats = title, ok, list(failures), stats
        self.head, self.lines = head, list(lines)

    def __bool__(self):
        return self.ok

    def __str__(self):
        tag = "" if self.title == "describe" else (" OK " if self.ok else " FAIL ")
        out = [f"{self.title}:{tag} {self.head}".rstrip()]
        out += [f"  - {f}" for f in self.failures] + [f"  {l}" for l in self.lines]
        return "\n".join(out)

    __repr__ = __str__


def components(mesh):
    """Connected components over edges: (root per vert, sorted list of component vert lists, isolated verts)."""
    n = len(mesh.verts)
    parent = list(range(n))

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in mesh.edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    used = {v for e in mesh.edges for v in e} | {v for f in mesh.faces for v in f}
    comps = {}
    for v in sorted(used):
        comps.setdefault(root(v), []).append(v)
    return [root(v) for v in range(n)], list(comps.values()), sorted(set(range(n)) - used)


def check(mesh, closed=False, contacts="auto", contact_tol=None, expect_genus=None, intersect=False, uvs=False):
    """Validate topology/geometry. Failures: non-manifold edges, inconsistent winding, inward normals (closed parts
    with negative signed volume), degenerate faces, duplicate faces, isolated verts; plus boundary edges if closed=True,
    a genus other than expect_genus (genus = handles, from chi; catches parts fused into loops), and with intersect=True
    faces passing through each other (within a part, or between parts; intersect='self' only within a part).
    contacts: 'auto' (parts, when there are 2-16), 'parts', 'groups', a list of group names, a list of (a, b) pairs, or
    None; adds info lines (touch / penetrate depth / gap dist) and stats['contacts'], never failures.
    uvs=True adds UV checks: failures for no / missing UVs, flipped or zero-area UV faces, islands overlapping (each
    other or themselves) and UVs outside 0..1; a 'uv:' line with islands, seams, stretch (worst and p95 of the
    UV-vs-3D area ratio, 1 = ideal) and texture coverage; stats['uv'].
    Returns a Report (truthy when OK)."""
    V, F = mesh.verts, mesh.faces
    ef = mesh.edge_faces
    boundary = [e for e, fs in ef.items() if len(fs) == 1]
    nonman = [e for e, fs in ef.items() if len(fs) > 2]
    seen, bad_wind = {}, []
    for f in F:
        for a, b in face_edges(f):
            k = edge_key(a, b)
            if len(ef.get(k, ())) == 2 and seen.get(k) == (a, b):
                bad_wind.append(k)
            seen[k] = (a, b)
    tol_area = 1e-10 * mesh.diag ** 2
    degen = [i for i, f in enumerate(F) if len(set(f)) < len(f) or mesh.face_areas[i] <= tol_area]
    dup, keys = [], {}
    for i, f in enumerate(F):
        k = frozenset(f)
        if k in keys:
            dup.append(i)
        keys.setdefault(k, i)
    roots, comps, isolated = components(mesh)
    inward = []
    by_comp = {}
    for i, f in enumerate(F):
        by_comp.setdefault(roots[f[0]], []).append(i)
    for ci, (r, fis) in enumerate(sorted(by_comp.items())):
        if all(len(ef[edge_key(a, b)]) == 2 for i in fis for a, b in face_edges(F[i]) if a != b):
            vol = signed_volume(V, [F[i] for i in fis])
            if vol < 0:
                inward.append(f"comp{ci}(vol {fmt(vol)})")
    sizes = [len(f) for f in F]
    stats = dict(V=len(V), E=len(mesh.edges), F=len(F), chi=len(V) - len(mesh.edges) + len(F), comps=len(comps),
                 quads=sizes.count(4), tris=sizes.count(3), ngons=sum(s > 4 for s in sizes),
                 boundary=len(boundary), wire=len(mesh.loose_edges), nonmanifold=len(nonman),
                 bad_winding=len(bad_wind), inward=len(inward), degenerate=len(degen), duplicate=len(dup),
                 isolated=len(isolated))
    fails = []
    if nonman:
        fails.append(f"non-manifold edges: {len(nonman)} e.g. {_eg(sorted(nonman))}")
    if bad_wind:
        fails.append(f"inconsistent winding: {len(bad_wind)} edges e.g. {_eg(sorted(set(bad_wind)))}")
    if inward:
        fails.append(f"inward normals (signed volume < 0): {_eg(inward)}")
    if degen:
        fails.append(f"degenerate faces: {len(degen)} e.g. {_eg(f'f{i}' for i in degen)}")
    if dup:
        fails.append(f"duplicate faces: {len(dup)} e.g. {_eg(f'f{i}' for i in dup)}")
    if isolated:
        fails.append(f"isolated verts: {len(isolated)} e.g. {_eg(f'v{i}' for i in isolated)}")
    if closed and boundary:
        fails.append(f"boundary edges: {len(boundary)} (expected closed) e.g. {_eg(sorted(boundary))}")
    genus = None
    if not nonman and not mesh.loose_edges and not isolated and (not boundary or expect_genus is not None):
        from .query import boundary_loops
        loops = len(boundary_loops(mesh)) if boundary else 0
        genus = (2 * len(by_comp) - stats["chi"] - loops) // 2  # chi = sum over comps of 2 - 2g - loops
    stats["genus"] = genus
    if expect_genus is not None and genus != expect_genus:
        why = ("non-manifold, genus undefined" if genus is None else
               "extra handles: parts fused into a loop? find_handles(m) locates them" if genus > expect_genus else
               "fewer handles than expected: a hole closed up?")
        fails.append(f"genus {genus}, expected {expect_genus} (chi={stats['chi']} comps={stats['comps']}): {why}")
    if intersect:
        fails += _intersections(mesh, roots, intersect, stats)
    uv_line = _uv_checks(mesh, stats, fails) if uvs else None
    g = "" if genus is None else f" genus={genus}"
    head = (f"V={stats['V']} E={stats['E']} F={stats['F']} chi={stats['chi']}{g} comps={stats['comps']} | "
            f"quad={stats['quads']} tri={stats['tris']} ngon={stats['ngons']} | "
            f"boundary={stats['boundary']} wire={stats['wire']}")
    lines = []
    by = "parts" if contacts == "auto" and 2 <= len(by_comp) <= 16 else (None if contacts == "auto" else contacts)
    if by:
        pairs = None
        if not isinstance(by, str) and all(not isinstance(x, str) for x in by):
            by, pairs = "parts", list(by)
        recs = find_contacts(mesh, by, contact_tol, pairs=pairs)
        stats["contacts"] = recs
        if recs:
            lines = contact_lines(recs, "pairs" if pairs else by if isinstance(by, str) else "groups")
    if uv_line:
        lines.insert(0, uv_line)
    return Report("check", not fails, fails, stats, head, lines)


def _uv_checks(mesh, stats, fails):
    """Append UV failures; return the 'uv:' info line."""
    from .uv import uv_stats
    st = uv_stats(mesh)
    stats["uv"] = st
    if not st["has"]:
        fails.append("no UVs: unwrap(m), or build from primitives / sweeps (they come with UVs)")
        return None
    fix = "project_uvs(m, faces=...) or unwrap(m)"
    if st["missing"]:
        fails.append(f"missing UVs: {st['missing']} faces e.g. {_eg(f'f{i}' for i in st['missing_faces'])} ({fix})")
    if st["flipped"]:
        fails.append(f"flipped UV faces (mirrored in UV): {st['flipped']} e.g. "
                     f"{_eg(f'f{i}' for i in st['flipped_faces'])} ({fix})")
    if st["zero"]:
        fails.append(f"zero-area UV faces: {st['zero']} (e.g. extrude walls; {fix})")
    if st["overlap_texels"]:
        top = sorted(st["overlap_pairs"].items(), key=lambda kv: -kv[1])[:4]
        fails.append(f"overlapping UV islands: {st['overlap_texels'] / st['res'] ** 2:.2%} of texels, islands "
                     + ", ".join(f"{a}-{b}" for (a, b), _ in top) + (" ..." if len(st["overlap_pairs"]) > 4 else "")
                     + " (pack(m))")
    if st["self_overlap_texels"]:
        fails.append(f"UV islands folding onto themselves: {st['self_overlap_texels'] / st['res'] ** 2:.2%} of "
                     f"texels ({fix})")
    if st["outside"]:
        fails.append(f"UVs outside 0..1: {st['outside']} faces (pack(m))")
    return (f"uv: islands={st['islands']} seams={st['seams']} stretch worst={fmt(st['stretch_worst'])} "
            f"p95={fmt(st['stretch_p95'])} coverage={st['coverage']:.0%}")


def _intersections(mesh, roots, mode, stats):
    """Failure lines for faces passing through each other, split into within-part and between-part pairs."""
    pairs = self_intersections(mesh)
    name = {}
    for n, faces in _units(mesh, "parts"):
        for f in faces:
            name[f] = n
    within = [(a, b) for a, b in pairs if roots[mesh.faces[a][0]] == roots[mesh.faces[b][0]]]
    between = {}
    for a, b in pairs:
        if roots[mesh.faces[a][0]] != roots[mesh.faces[b][0]]:
            k = tuple(sorted((name[a], name[b])))
            between[k] = between.get(k, 0) + 1
    stats["intersections"] = dict(within=len(within), between=sum(between.values()), pairs=pairs)
    out = []
    if within:
        parts = sorted({name[a] for a, _ in within})
        out.append(f"self-intersections: {len(within)} face pairs in {' '.join(parts)} e.g. "
                   + _eg(f"f{a}xf{b}" for a, b in within))
    if between and mode != "self":
        items = [f"{a}-{b} ({n})" for (a, b), n in sorted(between.items(), key=lambda kv: -kv[1])]
        out.append(f"parts pass through each other (face pairs): {', '.join(items)}")
    return out


def _nearest_dist(Q, P):
    """Distance from each Q row to its nearest P row (scipy's KD-tree if installed, else matmul candidates with the
    exact distance recomputed)."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(P).query(Q)[0]
    except ImportError:
        pass
    out = np.empty(len(Q))
    chunk = max(1, int(4e6 // max(len(P), 1)))
    p2 = (P * P).sum(1)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        j = np.argmin(p2[None, :] - 2 * q @ P.T, axis=1)
        out[i:i + chunk] = np.linalg.norm(q - P[j], axis=1)
    return out


def mirror_error(mesh, axis="x"):
    """p99 distance from each mirrored vert (plane through origin, axis 'x'|'y'|'z') to the nearest vert, as a
    fraction of the bbox diagonal; 0 = symmetric. Large meshes use an evenly strided subset of query verts."""
    axis = "xyz".index(axis) if isinstance(axis, str) else int(axis)
    P = mesh.verts
    if len(P) == 0:
        return 0.0
    Q = P[:: max(1, len(P) // 3000)].copy()
    Q[:, axis] *= -1
    return float(np.percentile(_nearest_dist(Q, P), 99)) / mesh.diag


def _centroid(mesh, kind, idx):
    if kind == "f" and len(idx):
        a = mesh.face_areas[idx]
        if a.sum() > 0:
            return (mesh.face_centers[idx] * a[:, None]).sum(0) / a.sum()
    vs = sorted({v for f in idx for v in mesh.faces[f]}) if kind == "f" else idx
    return mesh.verts[vs].mean(0)


def describe(mesh, groups=None, summary=True):
    """Terse numeric summary: bbox, dims, centroid (area-weighted surface; vert groups use the vertex mean), area,
    volume (closed meshes), mirror error per axis, then groups. summary=True stays under 20 lines whatever the group
    count: one line per group (at most 12) and centroid distances only when <= 5 groups are shown.
    groups = names to show; summary=False shows each group's bbox and every pairwise distance.
    stats hold every group (bbox, centroid, n) and pairwise distances (first 200 groups)."""
    V = mesh.verts
    stats = {}
    if len(V) == 0:
        return Report("describe", True, [], stats, "empty mesh")
    lo, hi = V.min(0), V.max(0)
    cen = _centroid(mesh, "f", list(range(len(mesh.faces)))) if mesh.faces else V.mean(0)
    area = float(mesh.face_areas.sum())
    closed = mesh.faces and all(len(fs) == 2 for fs in mesh.edge_faces.values())
    vol = signed_volume(V, mesh.faces) if closed else None
    mirr = {a: mirror_error(mesh, a) for a in "xyz"}
    stats.update(bbox=(lo, hi), dims=hi - lo, centroid=cen, area=area, volume=vol, mirror=mirr, groups={})
    head = (f"bbox={fvec(lo)}..{fvec(hi)} dims={'x'.join(fmt(d) for d in hi - lo)} centroid={fvec(cen)} "
            f"area={fmt(area)} vol={fmt(vol) if vol is not None else 'open'}")
    lines = ["mirror err/diag: " + " ".join(f"{a}={fmt(e)}" for a, e in mirr.items())]
    for name in sorted(mesh.groups):
        g = mesh.groups[name]
        idx = sorted(g.idx)
        if not idx:
            stats["groups"][name] = dict(kind=g.kind, n=0)
            continue
        vs = idx if g.kind == "v" else sorted({v for f in idx for v in mesh.faces[f]})
        stats["groups"][name] = dict(kind=g.kind, n=len(idx), bbox=(V[vs].min(0), V[vs].max(0)),
                                     centroid=_centroid(mesh, g.kind, idx))
    gs = stats["groups"]
    full = [n for n in gs if gs[n]["n"]][:200]  # pairwise distances for at most 200 groups
    stats["group_dist"] = {(a, b): float(np.linalg.norm(gs[a]["centroid"] - gs[b]["centroid"]))
                           for i, a in enumerate(full) for b in full[i + 1:]}
    names = sorted(gs) if groups is None else [groups] if isinstance(groups, str) else list(groups)
    for n in names:
        if n not in gs:
            raise KeyError(f"no group {n!r}; groups: {sorted(gs)}")
    shown = names[:12] if summary else names
    for n in shown:
        g = gs[n]
        if not g["n"]:
            lines.append(f"group {n}[{g['kind']}0]: empty")
            continue
        glo, ghi = g["bbox"]
        box = "" if summary else f"bbox={fvec(glo)}..{fvec(ghi)} "
        lines.append(f"group {n}[{g['kind']}{g['n']}]: {box}dims={'x'.join(fmt(d) for d in ghi - glo)} "
                     f"c={fvec(g['centroid'])}")
    if len(names) > len(shown):
        rest = names[len(shown):]
        lines.append(f"+{len(rest)} more groups: " + " ".join(rest[:10]) + (" ..." if len(rest) > 10 else "")
                     + " (describe(m, groups=[...]) to see them)")
    ok_names = [n for n in shown if gs[n]["n"]]
    pairs = [(a, b) for i, a in enumerate(ok_names) for b in ok_names[i + 1:]]
    dist = stats["group_dist"]
    if pairs and (not summary or len(ok_names) <= 5):
        def d(a, b):
            return dist.get((a, b), dist.get((b, a), float(np.linalg.norm(gs[a]["centroid"] - gs[b]["centroid"]))))
        items = [f"{a}-{b}={fmt(d(a, b))}" for a, b in pairs]
        for k in range(0, len(items), 8):
            lines.append(("group dist: " if k == 0 else "  ") + " ".join(items[k:k + 8]))
    return Report("describe", True, [], stats, head, lines)


# ---- plugins ------------------------------------------------------------------
_checks, _scorers = [], []


def register_check(fn):
    """Register fn(mesh) -> list of failure strings (empty = pass). Usable as a decorator."""
    _checks.append(fn)
    return fn


def register_scorer(fn):
    """Register fn(mesh) -> (name, float). Usable as a decorator."""
    _scorers.append(fn)
    return fn


def clear_plugins():
    """Forget all registered checks and scorers (the runner does this before each script)."""
    _checks.clear()
    _scorers.clear()


def run_plugins(mesh):
    """Run registered checks and scorers. Report.stats['scores'] = {name: value}; exceptions count as failures."""
    fails, scores = [], {}
    for fn in _checks:
        try:
            fails += [f"{fn.__name__}: {msg}" for msg in (fn(mesh) or [])]
        except Exception as e:  # a broken plugin must not hide the rest of the report
            fails.append(f"{fn.__name__} raised {type(e).__name__}: {e}")
    for fn in _scorers:
        try:
            name, val = fn(mesh)
            scores[str(name)] = float(val)
        except Exception as e:
            fails.append(f"{fn.__name__} raised {type(e).__name__}: {e}")
    head = f"{len(_checks)} checks, {len(_scorers)} scorers"
    if scores:
        head += " | " + " ".join(f"{k}={fmt(v)}" for k, v in scores.items())
    return Report("plugins", not fails, fails, {"scores": scores}, head)
