"""The iteration loop: `python -m meshkit run model.py` (or `meshkit run model.py`).

A model script defines `mesh` (a Mesh) or `build() -> Mesh`, optionally `check_opts = {"closed": True}`, and may
register plugin checks/scorers. Setting `sdf = <the SDF>` adds the parts-proximity report (`proximity_opts` are
passed on). Quad-dominant meshes (cage_from_sdf + retopo_project, quad_remesh) also get topo_report (with the
script's sdf for limb and sweep joints). Setting `texture = layers(...)` adds UV checks, bakes the maps (size follows draft / final; `bake_opts`
are passed on) and exports them with the mesh (GLB material, OBJ + MTL, Blender script). The runner prints build time (with to_mesh's share), check, describe and plugin results tersely, writes
exports to <script dir>/out/, and exits 0 (all OK), 1 (failures) or 2 (script error). `--final` meshes SDFs at the
final preset instead of draft; `--render [FOCUS]` writes a PNG contact sheet after the run. A .obj or .json path is
loaded and reported instead of run."""
import argparse
import os
import runpy
import sys
import time
import traceback

from . import sdf as _sdf
from .export import export_all, from_json, load_obj
from .gltf import load_glb
from .handles import find_handles
from .texture import BAKE_PRESETS, Texture, bake
from .mesh import Mesh
from .render import render_views
from .report import check, clear_plugins, describe, fmt, fvec, run_plugins, _checks, _scorers
from .retopo import topo_report


def short_traceback(exc, script):
    """One error line plus the script lines involved (and where in meshkit it failed)."""
    script = os.path.abspath(script)
    lines = [f"ERROR {type(exc).__name__}: {exc}"]
    if isinstance(exc, SyntaxError) and exc.filename and os.path.abspath(exc.filename) == script:
        return "\n".join(lines + [f"  at {os.path.basename(script)}:{exc.lineno}  {(exc.text or '').strip()}"])
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if os.path.abspath(f.filename) == script]
    for f in mine[-3:]:
        lines.append(f"  at {os.path.basename(script)}:{f.lineno}  {(f.line or '').strip()}")
    if frames and frames[-1] not in mine:
        f = frames[-1]
        lines.append(f"  in {f.name}() {os.path.basename(f.filename)}:{f.lineno}")
    return "\n".join(lines)


def _load(script):
    """(mesh, namespace) from a model script, or from a .obj / .json / .glb mesh file."""
    ext = os.path.splitext(script)[1].lower()
    if ext == ".obj":
        return load_obj(script), {}
    if ext == ".glb":
        return load_glb(script), {}
    if ext == ".json":
        return from_json(script), {}
    ns = runpy.run_path(script, run_name="__meshkit__")
    mesh = ns.get("mesh")
    if not isinstance(mesh, Mesh) and callable(ns.get("build")):  # a script that set mesh = build() runs once
        mesh = ns["build"]()
    if not isinstance(mesh, Mesh):
        raise TypeError("script must define `mesh` (a Mesh) or build() -> Mesh "
                        f"(got {type(mesh).__name__}; did you keep the (mesh, created) tuple?)")
    return mesh, ns


def _mesh_time(log):
    if not log:
        return ""
    t = sum(e["seconds"] for e in log)
    ev, dense = sum(e["evals"] for e in log), sum(e["dense"] for e in log)
    n = f" x{len(log)}" if len(log) > 1 else ""
    res = " ".join(sorted({f"{e['res']}({e['cells']})" if e["res"] in _sdf.PRESETS else f"{e['cells']} cells"
                           for e in log}))
    return f"  (to_mesh{n} {t:.2f}s: res {res}, {ev / dense:.0%} of grid evaluated)"


def _focus(text):
    """--render argument: True (whole mesh), 'x,y,z' (a point) or a group name."""
    if not isinstance(text, str):
        return text
    try:
        p = [float(x) for x in text.split(",")]
        return p if len(p) == 3 else text
    except ValueError:
        return text


def run(script, out_dir=None, export=True, out=sys.stdout, res=None, render=None, zoom=None):
    """Run a model script (or report a .obj/.json mesh) and print the terse report. res='final' meshes SDFs at the
    final preset; render=True or a focus (group name / point) writes <out>/<name>[_focus].png. Returns the exit
    code."""
    def say(s):
        print(s, file=out)

    name = os.path.splitext(os.path.basename(script))[0]
    is_file = os.path.splitext(script)[1].lower() in (".obj", ".json", ".glb")
    clear_plugins()
    _sdf._log.clear()
    saved = _sdf._defaults["res"]
    _sdf._defaults["res"] = res or saved
    t0 = time.perf_counter()
    try:
        mesh, ns = _load(script)
    except Exception as e:
        say(f"== {name}  FAILED after {time.perf_counter() - t0:.2f}s")
        say(short_traceback(e, script))
        return 2
    finally:
        _sdf._defaults["res"] = saved
    say(f"== {name}  {'loaded' if is_file else 'built'} in {time.perf_counter() - t0:.2f}s{_mesh_time(_sdf._log)}")
    opts = dict(ns.get("check_opts", {}))
    tex = ns.get("texture") if isinstance(ns.get("texture"), Texture) else None
    if tex is not None:
        opts.setdefault("uvs", True)  # a textured model needs clean UVs
    c = check(mesh, **opts)
    say(str(c))
    if opts.get("expect_genus") is not None and (c.stats["genus"] or 0) > opts["expect_genus"]:
        say(str(find_handles(mesh)))  # where the extra handles are
    say(str(describe(mesh)))
    if len(mesh.faces) and sum(len(f) == 4 for f in mesh.faces) >= 0.5 * len(mesh.faces):
        say(str(topo_report(mesh, ns.get("sdf") if isinstance(ns.get("sdf"), _sdf.SDF) else None)))
    ok = c.ok
    if isinstance(ns.get("sdf"), _sdf.SDF):
        p = _sdf.proximity(ns["sdf"], **ns.get("proximity_opts", {}))
        say(str(p))
        ok = ok and p.ok
    if _checks or _scorers:
        p = run_plugins(mesh)
        say(str(p))
        ok = ok and p.ok
    d = out_dir or os.path.join(os.path.dirname(os.path.abspath(script)), "out")
    baked = None
    if tex is not None and export:
        try:
            bo = dict(ns.get("bake_opts", {}))
            bo.setdefault("size", BAKE_PRESETS[res or saved])
            baked = bake(mesh, tex, out=d, name=name, **bo)
            maps = " ".join(k for k in ("albedo", "roughness", "normal", "height") if k in baked.paths)
            say(f"bake: {baked.size}px {maps} | coverage {baked.coverage:.0%} | {baked.seconds:.2f}s")
        except Exception as e:
            say(f"bake failed: {type(e).__name__}: {e}")
            ok = False
    if export and not is_file:
        say("wrote " + " ".join(_rel(p) for p in export_all(mesh, d, name, bake=baked)
                                if not p.endswith(".png")) + (" + maps" if baked else ""))
    focus = _focus(render)
    if focus is not None and focus is not False:
        os.makedirs(d, exist_ok=True)
        tag = "" if focus is True else f"_{focus}" if isinstance(focus, str) else "_focus"
        try:
            kw = dict(focus=None if focus is True else focus, zoom=zoom)
            paths = [render_views(mesh, os.path.join(d, f"{name}{tag}.png"), **kw)]
            if tex is not None:  # unlit albedo: through the bake's UVs when there is one
                paths.append(render_views(mesh, os.path.join(d, f"{name}{tag}_textured.png"),
                                          textured=baked if baked is not None else tex, **kw))
            what = "" if focus is True else f" focus={focus if isinstance(focus, str) else fvec(focus)}"
            say(f"render: {' '.join(_rel(p) for p in paths)}{what}{f' zoom={fmt(zoom)}' if zoom else ''}")
        except Exception as e:  # a bad focus must not hide the report above
            say(f"render failed: {type(e).__name__}: {e}")
            ok = False
    return 0 if ok else 1


def _rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:  # different drive on Windows
        return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="meshkit", description="Run a meshkit model script and report.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a model script (or report a .obj/.json/.glb): check, describe, plugins, "
                                    "export")
    r.add_argument("script")
    r.add_argument("--out", help="export directory (default: <script dir>/out)")
    r.add_argument("--no-export", action="store_true")
    r.add_argument("--final", action="store_true", help="mesh SDFs at the 'final' preset (default: draft)")
    r.add_argument("--render", nargs="?", const=True, metavar="FOCUS",
                   help="write a PNG contact sheet after the run; FOCUS = group name or x,y,z to zoom in on")
    r.add_argument("--zoom", type=float, help="render magnification (default: fit the focus)")
    a = ap.parse_args(argv)
    return run(a.script, a.out, not a.no_export, res="final" if a.final else None, render=a.render, zoom=a.zoom)
