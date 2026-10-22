"""run_demo.py (Person 7): end-to-end Layer 5 pipeline.

For each trajectory (orbit, flyover, lawnmower, follow) it runs, in order:

    cameras.make_trajectory -> render.render_rgb -> segment.render_ids
    -> per frame: loader.boxes_at, annotate.annotate_frame, segment.masks_from_id,
       segment.mask_to_rle, overlays (annotate.draw_2d / draw_3d, segment.draw_masks)
    -> export.export_all (coco_2d.json, coco_3d.json, mot/gt/gt.txt)
    -> drone_path.mp4 (render.make_video, needs ffmpeg)

It only calls the existing modules; it does not re-implement their logic.
Nothing is faked: if a render or export fails the run says so and exits non-zero.

Examples (run from the project folder):
    python run_demo.py --check                       # dependency / file check only
    python run_demo.py -t follow --limit 3           # quick smoke test
    python run_demo.py                               # all four trajectories, 60 frames each
    python run_demo.py -t orbit flyover --out-dir out_test

reid.py and plot3d.py are intentionally not part of this script (on hold).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KINDS = ("orbit", "flyover", "lawnmower", "follow")
SUBDIRS = ("rgb", "id", "overlay2d", "overlay3d", "overlay_mask", "mot")
PY_DEPS = {  # import name -> pip name
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "playwright": "playwright",
    "pycocotools": "pycocotools",
}
THREE_JS = HERE / "viewer" / "node_modules" / "three" / "build" / "three.module.js"
VIEWER_CITY = HERE / "viewer" / "city"

try:  # Windows consoles: never crash on a stray character
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def log(msg=""):
    print(msg, flush=True)


class PipelineError(RuntimeError):
    """A stage failed; the message says which one and why."""


# --------------------------------------------------------------------------
# Preflight: dependencies, input files, viewer assets
# --------------------------------------------------------------------------
def preflight(manifest_path: Path, need_render: bool):
    """Return (problems, warnings). Problems are fatal."""
    problems, warnings = [], []

    missing = [pip for mod, pip in PY_DEPS.items() if importlib.util.find_spec(mod) is None]
    if missing:
        problems.append("Missing Python packages: %s\n      Fix: python -m pip install %s"
                        % (", ".join(missing), " ".join(missing)))

    for name in ("loader.py", "cameras.py", "render.py", "segment.py", "annotate.py",
                 "export.py", "viewer/index.html", "scene.gltf", "scene.bin", "activity.bin"):
        if not (HERE / name).is_file():
            problems.append("Missing project file: %s (is this the repo root?)" % name)
    if not manifest_path.is_file():
        problems.append("Manifest not found: %s" % manifest_path)

    if need_render:
        if not THREE_JS.is_file():
            problems.append("three.js not installed (viewer/node_modules missing).\n"
                            "      Fix: cd viewer && npm install   (then cd ..)")
        if shutil.which("node") is None and not THREE_JS.is_file():
            warnings.append("Node.js/npm not found on PATH; needed once for 'npm install'.")
    if shutil.which("ffmpeg") is None:
        warnings.append("ffmpeg not found on PATH: drone_path.mp4 will be skipped "
                        "(install ffmpeg or pass --no-video).")
    return problems, warnings


def ensure_viewer_assets(manifest_path: Path):
    """Make viewer/city/ contain everything the browser loads.

    The viewer fetches city/scene.gltf (+ its .bin/texture files) and, for the
    id render, city/scene_manifest.json. viewer/city is git-ignored, so this
    copies the files from the repo root (a plain copy: no symlinks, works on
    Windows). The manifest copy is refreshed whenever it differs, so the browser
    always uses the same manifest as Python.
    """
    gltf_path = HERE / "scene.gltf"
    with open(gltf_path, "r", encoding="utf-8") as f:
        gltf = json.load(f)
    rel_files = ["scene.gltf"]
    for item in list(gltf.get("buffers", [])) + list(gltf.get("images", [])):
        uri = item.get("uri")
        if uri and not uri.startswith("data:"):
            rel_files.append(uri.replace("%20", " "))
    absent = [r for r in rel_files if not (HERE / r).is_file()]
    if absent:
        raise PipelineError("scene.gltf references files that are missing: %s%s"
                            % (", ".join(absent[:5]), " ..." if len(absent) > 5 else ""))
    copied = 0
    for rel in rel_files:
        src, dst = HERE / rel, VIEWER_CITY / rel
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    dst = VIEWER_CITY / "scene_manifest.json"
    if not dst.is_file() or dst.read_bytes() != manifest_path.read_bytes():
        VIEWER_CITY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, dst)
        copied += 1
    log("viewer/city ready (%d file(s) copied, %d checked)" % (copied, len(rel_files) + 1))


def validate_manifest(m):
    """Cheap sanity checks on the manifest; returns the set of objects with no source_node."""
    from loader import CATEGORY_IDS
    seen, no_node = set(), set()
    for o in m["objects"]:
        oid = o["object_id"]
        if not isinstance(oid, int) or not 1 <= oid <= 255:
            raise PipelineError("object_id %r must be an integer 1..255 (it is stored in 8 bits)" % (oid,))
        if oid in seen:
            raise PipelineError("duplicate object_id %d in manifest" % oid)
        seen.add(oid)
        if o["class"] not in CATEGORY_IDS:
            raise PipelineError("object %d has unknown class %r" % (oid, o["class"]))
        if o.get("source_node") is None:
            no_node.add(oid)
    return no_node


# --------------------------------------------------------------------------
# Small helpers (PIL based so non-ASCII Windows paths work)
# --------------------------------------------------------------------------
def read_bgr(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])


def save_bgr(path, bgr):
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(str(path))


def reset_dir(path: Path, root: Path):
    """Delete and recreate a generated folder (only ever inside this trajectory's folder)."""
    path, root = path.resolve(), root.resolve()
    if root not in path.parents:
        raise PipelineError("refusing to delete outside the output folder: %s" % path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def check_pngs(cams, folder: Path, label: str):
    """Every camera must have a PNG of exactly width x height, or the stage failed."""
    from PIL import Image
    bad = []
    for cam in cams:
        p = folder / ("%06d.png" % cam["frame_id"])
        if not p.is_file():
            bad.append("%s missing" % p.name)
            continue
        with Image.open(p) as im:
            if im.size != (cam["width"], cam["height"]):
                bad.append("%s is %dx%d, expected %dx%d"
                           % (p.name, im.size[0], im.size[1], cam["width"], cam["height"]))
    if bad:
        raise PipelineError("%s render incomplete (%d problem(s)): %s" % (label, len(bad), "; ".join(bad[:4])))


def mask_inside_box(mask, bbox, tol=5.0):
    import numpy as np
    ys, xs = np.nonzero(mask)
    bx, by, bw, bh = bbox
    return (xs.min() >= bx - tol and ys.min() >= by - tol
            and xs.max() + 1 <= bx + bw + tol and ys.max() + 1 <= by + bh + tol)


# --------------------------------------------------------------------------
# One trajectory
# --------------------------------------------------------------------------
def run_trajectory(kind, m, args, no_node):
    import numpy as np
    from annotate import annotate_frame, draw_2d, draw_3d
    from cameras import make_camera, make_trajectory, save_cams
    from export import export_all
    from loader import boxes_at
    from segment import draw_masks, mask_to_rle, masks_from_id

    t_start = time.time()
    traj_dir = Path(args.out_dir) / kind
    traj_dir.mkdir(parents=True, exist_ok=True)

    # 1. cameras ---------------------------------------------------------
    cams = make_trajectory(kind, 60, m)
    if args.limit:
        cams = cams[: args.limit]
    if args.size:
        cams = [make_camera(c["frame_id"], c["pos"], c["look_at"], args.size[0], args.size[1], c["fov_deg"])
                for c in cams]
    cams_path = save_cams(kind, cams, args.out_dir)
    log("[%s] 1/6 cameras: %d frame(s) -> %s" % (kind, len(cams), cams_path))

    # 2. RGB + ID renders ------------------------------------------------
    rgb_dir, id_dir = traj_dir / "rgb", traj_dir / "id"
    have = lambda d: all((d / ("%06d.png" % c["frame_id"])).is_file() for c in cams)
    if args.reuse_renders and have(rgb_dir) and have(id_dir):
        log("[%s] 2/6 renders: reusing existing rgb/ and id/ (--reuse-renders)" % kind)
    else:
        from render import render_rgb
        from segment import render_ids
        reset_dir(rgb_dir, traj_dir)
        reset_dir(id_dir, traj_dir)
        log("[%s] 2/6 rendering RGB (%d frames; ~1-2 s each on software GL)..." % (kind, len(cams)))
        render_rgb(cams, str(traj_dir))
        log("[%s]     rendering instance IDs..." % kind)
        render_ids(cams, str(traj_dir))
    check_pngs(cams, rgb_dir, "RGB")
    check_pngs(cams, id_dir, "ID")

    # 3. annotations + masks + overlays ------------------------------------
    for sub in ("overlay2d", "overlay3d", "overlay_mask", "mot"):
        reset_dir(traj_dir / sub, traj_dir)
    log("[%s] 3/6 annotating + masks + overlays..." % kind)
    frames = []
    stats = {"annotations": 0, "visible": 0, "invisible": 0, "unknown_visibility": 0,
             "masks_without_box": 0, "mask_outside_box": 0}
    for n, cam in enumerate(cams):
        fid = cam["frame_id"]
        fname = "%06d.png" % fid
        boxes = boxes_at(m, fid)
        by_id = {b["object_id"]: b for b in boxes}
        anns = annotate_frame(boxes, cam, include_static=args.static)
        masks = masks_from_id(str(id_dir / fname))  # {object_id: bool mask}, >= 20 px only

        for a in anns:
            oid = a["object_id"]
            if oid in no_node:  # not present in the ID render: visibility is unknown, not guessed
                a["visible"], a["mask_rle"] = None, None
                stats["unknown_visibility"] += 1
                continue
            mk = masks.get(oid)
            a["visible"] = mk is not None
            a["mask_rle"] = mask_to_rle(mk) if mk is not None else None
            stats["visible" if mk is not None else "invisible"] += 1
            if mk is not None and not mask_inside_box(mk, a["bbox_2d"]):
                stats["mask_outside_box"] += 1
        annotated = {a["object_id"] for a in anns}
        expected = {b["object_id"] for b in boxes if args.static or b["layer"] == "dynamic"}
        stats["masks_without_box"] += sum(1 for oid in masks if oid in expected and oid not in annotated)
        stats["annotations"] += len(anns)

        # overlays: only objects that are not known to be hidden
        shown = [a for a in anns if a["visible"] is not False]
        rgb = read_bgr(rgb_dir / fname)
        save_bgr(traj_dir / "overlay2d" / fname, draw_2d(rgb.copy(), shown))
        im3 = rgb.copy()
        for a in shown:
            draw_3d(im3, by_id[a["object_id"]], cam)
        save_bgr(traj_dir / "overlay3d" / fname, im3)
        draw_masks(rgb[:, :, ::-1], {a["object_id"]: masks[a["object_id"]]
                                     for a in shown if a["object_id"] in masks},
                   traj_dir / "overlay_mask" / fname)

        frames.append({"frame_id": fid, "file_name": "rgb/" + fname,
                       "width": cam["width"], "height": cam["height"],
                       "cam": cam, "annotations": anns})
        if (n + 1) % 10 == 0 or n + 1 == len(cams):
            log("[%s]     %d/%d frames annotated" % (kind, n + 1, len(cams)))

    # 4. exports -----------------------------------------------------------
    log("[%s] 4/6 exporting COCO 2D / COCO 3D / MOT..." % kind)
    export_all(frames, str(traj_dir), m)

    # 5. validation of what was actually written -----------------------------
    log("[%s] 5/6 validating exports..." % kind)
    stats.update(validate_exports(traj_dir, frames))

    # 6. drone path video ------------------------------------------------------
    video = "skipped"
    if args.no_video:
        log("[%s] 6/6 video skipped (--no-video)" % kind)
    elif shutil.which("ffmpeg") is None:
        log("[%s] 6/6 video skipped: ffmpeg not on PATH" % kind)
    else:
        log("[%s] 6/6 drone_path.mp4..." % kind)
        try:
            from render import make_video
            make_video(str(traj_dir))
            video = "drone_path.mp4"
        except Exception as exc:  # report, but the data products are already complete
            video = "FAILED: %s" % exc
            log("[%s]     WARNING: video failed: %s" % (kind, exc))

    stats.update(frames=len(frames), video=video, seconds=round(time.time() - t_start, 1))
    log("[%s] done in %.1fs: %d annotations (%d visible, %d hidden, %d unknown), out: %s"
        % (kind, stats["seconds"], stats["annotations"], stats["visible"], stats["invisible"],
           stats["unknown_visibility"], traj_dir))
    if stats["mask_outside_box"] or stats["masks_without_box"]:
        log("[%s]     NOTE: %d mask(s) extend >5px outside their projected 2D box and %d mask(s) have no 2D box;"
            " check camera/ID alignment (see README troubleshooting)."
            % (kind, stats["mask_outside_box"], stats["masks_without_box"]))
    return stats


def validate_exports(traj_dir: Path, frames):
    """Re-read the written files and check the cross-file invariants. Raises on failure."""
    from pycocotools.coco import COCO

    c2, c3 = traj_dir / "coco_2d.json", traj_dir / "coco_3d.json"
    gt = traj_dir / "mot" / "gt" / "gt.txt"
    for p in (c2, c3, gt, traj_dir / "mot" / "seqinfo.ini"):
        if not p.is_file():
            raise PipelineError("expected export missing: %s" % p)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(c2))  # raises if the file is not valid COCO
    with open(c3, "r", encoding="utf-8") as f:
        d3 = json.load(f)
    ids2 = sorted(coco.anns)
    ids3 = sorted(a["id"] for a in d3["annotations"])
    if ids2 != ids3:
        raise PipelineError("coco_2d.json and coco_3d.json annotation ids differ")
    if len(coco.imgs) != len(frames) or len(d3["images"]) != len(frames):
        raise PipelineError("image count in COCO files != number of frames")
    n_visible = sum(1 for fr in frames for a in fr["annotations"] if a["visible"] is not False)
    with open(gt, "r", encoding="utf-8") as f:
        n_mot = sum(1 for line in f if line.strip())
    if n_mot != n_visible:
        raise PipelineError("MOT lines (%d) != visible annotations (%d)" % (n_mot, n_visible))
    n_rle = sum(1 for a in coco.anns.values() if isinstance(a.get("segmentation"), dict))
    return {"coco_annotations": len(ids2), "mot_lines": n_mot, "annotations_with_mask": n_rle}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w <= 0 or h <= 0 or w % 2 or h % 2:
            raise ValueError
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT with even numbers, e.g. 960x600")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Run the Layer 5 pipeline (cameras, RGB, masks, 2D/3D boxes, overlays, COCO/MOT).")
    ap.add_argument("-t", "--trajectory", nargs="+", choices=KINDS + ("all",), default=["all"],
                    metavar="KIND", help="orbit flyover lawnmower follow, or all (default: all)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only process the first N frames of each trajectory (default: all 60)")
    ap.add_argument("--out-dir", default="out", help="output folder (default: out)")
    ap.add_argument("--manifest", default=str(HERE / "scene_manifest.json"),
                    help="manifest JSON (default: scene_manifest.json next to this script)")
    ap.add_argument("--size", type=parse_size, default=None, metavar="WxH",
                    help="override render size, e.g. 960x600 (cameras K/w2c are rebuilt to match)")
    ap.add_argument("--static", action="store_true", help="also annotate static/semi_static objects")
    ap.add_argument("--reuse-renders", action="store_true",
                    help="skip rendering if rgb/ and id/ PNGs already exist for the frames")
    ap.add_argument("--no-video", action="store_true", help="do not make drone_path.mp4")
    ap.add_argument("--check", action="store_true", help="check dependencies and input files, then exit")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        log("ERROR: --limit must be >= 1")
        return 2
    kinds = list(KINDS) if "all" in args.trajectory else list(dict.fromkeys(args.trajectory))
    manifest_path = Path(args.manifest).resolve()
    args.out_dir = str(Path(args.out_dir).resolve())

    log("Layer 5 demo | trajectories: %s | limit: %s | out: %s"
        % (", ".join(kinds), args.limit or "all", args.out_dir))
    problems, warnings = preflight(manifest_path, need_render=True)
    for w in warnings:
        log("WARNING: " + w)
    if problems:
        log("\nCannot start. Fix these first:")
        for p in problems:
            log("  - " + p)
        return 2

    try:
        from loader import load_manifest
        m = load_manifest(str(manifest_path))
        no_node = validate_manifest(m)
        ensure_viewer_assets(manifest_path)
    except PipelineError as exc:
        log("ERROR: %s" % exc)
        return 2
    n_dyn = sum(1 for o in m["objects"] if o["layer"] == "dynamic")
    log("manifest OK: %d objects (%d dynamic)%s" % (
        len(m["objects"]), n_dyn,
        "; no source_node for object(s) %s -> visibility unknown for them" % sorted(no_node) if no_node else ""))
    if args.check:
        log("Check passed. Chromium is only verified when rendering starts "
            "(if it is missing run: python -m playwright install chromium).")
        return 0

    results, failures = {}, {}
    for kind in kinds:
        try:
            results[kind] = run_trajectory(kind, m, args, no_node)
        except Exception as exc:  # report honestly, keep going with the others
            failures[kind] = str(exc)
            log("[%s] FAILED: %s" % (kind, exc))
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                log("       Hint: python -m playwright install chromium")

    summary_path = Path(args.out_dir) / "run_summary.json"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "failures": failures}, f, indent=2)

    log("\n===== SUMMARY =====")
    for kind in kinds:
        if kind in results:
            r = results[kind]
            log("OK      %-10s %3d frames, %4d annotations, %4d MOT lines, video: %s"
                % (kind, r["frames"], r["coco_annotations"], r["mot_lines"], r["video"]))
        else:
            log("FAILED  %-10s %s" % (kind, failures[kind]))
    log("summary written to %s" % summary_path)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
