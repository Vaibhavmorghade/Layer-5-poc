"""Person 6: COCO 2D, COCO 3D and MOT export.

Input ("frames") is a list of
    {frame_id, file_name, width, height, cam, annotations:[ann, ...]}
where ann is the contract annotation dict produced by annotate.annotate_frame
(Person 5) and filled with mask_rle / visible (Persons 4 and 7):
    {frame_id, object_id, class, category_id, layer, action,
     bbox_2d:[x,y,w,h], bbox_3d:{center,size,yaw[,corners]},
     mask_rle:{size,counts}|None, visible:bool|None, reid?}

API:
    export_coco(frames, out_path)                    -> coco_2d.json
    export_coco3d(frames, tracks, out_path, m=None)  -> coco_3d.json
    export_mot(frames, out_dir)                      -> mot/gt/gt.txt + mot/seqinfo.ini
    export_all(frames, out_dir, m=None, tracks=None) -> all three in one call
    build_frames(traj_dir, m)                        -> frames from files on disk
CLI:
    python export.py out/orbit            # builds frames from out/orbit/ and exports

Identifiers are shared across the three files:
    image id      = frame_id (0-based, same as the rgb/000000.png file name)
    annotation id = running counter, identical in coco_2d.json and coco_3d.json
    object_id, category_id and track_id (= object_id) are copied verbatim.
MOT frame numbers are 1-based, so MOT frame = frame_id + 1.
"""
import argparse
import json
import os

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "object"},
    {"id": 2, "name": "car", "supercategory": "object"},
    {"id": 3, "name": "building", "supercategory": "object"},
    {"id": 4, "name": "sign", "supercategory": "object"},
]
FPS = 10


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _is_visible(ann):
    """visible flag; None (nobody filled it in) is treated as visible."""
    v = ann.get("visible")
    return True if v is None else bool(v)


def _iter_anns(frames):
    """Yield (ann_id, frame, ann) with a deterministic running id.

    Both COCO files use this, so annotation ids match between them.
    Annotations without a 2D box are skipped (nothing to label).
    """
    ann_id = 0
    for fr in sorted(frames, key=lambda f: f["frame_id"]):
        for a in fr.get("annotations", []):
            if a.get("bbox_2d") is None:
                continue
            ann_id += 1
            yield ann_id, fr, a


def _write_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def _images(frames):
    return [{"id": int(f["frame_id"]), "file_name": f["file_name"],
             "width": int(f["width"]), "height": int(f["height"])}
            for f in sorted(frames, key=lambda f: f["frame_id"])]


def _rle_area(rle, bbox):
    """Mask area in pixels (pycocotools if present), else bbox area."""
    try:
        from pycocotools import mask as mu
        r = {"size": rle["size"], "counts": rle["counts"].encode("ascii")
             if isinstance(rle["counts"], str) else rle["counts"]}
        return float(mu.area(r))
    except Exception:
        return float(bbox[2] * bbox[3])


# --------------------------------------------------------------------------
# COCO 2D
# --------------------------------------------------------------------------
def export_coco(frames, out_path):
    """Write coco_2d.json: images, annotations (bbox + RLE mask), categories.

    Invisible objects are kept (attributes.visible = False) with an empty
    segmentation, as the plan asks.
    """
    anns = []
    for ann_id, fr, a in _iter_anns(frames):
        bbox = [float(v) for v in a["bbox_2d"]]
        rle = a.get("mask_rle")
        if rle:
            seg, area = rle, _rle_area(rle, bbox)
        else:
            seg, area = [], float(bbox[2] * bbox[3])
        attrs = {
            "object_id": int(a["object_id"]),
            "track_id": int(a["object_id"]),
            "action": a.get("action"),
            "visible": _is_visible(a),
        }
        if a.get("reid") is not None:
            attrs["reid"] = a["reid"]
        anns.append({
            "id": ann_id,
            "image_id": int(fr["frame_id"]),
            "category_id": int(a["category_id"]),
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "segmentation": seg,
            "attributes": attrs,
        })
    coco = {"images": _images(frames), "annotations": anns, "categories": CATEGORIES}
    return _write_json(coco, out_path)


# --------------------------------------------------------------------------
# COCO 3D
# --------------------------------------------------------------------------
def _source_nodes(m):
    if not m:
        return {}
    return {int(o["object_id"]): o.get("source_node") for o in m["objects"]}


def _tracks_json(tracks):
    """{object_id: [(frame, center, yaw)]} -> {"object_id": [{frame_id, center, yaw}]}."""
    out = {}
    for oid, pts in (tracks or {}).items():
        out[str(oid)] = [{"frame_id": int(f), "center": [float(c) for c in ctr],
                          "yaw": float(yaw)} for f, ctr, yaw in pts]
    return out


def export_coco3d(frames, tracks, out_path, m=None):
    """Write coco_3d.json.

    Same image_id / annotation id / object_id / category_id as coco_2d.json.
    Each annotation holds bbox_3d, source_node (3D segmentation reference, from
    the manifest `m`) and action. Each image holds its camera K and w2c.
    Top-level tracks_3d comes from loader.tracks_3d(m).
    """
    nodes = _source_nodes(m)
    images = []
    for im, fr in zip(_images(frames), sorted(frames, key=lambda f: f["frame_id"])):
        cam = fr.get("cam") or {}
        im = dict(im, K=cam.get("K"), w2c=cam.get("w2c"))
        images.append(im)
    anns = []
    for ann_id, fr, a in _iter_anns(frames):
        b3 = a.get("bbox_3d") or {}
        bbox_3d = {"center": b3.get("center"), "size": b3.get("size"), "yaw": b3.get("yaw")}
        if b3.get("corners") is not None:
            bbox_3d["corners"] = b3["corners"]
        anns.append({
            "id": ann_id,
            "image_id": int(fr["frame_id"]),
            "object_id": int(a["object_id"]),
            "category_id": int(a["category_id"]),
            "bbox_3d": bbox_3d,
            "source_node": nodes.get(int(a["object_id"])),
            "action": a.get("action"),
            "visible": _is_visible(a),
        })
    coco3d = {"images": images, "annotations": anns, "categories": CATEGORIES,
              "tracks_3d": _tracks_json(tracks)}
    return _write_json(coco3d, out_path)


# --------------------------------------------------------------------------
# MOT
# --------------------------------------------------------------------------
def export_mot(frames, out_dir):
    """Write out_dir/mot/gt/gt.txt and out_dir/mot/seqinfo.ini.

    gt line: frame,id,x,y,w,h,1,class_id,visibility  (frame is 1-based).
    Annotations with visible == False are skipped.
    """
    gt_dir = os.path.join(out_dir, "mot", "gt")
    os.makedirs(gt_dir, exist_ok=True)
    lines = []
    for _, fr, a in _iter_anns(frames):
        if not _is_visible(a):
            continue
        x, y, w, h = a["bbox_2d"]
        lines.append((int(fr["frame_id"]) + 1, int(a["object_id"]),
                      "%d,%d,%.2f,%.2f,%.2f,%.2f,1,%d,1.00" % (
                          int(fr["frame_id"]) + 1, int(a["object_id"]), x, y, w, h,
                          int(a["category_id"]))))
    lines.sort(key=lambda t: (t[0], t[1]))
    gt_path = os.path.join(gt_dir, "gt.txt")
    with open(gt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(l for _, _, l in lines) + ("\n" if lines else ""))

    first = frames[0] if frames else {"width": 0, "height": 0}
    name = os.path.basename(os.path.abspath(out_dir))
    ini = ("[Sequence]\nname=%s\nimDir=../rgb\nframeRate=%d\nseqLength=%d\n"
           "imWidth=%d\nimHeight=%d\nimExt=.png\n"
           % (name, FPS, len(frames), int(first["width"]), int(first["height"])))
    with open(os.path.join(out_dir, "mot", "seqinfo.ini"), "w", encoding="utf-8") as f:
        f.write(ini)
    return gt_path


# --------------------------------------------------------------------------
# One call for Person 7's run_demo.py
# --------------------------------------------------------------------------
def export_all(frames, out_dir, m=None, tracks=None):
    """Write coco_2d.json, coco_3d.json and mot/ into out_dir."""
    if tracks is None and m is not None:
        from loader import tracks_3d
        tracks = tracks_3d(m)
    export_coco(frames, os.path.join(out_dir, "coco_2d.json"))
    export_coco3d(frames, tracks, os.path.join(out_dir, "coco_3d.json"), m)
    export_mot(frames, out_dir)
    return out_dir


# --------------------------------------------------------------------------
# Build frames from files on disk (so Person 6 can run without run_demo.py)
# --------------------------------------------------------------------------
def build_frames(traj_dir, m, include_static=False, with_masks=True):
    """Make the `frames` list for one trajectory folder.

    Needs traj_dir/cams.json. If traj_dir/id/*.png exist, masks are read with
    segment.masks_from_id: mask_rle and visible are filled in from them
    (no mask => visible False). Otherwise visible stays None (= visible).
    """
    from annotate import annotate_frame
    from loader import boxes_at

    with open(os.path.join(traj_dir, "cams.json"), "r", encoding="utf-8") as f:
        cams = json.load(f)
    frames = []
    for cam in cams:
        fid = cam["frame_id"]
        anns = annotate_frame(boxes_at(m, fid), cam, include_static)
        id_png = os.path.join(traj_dir, "id", f"{fid:06d}.png")
        if with_masks and os.path.exists(id_png):
            from segment import masks_from_id, mask_to_rle
            masks = masks_from_id(id_png)
            for a in anns:
                mk = masks.get(a["object_id"])
                a["visible"] = mk is not None
                a["mask_rle"] = mask_to_rle(mk) if mk is not None else None
        frames.append({"frame_id": fid, "file_name": f"rgb/{fid:06d}.png",
                       "width": cam["width"], "height": cam["height"],
                       "cam": cam, "annotations": anns})
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export COCO 2D/3D + MOT for one trajectory folder")
    ap.add_argument("traj_dir", help="e.g. out/orbit (must contain cams.json)")
    ap.add_argument("--manifest", default="scene_manifest.json")
    ap.add_argument("--static", action="store_true", help="include static/semi_static boxes")
    a = ap.parse_args()
    from loader import load_manifest
    m = load_manifest(a.manifest)
    frames = build_frames(a.traj_dir, m, include_static=a.static)
    export_all(frames, a.traj_dir, m)
    n = sum(len(f["annotations"]) for f in frames)
    print(f"exported {len(frames)} frames, {n} annotations to {a.traj_dir}")
