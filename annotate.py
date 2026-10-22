"""Person 5: 2D box, 3D box and overlays.

API:
    bbox_2d(box, cam)                       -> [x, y, w, h] or None
    draw_2d(img, anns)                      -> img (cv2.rectangle + label)
    draw_3d(img, box, cam, color=None)      -> img (12 projected edges)
    annotate_frame(boxes, cam, include_static=False) -> [annotation dicts]
    save_overlays(rgb_dir, cams_json, m, out_dir, ...)  (CLI below)
"""
import argparse
import os

import cv2
import numpy as np

from cameras import project
from loader import box_corners, boxes_at, load_manifest

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
MIN_AREA = 4.0
COLORS = {  # BGR, one per class
    "person": (0, 200, 0),
    "car": (0, 0, 255),
    "building": (255, 128, 0),
    "sign": (0, 200, 255),
}


def _color(cls):
    return COLORS.get(cls, (255, 255, 255))


def bbox_2d(box, cam):
    """Tight 2D box [x, y, w, h] of the projected 3D box, clipped to the image.

    None if every corner is behind the camera, or the clipped area < 4 px.
    Corners behind the camera are ignored (their uv is NaN).
    """
    uv, depth = project(box_corners(box), cam)
    ok = depth > 0
    if not ok.any():
        return None
    pts = uv[ok]
    W, H = cam["width"], cam["height"]
    x0 = float(np.clip(pts[:, 0].min(), 0, W))
    x1 = float(np.clip(pts[:, 0].max(), 0, W))
    y0 = float(np.clip(pts[:, 1].min(), 0, H))
    y1 = float(np.clip(pts[:, 1].max(), 0, H))
    w, h = x1 - x0, y1 - y0
    if w * h < MIN_AREA:
        return None
    return [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)]


def draw_2d(img, anns, thickness=2):
    """Draw each annotation's bbox_2d with a per-class colour and a label like
    'person #21 walking'."""
    for a in anns:
        bb = a.get("bbox_2d")
        if bb is None:
            continue
        x, y, w, h = bb
        p1, p2 = (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h)))
        col = _color(a["class"])
        cv2.rectangle(img, p1, p2, col, thickness)
        label = f"{a['class']} #{a['object_id']} {a.get('action', '')}".strip()
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = p1[1] - 4 if p1[1] - th - base - 4 >= 0 else p1[1] + th + 4
        cv2.rectangle(img, (p1[0], ty - th - 2), (p1[0] + tw + 2, ty + base), col, -1)
        cv2.putText(img, label, (p1[0] + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_3d(img, box, cam, color=None, thickness=2):
    """Draw the 12 edges of the projected 3D box; skip edges with an end behind
    the camera."""
    uv, depth = project(box_corners(box), cam)
    col = color or _color(box["class"])
    for a, b in EDGES:
        if depth[a] <= 0 or depth[b] <= 0:
            continue
        pa = (int(round(uv[a, 0])), int(round(uv[a, 1])))
        pb = (int(round(uv[b, 0])), int(round(uv[b, 1])))
        cv2.line(img, pa, pb, col, thickness, cv2.LINE_AA)
    return img


def annotate_frame(boxes, cam, include_static=False):
    """Annotation dicts for one frame. Only layer == 'dynamic' by default;
    include_static=True also keeps static / semi_static boxes.

    mask_rle and visible are left None for Persons 4 and 7 to fill in.
    """
    anns = []
    for b in boxes:
        if b["layer"] != "dynamic" and not include_static:
            continue
        bb = bbox_2d(b, cam)
        if bb is None:
            continue
        anns.append({
            "frame_id": cam["frame_id"],
            "object_id": b["object_id"],
            "class": b["class"],
            "category_id": b["category_id"],
            "layer": b["layer"],
            "action": b["action"],
            "bbox_2d": bb,
            "bbox_3d": {"center": b["center"], "size": b["size"], "yaw": b["yaw"],
                        "corners": box_corners(b).round(4).tolist()},
            "mask_rle": None,   # Person 4
            "visible": None,    # Person 7
        })
    return anns


def save_overlays(rgb_dir, cams, m, out_dir, include_static=False, only_id=None):
    """Read rgb_dir/{frame:06d}.png, write out_dir/overlay2d/ and overlay3d/."""
    d2, d3 = (os.path.join(out_dir, n) for n in ("overlay2d", "overlay3d"))
    os.makedirs(d2, exist_ok=True)
    os.makedirs(d3, exist_ok=True)
    all_anns = []
    for cam in cams:
        fid = cam["frame_id"]
        src = os.path.join(rgb_dir, f"{fid:06d}.png")
        img = cv2.imread(src)
        if img is None:
            continue
        boxes = boxes_at(m, fid)
        anns = annotate_frame(boxes, cam, include_static)
        if only_id is not None:
            anns = [a for a in anns if a["object_id"] == only_id]
        draw_2d(img.copy(), anns)
        cv2.imwrite(os.path.join(d2, f"{fid:06d}.png"), draw_2d(img.copy(), anns))
        im3 = img.copy()
        ids = {a["object_id"] for a in anns}
        for b in boxes:
            if b["object_id"] in ids:
                draw_3d(im3, b, cam)
        cv2.imwrite(os.path.join(d3, f"{fid:06d}.png"), im3)
        all_anns.append(anns)
    return all_anns


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cams_json")
    ap.add_argument("--static", action="store_true", help="also draw static boxes")
    ap.add_argument("--only", type=int, default=None, help="only this object_id")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cams = json.load(open(a.cams_json))[: a.limit]
    out = os.path.dirname(os.path.abspath(a.cams_json))
    m = load_manifest("scene_manifest.json")
    res = save_overlays(os.path.join(out, "rgb"), cams, m, out, a.static, a.only)
    print(f"wrote overlays for {len(res)} frames to {out}")
