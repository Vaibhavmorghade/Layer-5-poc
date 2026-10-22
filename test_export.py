"""Person 6 test. Run: python test_export.py

1) Hand-made fake frames (no other module needed).
2) Real boxes/cameras/annotate (follow trajectory) with rectangle masks.
Checks COCO loads, ids match between 2D/3D, MOT line count == visible count.
"""
import json
import os
import tempfile

import numpy as np

from export import export_all, export_coco, export_coco3d, export_mot


def _rect_rle(h, w, bbox):
    """Rectangle mask -> RLE. Uses segment.mask_to_rle (pycocotools) if possible."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = [int(round(v)) for v in bbox]
    mask[y:y + bh, x:x + bw] = 1
    try:
        from segment import mask_to_rle
        return mask_to_rle(mask)
    except Exception:
        return {"size": [h, w], "counts": "fake"}  # pycocotools not installed


def _fake_frames():
    frames = []
    for fid in range(3):
        anns = [
            {"frame_id": fid, "object_id": 20, "class": "car", "category_id": 2,
             "layer": "dynamic", "action": "driving", "bbox_2d": [100.0, 80.0, 60.0, 40.0],
             "bbox_3d": {"center": [65, 98, 16], "size": [4, 1.7, 2], "yaw": 0.0},
             "mask_rle": _rect_rle(800, 1280, [100, 80, 60, 40]), "visible": True},
            {"frame_id": fid, "object_id": 21, "class": "person", "category_id": 1,
             "layer": "dynamic", "action": "walking", "bbox_2d": [300.0, 200.0, 10.0, 25.0],
             "bbox_3d": {"center": [70, 98, 20], "size": [0.6, 1.7, 0.6], "yaw": 0.5},
             "mask_rle": None, "visible": fid != 1},  # hidden in frame 1
        ]
        frames.append({"frame_id": fid, "file_name": f"rgb/{fid:06d}.png", "width": 1280,
                       "height": 800, "cam": {"K": np.eye(3).tolist(), "w2c": np.eye(4).tolist()},
                       "annotations": anns})
    return frames


def _check(out, frames, expect_tracks):
    d2 = json.load(open(os.path.join(out, "coco_2d.json")))
    d3 = json.load(open(os.path.join(out, "coco_3d.json")))
    assert set(d2) == {"images", "annotations", "categories"}
    assert [c["id"] for c in d2["categories"]] == [1, 2, 3, 4]
    assert len(d2["images"]) == len(frames)
    # same ids across 2D and 3D
    k2 = [(a["id"], a["image_id"], a["category_id"], a["attributes"]["object_id"]) for a in d2["annotations"]]
    k3 = [(a["id"], a["image_id"], a["category_id"], a["object_id"]) for a in d3["annotations"]]
    assert k2 == k3, "2D and 3D annotation ids differ"
    assert len({a["id"] for a in d2["annotations"]}) == len(d2["annotations"])
    assert all(i["K"] is not None and i["w2c"] is not None for i in d3["images"])
    assert expect_tracks == bool(d3["tracks_3d"])
    # MOT
    lines = open(os.path.join(out, "mot", "gt", "gt.txt")).read().strip().splitlines()
    vis = sum(1 for a in d2["annotations"] if a["attributes"]["visible"])
    assert len(lines) == vis, (len(lines), vis)
    assert all(len(l.split(",")) == 9 for l in lines)
    assert min(int(l.split(",")[0]) for l in lines) == 1  # 1-based
    ini = open(os.path.join(out, "mot", "seqinfo.ini")).read()
    assert "frameRate=10" in ini and f"seqLength={len(frames)}" in ini
    # real COCO API, if available
    try:
        from pycocotools.coco import COCO
        coco = COCO(os.path.join(out, "coco_2d.json"))
        assert len(coco.getAnnIds()) == len(d2["annotations"])
        print("  pycocotools COCO() loaded coco_2d.json OK")
    except ImportError:
        print("  (pycocotools not installed here: skipped COCO() load, run it on your machine)")
    return len(d2["annotations"]), len(lines)


def test_fake():
    frames = _fake_frames()
    with tempfile.TemporaryDirectory() as out:
        export_all(frames, out, tracks={20: [(0, [65, 98, 16], 0.0), (1, [66, 98, 16], 0.0)]})
        n, mot = _check(out, frames, True)
    print(f"fake input OK: {n} annotations, {mot} MOT lines (1 hidden skipped)")


def test_real():
    if not os.path.exists("scene_manifest.json"):
        print("real-data test skipped (no scene_manifest.json here)")
        return
    from annotate import annotate_frame
    from cameras import make_trajectory
    from loader import boxes_at, load_manifest, tracks_3d
    m = load_manifest("scene_manifest.json")
    frames = []
    for cam in make_trajectory("follow", n_frames=10, m=m):
        anns = annotate_frame(boxes_at(m, cam["frame_id"]), cam)
        for a in anns:
            a["mask_rle"] = _rect_rle(cam["height"], cam["width"], a["bbox_2d"])
            a["visible"] = a["object_id"] != 21
        frames.append({"frame_id": cam["frame_id"], "file_name": f"rgb/{cam['frame_id']:06d}.png",
                       "width": cam["width"], "height": cam["height"], "cam": cam, "annotations": anns})
    with tempfile.TemporaryDirectory() as out:
        export_all(frames, out, m, tracks_3d(m))
        n, mot = _check(out, frames, True)
        d3 = json.load(open(os.path.join(out, "coco_3d.json")))
        assert any(a["source_node"] is not None for a in d3["annotations"])
    print(f"real boxes OK: {n} annotations, {mot} MOT lines")


if __name__ == "__main__":
    test_fake()
    test_real()
    print("export.py tests passed")
