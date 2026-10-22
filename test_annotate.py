"""Test: car 20 under the follow camera; its 2D box should hug the car."""
import numpy as np
from annotate import annotate_frame, bbox_2d
from cameras import make_trajectory, project
from loader import boxes_at, box_corners, load_manifest

m = load_manifest("scene_manifest.json")
cams = make_trajectory("follow", 60, m)
for f in (0, 20, 40, 59):
    boxes = boxes_at(m, f)
    car = next(b for b in boxes if b["object_id"] == 20)
    bb = bbox_2d(car, cams[f])
    uv, _ = project(box_corners(car), cams[f])
    x, y, w, h = bb
    assert abs(x - uv[:, 0].min()) < 0.1 and abs(y - uv[:, 1].min()) < 0.1  # hugs corners
    cx, cy = x + w / 2, y + h / 2
    print(f"frame {f}: car 20 bbox {bb}, centre ({cx:.0f},{cy:.0f})")
    assert abs(cx - 640) < 150 and abs(cy - 400) < 150   # follow cam centres the car
    anns = annotate_frame(boxes, cams[f])
    assert all(a["layer"] == "dynamic" for a in anns)
    assert len(annotate_frame(boxes, cams[f], include_static=True)) >= len(anns)
# behind camera -> None
cam = cams[0]; far = dict(car, center=[-500, 100, 16.2])
assert bbox_2d(far, cam) is None
print("annotate OK")
