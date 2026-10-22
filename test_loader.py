"""Person 1 tests. Run: python test_loader.py  (or pytest test_loader.py)"""
import numpy as np
from loader import *

m = load_manifest("scene_manifest.json")

def test_counts():
    layers = [o["layer"] for o in m["objects"]]
    assert len(m["objects"]) == 27
    assert layers.count("static") == 19 and layers.count("dynamic") == 8

def test_boxes_and_corners():
    b = boxes_at(m, 0)
    assert len(b) == 27
    assert box_corners(b[0]).shape == (8, 3)
    car = next(x for x in b if x["object_id"] == 20)
    assert abs(car["center"][0] - 65) < 1 and abs(car["center"][2] - 16.2) < 1
    assert car["category_id"] == 2

def test_loop_wrap():
    assert boxes_at(m, 5)[19] == boxes_at(m, 125)[19]

def test_tracks():
    t = tracks_3d(m)
    assert len(t) == 8 and all(len(v) == 120 for v in t.values())

def test_parked_car():
    import copy
    m2 = copy.deepcopy(m)
    add_parked_car(m2)
    b = boxes_at(m2, 0)
    assert len(b) == 28 and b[-1]["layer"] == "semi_static" and b[-1]["action"] == "parked"

if __name__ == "__main__":
    for f in (test_counts, test_boxes_and_corners, test_loop_wrap, test_tracks, test_parked_car):
        f(); print("ok", f.__name__)
