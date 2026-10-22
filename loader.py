"""Person 1: manifest loader, 3D box corners, 3D tracks.

Contract (see plan): Y-up, metres, glTF world. yaw = rotation about +Y in
radians (0 = +X). Camera frame i shows manifest frame i % num_frames.
Box dict: {object_id, class, category_id, layer, action, center, size, yaw}
"""
import json
import numpy as np

CATEGORY_IDS = {"person": 1, "car": 2, "building": 3, "sign": 4}


def load_manifest(path="scene_manifest.json"):
    """Return the parsed manifest JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _box(obj, center, size, yaw, action):
    return {
        "object_id": obj["object_id"],
        "class": obj["class"],
        "category_id": CATEGORY_IDS[obj["class"]],
        "layer": obj["layer"],
        "action": action,
        "center": [float(c) for c in center],
        "size": [float(s) for s in size],
        "yaw": float(yaw),
    }


def boxes_at(m, frame):
    """All box dicts for camera frame `frame` (wraps with frame % num_frames).

    static / semi_static objects use their single box3d.
    dynamic objects use frames[frame % len(frames)].
    """
    boxes = []
    for obj in m["objects"]:
        if obj["layer"] == "dynamic":
            fr = obj["frames"]
            f = fr[frame % len(fr)]
            boxes.append(_box(obj, f["center"], f["size"], f["yaw"], f.get("action", "moving")))
        else:
            b = obj["box3d"]
            boxes.append(_box(obj, b["center"], b["size"], b.get("yaw", 0.0),
                              obj.get("action", "static")))
    return boxes


def box_corners(box):
    """8x3 world-space corners. size = [sx, sy, sz] (full extents).

    Order: 0-3 bottom face (y = -sy/2), 4-7 top face (y = +sy/2), with
    4..7 directly above 0..3, so edges (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7) draw the box.
    Rotation about +Y: x' = x*cos + z*sin, z' = -x*sin + z*cos.
    """
    sx, sy, sz = box["size"]
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz],
        [-hx,  hy, -hz], [hx,  hy, -hz], [hx,  hy, hz], [-hx,  hy, hz],
    ])
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return local @ R.T + np.asarray(box["center"], dtype=float)


def tracks_3d(m):
    """{object_id: [(frame, center, yaw), ...]} for dynamic objects."""
    tracks = {}
    for obj in m["objects"]:
        if obj["layer"] != "dynamic":
            continue
        tracks[obj["object_id"]] = [
            (f["frame"], list(f["center"]), f["yaw"]) for f in obj["frames"]
        ]
    return tracks


def add_parked_car(m, center=(108.0, 98.4, 16.2), size=(4.22, 1.76, 1.98), yaw=0.0):
    """Optional: add one hand-placed parked car as layer 'semi_static'.

    Returns the object dict (also appended to m["objects"]). Default sits in the lane car 20 and car 22 drive along (clear of both paths, x 90 to 133). Adjust `center`
    so it sits on a road, not inside a building.
    """
    new_id = max(o["object_id"] for o in m["objects"]) + 1
    obj = {
        "object_id": new_id, "class": "car", "layer": "semi_static",
        "source_node": None, "source_name": "Parked car (hand-placed)",
        "box3d": {"center": list(center), "size": list(size), "yaw": yaw},
        "action": "parked",
    }
    m["objects"].append(obj)
    return obj


if __name__ == "__main__":
    m = load_manifest("scene_manifest.json")
    layers = [o["layer"] for o in m["objects"]]
    assert len(m["objects"]) == 27
    assert layers.count("static") == 19 and layers.count("dynamic") == 8
    for o in m["objects"]:
        if o["layer"] == "dynamic":
            assert len(o["frames"]) == 120
    dyn = [o["class"] for o in m["objects"] if o["layer"] == "dynamic"]
    assert dyn.count("car") == 4 and dyn.count("person") == 4

    b0 = boxes_at(m, 0)
    assert len(b0) == 27
    assert box_corners(b0[0]).shape == (8, 3)
    car20 = next(b for b in b0 if b["object_id"] == 20)
    assert abs(car20["center"][0] - 65) < 1 and abs(car20["center"][2] - 16.2) < 1
    assert boxes_at(m, 120)[19]["center"] == b0[19]["center"]  # loop wrap
    # corners: centroid == center, and yaw=pi/2 maps +X extent onto -Z
    cs = box_corners(car20)
    assert np.allclose(cs.mean(0), car20["center"])
    t = dict(car20, yaw=np.pi / 2)
    sx, sy, sz = car20["size"]
    # local corner 1 = (+sx/2, -sy/2, -sz/2); after yaw=pi/2 it is (-sz/2, -sy/2, -sx/2)
    assert np.allclose(box_corners(t)[1] - np.array(t["center"]),
                       [-sz / 2, -sy / 2, -sx / 2], atol=1e-6)
    tr = tracks_3d(m)
    assert len(tr) == 8 and len(tr[20]) == 120
    print("loader.py OK: 27 boxes, corners (8,3), car 20 at", car20["center"])
