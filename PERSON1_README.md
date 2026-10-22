# Person 1: loader.py
Run test: `python test_loader.py`

API (import from loader):
- load_manifest(path) -> dict
- boxes_at(m, frame) -> [box dicts]   (frame wraps at 120)
- box_corners(box) -> (8,3) ndarray   (0-3 bottom, 4-7 top above 0-3)
- tracks_3d(m) -> {object_id: [(frame, center, yaw), ...]}
- add_parked_car(m) -> optional semi_static car (object_id 28)

Conventions: Y-up, metres. yaw about +Y, 0 = +X; positive yaw turns +X toward -Z.
Note: dynamic box `size` changes over the loop (car 20 goes from 4.22 x 1.98 to 2.9 x 4.47 footprint at yaw -0.63), taken as-is from the manifest.
Credit: Based on 'City map free' by Katydid (CC BY 4.0), https://sketchfab.com/3d-models/city-map-free-00c4aa34478b492bafa37bd198587d14
