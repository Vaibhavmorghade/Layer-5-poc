# Person 6: export.py (COCO 2D, COCO 3D, MOT)

Files: `export.py`, `test_export.py`

Test (no other module needed): `python test_export.py`

Real run for one trajectory (needs out/<traj>/cams.json, optionally out/<traj>/id/ for masks):
`python export.py out/orbit`  ->  coco_2d.json, coco_3d.json, mot/gt/gt.txt, mot/seqinfo.ini

For run_demo.py (Person 7):
```python
from export import export_all
export_all(frames, f"out/{kind}", m)   # frames = list of {frame_id, file_name, width, height, cam, annotations}
```
Or call `export_coco`, `export_coco3d`, `export_mot` separately.

Shared ids: image id = frame_id (0-based), annotation id identical in 2D and 3D, track_id = object_id, MOT frame = frame_id + 1.
visible None is treated as visible. visible False stays in COCO (attributes.visible) and is skipped in MOT.
Validate: `from pycocotools.coco import COCO; COCO('out/orbit/coco_2d.json')`
