# Layer 5 POC — Person 4 segmentation

## Person 4 files
- `segment.py`: ID-frame rendering launcher, per-object binary masks, COCO RLE, and translucent overlays.
- `viewer/index.html`: includes the `mode=id` rendering branch; `mode=rgb` remains the normal RGB renderer.
- `viewer/scene_manifest.json`: copy used by the browser ID renderer.
- `viewer/city/`: scene assets served by the browser.

## Setup (Windows PowerShell)
Run these commands from the extracted `Layer-5-poc-main` folder:

```powershell
py -m pip install -r requirements-person4.txt
cd viewer
npm install
cd ..
py -m playwright install chromium
```

The `viewer/package.json` pins Three.js. The city files are copied into `viewer/city/` so the local HTTP server can load them.

## Quick local test
```powershell
py segment.py
```
This runs a hand-made smoke test for the 20-pixel threshold and overlay saving. It does not require the browser renderer.

## Render ID masks
Generate cameras first if needed:
```powershell
py cameras.py
```
Then render ID frames (example: first 3 orbit frames):
```powershell
py render.py out/orbit/cams.json --mode id --limit 3
```
This saves ID images to `out/orbit/id/000000.png`, etc. To render all frames, omit `--limit 3`.

## Use the masks and RLE
```python
from segment import masks_from_id, mask_to_rle, draw_masks
from PIL import Image

masks = masks_from_id("out/orbit/id/000000.png")
for object_id, mask in masks.items():
    rle = mask_to_rle(mask)
    # Attach rle and visible=True to that object's annotation.
rgb = Image.open("out/orbit/rgb/000000.png").convert("RGB")
draw_masks(rgb, masks, "out/orbit/overlay_mask/000000.png")
```

`masks_from_id` ignores background ID 0 and masks smaller than 20 pixels. If an object ID has no returned mask, the caller should set its annotation's `visible` to `False`. `mask_to_rle` returns `{size, counts}` with `counts` as a string, ready for JSON serialization.

## Notes
- Run rendering from the project root, and ensure `viewer/city/scene.gltf` and its referenced assets exist.
- `pycocotools` may be harder to install on some Windows/Python combinations. If installation fails, use a compatible Python version/environment or install it through a supported conda environment.
- The browser-based ID rendering and full end-to-end scene render still need to be run in the target environment; the smoke test only validates the NumPy mask and overlay functions.
