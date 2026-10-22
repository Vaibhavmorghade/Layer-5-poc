"""Person 4: instance-ID rendering, binary masks, COCO RLE, and overlays.

Run from the project root. The renderer/viewer are supplied by Person 3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

# RGB colours used for the translucent visualization (building, person, car, sign).
CLASS_COLORS = {
    "building": (70, 130, 220),
    "person": (40, 210, 90),
    "car": (235, 80, 65),
    "sign": (235, 190, 40),
}
# Stable fallback palette for object IDs when no class mapping is passed.
_ID_COLORS = [
    (235, 80, 65), (40, 210, 90), (70, 130, 220), (235, 190, 40),
    (190, 80, 210), (40, 200, 210), (240, 140, 40), (160, 190, 50),
]


def _load_object_classes():
    manifest_path = Path(__file__).resolve().with_name("scene_manifest.json")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return {int(obj["object_id"]): obj.get("class", "") for obj in manifest.get("objects", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


_OBJECT_CLASSES = _load_object_classes()


def render_ids(cams, out_dir):
    """Render cameras as exact object-ID PNGs in out_dir/id/000000.png.

    Delegates to Person 3's render_frames implementation, which launches the
    viewer with ?mode=id. `cams` may be a list of camera dictionaries or a
    path to a cams JSON file.
    """
    try:
        from render import render_frames
    except ImportError as exc:
        raise ImportError(
            "Could not import render_frames from render.py. Run from the "
            "project root and install Person 3's Playwright dependencies."
        ) from exc

    if isinstance(cams, (str, os.PathLike)):
        with open(cams, "r", encoding="utf-8") as f:
            cams = json.load(f)
    return render_frames(cams, out_dir, mode="id", subdir="id")


def _read_id_image(id_png):
    """Return an RGB uint8 array from a filename, PIL image, or ndarray."""
    if isinstance(id_png, (str, os.PathLike, Path)):
        with Image.open(id_png) as im:
            arr = np.asarray(im.convert("RGB"))
    elif isinstance(id_png, Image.Image):
        arr = np.asarray(id_png.convert("RGB"))
    else:
        arr = np.asarray(id_png)
        if arr.ndim == 2:
            # A grayscale ID image is also accepted for small unit tests.
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] < 1:
            raise ValueError("id_png must be an image path or HxWxC array")
    return arr


def masks_from_id(id_png):
    """Extract {object_id: bool_mask} from the red channel.

    Background is ID 0. Objects with fewer than 20 pixels are ignored, as
    required by the project contract. A missing object therefore has no mask
    and should be annotated with visible=False by the caller.
    """
    arr = _read_id_image(id_png)
    red = arr[:, :, 0].astype(np.uint16, copy=False)
    ids, counts = np.unique(red, return_counts=True)
    masks = {}
    for object_id, count in zip(ids, counts):
        if int(object_id) == 0 or int(count) < 20:
            continue
        masks[int(object_id)] = red == object_id
    return masks


def mask_to_rle(mask):
    """Encode a 2-D mask in COCO RLE format with JSON-compatible counts."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for RLE encoding. Install it with "
            "`python -m pip install pycocotools`."
        ) from exc

    binary = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    if binary.ndim != 2:
        raise ValueError("mask_to_rle expects a 2-D mask")
    encoded = mask_utils.encode(binary)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(x) for x in encoded["size"]], "counts": counts}


def draw_masks(rgb, masks: Mapping[int, np.ndarray], out_path):
    """Tint each mask at 50% alpha over an RGB image and save the overlay.

    The output parent folder is created automatically. Returns the resulting
    RGB uint8 array so the caller can also use it in a pipeline.
    """
    if isinstance(rgb, (str, os.PathLike, Path)):
        with Image.open(rgb) as im:
            image = np.asarray(im.convert("RGB")).copy()
    elif isinstance(rgb, Image.Image):
        image = np.asarray(rgb.convert("RGB")).copy()
    else:
        image = np.asarray(rgb).copy()

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb must be an HxWx3 RGB image")
    image = np.clip(image[:, :, :3], 0, 255).astype(np.uint8)
    overlay = image.astype(np.float32)

    for object_id, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"mask for object {object_id} has shape {mask.shape}; "
                f"expected {image.shape[:2]}"
            )
        if not mask.any():
            continue
        object_class = _OBJECT_CLASSES.get(int(object_id), "")
        color = CLASS_COLORS.get(
            object_class,
            _ID_COLORS[(int(object_id) - 1) % len(_ID_COLORS)]
        )
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.asarray(color, dtype=np.float32)

    result = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(out_path)
    return result


def visible_for_object(masks, object_id):
    """Convenience helper: whether the object survived the 20-pixel threshold."""
    return int(object_id) in masks


if __name__ == "__main__":
    # Small deterministic smoke test that does not need a browser or city mesh.
    test_ids = np.zeros((32, 48, 3), dtype=np.uint8)
    test_ids[4:14, 5:17, 0] = 20  # 120 pixels: retained
    test_ids[20:23, 30:34, 0] = 21  # 12 pixels: ignored
    masks = masks_from_id(test_ids)
    assert list(masks) == [20], f"unexpected masks: {list(masks)}"
    assert int(masks[20].sum()) == 120
    test_rgb = np.full((32, 48, 3), 100, dtype=np.uint8)
    overlay_path = Path("out") / "person4_smoke_overlay.png"
    draw_masks(test_rgb, masks, overlay_path)
    assert overlay_path.exists()
    print("segment.py smoke test passed: threshold, binary mask, overlay save")
    print(f"Overlay written to: {overlay_path}")
