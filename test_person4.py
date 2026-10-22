
from pathlib import Path
import json
import numpy as np
from PIL import Image
import segment

# Use the ID image you already rendered
id_path = Path("id/000000.png")
out_dir = Path("out/person4_test")
out_dir.mkdir(parents=True, exist_ok=True)

# Extract object masks
masks = segment.masks_from_id(id_path)

print("ID image:", id_path)
print("Image size:", Image.open(id_path).size)
print("Object IDs found:", list(masks.keys()))

# Save masks and COCO RLE annotations
annotations = {}

for object_id, mask in masks.items():
    annotations[str(object_id)] = {
        "visible": segment.visible_for_object(masks, object_id),
        "mask_rle": segment.mask_to_rle(mask),
        "pixels": int(np.asarray(mask).sum())
    }

with open(out_dir / "annotations.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)

# Create an overlay using the RGB frame if available
rgb_path = Path("rgb/000000.png")
if rgb_path.exists():
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    segment.draw_masks(rgb, masks, out_dir / "overlay.png")
    print("Overlay saved:", out_dir / "overlay.png")

print("Annotations saved:", out_dir / "annotations.json")
print("Segmentation test completed.")
