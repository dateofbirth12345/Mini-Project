"""Convert per-image JSON or COCO JSON annotations into YOLO-format folders.

Usage:
  python scripts/convert_jsons_to_yolo.py --src "C:/path/to/jsons_or_coco.json_or_dir" --dst "data/raw_yolo" --img-dir "C:/path/to/images" 

Notes:
- This script attempts to handle two cases:
  1) A COCO-style single JSON file (contains 'images' and 'annotations').
  2) A directory of per-image JSON files, each containing bounding boxes in keys
     like 'objects' or 'annotations'. Adjust parsing heuristics below if your
     JSON schema differs.
- The script will create a YOLO-formatted structure at <dst> with `images/` and `labels/`.
- You must verify the class name -> index mapping below.

Adjust the CLASS_MAP to match your annotation labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Default mapping - change if your labels use different names
CLASS_MAP: Dict[str, int] = {
    "plastic": 0,
    "non_plastic": 1,
}

SUPPORTED_IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_image_for_stem(src_dirs: List[str], stem: str) -> Optional[str]:
    for d in src_dirs:
        for ext in SUPPORTED_IMG_EXTS:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                return p
    return None


def write_yolo_label(label_path: str, img_w: int, img_h: int, bboxes: List[Tuple[int, int, int, int, int]]):
    # bboxes: list of (class_idx, x1, y1, x2, y2)
    lines = []
    for cls_idx, x1, y1, x2, y2 in bboxes:
        # convert to cx, cy, w, h (normalized)
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_coco_json(coco_path: str, dst_images: str, dst_labels: str, img_dirs: List[str]):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    # Build category name -> id mapping
    cat_map = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    # If categories are numeric and correspond to plastic/non_plastic, user should confirm.

    for img_id, img in imgs.items():
        file_name = img.get("file_name")
        stem, _ = os.path.splitext(file_name)
        src_img = find_image_for_stem(img_dirs, stem) or None
        if src_img is None:
            # Try relative to coco json dir
            candidate = os.path.join(os.path.dirname(coco_path), file_name)
            if os.path.isfile(candidate):
                src_img = candidate
            else:
                print(f"Warning: image for {file_name} not found. Skipping.")
                continue
        # copy image
        shutil.copy2(src_img, os.path.join(dst_images, os.path.basename(src_img)))
        # load image size
        with Image.open(src_img) as im:
            w, h = im.size
        bboxes = []
        for ann in anns_by_image.get(img_id, []):
            bbox = ann.get("bbox")  # COCO bbox: [x,y,w,h]
            if not bbox:
                continue
            x, y, bw, bh = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + bw)
            y2 = int(y + bh)
            cat_name = cat_map.get(ann.get("category_id"), str(ann.get("category_id")))
            if cat_name not in CLASS_MAP:
                print(f"Warning: category '{cat_name}' not in CLASS_MAP; skipping")
                continue
            cls_idx = CLASS_MAP[cat_name]
            bboxes.append((cls_idx, x1, y1, x2, y2))
        label_path = os.path.join(dst_labels, stem + ".txt")
        write_yolo_label(label_path, w, h, bboxes)


def process_dir_of_jsons(src_dir: str, dst_images: str, dst_labels: str, img_dirs: List[str]):
    for fn in os.listdir(src_dir):
        if not fn.lower().endswith(".json"):
            continue
        json_path = os.path.join(src_dir, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Heuristics: find objects list
        objects = None
        for k in ("objects", "annotations", "labels", "detections", "objects_detected"):
            if k in data:
                objects = data[k]
                break
        if objects is None:
            # maybe top-level has bbox/label
            if "bbox" in data and ("label" in data or "category" in data):
                objects = [data]
        if objects is None:
            print(f"No objects found in {json_path}; skipping")
            continue
        # Determine image path (try same stem)
        stem, _ = os.path.splitext(fn)
        src_img = find_image_for_stem(img_dirs, stem)
        if src_img is None:
            # try looking for image with same stem in src_dir
            for ext in SUPPORTED_IMG_EXTS:
                candidate = os.path.join(src_dir, stem + ext)
                if os.path.isfile(candidate):
                    src_img = candidate
                    break
        if src_img is None:
            print(f"Image for {fn} not found; skipping")
            continue
        shutil.copy2(src_img, os.path.join(dst_images, os.path.basename(src_img)))
        with Image.open(src_img) as im:
            w, h = im.size
        bboxes = []
        for obj in objects:
            # Try multiple possible property names
            bbox = obj.get("bbox") or obj.get("bounding_box") or obj.get("box")
            label = obj.get("label") or obj.get("category") or obj.get("class")
            if bbox is None:
                # Maybe coordinates are x1,y1,x2,y2
                for keys in (("x1","y1","x2","y2"),("left","top","right","bottom")):
                    if all(k in obj for k in keys):
                        x1 = int(obj[keys[0]])
                        y1 = int(obj[keys[1]])
                        x2 = int(obj[keys[2]])
                        y2 = int(obj[keys[3]])
                        bbox = [x1, y1, x2-x1, y2-y1]
                        break
            if bbox is None:
                print(f"No bbox for object in {json_path}; skipping object")
                continue
            # If bbox is [x,y,w,h] or [x1,y1,x2,y2]
            if len(bbox) == 4:
                x, y, bw, bh = bbox
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + bw)
                y2 = int(y + bh)
            elif len(bbox) == 5:
                # maybe includes confidence
                x, y, bw, bh, _ = bbox
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + bw)
                y2 = int(y + bh)
            else:
                print(f"Unexpected bbox format in {json_path}: {bbox}; skipping")
                continue
            if label is None:
                print(f"No label for object in {json_path}; skipping")
                continue
            if isinstance(label, dict):
                label = label.get("name") or label.get("label")
            label = str(label)
            if label not in CLASS_MAP:
                print(f"Warning: category '{label}' not in CLASS_MAP; skipping")
                continue
            cls_idx = CLASS_MAP[label]
            bboxes.append((cls_idx, x1, y1, x2, y2))
        label_path = os.path.join(dst_labels, stem + ".txt")
        write_yolo_label(label_path, w, h, bboxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source JSON file or directory containing JSONs")
    parser.add_argument("--dst", default="data/raw_yolo", help="Destination root for YOLO-formatted data")
    parser.add_argument("--img-dir", default=None, help="Optional directory to search for images (if images are not alongside JSONs)")
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    img_dir = args.img_dir

    dst_images = os.path.join(dst, "images")
    dst_labels = os.path.join(dst, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    img_dirs = []
    if img_dir:
        img_dirs.append(img_dir)
    img_dirs.append(os.path.dirname(src) if os.path.isfile(src) else src)

    if os.path.isfile(src):
        # Could be a COCO json
        with open(src, "r", encoding="utf-8") as f:
            j = json.load(f)
        if "images" in j and "annotations" in j:
            process_coco_json(src, dst_images, dst_labels, img_dirs)
        else:
            # single per-image JSON? treat as dir
            print("Single JSON file provided but not COCO-format. Attempting to process as single-image JSON.")
            process_dir_of_jsons(os.path.dirname(src), dst_images, dst_labels, img_dirs)
    elif os.path.isdir(src):
        process_dir_of_jsons(src, dst_images, dst_labels, img_dirs)
    else:
        raise SystemExit("Source not found")

    print(f"Finished conversion. YOLO data at {dst} (images/, labels/). Update `configs/config.yaml` dataset.root to this path and then run `python src/preprocess.py`.")


if __name__ == "__main__":
    main()
