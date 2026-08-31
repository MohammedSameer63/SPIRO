#!/usr/bin/env python3
"""
SPIRO — setup_datasets.py
Downloads, extracts, maps, cleans and merges:
  • TACO (via GitHub)
  • ZeroWaste-f (via Roboflow public link)
  • TrashNet (via Kaggle API or GitHub mirror)
  • Garbage Classification (Kaggle GC via GitHub mirror)
  • Recyclable & Household Waste (via Roboflow public link)

Output: datasets/spiro_merged/ in YOLO format
         train/ val/ test/   each with images/ and labels/
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── SPIRO taxonomy mappings ───────────────────────────────────────────────────
# Maps source-dataset class names → SPIRO class ID
# SPIRO class IDs 0–108 defined in configs/taxonomy/spiro_taxonomy.yaml

TACO_TO_SPIRO: Dict[str, int] = {
    "Aluminium foil":            27,  # metal_foil
    "Battery":                   61,  # battery
    "Blister pack":             104,  # blister_pack
    "Bottle":                     0,  # plastic_bottle
    "Bottle cap":                 1,  # plastic_bottle_cap
    "Broken glass":              22,  # glass_broken
    "Can":                       27,  # metal_can
    "Carton":                    36,  # cardboard_carton
    "Cigarette":                 53,  # cigarette
    "Cup":                       30,  # paper_cup
    "Drink can":                 27,  # metal_can
    "Drink carton":              71,  # beverage_carton
    "Egg carton":                37,  # egg_carton
    "Foam cup":                  14,  # foam_cup
    "Foam food container":       15,  # foam_container
    "Food Can":                  28,  # food_can
    "Food waste":                47,  # food_waste_generic
    "Garbage bag":               12,  # plastic_bag_black
    "Glass bottle":              20,  # glass_bottle
    "Glass cup":                 23,  # glass_cup
    "Glass jar":                 21,  # glass_jar
    "Lid":                        5,  # plastic_lid
    "Magazine":                  34,  # magazine
    "Meal carton":               36,  # cardboard_carton
    "Metal bottle cap":          29,  # metal_lid
    "Metal lid":                 29,  # metal_lid
    "Normal paper":              31,  # paper_sheet
    "Other carton":              36,  # cardboard_carton
    "Paper":                     31,  # paper_sheet
    "Paper bag":                 35,  # paper_bag
    "Paper cup":                 30,  # paper_cup
    "Paper straw":               38,  # paper_straw
    "Pizza box":                 96,  # fast_food_box
    "Plastic bag & wrapper":     11,  # plastic_bag
    "Plastic bottle":             0,  # plastic_bottle
    "Plastic bottle cap":         1,  # plastic_bottle_cap
    "Plastic container":          6,  # plastic_container
    "Plastic cup":                7,  # plastic_cup
    "Plastic cup lid":            5,  # plastic_lid
    "Plastic glooves":           59,  # glove_disposable
    "Plastic straw":              9,  # plastic_straw
    "Plastic utensils":          10,  # plastic_cutlery
    "Pop tab":                   27,  # metal_can
    "Rope & strings":            93,  # rope
    "Scrap metal":               26,  # scrap_metal
    "Shoe":                      57,  # shoe
    "Shopping bag":              11,  # plastic_bag
    "Single-use carrier bag":    11,  # plastic_bag
    "Six pack rings":            13,  # plastic_ring
    "Spread tub":                 6,  # plastic_container
    "Squeezable tube":            4,  # plastic_tube
    "Straw":                      9,  # plastic_straw
    "Styrofoam piece":           14,  # foam_cup
    "Tissues":                   31,  # paper_sheet
    "Toilet tube":               33,  # toilet_roll
    "Trash bag":                 12,  # plastic_bag_black
    "Unlabeled litter":         108,  # unknown_litter
    "Wrapping paper":            32,  # wrapping_paper
    "Rope":                      93,  # rope
    "Other plastic":              8,  # plastic_other
    "Other plastic wrapper":      8,  # plastic_other
    "Other plastic container":    6,  # plastic_container
    "Other paper":               31,  # paper_sheet
    "Other glass":               24,  # glass_other
    "Other metal":               26,  # scrap_metal
    "Other paper & cardboard":   31,  # paper_sheet
}

TRASHNET_TO_SPIRO: Dict[str, int] = {
    "cardboard":  36,   # cardboard_carton
    "glass":      20,   # glass_bottle (dominant)
    "metal":      27,   # metal_can (dominant)
    "paper":      31,   # paper_sheet
    "plastic":     0,   # plastic_bottle (dominant)
    "trash":     108,   # unknown_litter
}

ZEROWASTE_TO_SPIRO: Dict[str, int] = {
    "aluminium_foil":  25,   # aluminium_foil
    "carton":          36,   # cardboard_carton
    "glass":           20,   # glass_bottle
    "other":          108,   # unknown_litter
    "plastic":          0,   # plastic_bottle
    "plastic_bag":     11,   # plastic_bag
    "transparent":      2,   # plastic_bottle_transparent
    "unknown":        108,   # unknown_litter
    "can":             27,   # metal_can
    "drinking_carton": 71,   # beverage_carton
    "cup":             30,   # paper_cup
    "lid":              5,   # plastic_lid
    "cardboard":       36,   # cardboard_carton
    "paper":           31,   # paper_sheet
    "soft_plastic":     8,   # plastic_other
    "rigid_plastic":    6,   # plastic_container
}

KAGGLE_GC_TO_SPIRO: Dict[str, int] = {
    "battery":        61,   # battery
    "biological":     47,   # food_waste_generic
    "brown-glass":    21,   # glass_jar (brown)
    "cardboard":      36,   # cardboard_carton
    "clothes":        56,   # textile_clothing
    "green-glass":    20,   # glass_bottle (green)
    "metal":          27,   # metal_can
    "paper":          31,   # paper_sheet
    "plastic":         0,   # plastic_bottle
    "shoes":          57,   # shoe
    "trash":         108,   # unknown_litter
    "white-glass":    23,   # glass_cup (white)
}

RECYCLABLE_HH_TO_SPIRO: Dict[str, int] = {
    "aerosol_cans":         27,   # metal_can
    "aluminum_food_cans":   28,   # food_can
    "aluminum_soda_cans":   27,   # metal_can
    "cardboard_boxes":      36,   # cardboard_carton
    "cardboard_packaging":  36,   # cardboard_carton
    "clothing":             56,   # textile_clothing
    "coffee_grounds":       51,   # coffee_grounds
    "disposable_plastic_cutlery": 10,  # plastic_cutlery
    "eggshells":            52,   # eggshell
    "food_waste":           47,   # food_waste_generic
    "glass_beverage_bottles": 20, # glass_bottle
    "glass_cosmetic_containers": 23,  # glass_cup
    "glass_food_jars":      21,   # glass_jar
    "magazines":            34,   # magazine
    "newspaper":            32,   # newspaper  -- was 32 wrapping but newspaper=32 too
    "office_paper":         31,   # paper_sheet
    "paper_cups":           30,   # paper_cup
    "plastic_beverage_bottles": 0, # plastic_bottle
    "plastic_cold_drink_cups":  7, # plastic_cup
    "plastic_detergent_bottles": 3, # plastic_bottle_opaque
    "plastic_food_containers":  6, # plastic_container
    "plastic_shopping_bags":   11, # plastic_bag
    "plastic_soda_bottles":     0, # plastic_bottle
    "plastic_straws":           9, # plastic_straw
    "plastic_trash_bags":      12, # plastic_bag_black
    "plastic_water_bottles":    0, # plastic_bottle
    "shoes":                   57, # shoe
    "steel_food_cans":         28, # food_can
    "styrofoam_cups":          14, # foam_cup
    "styrofoam_food_containers":15, # foam_container
    "tea_bags":                48, # tea_bag
}


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/content/spiro")
DATA = BASE / "datasets"
RAW  = DATA / "raw"
OUT  = DATA / "spiro_merged"

NUM_CLASSES = 109
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(cmd: str, check: bool = True) -> int:
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        print(f"  ⚠️  Command exited {r.returncode}")
    return r.returncode


def mkdir(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset downloaders
# =============================================================================

def download_taco() -> Path:
    """Download TACO dataset from GitHub."""
    dst = RAW / "taco"
    if (dst / "data").exists():
        print("  TACO already downloaded")
        return dst
    mkdir(dst)
    print("  Downloading TACO...")
    run(f"git clone --depth 1 https://github.com/pedropro/TACO.git {dst}")
    return dst


def download_trashnet() -> Path:
    """Download TrashNet dataset."""
    dst = RAW / "trashnet"
    if dst.exists() and any(dst.iterdir()):
        print("  TrashNet already downloaded")
        return dst
    mkdir(dst)
    print("  Downloading TrashNet...")
    # Use GitHub release mirror
    url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    run(f"wget -q -O {dst}/trashnet.zip '{url}'")
    if (dst / "trashnet.zip").exists():
        run(f"unzip -q {dst}/trashnet.zip -d {dst}")
        (dst / "trashnet.zip").unlink()
    return dst


def download_zerowaste() -> Path:
    """Download ZeroWaste-f via public Roboflow link."""
    dst = RAW / "zerowaste"
    if (dst / "train").exists():
        print("  ZeroWaste-f already downloaded")
        return dst
    mkdir(dst)
    print("  Downloading ZeroWaste-f via Roboflow...")
    # Public Roboflow dataset — no API key needed for public datasets
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="")  # public
        proj = rf.workspace("recycleai").project("zerowaste-f")
        dataset = proj.version(1).download("yolov8", location=str(dst))
        print("  ZeroWaste-f downloaded via Roboflow API")
    except Exception as e:
        print(f"  Roboflow API failed ({e}), trying direct download...")
        # Fallback: direct URL to public dataset
        url = "https://universe.roboflow.com/ds/bV1zBOoNxq?key=x3EEidCqDd"
        r = run(f"wget -q -O {dst}/zerowaste.zip '{url}'", check=False)
        if r == 0 and (dst / "zerowaste.zip").exists():
            run(f"unzip -q {dst}/zerowaste.zip -d {dst}")
    return dst


def download_kaggle_gc() -> Path:
    """
    Download Garbage Classification (Kaggle) dataset.
    Uses kaggle API if credentials available, else GitHub mirror.
    """
    dst = RAW / "kaggle_gc"
    if dst.exists() and any(dst.iterdir()):
        print("  Kaggle GC already downloaded")
        return dst
    mkdir(dst)

    # Try kaggle API first
    if shutil.which("kaggle"):
        r = run(
            f"kaggle datasets download -d asdasdasasdas/garbage-classification -p {dst} --unzip",
            check=False
        )
        if r == 0:
            return dst

    # Fallback: use public GitHub mirror / alternative source
    print("  Using alternative source for Garbage Classification...")
    # This dataset is also available from multiple public mirrors
    url = "https://github.com/victoresque/pytorch-template/raw/master/data/garbage.zip"
    r = run(f"wget -q -O {dst}/gc.zip '{url}'", check=False)
    if r != 0:
        # Try gdown if available
        run("pip install gdown -q", check=False)
        # Google Drive public link for dataset
        r = run(
            f"gdown --id 1XMaNx7sMVIFHsqUBv-8lhAYuZzn0nwI6 -O {dst}/gc.zip",
            check=False
        )
    if (dst / "gc.zip").exists():
        run(f"unzip -q {dst}/gc.zip -d {dst}")
        (dst / "gc.zip").unlink(missing_ok=True)
    return dst


def download_recyclable_hh() -> Path:
    """Download Recyclable and Household Waste Classification."""
    dst = RAW / "recyclable_hh"
    if dst.exists() and any(dst.iterdir()):
        print("  Recyclable HH already downloaded")
        return dst
    mkdir(dst)

    # Try kaggle first
    if shutil.which("kaggle"):
        r = run(
            f"kaggle datasets download -d alistairking/recyclable-and-household-waste-classification "
            f"-p {dst} --unzip",
            check=False
        )
        if r == 0:
            return dst

    # Roboflow public link
    print("  Using Roboflow for Recyclable HH...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="")
        proj = rf.workspace("recyclable").project("recyclable-and-household-waste-classification")
        dataset = proj.version(1).download("yolov8", location=str(dst))
    except Exception as e:
        print(f"  Could not download Recyclable HH: {e}")
        print("  Continuing with other datasets...")
    return dst


# =============================================================================
# Converters — each dataset → YOLO format with SPIRO class IDs
# =============================================================================

class YOLORecord:
    """One image + annotations in YOLO format."""
    def __init__(self, img_path: Path, labels: List[Tuple[int, float, float, float, float]]):
        self.img_path = img_path
        self.labels = labels  # [(class_id, cx, cy, w, h), ...]


def convert_taco(raw_dir: Path) -> List[YOLORecord]:
    """Convert TACO COCO annotations → YOLO + SPIRO IDs."""
    records: List[YOLORecord] = []
    ann_file = raw_dir / "data" / "annotations.json"
    if not ann_file.exists():
        print("  ⚠️  TACO annotations.json not found")
        return records

    with open(ann_file) as f:
        coco = json.load(f)

    # Build category id → spiro id
    cat_to_spiro: Dict[int, int] = {}
    for cat in coco.get("categories", []):
        name = cat.get("name", "")
        if name in TACO_TO_SPIRO:
            cat_to_spiro[cat["id"]] = TACO_TO_SPIRO[name]
        else:
            # Try supercategory
            sc = cat.get("supercategory", "")
            cat_to_spiro[cat["id"]] = TACO_TO_SPIRO.get(sc, 108)  # unknown_litter

    # Build image_id → file info
    id_to_img: Dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
    # Build image_id → annotations
    id_to_anns: Dict[int, list] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in id_to_img.items():
        img_path = raw_dir / "data" / img_info.get("file_name", "")
        if not img_path.exists():
            continue
        w = img_info.get("width", 1)
        h = img_info.get("height", 1)
        labels = []
        for ann in id_to_anns.get(img_id, []):
            cat_id = ann["category_id"]
            spiro_id = cat_to_spiro.get(cat_id, 108)
            x, y, bw, bh = ann["bbox"]  # COCO: x,y,w,h absolute
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            rw = bw / w
            rh = bh / h
            if rw > 0.001 and rh > 0.001:
                labels.append((spiro_id, cx, cy, rw, rh))
        if labels:
            records.append(YOLORecord(img_path, labels))

    print(f"  TACO: {len(records)} images with annotations")
    return records


def convert_trashnet(raw_dir: Path) -> List[YOLORecord]:
    """Convert TrashNet image-folder → YOLO (whole-image bbox)."""
    records: List[YOLORecord] = []
    dataset_dir = raw_dir / "dataset-resized"
    if not dataset_dir.exists():
        dataset_dir = raw_dir
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        spiro_id = TRASHNET_TO_SPIRO.get(class_dir.name.lower(), 108)
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in IMG_EXTS:
                # Whole-image bbox: cx=0.5, cy=0.5, w=0.9, h=0.9
                records.append(YOLORecord(img_file, [(spiro_id, 0.5, 0.5, 0.9, 0.9)]))
    print(f"  TrashNet: {len(records)} images")
    return records


def convert_yolo_dataset(
    raw_dir: Path,
    class_map: Dict[str, int],
    dataset_name: str,
) -> List[YOLORecord]:
    """Convert YOLO-format dataset with class name remapping."""
    records: List[YOLORecord] = []

    # Find data.yaml or obj.data
    yaml_file = None
    for fname in ["data.yaml", "dataset.yaml", "obj.data"]:
        p = raw_dir / fname
        if p.exists():
            yaml_file = p
            break
    if yaml_file is None:
        for p in raw_dir.rglob("data.yaml"):
            yaml_file = p
            break

    # Build idx → spiro_id from existing YOLO class names
    idx_to_spiro: Dict[int, int] = {}
    if yaml_file and yaml_file.suffix == ".yaml":
        with open(yaml_file) as f:
            ydata = yaml.safe_load(f)
        names = ydata.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        for i, name in enumerate(names):
            norm = name.lower().replace(" ", "_").replace("-", "_")
            spiro_id = (
                class_map.get(name) or
                class_map.get(norm) or
                class_map.get(name.lower()) or
                108
            )
            idx_to_spiro[i] = spiro_id

    # Walk all images
    for split in ["train", "valid", "val", "test", ""]:
        img_dir = raw_dir / split / "images" if split else raw_dir / "images"
        lbl_dir = raw_dir / split / "labels" if split else raw_dir / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            labels = []
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    orig_id = int(parts[0])
                    spiro_id = idx_to_spiro.get(orig_id, 108)
                    cx, cy, w, h = map(float, parts[1:])
                    if w > 0.001 and h > 0.001:
                        labels.append((spiro_id, cx, cy, w, h))
            if labels:
                records.append(YOLORecord(img_path, labels))

    print(f"  {dataset_name}: {len(records)} labelled images")
    return records


def convert_image_folder(
    raw_dir: Path,
    class_map: Dict[str, int],
    dataset_name: str,
) -> List[YOLORecord]:
    """Convert ImageFolder structure → YOLO whole-image bbox."""
    records: List[YOLORecord] = []
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.lower().replace(" ", "_").replace("-", "_")
        spiro_id = class_map.get(class_dir.name) or class_map.get(name) or 108
        for img_file in class_dir.rglob("*"):
            if img_file.suffix.lower() in IMG_EXTS:
                records.append(YOLORecord(img_file, [(spiro_id, 0.5, 0.5, 0.9, 0.9)]))
    print(f"  {dataset_name}: {len(records)} images")
    return records


# =============================================================================
# Writer
# =============================================================================

def write_split(
    records: List[YOLORecord],
    split_dir: Path,
    copy_images: bool = True,
) -> int:
    """Write YOLO-format split directory."""
    imgs_dir = split_dir / "images"
    lbls_dir = split_dir / "labels"
    mkdir(imgs_dir, lbls_dir)
    written = 0
    for rec in records:
        if not rec.img_path.exists():
            continue
        # Unique output name using hash
        h = hashlib.md5(str(rec.img_path).encode()).hexdigest()[:8]
        stem = f"{rec.img_path.stem}_{h}"
        ext = rec.img_path.suffix.lower() or ".jpg"
        out_img = imgs_dir / f"{stem}{ext}"
        out_lbl = lbls_dir / f"{stem}.txt"
        if copy_images:
            shutil.copy2(rec.img_path, out_img)
        with open(out_lbl, "w") as f:
            for cls_id, cx, cy, w, h in rec.labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        written += 1
    return written


def create_dataset_yaml(out_dir: Path) -> Path:
    """Write data.yaml for YOLO training."""
    # Load class names from taxonomy
    tax_path = BASE / "configs" / "taxonomy" / "spiro_taxonomy.yaml"
    if tax_path.exists():
        with open(tax_path) as f:
            tax = yaml.safe_load(f)
        class_names = tax.get("names", {i: f"class_{i}" for i in range(NUM_CLASSES)})
        if isinstance(class_names, dict):
            names_list = [class_names.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
        else:
            names_list = list(class_names)
    else:
        names_list = [f"class_{i}" for i in range(NUM_CLASSES)]

    data = {
        "path":  str(out_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    NUM_CLASSES,
        "names": names_list,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml written: {yaml_path}")
    return yaml_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SPIRO dataset setup")
    parser.add_argument("--all",      action="store_true", help="Download all datasets")
    parser.add_argument("--taco",     action="store_true")
    parser.add_argument("--trashnet", action="store_true")
    parser.add_argument("--zerowaste",action="store_true")
    parser.add_argument("--kaggle-gc",action="store_true")
    parser.add_argument("--recyclable",action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only convert existing raw data")
    args = parser.parse_args()

    do_all = args.all or not any([
        args.taco, args.trashnet, args.zerowaste, args.kaggle_gc, args.recyclable
    ])

    mkdir(RAW, OUT)
    all_records: List[YOLORecord] = []

    # ── TACO ─────────────────────────────────────────────────────────────────
    if do_all or args.taco:
        print("\n📦 TACO Dataset")
        raw = download_taco() if not args.skip_download else RAW / "taco"
        records = convert_taco(raw)
        all_records.extend(records)

    # ── TrashNet ─────────────────────────────────────────────────────────────
    if do_all or args.trashnet:
        print("\n📦 TrashNet Dataset")
        raw = download_trashnet() if not args.skip_download else RAW / "trashnet"
        if (raw / "dataset-resized").exists() or any(
            d.is_dir() for d in raw.iterdir() if d.name in TRASHNET_TO_SPIRO
        ):
            records = convert_trashnet(raw)
        else:
            records = convert_image_folder(raw, TRASHNET_TO_SPIRO, "TrashNet")
        all_records.extend(records)

    # ── ZeroWaste-f ──────────────────────────────────────────────────────────
    if do_all or args.zerowaste:
        print("\n📦 ZeroWaste-f Dataset")
        raw = download_zerowaste() if not args.skip_download else RAW / "zerowaste"
        if raw.exists():
            records = convert_yolo_dataset(raw, ZEROWASTE_TO_SPIRO, "ZeroWaste-f")
            all_records.extend(records)

    # ── Kaggle GC ────────────────────────────────────────────────────────────
    if do_all or args.kaggle_gc:
        print("\n📦 Garbage Classification (Kaggle)")
        raw = download_kaggle_gc() if not args.skip_download else RAW / "kaggle_gc"
        if raw.exists():
            # Try YOLO format first, then image folder
            records = convert_yolo_dataset(raw, KAGGLE_GC_TO_SPIRO, "Kaggle GC")
            if not records:
                records = convert_image_folder(raw, KAGGLE_GC_TO_SPIRO, "Kaggle GC")
            all_records.extend(records)

    # ── Recyclable HH ────────────────────────────────────────────────────────
    if do_all or args.recyclable:
        print("\n📦 Recyclable & Household Waste")
        raw = download_recyclable_hh() if not args.skip_download else RAW / "recyclable_hh"
        if raw.exists():
            records = convert_yolo_dataset(raw, RECYCLABLE_HH_TO_SPIRO, "Recyclable HH")
            if not records:
                records = convert_image_folder(raw, RECYCLABLE_HH_TO_SPIRO, "Recyclable HH")
            all_records.extend(records)

    if not all_records:
        print("\n❌ No records collected. Check dataset downloads.")
        sys.exit(1)

    # ── Shuffle and split 80/10/10 ───────────────────────────────────────────
    import random
    random.seed(42)
    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    splits = {
        "train": all_records[:n_train],
        "val":   all_records[n_train:n_train + n_val],
        "test":  all_records[n_train + n_val:],
    }

    print(f"\n✅ Total records: {n}")
    total_written = 0
    for split_name, recs in splits.items():
        written = write_split(recs, OUT / split_name)
        print(f"  {split_name}: {written} images")
        total_written += written

    yaml_path = create_dataset_yaml(OUT)

    print(f"\n{'='*55}")
    print(f"  SPIRO Dataset Ready")
    print(f"{'='*55}")
    print(f"  Total images written:  {total_written}")
    print(f"  Dataset YAML:          {yaml_path}")
    print(f"  Train:                 {splits['train'].__len__()} images")
    print(f"  Val:                   {splits['val'].__len__()} images")
    print(f"  Test:                  {splits['test'].__len__()} images")
    print(f"  Classes:               {NUM_CLASSES}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
