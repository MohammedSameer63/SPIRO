#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_validator.py
Validates a YOLO-format dataset directory for:
  - Image readability
  - Label format correctness
  - Bounding box range
  - Class ID validity
  - Resolution minimums
  - Annotation coverage

Usage
-----
    python training/dataset_validator.py --dir datasets/merged
    python training/dataset_validator.py --dir datasets/processed/train --report reports/val.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SPIRO YOLO dataset")
    p.add_argument("--dir", required=True, help="Dataset root (contains images/ labels/)")
    p.add_argument("--num-classes", type=int, default=109, help="Expected number of classes")
    p.add_argument("--min-width", type=int, default=64)
    p.add_argument("--min-height", type=int, default=64)
    p.add_argument("--report", default=None, help="Write JSON report to this path")
    p.add_argument("--strict", action="store_true", help="Exit 1 if any validation errors found")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    import cv2

    log = get_logger("dataset_validator")
    dataset_dir = Path(args.dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    report = {
        "dataset_dir": str(dataset_dir),
        "total_images": len(images),
        "errors": [],
        "warnings": [],
        "pass": True,
    }

    corrupt = 0
    bad_labels = 0
    low_res = 0
    missing_labels = 0
    bad_class = 0

    for img_path in images:
        # Readability
        img = cv2.imread(str(img_path))
        if img is None:
            report["errors"].append(f"Corrupt/unreadable: {img_path.name}")
            corrupt += 1
            continue

        ih, iw = img.shape[:2]
        if iw < args.min_width or ih < args.min_height:
            report["warnings"].append(f"Low resolution {iw}×{ih}: {img_path.name}")
            low_res += 1

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            report["warnings"].append(f"Missing label: {img_path.name}")
            missing_labels += 1
            continue

        with open(lbl_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    report["errors"].append(
                        f"{img_path.stem}.txt:{i} — expected 5 fields, got {len(parts)}"
                    )
                    bad_labels += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    report["errors"].append(f"{img_path.stem}.txt:{i} — non-numeric values")
                    bad_labels += 1
                    continue
                if cls_id < 0 or cls_id >= args.num_classes:
                    report["errors"].append(
                        f"{img_path.stem}.txt:{i} — class_id={cls_id} out of [0,{args.num_classes-1}]"
                    )
                    bad_class += 1
                for v, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
                    if not (0.0 <= v <= 1.0):
                        report["errors"].append(
                            f"{img_path.stem}.txt:{i} — {name}={v:.4f} out of [0,1]"
                        )
                        bad_labels += 1

    report["summary"] = {
        "corrupt_images": corrupt,
        "low_resolution": low_res,
        "missing_labels": missing_labels,
        "bad_label_lines": bad_labels,
        "bad_class_ids": bad_class,
    }

    if report["errors"]:
        report["pass"] = False

    # Log summary
    log.info(f"Validation results for {dataset_dir}:")
    log.info(f"  Images:          {len(images)}")
    log.info(f"  Corrupt:         {corrupt}")
    log.info(f"  Low resolution:  {low_res}")
    log.info(f"  Missing labels:  {missing_labels}")
    log.info(f"  Bad label lines: {bad_labels}")
    log.info(f"  Bad class IDs:   {bad_class}")
    log.info(f"  PASS:            {report['pass']}")

    if report["errors"]:
        for e in report["errors"][:20]:
            log.error(f"  ERROR: {e}")
        if len(report["errors"]) > 20:
            log.error(f"  ... and {len(report['errors']) - 20} more errors")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Report written to {out}")

    if args.strict and not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
