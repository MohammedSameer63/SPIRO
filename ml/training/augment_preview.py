#!/usr/bin/env python3
"""
SPIRO ML — training/augment_preview.py
Generates visual augmentation previews for a given image.
Useful for verifying pipeline settings before full training.

Usage
-----
    python training/augment_preview.py --image datasets/processed/train/images/sample.jpg
    python training/augment_preview.py --image sample.jpg --n 8 --output-dir previews/
    python training/augment_preview.py --image sample.jpg --labels sample.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview augmentation pipeline on an image")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--labels", default=None, help="YOLO label file path (optional)")
    p.add_argument("--n", type=int, default=6, help="Number of augmented previews")
    p.add_argument("--output-dir", default="reports/aug_previews")
    p.add_argument("--image-size", type=int, nargs=2, default=[640, 640],
                   metavar=("W", "H"), help="Augmentation target size")
    p.add_argument("--show-grid", action="store_true",
                   help="Combine previews into a single grid image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline
    import cv2
    import numpy as np

    log = get_logger("augment_preview")
    img_path = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        log.error(f"Image not found: {img_path}")
        sys.exit(1)

    # Determine label path
    lbl_path = None
    if args.labels:
        lbl_path = Path(args.labels)
    else:
        # Try sibling labels/
        candidate = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if candidate.exists():
            lbl_path = candidate

    log.info(f"Image:  {img_path}")
    log.info(f"Labels: {lbl_path or 'none'}")
    log.info(f"Generating {args.n} augmented previews → {out_dir}")

    aug = AugmentationPipeline(
        image_size=tuple(args.image_size),
        factor=args.n,
    )
    previews = aug.preview(
        image_path=img_path,
        labels_path=lbl_path,
        n_samples=args.n,
        output_dir=out_dir,
    )

    if args.show_grid and previews:
        # Build a grid image
        rows = 2
        cols = (args.n + 1) // 2
        h, w = previews[0].shape[:2]

        # Add original
        original = cv2.imread(str(img_path))
        if original is not None:
            original = cv2.resize(original, (w, h))
            previews.insert(0, original)

        # Pad to grid size
        while len(previews) < rows * cols:
            previews.append(np.zeros((h, w, 3), dtype=np.uint8))

        rows_imgs = []
        for r in range(rows):
            row_imgs = previews[r * cols: (r + 1) * cols]
            rows_imgs.append(np.hstack(row_imgs))
        grid = np.vstack(rows_imgs)

        grid_path = out_dir / "preview_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        log.info(f"Grid saved → {grid_path}")

    log.info(f"Done. {len(previews)} previews in {out_dir}")


if __name__ == "__main__":
    main()
