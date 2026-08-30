#!/usr/bin/env python3
"""
SPIRO ML — training/duplicate_detector.py
Detects exact and near-duplicate images using MD5 and perceptual hashing.

Usage
-----
    # Detect duplicates in a directory
    python training/duplicate_detector.py --dir datasets/merged

    # Detect near-duplicates with stricter threshold (lower = stricter)
    python training/duplicate_detector.py --dir datasets/merged --phash-threshold 4

    # Remove duplicates automatically
    python training/duplicate_detector.py --dir datasets/merged --remove

    # Report only
    python training/duplicate_detector.py --dir datasets/merged --report reports/dups.json
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect duplicate images in SPIRO dataset")
    p.add_argument("--dir", required=True, help="Dataset images directory (or root with images/)")
    p.add_argument("--phash-threshold", type=int, default=8,
                   help="Hamming distance for near-duplicate detection (0=exact match)")
    p.add_argument("--remove", action="store_true", help="Remove detected duplicates")
    p.add_argument("--report", default=None, help="Write JSON report to this path")
    return p.parse_args()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    return sum(int(b) << i for i, b in enumerate(diff.flatten()))


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from tqdm import tqdm

    log = get_logger("duplicate_detector")
    root = Path(args.dir)

    # Support both flat dir and root/images/
    if (root / "images").exists():
        images_dir = root / "images"
        labels_dir = root / "labels"
    else:
        images_dir = root
        labels_dir = None

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    log.info(f"Scanning {len(images)} images in {images_dir}")

    # ── MD5 exact dedup ──────────────────────────────────────────────────────
    hash_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for p in tqdm(images, desc="MD5 hashing"):
        h = md5(p)
        hash_to_paths[h].append(p)

    exact_groups = [g for g in hash_to_paths.values() if len(g) > 1]
    exact_dups = [p for g in exact_groups for p in g[1:]]  # keep first

    # ── dHash near-dedup ─────────────────────────────────────────────────────
    hashes: List = []
    good_images = [p for p in images if p not in set(exact_dups)]
    for p in tqdm(good_images, desc="dHash computing"):
        h = dhash(p)
        if h is not None:
            hashes.append((p, h))

    near_groups: List[List[Path]] = []
    used: Set[int] = set()
    for i, (pi, hi) in enumerate(tqdm(hashes, desc="Near-dup detection")):
        if i in used:
            continue
        group = [pi]
        for j in range(i + 1, len(hashes)):
            if j in used:
                continue
            pj, hj = hashes[j]
            dist = bin(hi ^ hj).count("1")
            if dist <= args.phash_threshold:
                group.append(pj)
                used.add(j)
        if len(group) > 1:
            near_groups.append(group)
            used.add(i)

    near_dups = [p for g in near_groups for p in g[1:]]

    # ── Report ───────────────────────────────────────────────────────────────
    report = {
        "total_images": len(images),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicates_to_remove": len(exact_dups),
        "near_duplicate_groups": len(near_groups),
        "near_duplicates_to_remove": len(near_dups),
        "phash_threshold": args.phash_threshold,
        "exact_groups": [[str(p) for p in g] for g in exact_groups],
        "near_groups": [[str(p) for p in g] for g in near_groups],
    }

    log.info(f"Results:")
    log.info(f"  Exact duplicate groups:     {len(exact_groups)}")
    log.info(f"  Exact duplicates to remove: {len(exact_dups)}")
    log.info(f"  Near-duplicate groups:      {len(near_groups)}")
    log.info(f"  Near-dups to remove:        {len(near_dups)}")

    if args.remove:
        all_to_remove = set(exact_dups) | set(near_dups)
        removed = 0
        for p in all_to_remove:
            try:
                p.unlink()
                if labels_dir:
                    lbl = labels_dir / f"{p.stem}.txt"
                    lbl.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                log.warning(f"Could not remove {p}: {e}")
        log.info(f"Removed {removed} duplicate images")
        report["removed"] = removed

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Report → {out}")


if __name__ == "__main__":
    main()
