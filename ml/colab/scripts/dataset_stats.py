#!/usr/bin/env python3
"""
SPIRO — dataset_stats.py
Prints a summary of the merged SPIRO dataset:
  - Total images per split
  - Class distribution (top/bottom 20)
  - Images per source dataset
  - Label file health check
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

BASE     = Path("/content/spiro")
DATA_DIR = BASE / "datasets" / "spiro_merged"
NUM_CLASSES = 109


def load_class_names() -> list:
    tax_path = BASE / "configs" / "taxonomy" / "spiro_taxonomy.yaml"
    if tax_path.exists():
        with open(tax_path) as f:
            tax = yaml.safe_load(f)
        names = tax.get("names", {})
        if isinstance(names, dict):
            return [names.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
        return list(names)

    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        names = d.get("names", [])
        if isinstance(names, dict):
            return [names.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
        return names
    return [f"class_{i}" for i in range(NUM_CLASSES)]


def count_split(split_dir: Path) -> tuple:
    """Returns (n_images, n_labels, class_counter, missing_labels)."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        return 0, 0, Counter(), 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
    cls_counter: Counter = Counter()
    missing = 0
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing += 1
            continue
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_counter[int(parts[0])] += 1
                except ValueError:
                    pass
    return len(images), sum(cls_counter.values()), cls_counter, missing


def bar(n: int, total: int, width: int = 30) -> str:
    filled = int(width * n / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    if not DATA_DIR.exists():
        print(f"❌ Dataset not found: {DATA_DIR}")
        print("   Run: python scripts/setup_datasets.py --all")
        sys.exit(1)

    class_names = load_class_names()
    total_cls_counter: Counter = Counter()
    split_stats = {}

    print(f"\n{'='*60}")
    print(f"  SPIRO Dataset Statistics")
    print(f"{'='*60}")

    for split in ["train", "val", "test"]:
        sd = DATA_DIR / split
        n_img, n_lbl, cls_cnt, missing = count_split(sd)
        split_stats[split] = {"images": n_img, "labels": n_lbl, "missing": missing}
        total_cls_counter.update(cls_cnt)
        print(f"  {split:<6}: {n_img:>6} images | {n_lbl:>8} annotations"
              f"{' | ⚠️ ' + str(missing) + ' missing labels' if missing else ''}")

    total_images = sum(s["images"] for s in split_stats.values())
    total_annots = sum(s["labels"] for s in split_stats.values())
    print(f"  {'TOTAL':<6}: {total_images:>6} images | {total_annots:>8} annotations")
    print(f"\n  Classes with annotations: {len(total_cls_counter)}/{NUM_CLASSES}")

    # data.yaml check
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        nc = d.get("nc", 0)
        status = "✅" if nc == NUM_CLASSES else "❌"
        print(f"  data.yaml nc={nc} {status}")
    else:
        print(f"  ⚠️  data.yaml not found")

    # Top 20 most common classes
    print(f"\n  {'─'*58}")
    print(f"  Top 20 classes by annotation count:")
    print(f"  {'─'*58}")
    top20 = total_cls_counter.most_common(20)
    max_cnt = top20[0][1] if top20 else 1
    for cls_id, cnt in top20:
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        pct  = cnt / max(total_annots, 1) * 100
        print(f"  {cls_id:>3} {name:<30} {cnt:>7} {pct:>5.1f}%  {bar(cnt, max_cnt, 15)}")

    # Bottom 20 (rare classes)
    print(f"\n  {'─'*58}")
    print(f"  Bottom 20 classes (rarest — may need oversampling):")
    print(f"  {'─'*58}")
    bottom20 = total_cls_counter.most_common()[:-21:-1]
    for cls_id, cnt in bottom20:
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        pct  = cnt / max(total_annots, 1) * 100
        print(f"  {cls_id:>3} {name:<30} {cnt:>7} {pct:>5.1f}%")

    # Classes with zero annotations
    all_ids = set(range(NUM_CLASSES))
    annotated = set(total_cls_counter.keys())
    zero_classes = all_ids - annotated
    if zero_classes:
        print(f"\n  ⚠️  {len(zero_classes)} classes with 0 annotations:")
        for cid in sorted(zero_classes):
            name = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
            print(f"     {cid:>3}: {name}")

    # 7-stream breakdown
    show_stream_breakdown(class_names, total_cls_counter)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()


def show_stream_breakdown(class_names: list, total_cls_counter) -> None:
    """Show annotation counts grouped by the 7 SPIRO disposal streams."""
    import yaml
    tax_path = BASE / "configs" / "taxonomy" / "spiro_taxonomy.yaml"
    if not tax_path.exists():
        return
    with open(tax_path) as f:
        tax = yaml.safe_load(f)
    streams = tax.get("streams", {})
    if not streams:
        return

    STREAM_ORDER = ["organic", "recoverable", "non_recoverable",
                    "hazardous", "sanitary", "e_waste", "reject"]

    print(f"\n  {'─'*58}")
    print(f"  Annotations by Disposal Stream:")
    print(f"  {'─'*58}")
    for sname in STREAM_ORDER:
        if sname not in streams:
            continue
        sdata = streams[sname]
        ids   = sdata.get("ids", [])
        label = sdata.get("label", sname)
        count = sum(total_cls_counter.get(i, 0) for i in ids)
        classes_present = sum(1 for i in ids if total_cls_counter.get(i, 0) > 0)
        print(f"  {label:<28} {count:>8} annotations  "
              f"({classes_present}/{len(ids)} classes present)")
