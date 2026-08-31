# SPIRO Dataset Engineering Guide

## Overview

This guide documents the complete dataset engineering pipeline for SPIRO ML.
The pipeline downloads, maps, cleans, merges, splits, augments, and versions
all supported waste datasets into a single unified SPIRO-format dataset.

---

## Dataset Sources

| Dataset | Classes | Images | Annotation | License | Notes |
|---|---|---|---|---|---|
| TACO | 60 | ~1,500 | COCO bbox | MIT | Real-world litter, Flickr images |
| TrashNet | 6 | 2,527 | Classification | CC BY 4.0 | No bboxes; pseudo-annotations generated |
| ZeroWaste-f | 4 | ~4,500 | COCO segmentation | CC BY-NC 4.0 | Recycling-bin level categories |
| OpenLitterMap | ~30 | ~40,000+ | OLM JSON bbox | CC BY 4.0 | Crowd-sourced; variable quality |
| Kaggle GC | 12 | 2,467 | Classification | CC0 1.0 | No bboxes; pseudo-annotations generated |
| MJU-Waste | 5 | 2,475 | COCO segmentation | CC BY-NC 4.0 | Bin-level, segmentation only |

---

## SPIRO Taxonomy (109 Classes)

All source datasets are mapped into the unified SPIRO taxonomy.
The taxonomy file is the **single source of truth**:

```
configs/taxonomy/spiro_taxonomy.yaml
```

### Taxonomy Groups

| Group | IDs | Examples |
|---|---|---|
| plastic | 0–21 | plastic_bottle, plastic_bag, plastic_straw |
| glass | 22–26 | glass_bottle, glass_jar, glass_fragment |
| metal | 27–35 | metal_can_beverage, metal_foil, aerosol_can |
| paper | 36–46 | cardboard_box, paper_cup, carton_drink |
| organic | 47–52 | food_waste_generic, fruit_peel, eggshell |
| cigarette | 53–55 | cigarette_butt, cigarette_pack, lighter |
| textile | 56–60 | clothing_item, shoe, mask_disposable |
| electronic | 61–65 | battery, cable_wire, mobile_phone |
| medical | 66–70 | syringe, medicine_blister, diaper |
| ... | ... | ... |
| misc | 106–108 | unknown_litter, wet_waste_generic |

Full taxonomy: `configs/taxonomy/spiro_taxonomy.yaml`

---

## Class Mapping Strategy

Each dataset has a mapping file in `configs/mappings/`:

| File | Dataset |
|---|---|
| `taco_to_spiro.yaml` | TACO |
| `trashnet_to_spiro.yaml` | TrashNet |
| `zerowaste_to_spiro.yaml` | ZeroWaste-f |
| `openlittermap_to_spiro.yaml` | OpenLitterMap |
| `kaggle_gc_to_spiro.yaml` | Kaggle GC |
| `supplementary_to_spiro.yaml` | MJU-Waste + WADE-ai |

### Mapping format

```yaml
mappings:
  "source_class_name":
    spiro_ids: [primary_id, secondary_id]
    notes: "Rationale"
```

The **first ID** in `spiro_ids` is the primary mapping used for detection.
Additional IDs indicate that a source class may represent multiple SPIRO classes.

### Classification → Detection Pseudo-Annotations

For TrashNet and Kaggle GC (which have no bounding boxes), the mapper
generates a whole-image pseudo-bounding-box:

```yaml
pseudo_bbox:
  centre_crop_fraction: 0.85  # bbox is 85% of image dimensions, centred
```

This is clearly labelled as a pseudo-annotation in the metadata.

---

## Pipeline Steps

### Full pipeline (recommended)

```bash
# Run the complete pipeline
python -c "
from src.lib.ml.dataset_engineering import DatasetOrchestrator
orch = DatasetOrchestrator(skip_download=False)
orch.run_full_pipeline()
"
```

### Step-by-step

```bash
# 1. Download all datasets
python training/dataset_download.py

# 2. Map to SPIRO taxonomy
python training/dataset_mapper.py --raw-dir datasets/raw

# 3. Clean mapped data
python training/dataset_cleaner.py --mapped-dir datasets/mapped

# 4. Merge all cleaned datasets
python training/dataset_merger.py

# 5. Split into train/val/test
python training/dataset_splitter.py --train 0.70 --val 0.15 --test 0.15

# 6. Compute statistics
python training/dataset_statistics.py --dir datasets/merged --charts

# 7. Create version snapshot
python training/dataset_versioning.py create --processed-dir datasets/processed
```

---

## Cleaning Pipeline

The `DataCleaner` performs these checks in order:

| Check | Method | Action |
|---|---|---|
| Corrupt images | OpenCV imread + file size | Remove |
| Exact duplicates | MD5 hash | Remove (keep first) |
| Near-duplicates | dHash (Hamming distance ≤ 8) | Remove (keep first) |
| Low resolution | Width/height threshold | Remove |
| Missing labels | File existence check | Flag |
| Orphan labels | No matching image | Remove |
| Out-of-range bbox | Coordinate clipping | Fix or remove |
| Invalid class ID | Range check [0, 108] | Remove annotation |
| Sub-minimum bbox area | `w×h < min_bbox_area` | Remove annotation |

### Perceptual hashing

Near-duplicate detection uses **dHash** (difference hash):
- Image is resized to 9×8 grayscale
- Horizontal pixel differences are encoded as 64 bits
- Two images are near-duplicates if Hamming distance ≤ threshold (default: 8)

Lower threshold = stricter (fewer false positives, may miss some dups).

---

## Augmentation Pipeline

`AugmentationPipeline` uses Albumentations with bounding-box synchronisation.

### Operations

**Geometric**
- Horizontal flip (p=0.5)
- Vertical flip (p=0.05)
- Random 90° rotation (p=0.2)
- Rotation ±20° (p=0.4)
- Shift/scale/rotate (p=0.4)
- Perspective distortion (p=0.3)
- Random resized crop (p=0.3)

**Color / Exposure**
- Color jitter: brightness, contrast, saturation, hue (p=0.5)
- Hue-saturation-value shift (p=0.4)
- CLAHE (p=0.2)
- RGB channel shift (p=0.2)
- Grayscale conversion (p=0.05)

**Blur / Noise**
- Gaussian blur, median blur, motion blur (p=0.2 each, one applied)
- Gaussian noise (p=0.2)
- ISO noise (p=0.15)

**Weather Effects**
- Random fog (p=0.1)
- Random rain (drizzle mode, p=0.1)
- Random shadow (p=0.15)
- Sun flare (p=0.05)

**Degradation**
- JPEG compression artifacts (p=0.15)
- Downscale (p=0.1)
- Coarse dropout / cutout (p=0.3)
- Grid dropout (p=0.1)

### Preview augmentation

```bash
python training/augment_preview.py \
    --image datasets/processed/train/images/sample.jpg \
    --n 8 \
    --output-dir reports/aug_previews \
    --show-grid
```

---

## Directory Structure

```
datasets/
├── raw/                    # Downloaded raw datasets (git-ignored)
│   ├── TACO/
│   │   ├── annotations.json
│   │   └── images/
│   ├── TrashNet/
│   │   └── dataset-resized/
│   ├── ZeroWaste/
│   ├── OpenLitterMap/
│   └── KaggleGC/
│
├── mapped/                 # SPIRO YOLO format per dataset
│   ├── TACO/
│   │   ├── images/
│   │   └── labels/
│   ├── TrashNet/
│   ├── ZeroWaste/
│   ├── OpenLitterMap/
│   └── KaggleGC/
│
├── merged/                 # All datasets merged, cross-deduped
│   ├── images/
│   ├── labels/
│   └── merge_meta.json
│
├── processed/              # Final train/val/test splits
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
│
└── dataset_versions.json   # Version history

reports/
├── dataset_statistics.json
├── dataset_report.md
├── cleaning_report.json
├── mapping_report.json
├── split_report.json
├── merge_report.json
├── dataset_version.json
├── quality_report.json
└── charts/
    ├── class_histogram.png
    ├── bbox_histogram.png
    ├── resolution_scatter.png
    ├── dataset_composition.png
    ├── split_distribution.png
    ├── group_coverage.png
    └── imbalance_heatmap.png
```

---

## Dataset Versioning

Every preprocessing run creates a version snapshot:

```python
from src.lib.ml.dataset_engineering.versioning import DatasetVersioning

dv = DatasetVersioning("datasets")
vid = dv.create_version(
    processed_dir=Path("datasets/processed"),
    sources=["taco", "trashnet", "zerowaste"],
    notes="Added ZeroWaste-f for v2",
)
```

Version ID format: `v{YYYYMMDD_HHMMSS}_{hash[:8]}`

### CLI

```bash
# Create
python training/dataset_versioning.py create --processed-dir datasets/processed

# List
python training/dataset_versioning.py list

# Show
python training/dataset_versioning.py show --version v20240115_143022_abc12345
```

---

## Output Formats

### YOLO annotation format

Each image has a paired `.txt` file:
```
<class_id> <cx> <cy> <width> <height>
```
All values normalised to [0, 1].

### dataset_statistics.json

```json
{
  "name": "SPIRO-Merged",
  "total_images": 42000,
  "total_annotations": 180000,
  "represented_classes": 87,
  "missing_classes": ["syringe", ...],
  "rare_classes": ["propane_tank_small", ...],
  "imbalance_ratio": 245.0,
  "class_distribution": {"plastic_bottle": 8500, ...},
  "group_coverage": {
    "plastic": {"coverage_pct": 95.5, "annotations": 65000},
    ...
  }
}
```

### cleaning_report.json

```json
{
  "dataset_dir": "datasets/mapped/TACO",
  "total_images": 1500,
  "corrupt_images": ["img_0042.jpg"],
  "exact_duplicates": [["img_a.jpg", "img_b.jpg"]],
  "near_duplicates": [],
  "low_resolution": ["img_tiny.jpg"],
  "removed_annotations": 12,
  "clean_images": 1487
}
```

---

## Adding a New Dataset

1. **Create a downloader** in `src/lib/ml/dataset_engineering/downloaders/base.py`
2. **Create a mapping file** in `configs/mappings/<name>_to_spiro.yaml`
3. **Create a mapper** in `src/lib/ml/dataset_engineering/mappers/mapper.py`
4. **Register** in `DatasetOrchestrator._ALL_DATASETS`
5. **Write tests** in `tests/unit/dataset_engineering/`

---

## Quality Scoring

`quality_checker.py` produces a quality score (0–100):

| Deduction | Condition |
|---|---|
| -5× corrupt_pct | Corrupt images |
| -2× exact_dup_pct | Exact duplicates |
| -1× near_dup_pct | Near-duplicates |
| -0.5 per missing class | Up to -20 |
| -0.2 per rare class | Up to -10 |
| -0.1 per 10x above baseline imbalance | Imbalance penalty |

Target: quality score ≥ 75 before training.
