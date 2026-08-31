"""
SPIRO ML — DatasetMapper
Converts each source dataset's annotations into SPIRO YOLO format.

Handles:
  - COCO JSON  (TACO, ZeroWaste, MJU-Waste)
  - Image folder classification  (TrashNet, KaggleGC)
  - OLM JSON   (OpenLitterMap)
  - YOLO txt   (WADE-ai)

All output goes to: datasets/mapped/<dataset_name>/
  images/    ← copied/symlinked images
  labels/    ← YOLO .txt files with SPIRO class IDs
  meta.json  ← per-image metadata
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

_MAPPED_DIR = Path("datasets/mapped")
_MAPPINGS_DIR = Path("configs/mappings")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# =============================================================================
# Base mapper
# =============================================================================

class BaseMapper:
    """
    Base class for all format-specific mappers.

    Subclasses implement `map()` which reads raw annotations,
    applies the SPIRO class mapping, and writes YOLO .txt files.
    """

    dataset_name: str = ""
    mapping_file: str = ""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Optional[Path] = None,
        mappings_dir: Path = _MAPPINGS_DIR,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir or (_MAPPED_DIR / self.dataset_name))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)

        self.taxonomy = TaxonomyLoader()
        self._mapping_cfg = self._load_mapping(mappings_dir / self.mapping_file)
        self._mappings: Dict[str, List[int]] = self._parse_mappings(self._mapping_cfg)
        self._excluded: set = set(self._mapping_cfg.get("excluded", []))
        self._report: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "images_converted": 0,
            "annotations_converted": 0,
            "annotations_skipped": 0,
            "class_distribution": {},
            "unmapped_classes": [],
        }

    @staticmethod
    def _load_mapping(path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _parse_mappings(cfg: Dict) -> Dict[str, List[int]]:
        """Extract source_class → [spiro_ids] from config."""
        raw = cfg.get("mappings") or cfg.get("mappings_wade", {})
        return {k.lower(): v["spiro_ids"] for k, v in raw.items()}

    def resolve_spiro_id(self, source_class: str) -> Optional[int]:
        """Return the primary SPIRO class ID for a source class name."""
        key = source_class.lower().strip()
        ids = self._mappings.get(key)
        if ids is None:
            if key not in self._report["unmapped_classes"]:
                self._report["unmapped_classes"].append(key)
            return None
        if not ids:
            return None  # explicitly excluded
        return ids[0]

    def _record_annotation(self, spiro_id: int) -> None:
        name = self.taxonomy.id_to_name(spiro_id)
        self._report["class_distribution"][name] = (
            self._report["class_distribution"].get(name, 0) + 1
        )
        self._report["annotations_converted"] += 1

    def map(self) -> Tuple[int, int]:
        """Run the mapping. Returns (images_converted, annotations_converted)."""
        raise NotImplementedError

    def report(self) -> Dict[str, Any]:
        return self._report

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_image(src: Path, dst: Path) -> bool:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            log.warning(f"Cannot copy {src}: {e}")
            return False

    @staticmethod
    def _write_yolo_label(path: Path, annotations: List[Tuple[int, float, float, float, float]]) -> None:
        with open(path, "w") as f:
            for cls_id, cx, cy, w, h in annotations:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    @staticmethod
    def _coco_bbox_to_yolo(
        x: float, y: float, w: float, h: float, img_w: int, img_h: int
    ) -> Tuple[float, float, float, float]:
        """COCO [x_min, y_min, width, height] → YOLO [cx, cy, w, h] normalised."""
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return (
            max(0.0, min(1.0, cx)),
            max(0.0, min(1.0, cy)),
            max(0.0, min(1.0, nw)),
            max(0.0, min(1.0, nh)),
        )

    @staticmethod
    def _seg_to_bbox(segmentation: List, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert COCO RLE or polygon segmentation to tight YOLO bbox."""
        if not segmentation:
            return (0.5, 0.5, 1.0, 1.0)
        if isinstance(segmentation[0], list):
            pts = np.array(segmentation[0]).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
        else:
            # RLE — fall back to full image
            return (0.5, 0.5, 1.0, 1.0)
        w = x2 - x1
        h = y2 - y1
        return BaseMapper._coco_bbox_to_yolo(x1, y1, w, h, img_w, img_h)


# =============================================================================
# COCO mapper (TACO, ZeroWaste, MJU-Waste)
# =============================================================================

class COCOMapper(BaseMapper):
    """
    Maps COCO-format annotations (bbox + optional segmentation) to SPIRO YOLO.
    Used by TACO, ZeroWaste-f, MJU-Waste.
    """

    def __init__(
        self,
        dataset_name: str,
        mapping_file: str,
        raw_dir: Path,
        annotations_json: Path,
        images_dir: Path,
        output_dir: Optional[Path] = None,
        use_segmentation: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.mapping_file = mapping_file
        super().__init__(raw_dir, output_dir)
        self._ann_json = annotations_json
        self._images_dir = images_dir
        self._use_seg = use_segmentation

    def map(self) -> Tuple[int, int]:
        log.info(f"Mapping {self.dataset_name} (COCO format)")
        with open(self._ann_json) as f:
            coco = json.load(f)

        # Build category_id → name
        cat_map: Dict[int, str] = {c["id"]: c["name"].lower() for c in coco["categories"]}

        # Build image_id → image info
        img_map: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}

        # Group annotations by image
        img_to_anns: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        for img_id, img_info in tqdm(img_map.items(), desc=f"Mapping {self.dataset_name}"):
            file_name = img_info.get("file_name", "")
            src_img = self._images_dir / file_name
            if not src_img.exists():
                # Try flat filename
                src_img = self._images_dir / Path(file_name).name
            if not src_img.exists():
                log.debug(f"Image not found: {file_name}")
                continue

            img_w = img_info.get("width", 0)
            img_h = img_info.get("height", 0)
            if img_w == 0 or img_h == 0:
                img_bgr = cv2.imread(str(src_img))
                if img_bgr is None:
                    continue
                img_h, img_w = img_bgr.shape[:2]

            yolo_anns: List[Tuple] = []
            for ann in img_to_anns.get(img_id, []):
                cat_name = cat_map.get(ann["category_id"], "")
                if cat_name in self._excluded:
                    self._report["annotations_skipped"] += 1
                    continue

                spiro_id = self.resolve_spiro_id(cat_name)
                if spiro_id is None:
                    self._report["annotations_skipped"] += 1
                    continue

                if self._use_seg and ann.get("segmentation"):
                    bbox = self._seg_to_bbox(ann["segmentation"], img_w, img_h)
                else:
                    x, y, w, h = ann["bbox"]
                    bbox = self._coco_bbox_to_yolo(x, y, w, h, img_w, img_h)

                yolo_anns.append((spiro_id, *bbox))
                self._record_annotation(spiro_id)

            # Write outputs
            stem = f"{self.dataset_name}_{img_id:08d}"
            dst_img = self.output_dir / "images" / f"{stem}{src_img.suffix.lower()}"
            dst_lbl = self.output_dir / "labels" / f"{stem}.txt"

            if self._copy_image(src_img, dst_img):
                self._write_yolo_label(dst_lbl, yolo_anns)
                self._report["images_converted"] += 1

        log.info(
            f"{self.dataset_name}: {self._report['images_converted']} images, "
            f"{self._report['annotations_converted']} annotations mapped"
        )
        return self._report["images_converted"], self._report["annotations_converted"]


# =============================================================================
# Classification folder mapper (TrashNet, KaggleGC)
# =============================================================================

class ClassificationFolderMapper(BaseMapper):
    """
    Maps image-folder classification datasets to SPIRO YOLO format.
    Generates a whole-image pseudo-bbox annotation.

    Directory structure assumed:
        root/
          class_name_a/
            img001.jpg
          class_name_b/
            img002.jpg
    """

    def __init__(
        self,
        dataset_name: str,
        mapping_file: str,
        root_dir: Path,
        output_dir: Optional[Path] = None,
        centre_crop_fraction: float = 0.85,
    ) -> None:
        self.dataset_name = dataset_name
        self.mapping_file = mapping_file
        super().__init__(root_dir, output_dir)
        self._root = Path(root_dir)
        self._crop_frac = centre_crop_fraction

    def map(self) -> Tuple[int, int]:
        log.info(f"Mapping {self.dataset_name} (classification folder format)")
        pseudo_cfg = self._mapping_cfg.get("pseudo_bbox", {})
        crop_frac = pseudo_cfg.get("centre_crop_fraction", self._crop_frac)

        for class_dir in sorted(self._root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            if class_name in self._excluded:
                continue

            spiro_id = self.resolve_spiro_id(class_name)
            if spiro_id is None:
                continue

            images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            for img_path in tqdm(images, desc=f"{self.dataset_name}/{class_name}", leave=False):
                stem = f"{self.dataset_name}_{class_name}_{img_path.stem}"
                dst_img = self.output_dir / "images" / f"{stem}{img_path.suffix.lower()}"
                dst_lbl = self.output_dir / "labels" / f"{stem}.txt"

                if not self._copy_image(img_path, dst_img):
                    continue

                # Pseudo-bbox: centre crop fraction
                margin = (1.0 - crop_frac) / 2.0
                cx, cy = 0.5, 0.5
                bw, bh = crop_frac, crop_frac
                self._write_yolo_label(dst_lbl, [(spiro_id, cx, cy, bw, bh)])
                self._record_annotation(spiro_id)
                self._report["images_converted"] += 1

        log.info(
            f"{self.dataset_name}: {self._report['images_converted']} images, "
            f"{self._report['annotations_converted']} annotations mapped"
        )
        return self._report["images_converted"], self._report["annotations_converted"]


# =============================================================================
# OLM JSON mapper (OpenLitterMap)
# =============================================================================

class OLMMapper(BaseMapper):
    """
    Maps OpenLitterMap custom JSON annotations to SPIRO YOLO format.

    OLM JSON schema (per image):
    {
      "filename": "abc.jpg",
      "annotations": [
        {"label": "cigarette_butt", "x": 0.1, "y": 0.2, "width": 0.05, "height": 0.05}
      ]
    }
    Coordinates are already normalised [0,1] in OLM format.
    """

    dataset_name = "OpenLitterMap"
    mapping_file = "openlittermap_to_spiro.yaml"

    def __init__(self, raw_dir: Path, annotations_json: Path, images_dir: Path,
                 output_dir: Optional[Path] = None) -> None:
        super().__init__(raw_dir, output_dir)
        self._ann_json = annotations_json
        self._images_dir = images_dir

    def map(self) -> Tuple[int, int]:
        log.info("Mapping OpenLitterMap")
        if not self._ann_json.exists():
            log.warning(f"OLM annotations not found at {self._ann_json}")
            return 0, 0

        with open(self._ann_json) as f:
            data = json.load(f)

        # OLM may be a list of records or a dict with "images" key
        records = data if isinstance(data, list) else data.get("images", [])

        for record in tqdm(records, desc="Mapping OpenLitterMap"):
            filename = record.get("filename") or record.get("file_name", "")
            src_img = self._images_dir / filename
            if not src_img.exists():
                src_img = self._images_dir / Path(filename).name
            if not src_img.exists():
                continue

            yolo_anns: List[Tuple] = []
            for ann in record.get("annotations", []):
                label = ann.get("label", "").lower().replace(" ", "_")
                if label in self._excluded:
                    self._report["annotations_skipped"] += 1
                    continue
                spiro_id = self.resolve_spiro_id(label)
                if spiro_id is None:
                    self._report["annotations_skipped"] += 1
                    continue

                # OLM uses normalised coords
                cx = float(ann.get("x", 0.5))
                cy = float(ann.get("y", 0.5))
                bw = float(ann.get("width", 0.1))
                bh = float(ann.get("height", 0.1))
                # Clip
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))

                yolo_anns.append((spiro_id, cx, cy, bw, bh))
                self._record_annotation(spiro_id)

            stem = f"olm_{Path(filename).stem}"
            dst_img = self.output_dir / "images" / f"{stem}{src_img.suffix.lower()}"
            dst_lbl = self.output_dir / "labels" / f"{stem}.txt"
            if self._copy_image(src_img, dst_img):
                self._write_yolo_label(dst_lbl, yolo_anns)
                self._report["images_converted"] += 1

        log.info(
            f"OLM: {self._report['images_converted']} images, "
            f"{self._report['annotations_converted']} annotations"
        )
        return self._report["images_converted"], self._report["annotations_converted"]


# =============================================================================
# YOLO txt mapper (WADE-ai)
# =============================================================================

class YOLOMapper(BaseMapper):
    """
    Maps existing YOLO-format annotations with a different class vocabulary
    into SPIRO YOLO format.
    """

    def __init__(
        self,
        dataset_name: str,
        mapping_file: str,
        raw_dir: Path,
        source_classes: List[str],
        output_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.mapping_file = mapping_file
        super().__init__(raw_dir, output_dir)
        self._source_classes = [c.lower() for c in source_classes]

    def map(self) -> Tuple[int, int]:
        log.info(f"Mapping {self.dataset_name} (YOLO format)")
        images_dir = self.raw_dir / "images"
        labels_dir = self.raw_dir / "labels"

        if not images_dir.exists():
            log.warning(f"No images/ dir at {self.raw_dir}")
            return 0, 0

        for img_path in tqdm(
            sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            desc=f"Mapping {self.dataset_name}",
        ):
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            yolo_anns: List[Tuple] = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    src_cls_id = int(parts[0])
                    if src_cls_id >= len(self._source_classes):
                        self._report["annotations_skipped"] += 1
                        continue
                    src_name = self._source_classes[src_cls_id]
                    spiro_id = self.resolve_spiro_id(src_name)
                    if spiro_id is None:
                        self._report["annotations_skipped"] += 1
                        continue

                    cx, cy, w, h = map(float, parts[1:])
                    yolo_anns.append((spiro_id, cx, cy, w, h))
                    self._record_annotation(spiro_id)

            stem = f"{self.dataset_name}_{img_path.stem}"
            dst_img = self.output_dir / "images" / f"{stem}{img_path.suffix.lower()}"
            dst_lbl = self.output_dir / "labels" / f"{stem}.txt"
            if self._copy_image(img_path, dst_img):
                self._write_yolo_label(dst_lbl, yolo_anns)
                self._report["images_converted"] += 1

        log.info(
            f"{self.dataset_name}: {self._report['images_converted']} images, "
            f"{self._report['annotations_converted']} annotations"
        )
        return self._report["images_converted"], self._report["annotations_converted"]


# =============================================================================
# Factory
# =============================================================================

def build_taco_mapper(raw_dir: Path, output_dir: Optional[Path] = None) -> COCOMapper:
    return COCOMapper(
        dataset_name="TACO",
        mapping_file="taco_to_spiro.yaml",
        raw_dir=raw_dir,
        annotations_json=raw_dir / "TACO" / "annotations.json",
        images_dir=raw_dir / "TACO" / "images",
        output_dir=output_dir,
        use_segmentation=False,
    )


def build_trashnet_mapper(raw_dir: Path, output_dir: Optional[Path] = None) -> ClassificationFolderMapper:
    return ClassificationFolderMapper(
        dataset_name="TrashNet",
        mapping_file="trashnet_to_spiro.yaml",
        root_dir=raw_dir / "TrashNet" / "dataset-resized",
        output_dir=output_dir,
    )


def build_zerowaste_mapper(raw_dir: Path, output_dir: Optional[Path] = None) -> COCOMapper:
    return COCOMapper(
        dataset_name="ZeroWaste",
        mapping_file="zerowaste_to_spiro.yaml",
        raw_dir=raw_dir,
        annotations_json=raw_dir / "ZeroWaste" / "zerowaste-f" / "train" / "annotations.json",
        images_dir=raw_dir / "ZeroWaste" / "zerowaste-f" / "train",
        output_dir=output_dir,
        use_segmentation=True,
    )


def build_olm_mapper(raw_dir: Path, output_dir: Optional[Path] = None) -> OLMMapper:
    return OLMMapper(
        raw_dir=raw_dir / "OpenLitterMap",
        annotations_json=raw_dir / "OpenLitterMap" / "annotations.json",
        images_dir=raw_dir / "OpenLitterMap" / "images",
        output_dir=output_dir,
    )


def build_kaggle_gc_mapper(raw_dir: Path, output_dir: Optional[Path] = None) -> ClassificationFolderMapper:
    return ClassificationFolderMapper(
        dataset_name="KaggleGC",
        mapping_file="kaggle_gc_to_spiro.yaml",
        root_dir=raw_dir / "KaggleGC",
        output_dir=output_dir,
    )
