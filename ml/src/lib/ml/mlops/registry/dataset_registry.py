"""
SPIRO ML — DatasetRegistry
Versioned, JSON-backed registry of SPIRO dataset snapshots.
Tracks lineage, checksums, split statistics, and source datasets.

Storage
-------
mlops/datasets/
  registry.json
  <dataset_id>/
    <version>/
      dataset_version.json
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger
from lib.ml.mlops.registry.dataset_version import DatasetVersion

log = get_logger(__name__)
_LOCK = threading.Lock()


class DatasetRegistry:
    """
    Versioned registry of SPIRO dataset snapshots.

    Example
    -------
    >>> registry = DatasetRegistry()
    >>> dv = DatasetVersion(dataset_id="spiro_v1", version="v1.0.0", ...)
    >>> registry.register(dv)
    >>> latest = registry.get_latest("spiro_v1")
    """

    def __init__(self, root: Path = Path("mlops/datasets")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry.json"
        self._index: Dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, dv: DatasetVersion) -> None:
        """Register a new dataset version."""
        with _LOCK:
            if not dv.checksum:
                dv.compute_checksum()

            ver_dir = self.root / dv.dataset_id / dv.version
            ver_dir.mkdir(parents=True, exist_ok=True)
            (ver_dir / "dataset_version.json").write_text(dv.to_json())

            if dv.dataset_id not in self._index:
                self._index[dv.dataset_id] = {}
            self._index[dv.dataset_id][dv.version] = {
                "created_at": dv.created_at,
                "image_count": dv.image_count,
                "annotation_count": dv.annotation_count,
                "checksum": dv.checksum,
                "parent_version": dv.parent_version,
                "path": str(ver_dir / "dataset_version.json"),
            }
            self._save_index()
        log.info(
            f"Registered dataset {dv.dataset_id} v{dv.version} "
            f"({dv.image_count} images, checksum={dv.checksum[:8]}...)"
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, dataset_id: str, version: str) -> Optional[DatasetVersion]:
        path = self.root / dataset_id / version / "dataset_version.json"
        if not path.exists():
            return None
        return DatasetVersion.from_json(path.read_text())

    def get_latest(self, dataset_id: str) -> Optional[DatasetVersion]:
        versions = sorted(self._index.get(dataset_id, {}).keys())
        if not versions:
            return None
        return self.get(dataset_id, versions[-1])

    def get_all_versions(self, dataset_id: str) -> List[DatasetVersion]:
        versions = sorted(self._index.get(dataset_id, {}).keys())
        return [v for ver in versions if (v := self.get(dataset_id, ver)) is not None]

    def list_datasets(self) -> List[str]:
        return list(self._index.keys())

    def next_version(self, dataset_id: str) -> str:
        """Auto-increment semantic version."""
        versions = sorted(self._index.get(dataset_id, {}).keys())
        if not versions:
            return "v1.0.0"
        last = versions[-1].lstrip("v")
        parts = last.split(".")
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"v{major}.{minor}.{patch + 1}"
        except (ValueError, IndexError):
            return f"v{len(versions) + 1}.0.0"

    # ------------------------------------------------------------------
    # Scan from disk
    # ------------------------------------------------------------------

    def snapshot_from_disk(
        self,
        dataset_id: str,
        dataset_root: Path,
        description: str = "",
        parent_version: Optional[str] = None,
        source_datasets: Optional[List[str]] = None,
    ) -> DatasetVersion:
        """
        Scan a YOLO-format dataset directory and create a DatasetVersion.
        Counts images and annotations per split and computes per-class stats.
        """
        dataset_root = Path(dataset_root)
        version = self.next_version(dataset_id)
        class_dist: Dict[str, int] = {}
        image_count = annotation_count = 0
        split_checksums: Dict[str, str] = {}

        for split in ["train", "val", "test"]:
            split_dir = dataset_root / split
            imgs_dir = split_dir / "images"
            lbls_dir = split_dir / "labels"
            if not imgs_dir.exists():
                continue
            imgs = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png"))
            image_count += len(imgs)
            # Compute split image-list hash
            h = hashlib.sha256()
            for img in sorted(imgs):
                h.update(img.name.encode())
            split_checksums[split] = h.hexdigest()
            # Count annotations
            if lbls_dir.exists():
                for lbl in lbls_dir.glob("*.txt"):
                    try:
                        lines = lbl.read_text().strip().splitlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                annotation_count += 1
                                cls_id = parts[0]
                                class_dist[cls_id] = class_dist.get(cls_id, 0) + 1
                    except Exception:
                        pass

        dv = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            description=description,
            image_count=image_count,
            annotation_count=annotation_count,
            class_distribution=class_dist,
            split_checksums=split_checksums,
            parent_version=parent_version,
            source_datasets=source_datasets or [],
            root_path=str(dataset_root),
        )
        dv.compute_checksum()
        return dv

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def export_versions_json(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/dataset_versions.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for did in self.list_datasets():
            for dv in self.get_all_versions(did):
                records.append(dv.to_dict())
        with open(output_path, "w") as f:
            json.dump({"datasets": records, "total": len(records)}, f, indent=2)
        log.info(f"Dataset versions report → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except json.JSONDecodeError:
                log.warning("dataset registry.json corrupt — resetting")
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))
