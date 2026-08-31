"""
SPIRO ML — DatasetVersioning
Creates immutable version snapshots of dataset preprocessing runs.

Each version records:
  - Version ID (hash-based)
  - Timestamp
  - Dataset hash (MD5 of all label files)
  - Image count, annotation count
  - Pipeline config snapshot
  - Source datasets included
  - Mapping report reference
  - Cleaning report reference
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_VERSION_FILE = "dataset_versions.json"


class DatasetVersioning:
    """
    Tracks dataset preprocessing versions.

    Parameters
    ----------
    versions_dir : Path
        Directory where version history is stored.

    Example
    -------
    >>> dv = DatasetVersioning(Path("datasets"))
    >>> vid = dv.create_version(
    ...     processed_dir=Path("datasets/processed"),
    ...     sources=["TACO", "TrashNet"],
    ...     config=cfg.as_dict(),
    ... )
    >>> dv.list_versions()
    """

    def __init__(self, versions_dir: Path = Path("datasets")) -> None:
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._version_file = self.versions_dir / _VERSION_FILE
        self._history: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_version(
        self,
        processed_dir: Path,
        sources: List[str],
        config: Optional[Dict] = None,
        notes: str = "",
        stats: Optional[Dict] = None,
    ) -> str:
        """
        Create a new dataset version entry.

        Parameters
        ----------
        processed_dir : Path
            Directory with train/ val/ test/ splits.
        sources : list[str]
            Dataset names included in this version.
        config : dict, optional
            Pipeline config snapshot.
        notes : str
            Human notes.
        stats : dict, optional
            Pre-computed statistics dict.

        Returns
        -------
        str — version ID
        """
        processed_dir = Path(processed_dir)
        dataset_hash = self._hash_labels(processed_dir)
        version_id = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{dataset_hash[:8]}"

        counts = self._count_images_and_anns(processed_dir)

        entry: Dict[str, Any] = {
            "version_id": version_id,
            "created_at": datetime.utcnow().isoformat(),
            "timestamp_unix": int(time.time()),
            "dataset_hash": dataset_hash,
            "sources": sources,
            "processed_dir": str(processed_dir),
            "image_counts": counts["images"],
            "annotation_counts": counts["annotations"],
            "total_images": sum(counts["images"].values()),
            "total_annotations": sum(counts["annotations"].values()),
            "config_snapshot": config or {},
            "notes": notes,
            "stats_summary": self._summarise_stats(stats) if stats else {},
        }

        self._history[version_id] = entry
        self._save()
        log.info(f"Dataset version created: {version_id}")
        return version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all versions sorted by creation time (newest first)."""
        return sorted(
            self._history.values(),
            key=lambda v: v["timestamp_unix"],
            reverse=True,
        )

    def get_version(self, version_id: str) -> Dict[str, Any]:
        if version_id not in self._history:
            raise KeyError(f"Version {version_id!r} not found")
        return self._history[version_id]

    def latest(self) -> Optional[Dict[str, Any]]:
        versions = self.list_versions()
        return versions[0] if versions else None

    def delete_version(self, version_id: str) -> None:
        if version_id not in self._history:
            raise KeyError(f"Version {version_id!r} not found")
        del self._history[version_id]
        self._save()
        log.info(f"Deleted version {version_id}")

    def save_json(self, path: Optional[Path] = None) -> None:
        path = path or (self.versions_dir / "dataset_version.json")
        latest = self.latest()
        with open(path, "w") as f:
            json.dump(
                {"latest": latest, "all_versions": self.list_versions()},
                f, indent=2,
            )
        log.info(f"Version history saved to {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_labels(processed_dir: Path) -> str:
        """
        Compute a deterministic MD5 over all label file contents
        (sorted by filename) to fingerprint the annotation set.
        """
        h = hashlib.md5()
        label_files = sorted(processed_dir.rglob("*.txt"))
        for lbl in label_files:
            h.update(lbl.name.encode())
            with open(lbl, "rb") as f:
                h.update(f.read())
        return h.hexdigest()

    @staticmethod
    def _count_images_and_anns(processed_dir: Path) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {
            "images": {}, "annotations": {}
        }
        for split in ["train", "val", "test"]:
            split_dir = processed_dir / split
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not img_dir.exists():
                continue
            img_count = sum(
                1 for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            ann_count = 0
            if lbl_dir.exists():
                for lbl in lbl_dir.iterdir():
                    with open(lbl) as f:
                        ann_count += sum(1 for l in f if l.strip())
            counts["images"][split] = img_count
            counts["annotations"][split] = ann_count
        return counts

    @staticmethod
    def _summarise_stats(stats: Dict) -> Dict:
        return {
            "total_images": stats.get("total_images", 0),
            "total_annotations": stats.get("total_annotations", 0),
            "represented_classes": stats.get("represented_classes", 0),
            "imbalance_ratio": stats.get("imbalance_ratio", 0),
            "rare_classes_count": len(stats.get("rare_classes", [])),
        }

    def _load(self) -> Dict:
        if self._version_file.exists():
            with open(self._version_file) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._version_file, "w") as f:
            json.dump(self._history, f, indent=2)
