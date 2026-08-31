"""
SPIRO ML — DatasetVersion
Canonical record for a versioned SPIRO dataset snapshot.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetVersion:
    """Complete record for one versioned SPIRO dataset."""

    # Identity
    dataset_id: str
    version: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    author: str = "spiro_ml"
    description: str = ""

    # Statistics
    image_count: int = 0
    annotation_count: int = 0
    class_count: int = 109
    class_distribution: Dict[str, int] = field(default_factory=dict)  # class_name → count

    # Integrity
    checksum: str = ""           # SHA-256 of dataset manifest
    split_checksums: Dict[str, str] = field(default_factory=dict)  # split → sha256

    # Lineage
    parent_version: Optional[str] = None
    source_datasets: List[str] = field(default_factory=list)   # taco, trashnet, etc.
    augmented: bool = False
    augmentation_config: Dict[str, Any] = field(default_factory=dict)

    # Paths
    root_path: str = ""
    manifest_path: str = ""

    # Tags
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetVersion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, text: str) -> "DatasetVersion":
        return cls.from_dict(json.loads(text))

    def compute_checksum(self) -> str:
        """Compute a deterministic hash from key statistics."""
        payload = json.dumps({
            "dataset_id": self.dataset_id,
            "version": self.version,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "class_distribution": dict(sorted(self.class_distribution.items())),
        }, sort_keys=True).encode()
        self.checksum = hashlib.sha256(payload).hexdigest()
        return self.checksum

    def lineage_chain(self, registry: "DatasetRegistry") -> List["DatasetVersion"]:  # type: ignore[name-defined]
        """Return the full lineage from this version back to the root."""
        chain = [self]
        current = self
        while current.parent_version:
            parent = registry.get(current.dataset_id, current.parent_version)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain
