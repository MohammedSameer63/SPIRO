"""
SPIRO ML — TaxonomyLoader
Loads and validates the SPIRO 109-class taxonomy from YAML.
Provides fast lookups: id→name, name→id, group→ids.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_DEFAULT_PATH = Path(__file__).parents[5] / "configs" / "taxonomy" / "spiro_taxonomy.yaml"


class TaxonomyLoader:
    """
    Singleton-style loader for the SPIRO taxonomy YAML.

    Example
    -------
    >>> tax = TaxonomyLoader()
    >>> tax.id_to_name(0)
    'plastic_bottle'
    >>> tax.name_to_id("cigarette_butt")
    53
    >>> tax.group_ids("plastic")
    [0, 1, 2, ..., 21]
    >>> len(tax.all_names())
    109
    """

    _instance: Optional["TaxonomyLoader"] = None

    def __new__(cls, path: Optional[Path] = None) -> "TaxonomyLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, path: Optional[Path] = None) -> None:
        if self._loaded:
            return
        path = path or _DEFAULT_PATH
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Support both old format (taxonomy.classes) and new format (root classes)
        if "classes" in raw:
            classes_raw = raw["classes"]
        else:
            classes_raw = raw["taxonomy"]["classes"]

        # YAML keys may come as int or str
        self._id_to_name: Dict[int, str] = {
            int(k): str(v) for k, v in classes_raw.items()
        }
        self._name_to_id: Dict[str, int] = {v: k for k, v in self._id_to_name.items()}

        groups_raw = raw.get("groups", raw.get("taxonomy", {}).get("groups", {}))
        self._groups: Dict[str, List[int]] = {
            g: list(ids) for g, ids in groups_raw.items()
        }

        # Load 7 disposal streams
        streams_raw = raw.get("streams", {})
        self._streams: Dict[str, Dict] = {}
        for sname, sdata in streams_raw.items():
            if isinstance(sdata, dict):
                self._streams[sname] = {
                    "label":       sdata.get("label", sname),
                    "description": sdata.get("description", ""),
                    "bin":         sdata.get("bin_color", ""),
                    "color":       sdata.get("color", "#9E9E9E"),
                    "icon":        sdata.get("icon", ""),
                    "ids":         list(sdata.get("ids", [])),
                }

        # Build id → stream lookup
        self._id_to_stream: Dict[int, str] = {}
        for sname, sdata in self._streams.items():
            for cid in sdata.get("ids", []):
                self._id_to_stream[int(cid)] = sname

        total_raw = raw.get("taxonomy", {}).get("total_classes", 109)
        self._total = total_raw
        assert len(self._id_to_name) == self._total, (
            f"Taxonomy mismatch: expected {self._total}, got {len(self._id_to_name)}"
        )
        self._loaded = True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def id_to_name(self, class_id: int) -> str:
        if class_id not in self._id_to_name:
            raise KeyError(f"Class ID {class_id} not in SPIRO taxonomy")
        return self._id_to_name[class_id]

    def name_to_id(self, name: str) -> int:
        if name not in self._name_to_id:
            raise KeyError(f"Class name '{name}' not in SPIRO taxonomy")
        return self._name_to_id[name]

    def group_ids(self, group: str) -> List[int]:
        if group not in self._groups:
            raise KeyError(f"Group '{group}' not found. Available: {list(self._groups)}")
        return self._groups[group]

    def all_names(self) -> List[str]:
        return [self._id_to_name[i] for i in range(self._total)]

    def all_ids(self) -> List[int]:
        return list(range(self._total))

    def group_names(self) -> List[str]:
        return list(self._groups.keys())

    def id_to_group(self, class_id: int) -> Optional[str]:
        for group, ids in self._groups.items():
            if class_id in ids:
                return group
        return None

    @property
    def num_classes(self) -> int:
        return self._total

    def validate_id(self, class_id: int) -> bool:
        return 0 <= class_id < self._total

    def as_dict(self) -> Dict[int, str]:
        return dict(self._id_to_name)

    # ------------------------------------------------------------------
    # 7-stream disposal lookups
    # ------------------------------------------------------------------

    def id_to_stream(self, class_id: int) -> str:
        """Return the disposal stream name for a class ID.
        Returns one of: organic, recoverable, non_recoverable,
        hazardous, sanitary, e_waste, reject
        """
        return self._id_to_stream.get(class_id, "reject")

    def stream_label(self, stream: str) -> str:
        """Return human-readable label for a stream."""
        return self._streams.get(stream, {}).get("label", stream)

    def stream_ids(self, stream: str) -> List[int]:
        """Return all class IDs belonging to a stream."""
        return self._streams.get(stream, {}).get("ids", [])

    def all_streams(self) -> Dict[str, Dict]:
        """Return all 7 stream definitions."""
        return dict(self._streams)

    def stream_names(self) -> List[str]:
        """Return list of stream names."""
        return list(self._streams.keys())
