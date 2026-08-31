"""
SPIRO ML — ContaminationEngine
Detects contamination scenarios from fused waste classifications:
  - Food residue on recyclables
  - Wet/liquid contamination
  - Mixed organic + recoverable
  - Organic + hazardous mixing
  - Sanitary contamination
  - Reject waste contamination

Each detected scenario contributes to a contamination score (0–1)
and generates a human-readable explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

# ─── SPIRO 7-stream waste groupings ───────────────────────────────────────────
# These sets mirror GuidanceEngine._CLASS_TO_STREAM exactly
ORGANIC_IDS:       Set[int] = {47,48,49,50,51,72,79,80,81}
RECOVERABLE_IDS:   Set[int] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,
                                22,23,25,
                                27,28,29,30,31,35,
                                36,37,38,40,41,42,44,45,46,
                                71,76,103}
NON_RECOVERABLE_IDS: Set[int] = {15,16,17,19,20,21,24,26,32,33,
                                  39,43,52,53,54,55,56,57,58,
                                  73,74,77,78,85,86,87,88,89,90,91,92,
                                  93,94,95,96,97,98,99,100,101,102,
                                  104,105,107,108}
HAZARDOUS_IDS:     Set[int] = {34,75,82,83,84}
SANITARY_IDS:      Set[int] = {59,60,66,67,68,69,70}
EWASTE_IDS:        Set[int] = {61,62,63,64,65}
REJECT_IDS:        Set[int] = {106}
FOOD_RESIDUE_IDS:  Set[int] = {47,48,49,50,51,52}


@dataclass
class ContaminationFlag:
    """A single contamination scenario detected."""
    code: str
    description: str
    severity: str        # "low" | "medium" | "high" | "critical"
    penalty: float       # contribution to contamination score (0–1)
    affected_classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "severity": self.severity,
            "penalty": round(self.penalty, 3),
            "affected_classes": self.affected_classes,
        }


@dataclass
class ContaminationResult:
    """Result of contamination analysis for a full image."""
    contamination_score: float          # 0 = clean, 1 = heavily contaminated
    is_contaminated: bool
    flags: List[ContaminationFlag]
    dominant_stream: str                # primary waste stream detected
    streams_detected: List[str]         # all streams in the image
    explanation: str
    action_required: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contamination_score": round(self.contamination_score, 3),
            "is_contaminated": self.is_contaminated,
            "flags": [f.to_dict() for f in self.flags],
            "dominant_stream": self.dominant_stream,
            "streams_detected": self.streams_detected,
            "explanation": self.explanation,
            "action_required": self.action_required,
        }


class ContaminationEngine:
    """
    Analyses a set of fused classification results for contamination.

    Parameters
    ----------
    food_residue_threshold : float
        Fusion confidence above which organic class is treated as food residue.
    wet_indicator_threshold : float
        Confidence above which a class signals wetness.
    mixed_waste_penalty : float
        Score added per conflicting stream pair.

    Example
    -------
    >>> engine = ContaminationEngine()
    >>> result = engine.analyse(fused_results)
    >>> print(result.contamination_score, result.explanation)
    """

    _STREAM_MAP = {
        "organic":         ORGANIC_IDS,
        "recoverable":     RECOVERABLE_IDS,
        "non_recoverable": NON_RECOVERABLE_IDS,
        "hazardous":       HAZARDOUS_IDS,
        "sanitary":        SANITARY_IDS,
        "e_waste":         EWASTE_IDS,
        "reject":          REJECT_IDS,
    }

    # Class IDs that indicate liquid/moisture contamination
    _WET_INDICATORS = {47, 48, 51}  # food waste, fruit peel, coffee grounds

    def __init__(
        self,
        food_residue_threshold: float = 0.4,
        wet_indicator_threshold: float = 0.6,
        mixed_waste_penalty: float = 0.3,
        contamination_threshold: float = 0.25,
    ) -> None:
        self.food_residue_thr = food_residue_threshold
        self.wet_thr = wet_indicator_threshold
        self.mixed_penalty = mixed_waste_penalty
        self.contamination_thr = contamination_threshold
        self.taxonomy = TaxonomyLoader()

    def analyse(
        self,
        fused_results: List[Dict[str, Any]],
    ) -> ContaminationResult:
        """
        Run contamination analysis on a list of fused detection results.

        Parameters
        ----------
        fused_results : list of dicts, each with:
            - class_id : int
            - class_name : str
            - fused_confidence : float
            - yolo_confidence : float
            - effnet_confidence : float

        Returns
        -------
        ContaminationResult
        """
        if not fused_results:
            return ContaminationResult(
                contamination_score=0.0,
                is_contaminated=False,
                flags=[],
                dominant_stream="none",
                streams_detected=[],
                explanation="No objects detected.",
                action_required="No action required.",
            )

        # Classify each result into a waste stream
        class_ids = [r["class_id"] for r in fused_results]
        class_names = [r["class_name"] for r in fused_results]
        confs = [r.get("fused_confidence", r.get("confidence", 0.5)) for r in fused_results]

        streams_present: Dict[str, List[str]] = {s: [] for s in self._STREAM_MAP}
        for cls_id, cls_name in zip(class_ids, class_names):
            for stream, ids in self._STREAM_MAP.items():
                if cls_id in ids:
                    streams_present[stream].append(cls_name)

        active_streams = [s for s, names in streams_present.items() if names]
        dominant = self._dominant_stream(streams_present, confs, class_ids)

        flags: List[ContaminationFlag] = []

        # 1. Food residue on recyclables
        has_food = any(cid in FOOD_RESIDUE_IDS for cid in class_ids)
        has_recoverable = any(cid in RECOVERABLE_IDS for cid in class_ids)
        if has_food and has_recoverable:
            food_names = [n for cid, n in zip(class_ids, class_names) if cid in FOOD_RESIDUE_IDS]
            rec_names  = [n for cid, n in zip(class_ids, class_names) if cid in RECOVERABLE_IDS]
            flags.append(ContaminationFlag(
                code="FOOD_RESIDUE_ON_RECYCLABLE",
                description="Food residue detected alongside recyclable materials. "
                            "Recyclables must be rinsed before disposal.",
                severity="medium",
                penalty=0.35,
                affected_classes=food_names[:3] + rec_names[:3],
            ))

        # 2. Organic + hazardous mixing
        has_organic = any(cid in ORGANIC_IDS for cid in class_ids)
        has_hazardous = any(cid in HAZARDOUS_IDS for cid in class_ids)
        if has_organic and has_hazardous:
            haz_names = [n for cid, n in zip(class_ids, class_names) if cid in HAZARDOUS_IDS]
            flags.append(ContaminationFlag(
                code="ORGANIC_HAZARDOUS_MIX",
                description="Hazardous waste mixed with organic material. "
                            "This is a serious contamination requiring separate disposal.",
                severity="critical",
                penalty=0.80,
                affected_classes=haz_names[:3],
            ))

        # 3. Sanitary contamination
        has_sanitary = any(cid in SANITARY_IDS for cid in class_ids)
        if has_sanitary and (has_recoverable or has_organic):
            san_names = [n for cid, n in zip(class_ids, class_names) if cid in SANITARY_IDS]
            flags.append(ContaminationFlag(
                code="SANITARY_CONTAMINATION",
                description="Sanitary waste (masks, gloves, diapers) must never be "
                            "mixed with recoverable or organic waste.",
                severity="high",
                penalty=0.65,
                affected_classes=san_names[:3],
            ))

        # 4. Wet/moisture indicators on dry recyclables
        has_wet = any(
            cid in self._WET_INDICATORS and conf >= self.wet_thr
            for cid, conf in zip(class_ids, confs)
        )
        if has_wet and has_recoverable:
            flags.append(ContaminationFlag(
                code="WET_RECYCLABLES",
                description="Moisture or liquid residue detected near recyclable materials. "
                            "Wet recyclables degrade paper-fibre streams.",
                severity="medium",
                penalty=0.30,
                affected_classes=[],
            ))

        # 5. E-waste with general waste
        has_ewaste = any(cid in EWASTE_IDS for cid in class_ids)
        if has_ewaste and len(active_streams) > 1:
            ew_names = [n for cid, n in zip(class_ids, class_names) if cid in EWASTE_IDS]
            flags.append(ContaminationFlag(
                code="EWASTE_MIXED_WITH_GENERAL",
                description="E-waste (batteries, cables, electronics) must go to special "
                            "collection points — never mixed with general waste.",
                severity="high",
                penalty=0.60,
                affected_classes=ew_names[:3],
            ))

        # 6. Reject waste contamination
        has_reject = any(cid in REJECT_IDS for cid in class_ids)
        if has_reject and len(active_streams) > 1:
            flags.append(ContaminationFlag(
                code="REJECT_WASTE_PRESENT",
                description="Unclassifiable or reject waste detected. "
                            "This stream contaminates all others.",
                severity="medium",
                penalty=self.mixed_penalty,
                affected_classes=[],
            ))

        # 7. Multiple recoverable sub-streams (e.g. glass + paper)
        if len(active_streams) >= 3 and "recoverable" in active_streams:
            flags.append(ContaminationFlag(
                code="MULTI_STREAM_MIX",
                description="Multiple waste categories detected together. "
                            "Sort into separate streams for best recovery.",
                severity="low",
                penalty=0.15,
                affected_classes=[],
            ))

        # Compute total contamination score (capped at 1.0)
        score = min(1.0, sum(f.penalty for f in flags))

        explanation = self._build_explanation(flags, dominant, active_streams)
        action = self._build_action(flags, score)

        return ContaminationResult(
            contamination_score=score,
            is_contaminated=score >= self.contamination_thr,
            flags=flags,
            dominant_stream=dominant,
            streams_detected=active_streams,
            explanation=explanation,
            action_required=action,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dominant_stream(
        self,
        streams: Dict[str, List[str]],
        confs: List[float],
        class_ids: List[int],
    ) -> str:
        """Return the stream with the most detected objects."""
        counts = {s: len(names) for s, names in streams.items() if names}
        if not counts:
            return "unknown"
        return max(counts, key=lambda s: counts[s])

    @staticmethod
    def _build_explanation(
        flags: List[ContaminationFlag],
        dominant: str,
        streams: List[str],
    ) -> str:
        if not flags:
            return (
                f"Waste is predominantly {dominant}. "
                "No contamination detected."
            )
        parts = [f.description for f in flags]
        return f"Dominant stream: {dominant}. Issues: " + " | ".join(parts)

    @staticmethod
    def _build_action(flags: List[ContaminationFlag], score: float) -> str:
        if score == 0:
            return "Dispose normally according to waste category."
        critical = [f for f in flags if f.severity == "critical"]
        high     = [f for f in flags if f.severity == "high"]
        if critical:
            return "CRITICAL: Separate waste streams immediately before disposal. Contact waste authority."
        if high:
            return "HIGH: Do not mix these items. Sort into correct streams before disposal."
        if score < 0.3:
            return "LOW: Rinse recyclables and separate food waste before disposal."
        return "MEDIUM: Sort waste into separate streams. Recyclables must be clean and dry."
