"""
SPIRO ML — OutputBuilder
Assembles the complete structured JSON response for a pipeline inference call.

Schema
------
{
  "pipeline_version": "1.0.0",
  "status": "success" | "partial" | "rejected" | "error",
  "image": { quality, original_size, ... },
  "detections": [ { bbox, confidence, class, verification, fusion, guidance, ... } ],
  "contamination": { score, is_contaminated, flags, explanation, action },
  "guidance": { summary, items: [ { stream, preparation, warnings, ... } ] },
  "explainability": { evidence: [ ... ] },
  "timings": { preprocess_ms, detection_ms, verification_ms, fusion_ms, total_ms },
  "model_versions": { yolo, effnet },
  "warnings": [ ... ]
}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_PIPELINE_VERSION = "1.0.0"


@dataclass
class PipelineTimings:
    preprocess_ms: float = 0.0
    detection_ms: float = 0.0
    cropping_ms: float = 0.0
    verification_ms: float = 0.0
    fusion_ms: float = 0.0
    contamination_ms: float = 0.0
    guidance_ms: float = 0.0
    explainability_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


class OutputBuilder:
    """
    Assembles the final structured JSON inference response.

    Example
    -------
    >>> builder = OutputBuilder(include_probabilities=False, include_timings=True)
    >>> response = builder.build(
    ...     status="success",
    ...     image_meta=...,
    ...     detections=...,
    ...     fused_results=...,
    ...     guidance_items=...,
    ...     contamination=...,
    ...     evidence=...,
    ...     timings=...,
    ... )
    """

    def __init__(
        self,
        include_probabilities: bool = False,
        include_timings: bool = True,
        include_model_versions: bool = True,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.include_probabilities = include_probabilities
        self.include_timings = include_timings
        self.include_model_versions = include_model_versions
        self.model_versions = model_versions or {}

    def build(
        self,
        status: str,
        image_meta: Dict[str, Any],
        detections: List[Any],
        fused_results: List[Dict[str, Any]],
        guidance_items: List[Any],
        contamination: Any,
        evidence: List[Any],
        timings: PipelineTimings,
        warnings: Optional[List[str]] = None,
        annotated_image_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete structured response.

        Parameters
        ----------
        status : "success" | "partial" | "rejected" | "error"
        image_meta : dict from PreprocessResult metadata + quality
        detections : list of Detection objects
        fused_results : list of FusionResult-like dicts
        guidance_items : list of GuidanceItem objects
        contamination : ContaminationResult object
        evidence : list of DetectionEvidence objects
        timings : PipelineTimings
        warnings : list of warning strings
        annotated_image_path : optional path to annotated image

        Returns
        -------
        dict — fully structured JSON-serialisable response
        """
        response: Dict[str, Any] = {
            "pipeline_version": _PIPELINE_VERSION,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Image metadata
        response["image"] = image_meta

        # Detection results
        detection_list = []
        for idx, (det, fused) in enumerate(zip(detections, fused_results)):
            # Get stream info from guidance engine
            from lib.ml.pipeline.guidance.guidance_engine import (
                GuidanceEngine, STREAMS
            )
            final_cls_id = fused.get("class_id", det.class_id)
            stream = GuidanceEngine.stream_for_class(final_cls_id)
            stream_info = STREAMS.get(stream, STREAMS["reject"])

            entry: Dict[str, Any] = {
                "detection_id": idx,
                "bbox_xyxy": [round(v, 2) for v in det.bbox_xyxy],
                "yolo_class_id": det.class_id,
                "yolo_class_name": det.class_name,
                "yolo_confidence": round(det.confidence, 4),
                "final_class_id": final_cls_id,
                "final_class_name": fused.get("class_name", det.class_name),
                "final_confidence": round(fused.get("fused_confidence",
                                                     fused.get("confidence", det.confidence)), 4),
                "fusion_method": fused.get("method", "none"),
                "models_agree": fused.get("agreement", True),
                "waste_stream": stream,
                "waste_stream_label": stream_info["label"],
                "waste_stream_color": stream_info["color"],
                "waste_stream_bin": stream_info["bin"],
            }

            # EffNet verification
            effnet_conf = fused.get("effnet_confidence")
            if effnet_conf is not None:
                entry["effnet_class_id"] = fused.get("effnet_class_id", det.class_id)
                entry["effnet_class_name"] = fused.get("effnet_class_name", det.class_name)
                entry["effnet_confidence"] = round(effnet_conf, 4)

            # Top-5 from evidence
            if idx < len(evidence) and evidence[idx].effnet_top5:
                entry["top5_predictions"] = evidence[idx].effnet_top5[:5]

            detection_list.append(entry)

        response["detections"] = detection_list
        response["detection_count"] = len(detection_list)

        # Contamination
        if contamination is not None:
            response["contamination"] = contamination.to_dict()

        # Guidance
        guidance_list = [g.to_dict() for g in guidance_items]
        summary = ""
        if guidance_items:
            streams = list({g.stream for g in guidance_items})
            summary = (
                f"Dispose as {streams[0]}."
                if len(streams) == 1
                else f"Mixed waste — separate into: {', '.join(streams)}."
            )
        response["guidance"] = {
            "summary": summary,
            "items": guidance_list,
        }

        # Explainability
        if evidence:
            response["explainability"] = {
                "evidence": [ev.to_dict() for ev in evidence],
            }
            if annotated_image_path:
                response["explainability"]["annotated_image"] = str(annotated_image_path)

        # Timings
        if self.include_timings:
            response["timings"] = timings.to_dict()

        # Model versions
        if self.include_model_versions and self.model_versions:
            response["model_versions"] = self.model_versions

        # Warnings
        all_warnings = list(warnings or [])
        if contamination and contamination.is_contaminated:
            all_warnings.append(
                f"Contamination detected (score={contamination.contamination_score:.2f}): "
                f"{contamination.action_required}"
            )
        if response["detection_count"] == 0:
            all_warnings.append("No objects detected in this image.")
        response["warnings"] = all_warnings

        return response

    @staticmethod
    def error_response(message: str, code: str = "PIPELINE_ERROR") -> Dict[str, Any]:
        return {
            "pipeline_version": _PIPELINE_VERSION,
            "status": "error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": {"code": code, "message": message},
            "detections": [],
            "detection_count": 0,
        }

    @staticmethod
    def rejected_response(reason: str, quality_info: Dict) -> Dict[str, Any]:
        return {
            "pipeline_version": _PIPELINE_VERSION,
            "status": "rejected",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rejection_reason": reason,
            "image": {"quality": quality_info},
            "detections": [],
            "detection_count": 0,
            "warnings": [f"Image rejected: {reason}"],
        }
