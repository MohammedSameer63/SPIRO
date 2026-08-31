"""
SPIRO ML — PipelineExplainer
Generates explainability output for the full SPIRO inference pipeline:
  - Per-detection reasoning chain (YOLO + EffNet + fusion)
  - Confidence breakdown
  - Agreement analysis
  - Annotated image with boxes + confidence
  - Structured JSON explanation
  - Optional Grad-CAM (if VerifierModel available)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

# Colour palette (BGR) for SPIRO groups
_PALETTE = [
    (0, 114, 189), (217, 83, 25),  (237, 177, 32), (126, 47, 142),
    (119, 172, 48), (77, 190, 238), (162, 20, 47),  (76, 153, 0),
    (255, 128, 0),  (0, 176, 240),  (150, 100, 200),(200, 50, 100),
]


@dataclass
class DetectionEvidence:
    """Full evidence for a single detection's classification decision."""
    detection_idx: int
    yolo_class_id: int
    yolo_class_name: str
    yolo_confidence: float
    effnet_class_id: int
    effnet_class_name: str
    effnet_confidence: float
    effnet_top5: List[Dict[str, Any]]
    fused_class_id: int
    fused_class_name: str
    fused_confidence: float
    fusion_method: str
    agreement: bool
    reasoning_chain: List[str]
    waste_stream: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_idx": self.detection_idx,
            "yolo": {
                "class_id": self.yolo_class_id,
                "class_name": self.yolo_class_name,
                "confidence": round(self.yolo_confidence, 4),
            },
            "effnet": {
                "class_id": self.effnet_class_id,
                "class_name": self.effnet_class_name,
                "confidence": round(self.effnet_confidence, 4),
                "top5": self.effnet_top5,
            },
            "fusion": {
                "class_id": self.fused_class_id,
                "class_name": self.fused_class_name,
                "confidence": round(self.fused_confidence, 4),
                "method": self.fusion_method,
                "agreement": self.agreement,
            },
            "waste_stream": self.waste_stream,
            "reasoning_chain": self.reasoning_chain,
        }


class PipelineExplainer:
    """
    Generates explainability artefacts for the full SPIRO pipeline.

    Parameters
    ----------
    save_visualizations : bool
        If True, save annotated images to disk.
    visualization_dir : Path
        Output directory for visualisations.
    include_reasoning_chain : bool
        If True, generate step-by-step decision explanation.
    gradcam_model : VerifierModel, optional
        If provided, generate Grad-CAM maps for each crop.

    Example
    -------
    >>> explainer = PipelineExplainer()
    >>> evidence = explainer.build_evidence(detections, verifications, fused_results)
    >>> annotated = explainer.annotate_image(image_bgr, evidence)
    >>> explanation_json = explainer.to_json(evidence)
    """

    def __init__(
        self,
        save_visualizations: bool = False,
        visualization_dir: Path = Path("reports/pipeline_viz"),
        include_reasoning_chain: bool = True,
        gradcam_model=None,
    ) -> None:
        self.save_visualizations = save_visualizations
        self.visualization_dir = Path(visualization_dir)
        self.include_reasoning_chain = include_reasoning_chain
        self.gradcam_model = gradcam_model

        if save_visualizations:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build evidence
    # ------------------------------------------------------------------

    def build_evidence(
        self,
        detections: List[Any],
        verifications: List[Optional[Dict[str, Any]]],
        fused_results: List[Dict[str, Any]],
        stream_lookup: Optional[Dict[int, str]] = None,
    ) -> List[DetectionEvidence]:
        """
        Build DetectionEvidence for every detected object.

        Parameters
        ----------
        detections : list of Detection objects
        verifications : list of verify_inference results (parallel to detections)
        fused_results : list of FusionResult dicts (parallel to detections)
        stream_lookup : optional dict mapping class_id → stream name

        Returns
        -------
        List[DetectionEvidence]
        """
        evidence_list: List[DetectionEvidence] = []

        for idx, (det, verif, fused) in enumerate(
            zip(detections, verifications, fused_results)
        ):
            yolo_cls = det.class_id
            yolo_name = det.class_name
            yolo_conf = det.confidence

            # EfficientNetV2 output
            if verif is not None:
                effnet_cls = verif.get("class_id", yolo_cls)
                effnet_name = verif.get("class_name", yolo_name)
                effnet_conf = verif.get("confidence", 0.0)
                top5 = verif.get("top_k", [])[:5]
            else:
                effnet_cls, effnet_name, effnet_conf = yolo_cls, yolo_name, yolo_conf
                top5 = []

            fused_cls = fused.get("class_id", yolo_cls)
            fused_name = fused.get("class_name", yolo_name)
            fused_conf = fused.get("fused_confidence", fused.get("confidence", yolo_conf))
            fusion_method = fused.get("method", "unknown")
            agreement = fused.get("agreement", yolo_cls == effnet_cls)

            stream = "unknown"
            if stream_lookup:
                stream = stream_lookup.get(fused_cls, "reject_waste")

            chain = self._build_reasoning_chain(
                yolo_name, yolo_conf,
                effnet_name, effnet_conf,
                fused_name, fused_conf,
                fusion_method, agreement,
            ) if self.include_reasoning_chain else []

            evidence_list.append(DetectionEvidence(
                detection_idx=idx,
                yolo_class_id=yolo_cls,
                yolo_class_name=yolo_name,
                yolo_confidence=yolo_conf,
                effnet_class_id=effnet_cls,
                effnet_class_name=effnet_name,
                effnet_confidence=effnet_conf,
                effnet_top5=top5,
                fused_class_id=fused_cls,
                fused_class_name=fused_name,
                fused_confidence=fused_conf,
                fusion_method=fusion_method,
                agreement=agreement,
                reasoning_chain=chain,
                waste_stream=stream,
            ))

        return evidence_list

    # ------------------------------------------------------------------
    # Reasoning chain
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning_chain(
        yolo_name: str, yolo_conf: float,
        effnet_name: str, effnet_conf: float,
        fused_name: str, fused_conf: float,
        fusion_method: str,
        agreement: bool,
    ) -> List[str]:
        chain = [
            f"Step 1 — YOLOv11 Detection: identified as '{yolo_name}' "
            f"with confidence {yolo_conf:.3f}.",

            f"Step 2 — EfficientNetV2 Verification: crop classified as '{effnet_name}' "
            f"with confidence {effnet_conf:.3f}.",
        ]

        if agreement:
            chain.append(
                f"Step 3 — Agreement: both models agree on '{fused_name}'. "
                f"High confidence decision."
            )
        else:
            chain.append(
                f"Step 3 — Disagreement: YOLO predicted '{yolo_name}' "
                f"but EffNet predicted '{effnet_name}'. "
                f"Fusion resolves via {fusion_method}."
            )

        chain.append(
            f"Step 4 — Confidence Fusion ({fusion_method}): "
            f"final decision '{fused_name}' at {fused_conf:.3f}."
        )

        if fused_conf < 0.35:
            chain.append("Step 5 — Warning: low fusion confidence. Result may be unreliable.")
        elif fused_conf > 0.75:
            chain.append("Step 5 — High-confidence decision. Result is reliable.")
        else:
            chain.append("Step 5 — Moderate confidence. Manual verification recommended for critical decisions.")

        return chain

    # ------------------------------------------------------------------
    # Annotated image
    # ------------------------------------------------------------------

    def annotate_image(
        self,
        image_bgr: np.ndarray,
        evidence: List[DetectionEvidence],
        detections: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence on the image.

        Parameters
        ----------
        image_bgr : original BGR image
        evidence : list of DetectionEvidence
        detections : original Detection objects (for bbox coords)

        Returns
        -------
        np.ndarray — annotated BGR image
        """
        canvas = image_bgr.copy()

        for ev in evidence:
            if detections is None:
                continue
            if ev.detection_idx >= len(detections):
                continue

            det = detections[ev.detection_idx]
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = _PALETTE[ev.fused_class_id % len(_PALETTE)]

            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Agreement indicator
            marker = "✓" if ev.agreement else "?"
            label = (
                f"{marker} {ev.fused_class_name} "
                f"Y:{ev.yolo_confidence:.2f} "
                f"E:{ev.effnet_confidence:.2f} "
                f"→{ev.fused_confidence:.2f}"
            )

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_text = max(y1 - 4, th + 4)
            cv2.rectangle(canvas, (x1, y_text - th - 4), (x1 + tw, y_text), color, -1)
            cv2.putText(canvas, label, (x1, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return canvas

    def save_annotated(
        self,
        annotated: np.ndarray,
        stem: str = "inference",
    ) -> Path:
        out = self.visualization_dir / f"{stem}_annotated.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        log.debug(f"Annotated image → {out}")
        return out

    # ------------------------------------------------------------------
    # Grad-CAM (optional)
    # ------------------------------------------------------------------

    def generate_gradcam(
        self,
        crops: List[Any],
        evidence: List[DetectionEvidence],
        stem: str = "gradcam",
    ) -> List[Optional[Path]]:
        """
        Generate Grad-CAM for each crop if a VerifierModel is available.

        Returns list of paths (one per detection, None if skipped).
        """
        if self.gradcam_model is None:
            return [None] * len(evidence)

        try:
            from lib.ml.verification.explainability.gradcam import GradCAM
        except ImportError:
            return [None] * len(evidence)

        cam = GradCAM(self.gradcam_model)
        results = []
        for idx, (crop, ev) in enumerate(zip(crops, evidence)):
            if not crop.is_valid:
                results.append(None)
                continue
            try:
                out_dir = self.visualization_dir / "gradcam"
                paths = cam.save_explanation(
                    source=crop.image_bgr,
                    output_dir=out_dir,
                    class_idx=ev.fused_class_id,
                    stem=f"{stem}_det{idx}",
                )
                results.append(Path(paths["overlay"]))
            except Exception as e:
                log.debug(f"Grad-CAM failed for detection {idx}: {e}")
                results.append(None)
        cam.remove_hooks()
        return results

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def to_json(self, evidence: List[DetectionEvidence]) -> List[Dict[str, Any]]:
        return [ev.to_dict() for ev in evidence]
