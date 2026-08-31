"""
SPIRO ML — VerifierModel (EfficientNetV2)
Second-stage classification model that takes cropped object patches
and produces per-class probability distributions.

Features:
  - All 7 EfficientNetV2 variants via timm
  - Two-phase fine-tuning (freeze → unfreeze)
  - Top-1 / Top-5 prediction API
  - Probability calibration interface
  - Save / load checkpoints
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
from lib.ml.verification.verify_config import VerifyConfig, _VARIANT_INPUT_SIZE

log = get_logger(__name__)


class VerificationResult:
    """Container for a single verification result."""

    __slots__ = (
        "class_id", "class_name", "confidence",
        "top5_classes", "top5_names", "top5_confidences",
        "probabilities",
    )

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        top5_classes: List[int],
        top5_names: List[str],
        top5_confidences: List[float],
        probabilities: List[float],
    ) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.top5_classes = top5_classes
        self.top5_names = top5_names
        self.top5_confidences = top5_confidences
        self.probabilities = probabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "top5": [
                {"class_id": cid, "class_name": cname, "confidence": conf}
                for cid, cname, conf in zip(
                    self.top5_classes, self.top5_names, self.top5_confidences
                )
            ],
        }

    def __repr__(self) -> str:
        return (
            f"VerificationResult({self.class_name!r}, conf={self.confidence:.3f}, "
            f"top5={self.top5_names[:3]})"
        )


class VerifierModel(nn.Module):
    """
    EfficientNetV2 second-stage verifier for SPIRO.

    Parameters
    ----------
    cfg : VerifyConfig
    freeze_backbone : bool
        If True, only the classifier head is trainable (phase 1).

    Example
    -------
    >>> model = VerifierModel(cfg)
    >>> result = model.predict(crop_image)
    >>> results = model.predict_batch([img1, img2, img3])
    """

    def __init__(self, cfg: VerifyConfig, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.input_size = cfg.input_size
        self.timm_name = cfg.timm_name

        # Load taxonomy
        self.taxonomy = TaxonomyLoader()
        self.class_names = self.taxonomy.all_names()

        # Device
        self.device = self._resolve_device(cfg.training.device)

        log.info(f"Building VerifierModel: {self.timm_name} | {self.num_classes} classes")
        self.backbone = timm.create_model(
            self.timm_name,
            pretrained=cfg.model.pretrained,
            num_classes=self.num_classes,
            drop_rate=cfg.model.drop_rate,
            drop_path_rate=cfg.model.drop_path_rate,
            global_pool=cfg.model.global_pool,
        )

        if freeze_backbone:
            self._freeze_backbone()
            log.info("Backbone frozen — head only trainable (phase 1)")

        self.to(self.device)
        self._build_inference_transform()
        log.info(f"VerifierModel ready on {self.device}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        for name, p in self.backbone.named_parameters():
            if "classifier" not in name and "head" not in name:
                p.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 3) -> None:
        """Gradually unfreeze the last N blocks for phase 2."""
        all_params = list(self.backbone.named_parameters())
        cutoff = max(0, len(all_params) - last_n_blocks * 20)
        for i, (_, p) in enumerate(all_params):
            if i >= cutoff:
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Unfrozen last {last_n_blocks} blocks — trainable: {trainable:,}")

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        log.info("All parameters unfrozen")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_inference_transform(self) -> None:
        import timm.data
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self._infer_tf = timm.data.create_transform(**data_cfg, is_training=False)

    def _load_tensor(
        self,
        source: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            if source.dim() == 3:
                source = source.unsqueeze(0)
            return source.to(self.device)
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            import cv2
            if source.ndim == 3 and source.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(source)
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")
        return self._infer_tf(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> VerificationResult:
        """
        Verify a single image crop.

        Parameters
        ----------
        source : image path, BGR ndarray, or PIL Image
        top_k : int — number of top predictions to return
        temperature : float — temperature scaling (1.0 = no scaling)

        Returns
        -------
        VerificationResult
        """
        self.eval()
        tensor = self._load_tensor(source)
        logits = self(tensor).squeeze(0)

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=0).cpu().tolist()
        top_k = min(top_k, self.num_classes)
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]

        return VerificationResult(
            class_id=top_indices[0],
            class_name=self.class_names[top_indices[0]],
            confidence=float(probs[top_indices[0]]),
            top5_classes=top_indices,
            top5_names=[self.class_names[i] for i in top_indices],
            top5_confidences=[float(probs[i]) for i in top_indices],
            probabilities=probs,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        sources: List[Union[str, Path, np.ndarray, Image.Image]],
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> List[VerificationResult]:
        """Run verification on a list of images (stacked batch)."""
        self.eval()
        tensors = torch.cat([self._load_tensor(s) for s in sources], dim=0)
        logits = self(tensors)
        if temperature != 1.0:
            logits = logits / temperature
        probs_batch = F.softmax(logits, dim=1).cpu().tolist()
        results = []
        for probs in probs_batch:
            top_k_actual = min(top_k, len(probs))
            top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k_actual]
            results.append(VerificationResult(
                class_id=top_idx[0],
                class_name=self.class_names[top_idx[0]],
                confidence=float(probs[top_idx[0]]),
                top5_classes=top_idx,
                top5_names=[self.class_names[i] for i in top_idx],
                top5_confidences=[float(probs[i]) for i in top_idx],
                probabilities=probs,
            ))
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "timm_name": self.timm_name,
                "num_classes": self.num_classes,
                "input_size": self.input_size,
                "class_names": self.class_names,
            },
            str(path),
        )
        log.info(f"VerifierModel saved to {path}")

    @classmethod
    def load(cls, cfg: VerifyConfig, path: Union[str, Path]) -> "VerifierModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.class_names = ckpt.get("class_names", model.class_names)
        log.info(f"VerifierModel loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def feature_dim(self) -> int:
        """Return the backbone feature dimension (before classifier)."""
        return self.backbone.num_features
