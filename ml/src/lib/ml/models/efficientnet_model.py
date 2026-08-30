"""
SPIRO ML — SPIROClassifier (EfficientNetV2 via timm)
Fine-tuneable image classifier with:
  - Configurable backbone freeze/unfreeze
  - Label smoothing loss
  - Clean predict() and predict_proba() API
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

from lib.ml.core.config import ConfigManager
from lib.ml.core.device import resolve_device
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class ClassificationResult:
    """Container for classifier output."""

    __slots__ = ("class_id", "class_name", "confidence", "probabilities")

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        probabilities: List[float],
    ) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.probabilities = probabilities

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(class={self.class_name!r}, "
            f"conf={self.confidence:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class SPIROClassifier(nn.Module):
    """
    EfficientNetV2 fine-tuned for SPIRO classification.

    Parameters
    ----------
    cfg : ConfigManager
    freeze_backbone : bool
        If True, only the head trains initially (for rapid adaptation).

    Example
    -------
    >>> clf = SPIROClassifier(cfg)
    >>> result = clf.predict("image.jpg")
    >>> print(result)
    """

    def __init__(self, cfg: ConfigManager, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_names: List[str] = list(cfg.dataset.class_names)
        self.num_classes = cfg.efficientnetv2.num_classes
        self.device = resolve_device(cfg.training.device)

        ev2_cfg = cfg.efficientnetv2
        log.info(f"Building EfficientNetV2 backbone: {ev2_cfg.variant}")
        self.backbone = timm.create_model(
            ev2_cfg.variant,
            pretrained=ev2_cfg.pretrained,
            num_classes=self.num_classes,
            drop_rate=ev2_cfg.drop_rate,
            drop_path_rate=ev2_cfg.drop_path_rate,
            global_pool=ev2_cfg.global_pool,
        )

        if freeze_backbone:
            self._freeze_backbone()
            log.info("Backbone frozen — only classification head is trainable")

        self.to(self.device)
        self._build_inference_transform()
        log.info(f"SPIROClassifier ready — {self.num_classes} classes on {self.device}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 3) -> None:
        """Gradually unfreeze the last N blocks for fine-tuning stage 2."""
        blocks = list(self.backbone.named_parameters())
        cutoff = max(0, len(blocks) - unfreeze_last_n_blocks * 20)
        for i, (name, param) in enumerate(blocks):
            if i >= cutoff:
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Unfrozen last {unfreeze_last_n_blocks} blocks — trainable params: {trainable:,}")

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_inference_transform(self) -> None:
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self._infer_transform = timm.data.create_transform(**data_cfg, is_training=False)

    def _load_image_tensor(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source[..., ::-1] if source.ndim == 3 else source)
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        tensor = self._infer_transform(img).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> ClassificationResult:
        """Classify a single image."""
        self.eval()
        tensor = self._load_image_tensor(source)
        logits = self(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        class_id = int(np.argmax(probs))
        return ClassificationResult(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=float(probs[class_id]),
            probabilities=probs,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        sources: List[Union[str, Path, np.ndarray, Image.Image]],
    ) -> List[ClassificationResult]:
        """Classify a batch of images."""
        self.eval()
        tensors = torch.cat([self._load_image_tensor(s) for s in sources], dim=0)
        logits = self(tensors)
        probs_batch = F.softmax(logits, dim=1).cpu().tolist()
        results = []
        for probs in probs_batch:
            class_id = int(np.argmax(probs))
            results.append(
                ClassificationResult(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(probs[class_id]),
                    probabilities=probs,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "variant": self.cfg.efficientnetv2.variant,
            },
            path,
        )
        log.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, cfg: ConfigManager, path: Union[str, Path]) -> "SPIROClassifier":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.class_names = ckpt["class_names"]
        log.info(f"Classifier loaded from {path}")
        return model

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
