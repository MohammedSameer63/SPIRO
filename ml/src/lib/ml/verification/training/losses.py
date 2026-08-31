"""
SPIRO ML — Loss Functions for EfficientNetV2 Verifier

Supports:
  - CrossEntropyLoss (standard)
  - LabelSmoothingCrossEntropy
  - FocalLoss (handles severe class imbalance)
  - WeightedCrossEntropyLoss (per-class frequency weights)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents overconfident predictions.

    L = (1-ε) * CE(y_hard) + ε * CE(uniform)
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 109) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # Hard target contribution
        nll = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        # Smooth contribution: mean of all log-probs
        smooth = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard CE. Typical: 1–5.
    alpha : list of float or None
        Per-class weight α. None = uniform.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = probability of correct class
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with per-class frequency-inverse weights.
    Equivalent to nn.CrossEntropyLoss(weight=class_weights).
    """

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


def build_loss(cfg, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory that returns the correct loss from config.

    Parameters
    ----------
    cfg : VerifyConfig
    class_weights : Tensor, optional
        Shape [num_classes]. Used for weighted and focal modes.
    """
    name = cfg.loss.name
    nc = cfg.model.num_classes

    if name == "cross_entropy":
        log.info("Loss: CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    elif name == "label_smoothing":
        log.info(f"Loss: LabelSmoothingCrossEntropy (smoothing={cfg.loss.label_smoothing})")
        return LabelSmoothingCrossEntropy(
            smoothing=cfg.loss.label_smoothing,
            num_classes=nc,
        )

    elif name == "focal":
        alpha = list(cfg.loss.focal_alpha) if cfg.loss.focal_alpha else None
        log.info(f"Loss: FocalLoss (gamma={cfg.loss.focal_gamma})")
        return FocalLoss(gamma=cfg.loss.focal_gamma, alpha=alpha)

    elif name == "weighted":
        if class_weights is None:
            log.warning("weighted loss requested but no class_weights provided — using uniform CE")
            return nn.CrossEntropyLoss()
        log.info("Loss: WeightedCrossEntropyLoss")
        return WeightedCrossEntropyLoss(class_weights)

    else:
        raise ValueError(f"Unknown loss: {name!r}. Choose from: cross_entropy, label_smoothing, focal, weighted")
