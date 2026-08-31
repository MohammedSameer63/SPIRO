"""
SPIRO ML — EfficientNetTrainer
Custom PyTorch training loop for SPIROClassifier (EfficientNetV2).
Features:
  - Two-phase training (frozen backbone → full fine-tune)
  - Label-smoothing cross-entropy
  - Cosine / step / plateau LR schedulers
  - Early stopping
  - TensorBoard + MLflow logging
  - Checkpoint saving/loading
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from lib.ml.core.config import ConfigManager
from lib.ml.core.device import resolve_device
from lib.ml.core.logger import get_logger
from lib.ml.data.augmentation import build_train_transform, build_val_transform
from lib.ml.models.efficientnet_model import SPIROClassifier

log = get_logger(__name__)


class EfficientNetTrainer:
    """
    Full training loop for SPIROClassifier.

    Parameters
    ----------
    cfg : ConfigManager
    model : SPIROClassifier, optional
        If None, creates a fresh model from cfg.

    Example
    -------
    >>> trainer = EfficientNetTrainer(cfg)
    >>> history = trainer.train(
    ...     train_dir="datasets/processed/train",
    ...     val_dir="datasets/processed/val",
    ... )
    """

    def __init__(
        self,
        cfg: ConfigManager,
        model: Optional[SPIROClassifier] = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self.ckpt_dir = Path(cfg.paths.checkpoints_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = Path(cfg.paths.tensorboard_dir)

        self.model = model or SPIROClassifier(cfg)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }
        self._best_val_acc: float = 0.0
        self._best_ckpt: Optional[Path] = None
        self._patience_counter: int = 0

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _make_loaders(
        self, train_dir: str, val_dir: str
    ) -> Tuple[DataLoader, DataLoader]:
        """Build ImageFolder DataLoaders with Albumentations-compatible wrappers."""
        imgsz = tuple(self.cfg.dataset.image_size)

        # For classification, use torchvision transforms (timm handles normalisation)
        train_tf = transforms.Compose([
            transforms.Resize(imgsz),
            transforms.RandomHorizontalFlip(self.cfg.augmentation.horizontal_flip),
            transforms.ColorJitter(
                brightness=self.cfg.augmentation.brightness_limit,
                contrast=self.cfg.augmentation.contrast_limit,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        t_cfg = self.cfg.training
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=t_cfg.batch_size, shuffle=True,
            num_workers=t_cfg.workers, pin_memory=t_cfg.pin_memory, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=t_cfg.batch_size, shuffle=False,
            num_workers=t_cfg.workers, pin_memory=t_cfg.pin_memory,
        )
        log.info(
            f"DataLoaders ready — train: {len(train_ds)} | val: {len(val_ds)}"
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Optimiser & scheduler factory
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> optim.Optimizer:
        t_cfg = self.cfg.training
        params = [p for p in self.model.parameters() if p.requires_grad]
        if t_cfg.optimizer == "AdamW":
            return optim.AdamW(params, lr=t_cfg.learning_rate, weight_decay=t_cfg.weight_decay)
        if t_cfg.optimizer == "Adam":
            return optim.Adam(params, lr=t_cfg.learning_rate, weight_decay=t_cfg.weight_decay)
        return optim.SGD(
            params, lr=t_cfg.learning_rate,
            momentum=t_cfg.momentum, weight_decay=t_cfg.weight_decay, nesterov=True,
        )

    def _build_scheduler(self, optimizer: optim.Optimizer, steps_per_epoch: int):
        t_cfg = self.cfg.training
        if t_cfg.lr_scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=t_cfg.epochs, eta_min=1e-6)
        if t_cfg.lr_scheduler == "step":
            return StepLR(optimizer, step_size=30, gamma=0.1)
        if t_cfg.lr_scheduler == "plateau":
            return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
        return None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_dir: str,
        val_dir: str,
        two_phase: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run full training.

        Parameters
        ----------
        train_dir : str
            Path to ImageFolder-structured training directory.
        val_dir : str
            Path to ImageFolder-structured validation directory.
        two_phase : bool
            Phase 1: frozen backbone. Phase 2: full fine-tune.
        """
        exp_name = getattr(
            getattr(self.cfg, "experiment", None), "name", "spiro_efficientnet"
        )
        writer = SummaryWriter(log_dir=str(self.tb_dir / exp_name))

        mlflow.set_tracking_uri(str(Path(self.cfg.paths.mlflow_dir).resolve()))
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=exp_name)

        try:
            train_loader, val_loader = self._make_loaders(train_dir, val_dir)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.training.amp)

            # Phase 1: frozen backbone
            if two_phase:
                log.info("=== Phase 1: Training head only (frozen backbone) ===")
                self.model._freeze_backbone()
                self._run_phase(
                    train_loader, val_loader, criterion, scaler, writer,
                    num_epochs=min(10, self.cfg.training.epochs // 5),
                    phase_name="phase1",
                )

            # Phase 2: full fine-tune
            log.info("=== Phase 2: Full fine-tune ===")
            self.model.unfreeze_all()
            self._run_phase(
                train_loader, val_loader, criterion, scaler, writer,
                num_epochs=self.cfg.training.epochs,
                phase_name="phase2",
            )

            mlflow.log_metrics({
                "best_val_acc": self._best_val_acc,
                "final_train_loss": self.history["train_loss"][-1],
            })

        finally:
            mlflow.end_run()
            writer.close()

        log.info(f"Training complete — best val_acc: {self._best_val_acc:.4f}")
        return self.history

    def _run_phase(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        writer: SummaryWriter,
        num_epochs: int,
        phase_name: str,
    ) -> None:
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, len(train_loader))

        for epoch in range(1, num_epochs + 1):
            t_loss, t_acc = self._train_epoch(train_loader, criterion, optimizer, scaler)
            v_loss, v_acc = self._val_epoch(val_loader, criterion)

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(v_acc)
                else:
                    scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"[{phase_name}] Epoch {epoch}/{num_epochs} | "
                f"train_loss={t_loss:.4f} train_acc={t_acc:.4f} | "
                f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} | lr={lr:.2e}"
            )

            writer.add_scalars(f"{phase_name}/loss", {"train": t_loss, "val": v_loss}, epoch)
            writer.add_scalars(f"{phase_name}/acc", {"train": t_acc, "val": v_acc}, epoch)
            writer.add_scalar(f"{phase_name}/lr", lr, epoch)

            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            # Checkpoint best
            if v_acc > self._best_val_acc:
                self._best_val_acc = v_acc
                self._patience_counter = 0
                ckpt = self.ckpt_dir / f"{phase_name}_best.pt"
                self.model.save(ckpt)
                self._best_ckpt = ckpt
                log.info(f"  ✓ New best val_acc={v_acc:.4f} — saved to {ckpt}")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.cfg.training.patience:
                    log.info(f"Early stopping after {self.cfg.training.patience} epochs without improvement")
                    break

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                logits = self.model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if self.cfg.training.gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                logits = self.model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_loss / total, correct / total

    @property
    def best_checkpoint(self) -> Optional[Path]:
        return self._best_ckpt
