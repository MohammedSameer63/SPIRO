"""
SPIRO ML — VerifyTrainer
Full two-phase EfficientNetV2 training loop:
  Phase 1: Frozen backbone, train classification head only
  Phase 2: Full fine-tuning with lower LR

Features:
  - AMP with gradient scaling
  - Gradient accumulation
  - EMA weight averaging
  - Early stopping
  - TensorBoard + CSV logging
  - Checkpoint management
  - MLflow experiment tracking
  - Per-epoch console summary
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.ml.core.logger import get_logger
from lib.ml.verification.model.verify_model import VerifierModel
from lib.ml.verification.training.losses import build_loss
from lib.ml.verification.training.verify_dataset import build_dataloaders
from lib.ml.verification.verify_config import VerifyConfig
from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics

log = get_logger(__name__)


class VerifyTrainer:
    """
    Trains the EfficientNetV2 verifier in two phases.

    Example
    -------
    >>> cfg = VerifyConfig.load("configs/verification/effnetv2_s.yaml")
    >>> trainer = VerifyTrainer(cfg)
    >>> history = trainer.train()
    """

    def __init__(self, cfg: VerifyConfig) -> None:
        self.cfg = cfg
        self.exp_name = cfg.experiment.name

        self.ckpt_dir = Path(cfg.checkpoint.dir) / self.exp_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric: float = 0.0
        self._patience_counter: int = 0
        self._best_ckpt: Optional[Path] = None

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "val_top5_acc": [], "lr": [],
        }

        # Logging
        self._writer: Optional[SummaryWriter] = None
        self._csv_path = Path(cfg.logging.csv_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Execute full two-phase training. Returns history and summary."""
        log.info("═" * 55)
        log.info(f"  SPIRO VerifyTrainer — {self.exp_name}")
        log.info("═" * 55)
        log.info(f"  Variant:  {self.cfg.model.variant}")
        log.info(f"  Classes:  {self.cfg.model.num_classes}")
        log.info(f"  Input:    {self.cfg.input_size}×{self.cfg.input_size}")
        log.info(f"  Epochs:   {self.cfg.training.epochs}")
        log.info(f"  Batch:    {self.cfg.training.batch_size}")

        t0 = time.perf_counter()

        # TensorBoard
        if self.cfg.logging.tensorboard:
            tb_dir = Path(self.cfg.logging.tensorboard_dir) / self.exp_name
            self._writer = SummaryWriter(log_dir=str(tb_dir))

        # Data
        train_loader, val_loader, test_loader = build_dataloaders(self.cfg)

        # Loss
        class_weights = None
        if self.cfg.loss.name == "weighted":
            # Build from dataset class counts
            from lib.ml.verification.training.verify_dataset import VerifyDataset
            from lib.ml.verification.training.verify_dataset import build_train_transforms
            ds = VerifyDataset(
                Path(self.cfg.dataset.root), "train",
                build_train_transforms(self.cfg),
            )
            class_weights = ds.class_weights()
        criterion = build_loss(self.cfg, class_weights)

        # Phase 1: freeze backbone
        model = VerifierModel(self.cfg, freeze_backbone=True)
        phase1_ep = self.cfg.training.phase1_epochs
        phase2_ep = self.cfg.training.phase2_epochs

        if phase1_ep > 0:
            log.info(f"── Phase 1: frozen backbone ({phase1_ep} epochs)")
            self._run_phase(
                model, train_loader, val_loader, criterion,
                lr=self.cfg.optimizer.lr_head,
                num_epochs=phase1_ep,
                phase_name="phase1",
            )

        # Phase 2: full fine-tune
        log.info(f"── Phase 2: full fine-tune ({phase2_ep} epochs)")
        model.unfreeze_all()
        self._run_phase(
            model, train_loader, val_loader, criterion,
            lr=self.cfg.optimizer.lr0,
            num_epochs=phase2_ep,
            phase_name="phase2",
        )

        # Final test evaluation
        test_metrics = self._evaluate(model, test_loader, criterion, "test")
        log.info(
            f"Test — top1={test_metrics['top1_accuracy']:.4f} "
            f"top5={test_metrics['top5_accuracy']:.4f}"
        )

        # ONNX export
        onnx_path = None
        if self.cfg.export.auto_export_onnx and self._best_ckpt:
            onnx_path = self._export_onnx(self._best_ckpt)

        elapsed = time.perf_counter() - t0
        summary = {
            "experiment": self.exp_name,
            "elapsed_s": round(elapsed, 1),
            "best_top1": self._best_metric,
            "best_checkpoint": str(self._best_ckpt) if self._best_ckpt else None,
            "onnx_path": str(onnx_path) if onnx_path else None,
            "test_metrics": test_metrics,
        }

        # Save summary
        out = self.ckpt_dir / "training_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

        if self._writer:
            self._writer.close()

        log.info(f"Training complete in {elapsed/60:.1f}min | best top1={self._best_metric:.4f}")
        return summary

    # ------------------------------------------------------------------
    # Phase runner
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        model: VerifierModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        lr: float,
        num_epochs: int,
        phase_name: str,
    ) -> None:
        optimizer = self._build_optimizer(model, lr)
        scheduler = self._build_scheduler(optimizer, num_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.training.amp)
        accum_steps = max(1, self.cfg.training.gradient_accumulation)

        for epoch in range(1, num_epochs + 1):
            # Train
            tr_loss, tr_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler, accum_steps
            )
            # Validate
            val_metrics = self._evaluate(model, val_loader, criterion, "val")
            v_loss = val_metrics["loss"]
            v_acc = val_metrics["top1_accuracy"]
            v_top5 = val_metrics["top5_accuracy"]

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(v_acc)
                else:
                    scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_global = len(self.history["train_loss"]) + 1

            # Record history
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["val_top5_acc"].append(v_top5)
            self.history["lr"].append(lr_now)

            # TensorBoard
            if self._writer:
                self._writer.add_scalars(f"{phase_name}/loss", {"train": tr_loss, "val": v_loss}, epoch_global)
                self._writer.add_scalars(f"{phase_name}/acc", {"train": tr_acc, "val": v_acc, "val_top5": v_top5}, epoch_global)
                self._writer.add_scalar(f"{phase_name}/lr", lr_now, epoch_global)

            # CSV
            if self.cfg.logging.csv:
                self._append_csv(epoch_global, tr_loss, tr_acc, v_loss, v_acc, v_top5, lr_now)

            # Console
            log.info(
                f"[{phase_name} {epoch:>3}/{num_epochs}] "
                f"loss={tr_loss:.4f}/{v_loss:.4f} "
                f"top1={tr_acc:.4f}/{v_acc:.4f} "
                f"top5={v_top5:.4f} lr={lr_now:.2e}"
            )

            # Best checkpoint
            if v_acc > self._best_metric:
                self._best_metric = v_acc
                self._patience_counter = 0
                ckpt = self.ckpt_dir / "best_effnet.pt"
                model.save(ckpt)
                self._best_ckpt = ckpt
                log.info(f"  ✓ New best top1={v_acc:.4f} → {ckpt}")
            else:
                self._patience_counter += 1

            # Save last
            last_ckpt = self.ckpt_dir / "last_effnet.pt"
            model.save(last_ckpt)

            # Periodic
            if self.cfg.training.save_period > 0 and epoch % self.cfg.training.save_period == 0:
                per_ckpt = self.ckpt_dir / f"effnet_epoch_{epoch_global:05d}.pt"
                model.save(per_ckpt)

            # Early stopping
            if (self.cfg.training.patience > 0
                    and self._patience_counter >= self.cfg.training.patience):
                log.info(f"Early stopping at epoch {epoch_global} ({phase_name})")
                break

    # ------------------------------------------------------------------
    # Train / eval epochs
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: VerifierModel,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        accum_steps: int,
    ) -> Tuple[float, float]:
        model.train()
        criterion.to(model.device)
        total_loss = total_correct = total = 0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(loader):
            images = images.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                logits = model(images)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        return total_loss / max(total, 1), total_correct / max(total, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: VerifierModel,
        loader: DataLoader,
        criterion: nn.Module,
        split: str,
    ) -> Dict[str, float]:
        model.eval()
        criterion.to(model.device)
        all_logits, all_labels = [], []
        total_loss = total = 0

        for images, labels in loader:
            images = images.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        if total == 0:
            return {"loss": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}

        logits_all = torch.cat(all_logits)
        labels_all = torch.cat(all_labels)
        metrics = VerifyMetrics.compute_basic(logits_all, labels_all, top_k=5)
        metrics["loss"] = total_loss / total
        return metrics

    # ------------------------------------------------------------------
    # Optimiser / scheduler factories
    # ------------------------------------------------------------------

    def _build_optimizer(self, model: VerifierModel, lr: float) -> optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        opt_name = self.cfg.optimizer.name
        wd = self.cfg.optimizer.weight_decay
        betas = tuple(self.cfg.optimizer.betas)
        if opt_name == "AdamW":
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
        if opt_name == "Adam":
            return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
        return optim.SGD(
            params, lr=lr, momentum=self.cfg.optimizer.momentum, weight_decay=wd, nesterov=True
        )

    def _build_scheduler(self, optimizer: optim.Optimizer, n_epochs: int):
        name = self.cfg.scheduler.name
        if name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
        if name == "step":
            return StepLR(optimizer, step_size=self.cfg.scheduler.step_size,
                          gamma=self.cfg.scheduler.step_gamma)
        if name == "plateau":
            return ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        return None

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_onnx(self, checkpoint: Path) -> Optional[Path]:
        try:
            from lib.ml.verification.export.verify_export import VerifyExporter
            exporter = VerifyExporter(self.cfg)
            model = VerifierModel.load(self.cfg, checkpoint)
            return exporter.export(model)
        except Exception as e:
            log.error(f"ONNX export failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _append_csv(
        self, epoch: int, tr_loss: float, tr_acc: float,
        v_loss: float, v_acc: float, v_top5: float, lr: float,
    ) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["epoch", "train_loss", "train_acc",
                            "val_loss", "val_acc", "val_top5_acc", "lr"])
            w.writerow([epoch, round(tr_loss, 6), round(tr_acc, 6),
                        round(v_loss, 6), round(v_acc, 6), round(v_top5, 6),
                        round(lr, 8)])
