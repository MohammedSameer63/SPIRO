"""
SPIRO ML — ModelPromoter
Automated gate that decides whether to promote a candidate model to
production. Runs validation, comparison, and policy checks before
calling ModelDeployer.

Promotion policy
----------------
1. ModelValidator must pass all checks
2. Candidate must beat production on primary metric (min_improvement)
3. Latency regression must be within threshold
4. No critical secondary metric regression
5. Taxonomy must be consistent (109 classes)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger
from lib.ml.mlops.deployment.model_deployer import ModelDeployer
from lib.ml.mlops.registry.model_registry import ModelRegistry
from lib.ml.mlops.registry.model_metadata import ModelMetadata
from lib.ml.mlops.validation.model_validator import ModelValidator, ValidationReport
from lib.ml.mlops.validation.model_compare import ModelComparator, ModelComparisonReport

log = get_logger(__name__)


@dataclass
class PromotionDecision:
    model_id: str
    version: str
    decided_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    approved: bool = False
    stage: str = ""
    validation_passed: bool = False
    comparison_recommendation: str = ""
    reasons: List[str] = field(default_factory=list)
    deployed: bool = False
    deployment_strategy: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ModelPromoter:
    """
    Automated model promotion pipeline.

    Parameters
    ----------
    registry : ModelRegistry
    validator : ModelValidator
    comparator : ModelComparator
    deployer : ModelDeployer
    auto_deploy : bool
        If True, deploy automatically when promotion criteria are met.
    primary_metric : str
    min_improvement : float
    deployment_strategy : str

    Example
    -------
    >>> promoter = ModelPromoter(auto_deploy=True)
    >>> decision = promoter.evaluate_and_promote(
    ...     model_id="yolov11s",
    ...     candidate_version="v2.0.0",
    ... )
    >>> print(decision.approved, decision.deployed)
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        validator: Optional[ModelValidator] = None,
        comparator: Optional[ModelComparator] = None,
        deployer: Optional[ModelDeployer] = None,
        auto_deploy: bool = False,
        primary_metric: str = "mAP50_95",
        min_improvement: float = 0.001,
        deployment_strategy: str = "blue_green",
        report_dir: Path = Path("mlops/reports"),
    ) -> None:
        self.registry   = registry   or ModelRegistry()
        self.validator  = validator  or ModelValidator()
        self.comparator = comparator or ModelComparator(
            primary_metric=primary_metric, min_improvement=min_improvement
        )
        self.deployer   = deployer   or ModelDeployer(registry=self.registry, validator=self.validator)
        self.auto_deploy = auto_deploy
        self.primary_metric = primary_metric
        self.min_improvement = min_improvement
        self.deployment_strategy = deployment_strategy
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_promote(
        self,
        model_id: str,
        candidate_version: str,
        task: str = "detection",
        deployed_by: str = "system",
        notes: str = "",
    ) -> PromotionDecision:
        """
        Run the full promotion pipeline for a candidate version.

        1. Load candidate metadata from registry
        2. Run ModelValidator
        3. Compare against current production model
        4. Make promotion decision
        5. Deploy if auto_deploy=True and approved

        Returns
        -------
        PromotionDecision
        """
        decision = PromotionDecision(
            model_id=model_id,
            version=candidate_version,
            notes=notes,
        )

        # ── Step 1: Load candidate ────────────────────────────────────
        candidate = self.registry.get(model_id, candidate_version)
        if candidate is None:
            decision.reasons.append(f"Candidate {model_id} v{candidate_version} not found in registry")
            return decision

        onnx_path = Path(candidate.onnx_path) if candidate.onnx_path else None
        if onnx_path is None or not onnx_path.exists():
            decision.reasons.append(f"ONNX not found: {candidate.onnx_path}")
            return decision

        # ── Step 2: Validation ────────────────────────────────────────
        val_report = self.validator.validate(
            onnx_path=onnx_path,
            model_id=model_id,
            version=candidate_version,
            input_size=candidate.input_size,
            num_classes=candidate.num_classes,
            task=candidate.task,
        )
        self.validator.save_report(val_report, self.report_dir / "validation")
        decision.validation_passed = val_report.overall_passed

        if not val_report.overall_passed:
            decision.approved = False
            decision.reasons.append(f"Validation failed: {val_report.summary}")
            self._save_decision(decision)
            log.warning(f"Promotion REJECTED [{model_id} v{candidate_version}]: {decision.reasons}")
            return decision

        decision.reasons.append(f"Validation passed ({val_report.summary})")

        # ── Step 3: Comparison vs production ─────────────────────────
        production = self.registry.get_production_model(model_id)
        if production is not None:
            cmp_report = self.comparator.compare(
                candidate=candidate,
                production=production,
                run_latency_benchmark=bool(onnx_path.exists() and
                                           Path(production.onnx_path or "").exists()),
            )
            decision.comparison_recommendation = cmp_report.recommendation
            decision.reasons.extend(cmp_report.reasons)

            if cmp_report.recommendation != "promote":
                decision.approved = False
                decision.stage = "rejected"
                self._save_decision(decision)
                log.info(f"Promotion REJECTED [{model_id} v{candidate_version}]")
                return decision
        else:
            decision.reasons.append("No production model — first deployment, skipping comparison")

        # ── Step 4: Approve ───────────────────────────────────────────
        decision.approved = True
        decision.stage = "approved"

        # ── Step 5: Deploy ────────────────────────────────────────────
        if self.auto_deploy:
            try:
                self.deployer.deploy(
                    model_id=model_id,
                    version=candidate_version,
                    onnx_path=onnx_path,
                    strategy=self.deployment_strategy,
                    task=task,
                    deployed_by=deployed_by,
                    validate=False,   # already validated above
                )
                decision.deployed = True
                decision.deployment_strategy = self.deployment_strategy
                decision.stage = "production"
                decision.reasons.append(
                    f"Deployed via {self.deployment_strategy} strategy"
                )
            except Exception as e:
                decision.deployed = False
                decision.reasons.append(f"Deployment failed: {e}")
        else:
            decision.reasons.append(
                "auto_deploy=False — manual deployment required: "
                f"python training/mlops/model_deployer.py deploy "
                f"--model-id {model_id} --version {candidate_version}"
            )

        self._save_decision(decision)
        log.info(
            f"Promotion {'APPROVED+DEPLOYED' if decision.deployed else 'APPROVED'} "
            f"[{model_id} v{candidate_version}]"
        )
        return decision

    def _save_decision(self, decision: PromotionDecision) -> Path:
        out = (
            self.report_dir / "promotions" /
            f"{decision.model_id}_{decision.version}_promotion.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(decision.to_json())
        return out
