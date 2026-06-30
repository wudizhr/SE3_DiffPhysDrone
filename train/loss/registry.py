from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Type

from se3diff_config.schema import LossConfig

from .context import LossContext
from .result import TrainingLossResult
from .terms.base import LossTerm
from .terms.attitude import TargetYawAlignmentLoss, YawAlignmentLoss
from .terms.control import ActionL2Loss, CtbrSmoothnessLoss, JerkLoss, ThrustRegularizationLoss
from .terms.obstacle import CollisionLoss, ObjectAvoidanceLoss
from .terms.velocity import VelocityPredictionLoss, VelocityTrackingLoss


TERM_CLASSES: Dict[str, Type[LossTerm]] = {
    "velocity_tracking": VelocityTrackingLoss,
    "velocity_prediction": VelocityPredictionLoss,
    "object_avoidance": ObjectAvoidanceLoss,
    "collision": CollisionLoss,
    "action_l2": ActionL2Loss,
    "jerk": JerkLoss,
    "yaw_alignment": YawAlignmentLoss,
    "target_yaw_alignment": TargetYawAlignmentLoss,
    "thrust_regularization": ThrustRegularizationLoss,
    "ctbr_smoothness": CtbrSmoothnessLoss,
}

LEGACY_TERM_CONFIG = OrderedDict([
    ("velocity_tracking", "coef_v"),
    ("object_avoidance", "coef_obj_avoidance"),
    ("action_l2", "coef_d_acc"),
    ("jerk", "coef_d_jerk"),
    ("velocity_prediction", "coef_v_pred"),
    ("collision", "coef_collide"),
])


@dataclass
class LossWeightSchedule:
    """Optional linear curriculum for one loss weight."""

    base_weight: float
    start_weight: float | None = None
    end_weight: float | None = None
    start_iter: int = 0
    end_iter: int = 0

    def weight_at(self, step: int | None) -> float:
        if self.start_weight is None or self.end_weight is None:
            return self.base_weight
        if step is None:
            return self.base_weight
        if self.end_iter <= self.start_iter:
            return self.end_weight if step >= self.end_iter else self.start_weight
        if step <= self.start_iter:
            return self.start_weight
        if step >= self.end_iter:
            return self.end_weight
        progress = (step - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_weight + progress * (self.end_weight - self.start_weight)


@dataclass
class LossRegistry:
    terms: list[LossTerm] = field(default_factory=list)
    schedules: dict[str, LossWeightSchedule] = field(default_factory=dict)

    def register(self, term: LossTerm, schedule: LossWeightSchedule | None = None) -> None:
        self.terms.append(term)
        self.schedules[term.name] = schedule or LossWeightSchedule(base_weight=term.weight)

    def extend(self, terms: Iterable[tuple[LossTerm, LossWeightSchedule]]) -> None:
        for term, schedule in terms:
            self.register(term, schedule)

    def required_history(self) -> set[str]:
        required = set()
        for term in self.terms:
            if term.enabled:
                required.update(term.required_history)
        return required

    def compute(self, context: LossContext, step: int | None = None) -> TrainingLossResult:
        import torch

        terms = OrderedDict()
        weighted_terms = OrderedDict()
        weights = OrderedDict()
        total = context.v.new_zeros(())
        for term in self.terms:
            if not term.enabled:
                continue
            value = term.compute(context)
            terms[term.log_name] = value
            weight = self.schedules[term.name].weight_at(step)
            weights[term.log_name] = weight
            weighted = value * weight
            weighted_terms[term.log_name] = weighted
            total = total + weighted
        if not terms:
            total = torch.zeros((), device=context.v.device, dtype=context.v.dtype)
        return TrainingLossResult(
            total=total,
            terms=dict(terms),
            weighted_terms=dict(weighted_terms),
            weights=dict(weights),
            distance=context.distance,
            speed_history=context.speed_history,
            v_history=context.v,
            act_buffer=context.action,
        )


def _schedule_from_config(weight: float, config: dict) -> LossWeightSchedule:
    curriculum = config.get("curriculum") or {}
    if not curriculum:
        return LossWeightSchedule(base_weight=weight)
    return LossWeightSchedule(
        base_weight=weight,
        start_weight=float(curriculum.get("start_weight", 0.0)),
        end_weight=float(curriculum.get("end_weight", weight)),
        start_iter=int(curriculum.get("start_iter", 0)),
        end_iter=int(curriculum.get("end_iter", 0)),
    )


def _term_from_config(name: str, config: dict) -> tuple[LossTerm, LossWeightSchedule]:
    try:
        cls = TERM_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown loss term '{name}'") from exc
    weight = float(config.get("weight", 1.0))
    return (
        cls(weight=weight, enabled=bool(config.get("enabled", True))),
        _schedule_from_config(weight, config),
    )


def _legacy_terms(loss_config: LossConfig) -> list[tuple[LossTerm, LossWeightSchedule]]:
    terms = []
    for term_name, coef_name in LEGACY_TERM_CONFIG.items():
        cls = TERM_CLASSES[term_name]
        weight = float(getattr(loss_config, coef_name))
        terms.append((cls(weight=weight, enabled=True), LossWeightSchedule(base_weight=weight)))
    return terms


def build_loss_registry(loss_config: LossConfig) -> LossRegistry:
    registry = LossRegistry()
    if loss_config.terms:
        registry.extend(_term_from_config(name, term_config or {}) for name, term_config in loss_config.terms.items())
    else:
        registry.extend(_legacy_terms(loss_config))
    return registry
