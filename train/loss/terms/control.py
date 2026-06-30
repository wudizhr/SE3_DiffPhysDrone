from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import LossTerm


@dataclass
class ActionL2Loss(LossTerm):
    name: ClassVar[str] = "action_l2"
    log_name: ClassVar[str] = "loss_d_acc"

    def compute(self, context):
        return context.action.pow(2).sum(-1).mean()


@dataclass
class JerkLoss(LossTerm):
    name: ClassVar[str] = "jerk"
    log_name: ClassVar[str] = "loss_d_jerk"

    def compute(self, context):
        jerk_history = context.action.diff(1, 0).mul(15)
        return jerk_history.pow(2).sum(-1).mean()


@dataclass
class ThrustRegularizationLoss(LossTerm):
    name: ClassVar[str] = "thrust_regularization"
    log_name: ClassVar[str] = "loss_thrust_regularization"

    def compute(self, context):
        if context.action_mode != "ctbr":
            raise ValueError("thrust_regularization loss requires CTBR actions")
        if context.mass is None:
            raise ValueError("thrust_regularization loss requires mass in LossContext")
        thrust_acc = context.action[..., :1] / context.mass.reshape(1, -1, 1)
        return (thrust_acc - 9.80665).abs().mean()


@dataclass
class CtbrSmoothnessLoss(LossTerm):
    name: ClassVar[str] = "ctbr_smoothness"
    log_name: ClassVar[str] = "loss_ctbr_smoothness"

    def compute(self, context):
        import torch

        if context.action_mode != "ctbr":
            raise ValueError("ctbr_smoothness loss requires CTBR actions")
        omega_norm = torch.norm(context.action[..., 1:4], 2, -1).mean()
        if context.action.size(0) > 1:
            action_delta = torch.norm(context.action.diff(1, 0), 2, -1).mean()
        else:
            action_delta = context.action.new_zeros(())
        return omega_norm + action_delta
