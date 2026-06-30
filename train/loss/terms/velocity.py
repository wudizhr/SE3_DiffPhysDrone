from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import LossTerm


@dataclass
class VelocityTrackingLoss(LossTerm):
    name: ClassVar[str] = "velocity_tracking"
    log_name: ClassVar[str] = "loss_v"

    def compute(self, context):
        import torch
        from torch.nn import functional as F

        v_history_cum = context.v.cumsum(0)
        v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
        delta_v = torch.norm(v_history_avg - context.target_v[1:1 - 30], 2, -1)
        return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))


@dataclass
class VelocityPredictionLoss(LossTerm):
    name: ClassVar[str] = "velocity_prediction"
    log_name: ClassVar[str] = "loss_v_pred"

    def compute(self, context):
        from torch.nn import functional as F

        if context.v_pred is None:
            return context.v.new_zeros(())
        return F.mse_loss(context.v_pred, context.v.detach())
