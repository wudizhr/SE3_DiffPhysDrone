from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import LossTerm


@dataclass
class YawAlignmentLoss(LossTerm):
    """Penalize mismatch between body x-axis and velocity direction."""

    name: ClassVar[str] = "yaw_alignment"
    log_name: ClassVar[str] = "loss_yaw_alignment"

    def compute(self, context):
        import torch
        from torch.nn import functional as F

        if context.R is None:
            raise ValueError("yaw_alignment loss requires R_history in LossContext")
        body_x = context.R[:, :, :, 0]
        speed = torch.norm(context.v, 2, -1, keepdim=True)
        valid = speed.squeeze(-1) > 1e-6
        v_unit = F.normalize(context.v, p=2, dim=-1, eps=1e-6)
        alignment = (body_x * v_unit).sum(-1).clamp(-1.0, 1.0)
        loss = 1.0 - alignment
        if valid.any():
            return loss[valid].mean()
        return context.v.new_zeros(())


@dataclass
class TargetYawAlignmentLoss(LossTerm):
    """Penalize horizontal yaw mismatch between body x-axis and target velocity."""

    name: ClassVar[str] = "target_yaw_alignment"
    log_name: ClassVar[str] = "loss_target_yaw_alignment"

    def compute(self, context):
        import torch
        from torch.nn import functional as F

        if context.R is None:
            raise ValueError("target_yaw_alignment loss requires R_history in LossContext")
        body_x_xy = context.R[:, :, :2, 0]
        target_xy = context.target_v[:, :, :2]
        target_norm = torch.norm(target_xy, 2, -1, keepdim=True)
        valid = target_norm.squeeze(-1) > 1e-6
        body_yaw = F.normalize(body_x_xy, p=2, dim=-1, eps=1e-6)
        target_yaw = F.normalize(target_xy, p=2, dim=-1, eps=1e-6)
        alignment = (body_yaw * target_yaw).sum(-1).clamp(-1.0, 1.0)
        loss = 1.0 - alignment
        if valid.any():
            return loss[valid].mean()
        return context.v.new_zeros(())
