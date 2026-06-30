from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import LossTerm


def barrier(x, v_to_pt):
    """Obstacle barrier loss, weighted more heavily when moving toward obstacles."""

    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def obstacle_distance_and_weight(context):
    import torch

    distance = context.distance
    if distance.size(1) > 1:
        with torch.no_grad():
            v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
        obstacle_distance = distance[:, 1:]
    else:
        v_to_pt = torch.ones_like(distance)
        obstacle_distance = distance
    return obstacle_distance, v_to_pt


@dataclass
class ObjectAvoidanceLoss(LossTerm):
    name: ClassVar[str] = "object_avoidance"
    log_name: ClassVar[str] = "loss_obj_avoidance"

    def compute(self, context):
        obstacle_distance, v_to_pt = obstacle_distance_and_weight(context)
        return barrier(obstacle_distance, v_to_pt)


@dataclass
class CollisionLoss(LossTerm):
    name: ClassVar[str] = "collision"
    log_name: ClassVar[str] = "loss_collide"

    def compute(self, context):
        from torch.nn import functional as F

        obstacle_distance, v_to_pt = obstacle_distance_and_weight(context)
        return F.softplus(obstacle_distance.mul(-32)).mul(v_to_pt).mean()
