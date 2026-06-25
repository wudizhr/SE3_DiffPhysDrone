from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

from se3diff_config.schema import LossConfig

if TYPE_CHECKING:
    import torch


@dataclass
class TrainingLossResult:
    loss: torch.Tensor
    loss_v: torch.Tensor
    loss_v_pred: torch.Tensor
    loss_obj_avoidance: torch.Tensor
    loss_d_acc: torch.Tensor
    loss_d_jerk: torch.Tensor
    loss_collide: torch.Tensor
    distance: torch.Tensor
    speed_history: torch.Tensor
    v_history: torch.Tensor
    act_buffer: torch.Tensor

    def tensorboard_scalars(self, success: torch.Tensor, avg_speed: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "loss": self.loss,
            "loss_v": self.loss_v,
            "loss_v_pred": self.loss_v_pred,
            "loss_obj_avoidance": self.loss_obj_avoidance,
            "loss_d_acc": self.loss_d_acc,
            "loss_d_jerk": self.loss_d_jerk,
            "loss_collide": self.loss_collide,
            "success": success,
            "max_speed": self.speed_history.max(0).values.mean(),
            "avg_speed": avg_speed.mean(),
            "ar": (success * avg_speed).mean(),
        }


def barrier(x: torch.Tensor, v_to_pt: torch.Tensor) -> torch.Tensor:
    """Obstacle barrier loss, weighted more heavily when moving toward obstacles."""

    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def compute_training_loss(
    *,
    loss_config: LossConfig,
    v_history: List[torch.Tensor],
    target_v_history: List[torch.Tensor],
    v_preds: List[torch.Tensor],
    act_buffer: List[torch.Tensor],
    vec_to_pt_history: List[torch.Tensor],
    margin: torch.Tensor,
) -> TrainingLossResult:
    import torch
    from torch.nn import functional as F

    # 速度跟踪 loss：用 30 步滑动平均速度和目标速度比较，降低瞬时噪声影响。
    v_history_tensor = torch.stack(v_history)
    v_history_cum = v_history_tensor.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history_tensor = torch.stack(target_v_history)
    delta_v = torch.norm(v_history_avg - target_v_history_tensor[1:1 - 30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    # 训练网络的速度预测头，让无里程计模式也能从视觉/隐藏状态估计速度。
    v_preds_tensor = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds_tensor, v_history_tensor.detach())

    # 控制正则：约束动作幅值和 jerk，避免控制过激。
    act_buffer_tensor = torch.stack(act_buffer)
    jerk_history = act_buffer_tensor.diff(1, 0).mul(15)
    loss_d_acc = act_buffer_tensor.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

    # 避障/碰撞 loss：使用未来短时间采样点到最近障碍物的距离。
    vec_to_pt_history_tensor = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history_tensor, 2, -1)
    distance = distance - margin
    if distance.size(1) > 1:
        with torch.no_grad():
            # 如果距离正在快速变小，说明正朝障碍物接近，提高该位置的惩罚权重。
            v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
        obstacle_distance = distance[:, 1:]
    else:
        v_to_pt = torch.ones_like(distance)
        obstacle_distance = distance
    loss_obj_avoidance = barrier(obstacle_distance, v_to_pt)
    loss_collide = F.softplus(obstacle_distance.mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history_tensor.norm(2, -1)

    # 总损失由速度跟踪、避障、碰撞、控制平滑和辅助预测项加权组成。
    loss = loss_config.coef_v * loss_v + \
        loss_config.coef_obj_avoidance * loss_obj_avoidance + \
        loss_config.coef_d_acc * loss_d_acc + \
        loss_config.coef_d_jerk * loss_d_jerk + \
        loss_config.coef_v_pred * loss_v_pred + \
        loss_config.coef_collide * loss_collide

    return TrainingLossResult(
        loss=loss,
        loss_v=loss_v,
        loss_v_pred=loss_v_pred,
        loss_obj_avoidance=loss_obj_avoidance,
        loss_d_acc=loss_d_acc,
        loss_d_jerk=loss_d_jerk,
        loss_collide=loss_collide,
        distance=distance,
        speed_history=speed_history,
        v_history=v_history_tensor,
        act_buffer=act_buffer_tensor,
    )
