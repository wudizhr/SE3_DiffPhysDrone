from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActionAdapterResult:
    control: Any
    a_pred: Any | None = None
    v_pred: Any | None = None


class AccelVelocityActionAdapter:
    """Convert the current 6D policy output into Env.run acceleration control."""

    mode = "accel_velocity"

    def to_control(self, raw_action: Any, env: Any, policy_frame: Any) -> ActionAdapterResult:
        a_pred, v_pred, *_ = (policy_frame @ raw_action.reshape(env.batch_size, 3, -1)).unbind(-1)
        control = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        return ActionAdapterResult(control=control, a_pred=a_pred, v_pred=v_pred)


def create_action_adapter(action_mode: str | None = None):
    mode = action_mode or "accel_velocity"
    if mode == "accel_velocity":
        return AccelVelocityActionAdapter()
    raise ValueError(f"Unknown action_mode '{mode}'")
