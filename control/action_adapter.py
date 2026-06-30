from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActionAdapterResult:
    control: Any
    a_pred: Any | None = None
    v_pred: Any | None = None
    omega_cmd: Any | None = None
    thrust_cmd: Any | None = None


class AccelVelocityActionAdapter:
    """Convert the current 6D policy output into Env.run acceleration control."""

    mode = "accel_velocity"

    def initial_control(self, env: Any) -> Any:
        return env.act

    def to_control(self, raw_action: Any, env: Any, policy_frame: Any) -> ActionAdapterResult:
        a_pred, v_pred, *_ = (policy_frame @ raw_action.reshape(env.batch_size, 3, -1)).unbind(-1)
        control = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        return ActionAdapterResult(control=control, a_pred=a_pred, v_pred=v_pred)


class CtbrActionAdapter:
    """Convert raw CTBR policy output into commanded body rates and thrust."""

    mode = "ctbr"

    def initial_control(self, env: Any) -> Any:
        import torch

        return torch.cat([env.collective_thrust, env.omega], -1)

    def to_control(self, raw_action: Any, env: Any, policy_frame: Any) -> ActionAdapterResult:
        import torch

        thrust_raw = raw_action[:, :1]
        body_rate_raw = raw_action[:, 1:4]
        thrust_center = 0.5 * (env.ctbr_thrust_min + env.ctbr_thrust_max)
        thrust_half_range = 0.5 * (env.ctbr_thrust_max - env.ctbr_thrust_min)
        thrust_cmd = thrust_center + thrust_half_range * torch.tanh(thrust_raw)
        omega_cmd = env.ctbr_body_rate_limit * torch.tanh(body_rate_raw)
        control = torch.cat([thrust_cmd, omega_cmd], -1)
        return ActionAdapterResult(control=control, omega_cmd=omega_cmd, thrust_cmd=thrust_cmd)


def create_action_adapter(action_mode: str | None = None):
    mode = action_mode or "accel_velocity"
    if mode == "accel_velocity":
        return AccelVelocityActionAdapter()
    if mode == "ctbr":
        return CtbrActionAdapter()
    raise ValueError(f"Unknown action_mode '{mode}'")
