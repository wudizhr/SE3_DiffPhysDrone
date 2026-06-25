from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from control import create_action_adapter


@dataclass
class PolicyRunnerConfig:
    ctl_dt: float
    max_steps: int
    no_odom: bool = False
    yaw_target_correction: bool = False
    depth_pool_kernel: int = 4
    deterministic_visualization: bool = True
    sensor_name: str = "depth_odom"
    action_mode: str = "accel_velocity"


class PolicyRunner:
    def __init__(self, env, model, config: PolicyRunnerConfig, observation_builder=None):
        self.env = env
        self.model = model
        self.config = config
        self.observation_builder = observation_builder
        self.action_adapter = create_action_adapter(config.action_mode)
        self.hidden_state = None
        self.act_buffer = [env.act.clone(), env.act.clone()]

    def reset_model(self) -> None:
        self.hidden_state = None
        self.model.reset()

    def step(self, step_idx: int) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        env = self.env
        depth, _ = env.render(self.config.ctl_dt)
        target_v_raw = env.p_target - env.p.detach()
        yaw_correction_vec = target_v_raw if self.config.yaw_target_correction else env.v
        env.run(self.act_buffer[step_idx], self.config.ctl_dt, yaw_correction_vec)

        R, state, local_v, target_v = self.build_state(
            target_v_raw,
            full_attitude=self.config.sensor_name == "mid360",
        )
        if self.observation_builder is not None and self.config.sensor_name == "mid360":
            sensor_inputs = self.observation_builder.render_inputs(
                env,
                self.config.ctl_dt,
                include_debug_outputs=True,
            )
            obs = self.observation_builder.build(sensor_inputs=sensor_inputs, state=state)
            action, _, self.hidden_state = self.model(obs, hx=self.hidden_state)
            pooled = obs["mid360_pseudo_image"]
            mid360_pseudo_image = pooled
            mid360_points = sensor_inputs.get("mid360_points")
            mid360_world_points = sensor_inputs.get("mid360_world_points")
            mid360_ranges = sensor_inputs.get("mid360_ranges")
        else:
            model_input = 3 / depth.clamp(0.3, 24) - 0.6
            pooled = F.max_pool2d(
                model_input[:, None],
                self.config.depth_pool_kernel,
                self.config.depth_pool_kernel,
            )
            action, _, self.hidden_state = self.model(pooled, state, self.hidden_state)
            mid360_pseudo_image = None
            mid360_points = None
            mid360_world_points = None
            mid360_ranges = None

        adapted_action = self.action_adapter.to_control(action, env, R)
        act = adapted_action.control
        self.act_buffer.append(act)

        result = {
            "depth": depth,
            "pooled_depth": pooled,
            "raw_action": action,
            "action": act,
            "rotation": env.R,
            "position": env.p,
            "velocity": env.v,
        }
        if mid360_pseudo_image is not None:
            result["mid360_pseudo_image"] = mid360_pseudo_image
        if mid360_points is not None:
            result["mid360_points"] = mid360_points
            result["mid360_world_points"] = mid360_world_points
            result["mid360_ranges"] = mid360_ranges
        return result

    def build_state(self, target_v_raw, *, full_attitude: bool = False):
        import torch
        import torch.nn.functional as F

        env = self.env
        R_body = env.R
        fwd = R_body[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True).clamp_min(1e-6)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

        target_v_local = torch.squeeze(target_v[:, None] @ R, 1)
        attitude = env.R.reshape(env.batch_size, 9) if full_attitude else env.R[:, 2]
        state = [target_v_local, attitude, env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not self.config.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)
        return R, state, local_v, target_v
