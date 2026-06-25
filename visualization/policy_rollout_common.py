"""Torch/CUDA helpers shared by offline rollout export."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_DIR, VIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from se3diff_config.io import load_checkpoint_training_config, merge_checkpoint_with_user_config
from visualization_common import normalize_checkpoint_keys


def as_1_batch_tensor(value: Any, device: torch.device) -> torch.Tensor:
    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=torch.float32).unsqueeze(0).clone()


def as_scalar_tensor(value: Any, device: torch.device, shape=(1, 1)) -> torch.Tensor:
    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=torch.float32).reshape(shape).clone()


def depth_pool_kernel_for_model() -> int:
    return 4


def load_training_inference_config(checkpoint_path: Path) -> Dict[str, Any]:
    return load_checkpoint_training_config(checkpoint_path)


def merge_training_inference_config(config: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    return merge_checkpoint_with_user_config(config, checkpoint_path)


def create_env_from_snapshot(snapshot: Dict[str, Any], config: Dict[str, Any], device: torch.device) -> Env:
    import torch
    from env import Env

    env = Env(
        1,
        int(snapshot.get("width", 64)),
        int(snapshot.get("height", 48)),
        float(snapshot.get("grad_decay", config.get("grad_decay", 0.4))),
        device,
        fov_x_half_tan=float(snapshot.get("fov_x_half_tan", config.get("fov_x_half_tan", 0.53))),
        single=True,
        gate=bool(snapshot.get("gate", False)),
        ground_voxels=bool(snapshot.get("ground_voxels", False)),
        ceiling=bool(snapshot.get("ceiling", False)),
        ceiling_height=float(snapshot.get("ceiling_height", 3.0)),
        scaffold=bool(snapshot.get("scaffold", False)),
        speed_mtp=float(snapshot.get("speed_mtp", 1.0)),
        random_rotation=bool(snapshot.get("random_rotation", False)),
        cam_angle=int(snapshot.get("cam_angle", config.get("cam_angle", 10))),
        start=snapshot.get("p"),
        target=snapshot.get("p_target"),
        max_speed=float(snapshot.get("max_speed", 4.0)),
        margin=float(snapshot.get("margin", 0.25)),
        gap=bool(snapshot.get("gap", False)),
        gap_prob=float(snapshot.get("gap_prob", 0.0)),
        n_balls=int(snapshot.get("n_balls", 30)),
        n_voxels=int(snapshot.get("n_voxels", 30)),
        n_cyl=int(snapshot.get("n_cyl", 30)),
        n_cyl_h=int(snapshot.get("n_cyl_h", 2)),
        n_ground_voxels=int(snapshot.get("n_ground_voxels", 10)),
        is_scale=bool(snapshot.get("is_scale", True)),
    )

    env.balls = as_1_batch_tensor(snapshot["balls"], device)
    env.voxels = as_1_batch_tensor(snapshot["voxels"], device)
    env.cyl = as_1_batch_tensor(snapshot["cyl"], device)
    env.cyl_h = as_1_batch_tensor(snapshot["cyl_h"], device)
    env.R_cam = as_1_batch_tensor(snapshot["R_cam"], device)
    env._fov_x_half_tan = float(snapshot.get("_fov_x_half_tan", snapshot.get("fov_x_half_tan", env.fov_x_half_tan)))

    env.p = as_1_batch_tensor(snapshot["p"], device)
    env.p_target = as_1_batch_tensor(snapshot["p_target"], device)
    env.v = as_1_batch_tensor(snapshot.get("v", [0.0, 0.0, 0.0]), device)
    env.v_wind = as_1_batch_tensor(snapshot.get("v_wind", [0.0, 0.0, 0.0]), device)
    env.act = as_1_batch_tensor(snapshot.get("act", [0.0, 0.0, 0.0]), device)
    env.a = as_1_batch_tensor(snapshot.get("a", [0.0, 0.0, 0.0]), device)
    env.dg = as_1_batch_tensor(snapshot.get("dg", [0.0, 0.0, 0.0]), device)
    env.R = as_1_batch_tensor(snapshot["R"], device)
    env.R_old = as_1_batch_tensor(snapshot.get("R_old", snapshot["R"]), device)
    env.p_old = as_1_batch_tensor(snapshot.get("p_old", snapshot["p"]), device)

    env.max_speed = as_scalar_tensor(snapshot.get("max_speed", [[4.0]]), device)
    env.margin = as_scalar_tensor(snapshot.get("margin", [0.25]), device, shape=(1,))
    env.pitch_ctl_delay = as_scalar_tensor(snapshot.get("pitch_ctl_delay", [[12.0]]), device)
    env.yaw_ctl_delay = as_scalar_tensor(snapshot.get("yaw_ctl_delay", [[6.0]]), device)
    env.thr_est_error = as_scalar_tensor(snapshot.get("thr_est_error", [1.0]), device, shape=(1,))
    env.drag_2 = as_1_batch_tensor(snapshot.get("drag_2", [0.0, 0.3]), device)
    env.z_drag_coef = as_scalar_tensor(snapshot.get("z_drag_coef", [[1.0]]), device)
    env.drone_radius = float(snapshot.get("drone_radius", 0.15))
    env.n_drones_per_group = int(snapshot.get("n_drones_per_group", 1))

    mission = snapshot.get("mission", {})
    start = config.get("start", mission.get("start") if isinstance(mission, dict) else None)
    target = config.get("target", mission.get("target") if isinstance(mission, dict) else None)

    if start is not None:
        env.p = torch.as_tensor(start, device=device, dtype=torch.float32).reshape(1, 3)
        env.p_old = env.p.clone()
    if target is not None:
        env.p_target = torch.as_tensor(target, device=device, dtype=torch.float32).reshape(1, 3)

    import torch.nn.functional as F
    import quadsim_cuda

    initial_v_pred = F.normalize(env.p_target - env.p, dim=-1)
    env.R = quadsim_cuda.update_state_vec(env.R, env.act, initial_v_pred, torch.zeros_like(env.yaw_ctl_delay), 5)
    env.R_old = env.R.clone()
    return env


def load_model(checkpoint_path: Path, no_odom: bool, device: torch.device, config: Dict[str, Any] | None = None):
    import torch
    from model import create_model
    from se3diff_config.schema import ModelConfig

    config = config or {}
    model_name = str(config.get("model_name", config.get("model_type", "pm_model")))
    model_config = ModelConfig(
        name=model_name,
        model_type=str(config.get("model_type", model_name)),
        model_class=str(config.get("model_class", "Model")),
        dim_obs=int(config.get("dim_obs", 7 if no_odom else 10)),
        dim_action=int(config.get("dim_action", 6)),
        depth_pool_kernel=int(config.get("depth_pool_kernel", depth_pool_kernel_for_model())),
        hidden_dim=int(config.get("hidden_dim", 192)),
        action_mode=str(config.get("action_mode", "accel_velocity")),
    )
    model = create_model(model_config).to(device).eval()
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_checkpoint_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("missing_keys:", missing)
    if unexpected:
        print("unexpected_keys:", unexpected)
    return model
