#!/usr/bin/env python3
"""Export a single DiffPhysDrone Env scene snapshot."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, TYPE_CHECKING


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_DIR, VIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visualization_common import apply_clearance_arrays, load_yaml, resolve_path
from se3diff_config.env_factory import create_env
from se3diff_config.schema import EnvConfig

if TYPE_CHECKING:
    import torch
    from env import Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML scene generation config")
    parser.add_argument("--output", default=None, help="Path to the exported .pt scene snapshot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--grad_decay", type=float, default=None)
    parser.add_argument("--speed_mtp", type=float, default=None)
    parser.add_argument("--max_speed", "--max-speed", dest="max_speed", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--cam_angle", type=int, default=None)
    parser.add_argument("--fov_x_half_tan", type=float, default=None)
    parser.add_argument("--gap", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gap_prob", type=float, default=None)
    parser.add_argument("--n_balls", type=int, default=None)
    parser.add_argument("--n_voxels", type=int, default=None)
    parser.add_argument("--n_cyl", type=int, default=None)
    parser.add_argument("--n_cyl_h", type=int, default=None)
    parser.add_argument("--n_ground_voxels", type=int, default=None)
    parser.add_argument("--ceiling_height", "--ceiling-height", dest="ceiling_height", type=float, default=None)
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--single", default=None, action=bool_action)
    parser.add_argument("--gate", default=None, action=bool_action)
    parser.add_argument("--ground_voxels", "--ground-voxels", dest="ground_voxels", default=None, action=bool_action)
    parser.add_argument("--ceiling", default=None, action=bool_action)
    parser.add_argument("--scaffold", default=None, action=bool_action)
    parser.add_argument("--random_rotation", "--random-rotation", dest="random_rotation", default=None, action=bool_action)
    return parser.parse_args()


def choose(cli_value: Any, config: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config.get(key, default)


def tensor_from_config(value: Any, *, device: "torch.device", name: str, expected_len: int) -> "torch.Tensor":
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {tensor.numel()}")
    return tensor.reshape(expected_len)


def apply_obstacle_overrides(env: "Env", obstacles: Mapping[str, Any]) -> bool:
    changed = False
    specs = {
        "ball_w": 4,
        "ball_b": 4,
        "voxel_w": 6,
        "voxel_b": 6,
        "ground_voxel_w": 6,
        "ground_voxel_b": 6,
        "cyl_w": 3,
        "cyl_b": 3,
        "cyl_h_w": 3,
        "cyl_h_b": 3,
        "gate_w": 4,
        "gate_b": 4,
    }
    for name, expected_len in specs.items():
        if name not in obstacles:
            continue
        setattr(env, name, tensor_from_config(obstacles[name], device=env.device, name=name, expected_len=expected_len))
        changed = True
    return changed


def get_mission_points(mission: Mapping[str, Any]) -> torch.Tensor | None:
    import torch

    points = []
    for key in ("start", "target"):
        if key in mission:
            points.append(mission[key])
    if not points:
        return None
    return torch.as_tensor(points, dtype=torch.float32)


def apply_clearance(env: "Env", mission: Mapping[str, Any], clearance: Mapping[str, Any]) -> None:
    import torch

    if not bool(clearance.get("enabled", False)):
        return
    points = get_mission_points(mission)
    if points is None:
        return
    radius = float(clearance.get("radius", 2.5))
    balls = env.balls.detach().cpu().numpy().copy()
    voxels = env.voxels.detach().cpu().numpy().copy()
    cyl = env.cyl.detach().cpu().numpy().copy()
    cyl_h = env.cyl_h.detach().cpu().numpy().copy()
    apply_clearance_arrays(
        balls,
        voxels,
        cyl,
        cyl_h,
        points=points.numpy(),
        radius=radius,
    )
    env.balls = torch.as_tensor(balls, device=env.device, dtype=env.balls.dtype)
    env.voxels = torch.as_tensor(voxels, device=env.device, dtype=env.voxels.dtype)
    env.cyl = torch.as_tensor(cyl, device=env.device, dtype=env.cyl.dtype)
    env.cyl_h = torch.as_tensor(cyl_h, device=env.device, dtype=env.cyl_h.dtype)


def cpu_clone(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def empty_cli_args() -> argparse.Namespace:
    keys = (
        "seed",
        "device",
        "grad_decay",
        "speed_mtp",
        "max_speed",
        "margin",
        "cam_angle",
        "fov_x_half_tan",
        "gap",
        "gap_prob",
        "n_balls",
        "n_voxels",
        "n_cyl",
        "n_cyl_h",
        "n_ground_voxels",
        "ceiling_height",
        "single",
        "gate",
        "ground_voxels",
        "ceiling",
        "scaffold",
        "random_rotation",
    )
    return argparse.Namespace(**{key: None for key in keys})


def export_scene_snapshot(
    config: Dict[str, Any],
    config_path: Path | None = None,
    *,
    output_override: str | Path | None = None,
    cli_args: argparse.Namespace | None = None,
) -> Path:
    import torch
    from env import Env

    args = cli_args or empty_cli_args()
    env_config = config.get("env", {})
    obstacle_config = config.get("obstacles", {})
    mission_config = config.get("mission", {})
    clearance_config = config.get("clearance", {})
    if not isinstance(env_config, dict):
        raise ValueError("scene config key 'env' must be a mapping")
    if not isinstance(obstacle_config, dict):
        raise ValueError("scene config key 'obstacles' must be a mapping")
    if not isinstance(mission_config, dict):
        raise ValueError("scene config key 'mission' must be a mapping")
    if not isinstance(clearance_config, dict):
        raise ValueError("scene config key 'clearance' must be a mapping")

    seed = choose(args.seed, config, "seed", None)
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    device = torch.device(choose(args.device, config, "device", "cuda"))
    shared_env_config = EnvConfig(
        width=int(env_config.get("width", 64)),
        height=int(env_config.get("height", 48)),
        is_scale=bool(env_config.get("is_scale", True)),
        grad_decay=float(choose(args.grad_decay, env_config, "grad_decay", 0.4)),
        fov_x_half_tan=float(choose(args.fov_x_half_tan, env_config, "fov_x_half_tan", 0.53)),
        single=bool(choose(args.single, env_config, "single", False)),
        gate=bool(choose(args.gate, env_config, "gate", False)),
        ground_voxels=bool(choose(args.ground_voxels, env_config, "ground_voxels", False)),
        ceiling=bool(choose(args.ceiling, env_config, "ceiling", False)),
        ceiling_height=float(choose(args.ceiling_height, env_config, "ceiling_height", 3.0)),
        scaffold=bool(choose(args.scaffold, env_config, "scaffold", False)),
        speed_mtp=float(choose(args.speed_mtp, env_config, "speed_mtp", 1.0)),
        max_speed=choose(args.max_speed, env_config, "max_speed", None),
        margin=choose(args.margin, env_config, "margin", None),
        random_rotation=bool(choose(args.random_rotation, env_config, "random_rotation", False)),
        cam_angle=int(choose(args.cam_angle, env_config, "cam_angle", 10)),
        gap=bool(choose(args.gap, env_config, "gap", False)),
        gap_prob=float(choose(args.gap_prob, env_config, "gap_prob", 0.0)),
        n_balls=int(choose(args.n_balls, env_config, "n_balls", 30)),
        n_voxels=int(choose(args.n_voxels, env_config, "n_voxels", 30)),
        n_cyl=int(choose(args.n_cyl, env_config, "n_cyl", 30)),
        n_cyl_h=int(choose(args.n_cyl_h, env_config, "n_cyl_h", 2)),
        n_ground_voxels=int(choose(args.n_ground_voxels, env_config, "n_ground_voxels", 10)),
    )
    env = create_env(
        Env,
        shared_env_config,
        batch_size=1,
        device=device,
        start=mission_config.get("start"),
        target=mission_config.get("target"),
    )
    if apply_obstacle_overrides(env, obstacle_config):
        env.reset()
    apply_clearance(env, mission_config, clearance_config)

    snapshot = {
        "version": 1,
        "width": env.width,
        "height": env.height,
        "grad_decay": env.grad_decay,
        "fov_x_half_tan": env.fov_x_half_tan,
        "_fov_x_half_tan": env._fov_x_half_tan,
        "single": env.single,
        "gate": env.gate,
        "ground_voxels": env.ground_voxels,
        "ceiling": env.ceiling,
        "ceiling_height": env.ceiling_height,
        "scaffold": env.scaffold,
        "speed_mtp": env.speed_mtp,
        "random_rotation": env.random_rotation,
        "cam_angle": env.cam_angle,
        "gap": env.gap,
        "gap_prob": env.gap_prob,
        "n_balls": env.n_balls,
        "n_voxels": env.n_voxels,
        "n_cyl": env.n_cyl,
        "n_cyl_h": env.n_cyl_h,
        "n_ground_voxels": env.n_ground_voxels,
        "balls": cpu_clone(env.balls[0]),
        "voxels": cpu_clone(env.voxels[0]),
        "cyl": cpu_clone(env.cyl[0]),
        "cyl_h": cpu_clone(env.cyl_h[0]),
        "R_cam": cpu_clone(env.R_cam[0]),
        "p": cpu_clone(env.p[0]),
        "p_target": cpu_clone(env.p_target[0]),
        "v": cpu_clone(env.v[0]),
        "v_wind": cpu_clone(env.v_wind[0]),
        "act": cpu_clone(env.act[0]),
        "a": cpu_clone(env.a[0]),
        "dg": cpu_clone(env.dg[0]),
        "R": cpu_clone(env.R[0]),
        "R_old": cpu_clone(env.R_old[0]),
        "p_old": cpu_clone(env.p_old[0]),
        "max_speed": cpu_clone(env.max_speed[0]),
        "margin": cpu_clone(env.margin[0]),
        "pitch_ctl_delay": cpu_clone(env.pitch_ctl_delay[0]),
        "yaw_ctl_delay": cpu_clone(env.yaw_ctl_delay[0]),
        "thr_est_error": cpu_clone(env.thr_est_error[0]),
        "drag_2": cpu_clone(env.drag_2[0]),
        "z_drag_coef": cpu_clone(env.z_drag_coef[0]),
        "drone_radius": float(env.drone_radius),
        "n_drones_per_group": int(env.n_drones_per_group),
        "is_scale": bool(env.is_scale),
        "mission": mission_config,
        "clearance": clearance_config,
        "scene_config": config,
    }

    output_value = output_override if output_override is not None else config.get("output")
    if output_value is None:
        raise ValueError("Output path is required; pass --output or set output in the scene config")
    output = resolve_path(output_value, config_path.parent if config_path else REPO_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, output)
    print(f"Exported scene snapshot to {output.resolve()}")
    return output


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config: Dict[str, Any] = load_yaml(config_path) if config_path else {}
    export_scene_snapshot(config, config_path, output_override=args.output, cli_args=args)


if __name__ == "__main__":
    main()
