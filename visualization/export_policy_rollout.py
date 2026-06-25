#!/usr/bin/env python3
"""Export a policy rollout to NPZ for offline visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_DIR, VIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visualization_common import load_yaml, resolve_path, save_rollout_npz
from se3diff_config.io import flatten_user_config

if TYPE_CHECKING:
    import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to offline rollout YAML config")
    parser.add_argument("--output", default=None, help="Optional output .npz path overriding rollout_path")
    return parser.parse_args()


def require_path(config: Dict[str, Any], key: str, config_path: Path) -> Path:
    paths = config.get("paths", {})
    value = config.get(key)
    if value is None and isinstance(paths, dict):
        value = paths.get(key)
    if value is None:
        raise KeyError(f"Missing required YAML key '{key}' in {config_path}")
    path = resolve_path(value, config_path.parent)
    if not path.exists():
        raise FileNotFoundError(f"Configured {key} does not exist: {path}")
    return path


def find_latest_final_checkpoint(checkpoints_root: str | Path | None = None) -> Path:
    root = Path(checkpoints_root).expanduser() if checkpoints_root is not None else REPO_ROOT / "checkpoints"
    if not root.is_absolute():
        root = REPO_ROOT / root
    candidates = [path for path in root.glob("*/checkpoint_final.pth") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_final.pth found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).resolve()


def default_checkpoints_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / "checkpoints"
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / "checkpoints"


def resolve_checkpoint_path(config: Dict[str, Any], config_path: Path) -> Path:
    paths = config.get("paths", {})
    value = config.get("checkpoint_path")
    if value is None and isinstance(paths, dict):
        value = paths.get("checkpoint_path")
    if value is None:
        checkpoint = find_latest_final_checkpoint(default_checkpoints_root(config_path))
        print(f"Using latest final checkpoint: {checkpoint}")
        return checkpoint
    path = resolve_path(value, config_path.parent)
    if not path.exists():
        raise FileNotFoundError(f"Configured checkpoint_path does not exist: {path}")
    return path


def get_ctl_dt(config: Dict[str, Any]) -> float:
    if "ctl_freq" in config:
        ctl_freq = float(config["ctl_freq"])
        if ctl_freq <= 0:
            raise ValueError(f"ctl_freq must be positive, got {ctl_freq}")
        ctl_dt = 1.0 / ctl_freq
    elif "ctl_dt" in config:
        ctl_dt = float(config["ctl_dt"])
    else:
        ctl_dt = 1.0 / float(config.get("rate_hz", 15.0))
    if ctl_dt <= 0:
        raise ValueError(f"ctl_dt must be positive, got {ctl_dt}")
    return ctl_dt


def get_rate_hz(config: Dict[str, Any]) -> float:
    if "ctl_freq" in config:
        rate_hz = float(config["ctl_freq"])
    elif "rate_hz" in config:
        rate_hz = float(config["rate_hz"])
    else:
        rate_hz = 1.0 / get_ctl_dt(config)
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    return rate_hz


def mid360_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sensor_name = str(config.get("sensor_name", config.get("name", "depth_odom")))
    record_pseudo_default = sensor_name == "mid360"
    return {
        "record_mid360": bool(config.get("record_mid360", False)),
        "record_mid360_pseudo_image": bool(config.get("record_mid360_pseudo_image", record_pseudo_default)),
        "mid360_points_per_scan": int(config.get("mid360_points_per_scan", 20000)),
        "mid360_vertical_channels": int(config.get("mid360_vertical_channels", 64)),
        "mid360_min_range": float(config.get("mid360_min_range", 0.1)),
        "mid360_max_range": float(config.get("mid360_max_range", 70.0)),
        "mid360_vertical_min_deg": float(config.get("mid360_vertical_min_deg", -7.0)),
        "mid360_vertical_max_deg": float(config.get("mid360_vertical_max_deg", 52.0)),
    }


def render_mid360_scan(env, config: Dict[str, Any], torch):
    import quadsim_cuda

    scan_config = mid360_config(config)
    points = torch.empty((1, scan_config["mid360_points_per_scan"], 3), device=env.device)
    ranges = torch.empty((1, scan_config["mid360_points_per_scan"]), device=env.device)
    quadsim_cuda.render_mid360(
        points,
        ranges,
        env.balls,
        env.cyl,
        env.cyl_h,
        env.voxels,
        env.R,
        env.p,
        int(env.n_drones_per_group),
        scan_config["mid360_vertical_channels"],
        scan_config["mid360_min_range"],
        scan_config["mid360_max_range"],
        scan_config["mid360_vertical_min_deg"],
        scan_config["mid360_vertical_max_deg"],
        bool(getattr(env, "ceiling", False)),
        float(getattr(env, "ceiling_height", 3.0)),
    )
    return points[0], ranges[0]


def collect_rollout(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    import numpy as np
    import torch

    from visualization_common import min_pool_depth, tensor_to_numpy
    from policy_rollout_common import (
        create_env_from_snapshot,
        depth_pool_kernel_for_model,
        load_model,
        merge_training_inference_config,
    )
    from rollout import PolicyRunner, PolicyRunnerConfig
    from sensors import create_observation_builder
    from se3diff_config.schema import SensorConfig

    scene_path = require_path(config, "scene_path", config_path)
    checkpoint_path = resolve_checkpoint_path(config, config_path)
    config = merge_training_inference_config(config, checkpoint_path)
    device = torch.device(config.get("device", "cuda"))
    no_odom = bool(config.get("no_odom", False))
    sensor_name = str(config.get("sensor_name", config.get("name", "depth_odom")))
    if sensor_name not in {"depth", "depth_odom", "mid360"}:
        sensor_name = "depth_odom"
    ctl_dt = get_ctl_dt(config)
    rate_hz = get_rate_hz(config)
    max_steps = int(config.get("max_steps", 450))
    scan_config = mid360_config(config)

    snapshot = torch.load(scene_path, map_location="cpu")
    mission = snapshot.get("mission", {})
    target_reached_radius = float(
        config.get(
            "target_reached_radius",
            mission.get("target_reached_radius", 0.5) if isinstance(mission, dict) else 0.5,
        )
    )
    env = create_env_from_snapshot(snapshot, config, device)
    if bool(config.get("deterministic_visualization", True)):
        from rollout.randomization import disable_visualization_randomization

        disable_visualization_randomization(env)
    model = load_model(checkpoint_path, no_odom, device, config)
    depth_pool_kernel = int(config.get("depth_pool_kernel", depth_pool_kernel_for_model()))
    observation_builder = None
    if sensor_name == "mid360":
        observation_builder = create_observation_builder(
            SensorConfig(
                name="mid360",
                use_odom=not no_odom,
                depth_pool_kernel=depth_pool_kernel,
                mid360_points_per_scan=int(config.get("mid360_points_per_scan", 20000)),
                mid360_vertical_channels=int(config.get("mid360_vertical_channels", 64)),
                mid360_min_range=float(config.get("mid360_min_range", 0.1)),
                mid360_max_range=float(config.get("mid360_max_range", 70.0)),
                mid360_vertical_min_deg=float(config.get("mid360_vertical_min_deg", -7.0)),
                mid360_vertical_max_deg=float(config.get("mid360_vertical_max_deg", 52.0)),
                mid360_theta_min_deg=float(config.get("mid360_theta_min_deg", -180.0)),
                mid360_theta_max_deg=float(config.get("mid360_theta_max_deg", 180.0)),
                mid360_phi_min_deg=float(config.get("mid360_phi_min_deg", 38.0)),
                mid360_phi_max_deg=float(config.get("mid360_phi_max_deg", 97.0)),
                mid360_theta_resolution_deg=float(config.get("mid360_theta_resolution_deg", 2.0)),
                mid360_phi_resolution_deg=float(config.get("mid360_phi_resolution_deg", 1.0)),
            )
        )
    runner = PolicyRunner(
        env,
        model,
        PolicyRunnerConfig(
            ctl_dt=ctl_dt,
            max_steps=max_steps,
            no_odom=no_odom,
            yaw_target_correction=bool(config.get("yaw_target_correction", False)),
            depth_pool_kernel=depth_pool_kernel,
            deterministic_visualization=bool(config.get("deterministic_visualization", True)),
            sensor_name=sensor_name,
            action_mode=str(config.get("action_mode", "accel_velocity")),
        ),
        observation_builder=observation_builder,
    )
    runner.reset_model()

    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    raw_depths: List[np.ndarray] = []
    pooled_depths: List[np.ndarray] = []
    pooled_raw_depths: List[np.ndarray] = []
    mid360_points: List[np.ndarray] = []
    mid360_ranges: List[np.ndarray] = []
    mid360_pseudo_images: List[np.ndarray] = []
    clearance_distance_history: List[torch.Tensor] = []

    with torch.no_grad():
        for step_idx in range(max_steps):
            step = runner.step(step_idx)
            vec_to_pt = env.find_vec_to_nearest_pt(
                use_future_samples=bool(config.get("use_future_collision_samples", True))
            )
            clearance_distance_history.append(torch.norm(vec_to_pt, 2, -1) - env.margin)

            positions.append(tensor_to_numpy(env.p[0]).astype(np.float32).copy())
            velocities.append(tensor_to_numpy(env.v[0]).astype(np.float32).copy())
            actions.append(tensor_to_numpy(step["action"][0]).astype(np.float32).copy())
            rotations.append(tensor_to_numpy(env.R[0]).astype(np.float32).copy())
            raw_depth_np = tensor_to_numpy(step["depth"][0]).astype(np.float32).copy()
            raw_depths.append(raw_depth_np)
            pooled_depths.append(tensor_to_numpy(step["pooled_depth"][0, 0]).astype(np.float32).copy())
            pooled_raw_depths.append(min_pool_depth(raw_depth_np, depth_pool_kernel))
            if scan_config["record_mid360_pseudo_image"] and "mid360_pseudo_image" in step:
                pseudo_image = tensor_to_numpy(step["mid360_pseudo_image"][0, 0]).astype(np.float32).copy()
                mid360_pseudo_images.append(pseudo_image)
            if scan_config["record_mid360"]:
                if "mid360_points" in step:
                    scan_points = step.get("mid360_world_points", step["mid360_points"])[0]
                    scan_ranges = step["mid360_ranges"][0]
                else:
                    scan_points, scan_ranges = render_mid360_scan(env, config, torch)
                mid360_points.append(tensor_to_numpy(scan_points).astype(np.float32).copy())
                mid360_ranges.append(tensor_to_numpy(scan_ranges).astype(np.float32).copy())
            distance_to_target = torch.norm(env.p - env.p_target, 2, -1).item()
            if distance_to_target <= target_reached_radius:
                print(f"Reached target at step {step_idx + 1}: distance={distance_to_target:.3f} m")
                break

    clearance_distance = torch.stack(clearance_distance_history)
    success = torch.all(clearance_distance.flatten(0, 1) > 0, 0)
    success_rate = success.float().mean()
    print(f"maxspeed: {env.max_speed[0].item():.3f}, margin: {env.margin[0].item():.3f}")
    # print(f"start: {tensor_to_numpy(env.p[0]).astype(np.float32)}, target: {tensor_to_numpy(env.p_target[0]).astype(np.float32)}")
    print(f"Rollout success_rate={float(success_rate.item()):.3f} ({int(success.sum().item())}/{success.numel()})")

    rollout = {
        "positions": np.stack(positions),
        "velocities": np.stack(velocities),
        "actions": np.stack(actions),
        "rotations": np.stack(rotations),
        "raw_depth": np.stack(raw_depths),
        "pooled_depth": np.stack(pooled_depths),
        "pooled_raw_depth": np.stack(pooled_raw_depths),
        "target": tensor_to_numpy(env.p_target[0]).astype(np.float32),
        "balls": tensor_to_numpy(snapshot["balls"]).astype(np.float32),
        "voxels": tensor_to_numpy(snapshot["voxels"]).astype(np.float32),
        "cyl": tensor_to_numpy(snapshot["cyl"]).astype(np.float32),
        "cyl_h": tensor_to_numpy(snapshot["cyl_h"]).astype(np.float32),
        "rate_hz": np.array(rate_hz, dtype=np.float32),
        "frame_id": np.array(str(config.get("frame_id", "map"))),
        "clearance_distance": tensor_to_numpy(clearance_distance).astype(np.float32),
        "success": tensor_to_numpy(success).astype(np.bool_),
        "success_rate": np.array(float(success_rate.item()), dtype=np.float32),
    }
    if mid360_pseudo_images:
        pseudo_stack = np.stack(mid360_pseudo_images)
        rollout.update(
            {
                "mid360_pseudo_image": pseudo_stack,
                "mid360_pseudo_image_shape": np.asarray(pseudo_stack.shape[1:], dtype=np.int32),
                "mid360_pseudo_image_min_range": np.array(scan_config["mid360_min_range"], dtype=np.float32),
                "mid360_pseudo_image_max_range": np.array(scan_config["mid360_max_range"], dtype=np.float32),
            }
        )
    if scan_config["record_mid360"]:
        rollout.update(
            {
                "mid360_points": np.stack(mid360_points),
                "mid360_ranges": np.stack(mid360_ranges),
                "mid360_min_range": np.array(scan_config["mid360_min_range"], dtype=np.float32),
                "mid360_max_range": np.array(scan_config["mid360_max_range"], dtype=np.float32),
                "mid360_vertical_channels": np.array(scan_config["mid360_vertical_channels"], dtype=np.int32),
            }
        )
    return rollout


def export_policy_rollout(
    config: Dict[str, Any],
    config_path: Path,
    *,
    output_override: str | Path | None = None,
) -> Path:
    flat_config = flatten_user_config(config)
    rollout_path = flat_config.get("rollout_path")
    if rollout_path is None:
        raise KeyError(f"Missing required YAML key 'rollout_path' in {config_path}")
    output = Path(output_override).expanduser() if output_override else resolve_path(rollout_path, config_path.parent)
    rollout = collect_rollout(config, config_path)
    save_rollout_npz(output, rollout)
    print(f"Exported rollout to {output.resolve()}")
    return output


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    export_policy_rollout(config, config_path, output_override=args.output)


if __name__ == "__main__":
    main()
