"""Shared helpers for offline visualization export scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_REQUIRED_KEYS = (
    "positions",
    "velocities",
    "actions",
    "rotations",
    "raw_depth",
    "pooled_depth",
    "pooled_raw_depth",
    "target",
    "balls",
    "voxels",
    "cyl",
    "cyl_h",
    "rate_hz",
    "frame_id",
)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = REPO_ROOT
    return Path(base_dir).expanduser() / p


def save_rollout_npz(path: str | Path, data: Mapping[str, Any]) -> None:
    missing = [key for key in ROLLOUT_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Rollout data missing required keys: {missing}")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)


def min_pool_depth(depth: np.ndarray, kernel_size: int = 4) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape {arr.shape}")
    height, width = arr.shape
    if height % kernel_size != 0 or width % kernel_size != 0:
        raise ValueError(f"Image shape {arr.shape} is not divisible by kernel_size={kernel_size}")
    return arr.reshape(height // kernel_size, kernel_size, width // kernel_size, kernel_size).min(axis=(1, 3))


def apply_clearance_arrays(
    balls: np.ndarray,
    voxels: np.ndarray,
    cyl: np.ndarray,
    cyl_h: np.ndarray,
    *,
    points: np.ndarray,
    radius: float,
    far_x: float = -1000.0,
) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    radius = float(radius)
    for point in pts:
        if balls.size:
            dist = np.linalg.norm(balls[..., :3] - point, axis=-1)
            balls[dist < radius + balls[..., 3], 0] = far_x

        if voxels.size:
            delta = np.maximum(np.abs(voxels[..., :3] - point) - voxels[..., 3:6], 0.0)
            dist = np.linalg.norm(delta, axis=-1)
            voxels[dist < radius, 0] = far_x

        if cyl.size:
            dist_xy = np.linalg.norm(cyl[..., :2] - point[:2], axis=-1)
            cyl[dist_xy < radius + cyl[..., 2], 0] = far_x

        if cyl_h.size:
            xz = np.stack([cyl_h[..., 0], cyl_h[..., 1]], axis=-1)
            dist_xz = np.linalg.norm(xz - point[[0, 2]], axis=-1)
            cyl_h[dist_xz < radius + cyl_h[..., 2], 0] = far_x


def normalize_checkpoint_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    for wrapper_key in ("state_dict", "model_state_dict", "model"):
        wrapped = state_dict.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            state_dict = wrapped
            break

    key_map = {
        "v_proj.weight": "observation_fc.weight",
        "v_proj.bias": "observation_fc.bias",
        "fc.weight": "action_fc.weight",
    }
    return {key_map.get(k, k): v for k, v in state_dict.items()}


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
