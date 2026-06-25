from __future__ import annotations

from typing import Any

from .schema import EnvConfig


def create_env(env_cls: type, env_config: EnvConfig, *, batch_size: int, device: Any, start=None, target=None):
    return env_cls(
        batch_size,
        int(env_config.width),
        int(env_config.height),
        float(env_config.grad_decay),
        device,
        fov_x_half_tan=float(env_config.fov_x_half_tan),
        single=bool(env_config.single),
        gate=bool(env_config.gate),
        ground_voxels=bool(env_config.ground_voxels),
        ceiling=bool(env_config.ceiling),
        ceiling_height=float(env_config.ceiling_height),
        scaffold=bool(env_config.scaffold),
        speed_mtp=float(env_config.speed_mtp),
        random_rotation=bool(env_config.random_rotation),
        cam_angle=int(env_config.cam_angle),
        start=start,
        target=target,
        max_speed=env_config.max_speed,
        margin=env_config.margin,
        gap=bool(env_config.gap),
        gap_prob=float(env_config.gap_prob),
        n_balls=int(env_config.n_balls),
        n_voxels=int(env_config.n_voxels),
        n_cyl=int(env_config.n_cyl),
        n_cyl_h=int(env_config.n_cyl_h),
        n_ground_voxels=int(env_config.n_ground_voxels),
        is_scale=bool(env_config.is_scale),
    )
