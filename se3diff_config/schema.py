from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EnvConfig:
    width: int = 64
    height: int = 48
    grad_decay: float = 0.4
    fov_x_half_tan: float = 0.53
    single: bool = False
    gate: bool = False
    ground_voxels: bool = False
    ceiling: bool = False
    ceiling_height: float = 3.0
    scaffold: bool = False
    speed_mtp: float = 1.0
    random_rotation: bool = False
    cam_angle: int = 10
    gap: bool = False
    gap_prob: float = 0.0
    n_balls: int = 30
    n_voxels: int = 30
    n_cyl: int = 30
    n_cyl_h: int = 2
    n_ground_voxels: int = 10
    max_speed: float | None = None
    margin: float | None = None
    is_scale: bool = True


@dataclass
class TrainConfig:
    resume: str | None = None
    save_dir: str = "."
    num_envs: int | None = None
    batch_size: int = 64
    num_iters: int = 50000
    lr: float = 1e-3
    timesteps: int = 150
    yaw_drift: bool = False
    use_future_collision_samples: bool = True


@dataclass
class LossConfig:
    coef_v: float = 1.0
    coef_v_pred: float = 2.0
    coef_collide: float = 2.0
    coef_obj_avoidance: float = 1.5
    coef_d_acc: float = 0.01
    coef_d_jerk: float = 0.001


@dataclass
class InferenceConfig:
    device: str = "cuda"
    no_odom: bool = False
    yaw_target_correction: bool = False
    ctl_freq: float = 15.0
    max_steps: int = 450
    frame_id: str = "map"
    deterministic_visualization: bool = True


@dataclass
class ModelConfig:
    name: str = "pm_model"
    model_type: str = "pm_model"
    model_class: str = "Model"
    dim_obs: int | None = None
    dim_action: int = 6
    depth_pool_kernel: int = 4
    hidden_dim: int = 192
    action_mode: str = "accel_velocity"


@dataclass
class SensorConfig:
    name: str = "depth_odom"
    depth_pool_kernel: int = 4
    use_depth: bool = True
    use_odom: bool = True
    use_flow: bool = False
    use_mid360: bool = False
    mid360_points_per_scan: int = 20000
    mid360_vertical_channels: int = 64
    mid360_min_range: float = 0.1
    mid360_max_range: float = 70.0
    mid360_vertical_min_deg: float = -7.0
    mid360_vertical_max_deg: float = 52.0
    mid360_theta_min_deg: float = -90.0
    mid360_theta_max_deg: float = 90.0
    mid360_phi_min_deg: float = 38.0
    mid360_phi_max_deg: float = 97.0
    mid360_theta_resolution_deg: float = 1.0
    mid360_phi_resolution_deg: float = 1.0
    mid360_encoding: str = "pseudo_image"


@dataclass
class PathsConfig:
    scene_path: str | None = None
    rollout_path: str | None = None
    checkpoint_path: str | None = None


@dataclass
class PlaybackConfig:
    playback_speed: float = 1.0
    publish_prefix: str = "/diffphys"
    body_axis_length: float = 0.6
    body_axis_radius: float = 0.03
    loop: bool = False
    exit_on_finish: bool = True
    frame_id: str = "map"
    rate_hz: float | None = None
    ctl_freq: float | None = None
    ctl_dt: float | None = None


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    scene: Dict[str, Any] = field(default_factory=dict)
    rollout: Dict[str, Any] = field(default_factory=dict)
