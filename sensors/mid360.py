from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import ObservationBatch, ObservationBuilder
from se3diff_config.schema import SensorConfig


class Mid360ObservationBuilder(ObservationBuilder):
    name = "mid360"
    model_input_keys = ("mid360_pseudo_image", "state")
    requires_depth = False

    def __init__(self, config: SensorConfig):
        self.config = config
        self.depth_pool_kernel = int(config.depth_pool_kernel)
        self.dim_state = 16 if bool(config.use_odom) else 13
        self.points_per_scan = int(config.mid360_points_per_scan)
        self.vertical_channels = int(config.mid360_vertical_channels)
        self.min_range = float(config.mid360_min_range)
        self.max_range = float(config.mid360_max_range)
        self.vertical_min_deg = float(config.mid360_vertical_min_deg)
        self.vertical_max_deg = float(config.mid360_vertical_max_deg)
        self.theta_min = math.radians(float(config.mid360_theta_min_deg))
        self.theta_max = math.radians(float(config.mid360_theta_max_deg))
        self.phi_min = math.radians(float(config.mid360_phi_min_deg))
        self.phi_max = math.radians(float(config.mid360_phi_max_deg))
        self.theta_resolution = math.radians(float(config.mid360_theta_resolution_deg))
        self.phi_resolution = math.radians(float(config.mid360_phi_resolution_deg))
        if self.theta_resolution <= 0 or self.phi_resolution <= 0:
            raise ValueError("MID360 pseudo-image angular resolutions must be positive")
        if self.theta_max <= self.theta_min or self.phi_max <= self.phi_min:
            raise ValueError("MID360 pseudo-image angular ranges must be increasing")
        self.n_theta = int(round((self.theta_max - self.theta_min) / self.theta_resolution))
        self.n_phi = int(round((self.phi_max - self.phi_min) / self.phi_resolution))
        if self.n_theta <= 0 or self.n_phi <= 0:
            raise ValueError("MID360 pseudo-image grid must have at least one bin per axis")

    def points_to_pseudo_image(self, points: Any, ranges: Any | None = None):
        if _is_torch_tensor(points):
            return self._points_to_pseudo_image_torch(points, ranges)
        return self._points_to_pseudo_image_numpy(points, ranges)

    def render_inputs(self, env, ctl_dt, *, include_debug_outputs: bool = False) -> dict[str, Any]:
        import torch
        import quadsim_cuda

        points = torch.empty((env.batch_size, self.points_per_scan, 3), device=env.device)
        ranges = torch.empty((env.batch_size, self.points_per_scan), device=env.device)
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
            self.vertical_channels,
            self.min_range,
            self.max_range,
            self.vertical_min_deg,
            self.vertical_max_deg,
            bool(getattr(env, "ceiling", False)),
            float(getattr(env, "ceiling_height", 3.0)),
        )
        with torch.no_grad():
            body_points = self.world_points_to_body(points, env.p, env.R)
            pseudo_image = torch.empty((env.batch_size, 1, self.n_phi, self.n_theta), device=env.device)
            pseudo_image = self._points_to_pseudo_image_cuda_nchw(body_points, ranges, pseudo_image)
        sensor_inputs = {
            "mid360_pseudo_image": pseudo_image,
        }
        if include_debug_outputs:
            sensor_inputs["mid360_points"] = body_points
            sensor_inputs["mid360_world_points"] = points
            sensor_inputs["mid360_ranges"] = ranges
        return sensor_inputs

    def build(self, *, sensor_inputs, state=None) -> dict[str, Any]:
        if state is None:
            raise ValueError("Mid360ObservationBuilder.build requires a precomputed state")
        return {"mid360_pseudo_image": sensor_inputs["mid360_pseudo_image"], "state": state}

    def build_batch(self, env, *, state=None) -> ObservationBatch:
        sensor_inputs = self.render_inputs(env, None, include_debug_outputs=True)
        if state is None:
            state = self.full_attitude_state(env)
        obs = self.build(sensor_inputs=sensor_inputs, state=state)
        return ObservationBatch(
            depth=obs["mid360_pseudo_image"],
            state=obs["state"],
            point_cloud=sensor_inputs["mid360_points"],
        )

    def full_attitude_state(self, env, target_v_local=None, local_v=None):
        import torch

        if target_v_local is None:
            target_v_local = torch.zeros((env.batch_size, 3), device=env.device, dtype=env.R.dtype)
        attitude = env.R.reshape(env.batch_size, 9)
        margin = env.margin[:, None]
        state = [target_v_local, attitude, margin]
        if bool(self.config.use_odom):
            if local_v is None:
                local_v = torch.zeros((env.batch_size, 3), device=env.device, dtype=env.R.dtype)
            state.insert(0, local_v)
        return torch.cat(state, -1)

    def world_points_to_body(self, points: Any, position: Any, rotation: Any):
        if _is_torch_tensor(points):
            return self._world_points_to_body_torch(points, position, rotation)
        pts = np.asarray(points, dtype=np.float32)
        pos = np.asarray(position, dtype=np.float32)
        rot = np.asarray(rotation, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[None, ...]
        if pos.ndim == 1:
            pos = pos[None, ...]
        if rot.ndim == 2:
            rot = rot[None, ...]
        return np.einsum("bij,bnj->bni", np.swapaxes(rot, 1, 2), pts - pos[:, None, :])

    def _world_points_to_body_torch(self, points: Any, position: Any, rotation: Any):
        import torch

        return torch.matmul(points - position[:, None, :], rotation)

    def _points_to_pseudo_image_numpy(self, points: Any, ranges: Any | None = None) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[None, ...]
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"points must have shape [N, 3] or [B, N, 3], got {pts.shape}")

        r = np.linalg.norm(pts, axis=-1).astype(np.float32)
        if ranges is not None:
            provided_ranges = np.asarray(ranges, dtype=np.float32)
            if provided_ranges.ndim == 1:
                provided_ranges = provided_ranges[None, ...]
            valid = provided_ranges > 0
            r = np.where(valid, provided_ranges, r)
        else:
            valid = r > 0
        valid = valid & (r >= self.min_range) & (r <= self.max_range)

        theta = np.arctan2(pts[..., 1], pts[..., 0])
        safe_r = np.maximum(r, 1e-6)
        phi = np.arccos(np.clip(pts[..., 2] / safe_r, -1.0, 1.0))
        theta_idx = np.floor((theta - self.theta_min) / self.theta_resolution).astype(np.int64)
        phi_idx = np.floor((phi - self.phi_min) / self.phi_resolution).astype(np.int64)
        valid = valid & (theta_idx >= 0) & (theta_idx < self.n_theta) & (phi_idx >= 0) & (phi_idx < self.n_phi)

        image = np.full((pts.shape[0], self.n_phi, self.n_theta, 1), self.max_range, dtype=np.float32)
        for batch_idx in range(pts.shape[0]):
            batch_valid = valid[batch_idx]
            if not np.any(batch_valid):
                continue
            np.minimum.at(
                image[batch_idx, :, :, 0],
                (phi_idx[batch_idx, batch_valid], theta_idx[batch_idx, batch_valid]),
                r[batch_idx, batch_valid],
            )
        return image

    def _points_to_pseudo_image_torch(self, points: Any, ranges: Any | None = None):
        import torch

        pts = points
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"points must have shape [N, 3] or [B, N, 3], got {tuple(pts.shape)}")

        r = torch.linalg.norm(pts, dim=-1)
        if ranges is not None:
            provided_ranges = ranges
            if provided_ranges.ndim == 1:
                provided_ranges = provided_ranges.unsqueeze(0)
            valid = provided_ranges > 0
            r = torch.where(valid, provided_ranges.to(device=pts.device, dtype=pts.dtype), r)
        else:
            valid = r > 0
        valid = valid & (r >= self.min_range) & (r <= self.max_range)

        theta = torch.atan2(pts[..., 1], pts[..., 0])
        safe_r = torch.clamp(r, min=1e-6)
        phi = torch.acos(torch.clamp(pts[..., 2] / safe_r, -1.0, 1.0))
        theta_idx = torch.floor((theta - self.theta_min) / self.theta_resolution).to(torch.long)
        phi_idx = torch.floor((phi - self.phi_min) / self.phi_resolution).to(torch.long)
        valid = valid & (theta_idx >= 0) & (theta_idx < self.n_theta) & (phi_idx >= 0) & (phi_idx < self.n_phi)

        image = torch.full(
            (pts.shape[0], self.n_phi, self.n_theta, 1),
            self.max_range,
            device=pts.device,
            dtype=pts.dtype,
        )
        flat = image[..., 0].reshape(pts.shape[0], self.n_phi * self.n_theta)
        flat_idx = phi_idx * self.n_theta + theta_idx
        for batch_idx in range(pts.shape[0]):
            batch_valid = valid[batch_idx]
            if bool(batch_valid.any()):
                flat[batch_idx].scatter_reduce_(
                    0,
                    flat_idx[batch_idx, batch_valid],
                    r[batch_idx, batch_valid],
                    reduce="amin",
                    include_self=True,
                )
        return image

    def _points_to_pseudo_image_cuda_nchw(self, points: Any, ranges: Any, output: Any | None = None):
        import torch
        import quadsim_cuda

        pts = points
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"points must have shape [N, 3] or [B, N, 3], got {tuple(pts.shape)}")
        if ranges is None:
            ranges = torch.zeros(pts.shape[:2], device=pts.device, dtype=pts.dtype)
        elif ranges.ndim == 1:
            ranges = ranges.unsqueeze(0)
        if output is None:
            output = torch.empty((pts.shape[0], 1, self.n_phi, self.n_theta), device=pts.device, dtype=pts.dtype)
        output.fill_(self.max_range)
        quadsim_cuda.points_to_pseudo_image_mid360(
            output,
            pts.contiguous(),
            ranges.to(device=pts.device, dtype=pts.dtype).contiguous(),
            self.min_range,
            self.max_range,
            self.theta_min,
            self.phi_min,
            self.theta_resolution,
            self.phi_resolution,
        )
        return output


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device")
