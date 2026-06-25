from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class GapWallGenerator(ObstacleGenerator):
    prob: float = 0.0
    name: str = "gap_wall"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import torch
        import torch.nn.functional as F

        prob = float(self.prob)
        if prob <= 0:
            return scene

        env = ctx.env
        B = ctx.batch_size
        device = ctx.device
        dtype = scene.voxels.dtype
        active = torch.rand(B, device=device) < prob
        gap_voxels = torch.empty((B, 2, 6), device=device, dtype=dtype)
        gap_voxels[:] = torch.tensor([-1000., 0., 0., 0.1, 0.1, 0.1], device=device, dtype=dtype)

        if not active.any():
            scene.voxels = torch.cat([scene.voxels, gap_voxels], 1)
            return scene

        delta = env.p_target - env.p
        path = F.normalize(delta, 2, -1)
        midpoint = env.p + delta * (0.48 + 0.08 * torch.rand((B, 1), device=device, dtype=dtype))
        midpoint = midpoint + path * ((torch.rand((B, 1), device=device, dtype=dtype) - 0.5) * 1.0)
        x_major = delta[:, 0].abs() > delta[:, 1].abs()

        wall_half_width = 4.5
        wall_z_center = 1.5
        wall_z_half = 2.5
        thickness = 0.35
        gap_half_width = 0.25 + 0.25 * torch.rand((B,), device=device, dtype=dtype)
        gap_lateral = (torch.rand((B,), device=device, dtype=dtype) - 0.5) * 2.6
        gap_lateral = torch.minimum(
            torch.maximum(gap_lateral, -wall_half_width + gap_half_width + 0.1),
            wall_half_width - gap_half_width - 0.1,
        )

        pieces = [
            (-wall_half_width, gap_lateral - gap_half_width),
            (gap_lateral + gap_half_width, wall_half_width),
        ]

        for piece_i, (lat_min, lat_max) in enumerate(pieces):
            if not torch.is_tensor(lat_min):
                lat_min = torch.full((B,), lat_min, device=device, dtype=dtype)
            if not torch.is_tensor(lat_max):
                lat_max = torch.full((B,), lat_max, device=device, dtype=dtype)
            lat_center = (lat_min + lat_max) * 0.5
            lat_half = (lat_max - lat_min).mul(0.5).clamp_min(0.05)

            center = gap_voxels[:, piece_i, :3]
            radius = gap_voxels[:, piece_i, 3:]
            center[active, 2] = wall_z_center
            radius[active, 2] = wall_z_half

            y_wall = ~x_major
            y_mask = active & y_wall
            if y_mask.any():
                center[y_mask, 0] = midpoint[y_mask, 0] + lat_center[y_mask]
                center[y_mask, 1] = midpoint[y_mask, 1]
                radius[y_mask, 0] = lat_half[y_mask]
                radius[y_mask, 1] = thickness

            x_mask = active & x_major
            if x_mask.any():
                center[x_mask, 0] = midpoint[x_mask, 0]
                center[x_mask, 1] = midpoint[x_mask, 1] + lat_center[x_mask]
                radius[x_mask, 0] = thickness
                radius[x_mask, 1] = lat_half[x_mask]

        scene.voxels = torch.cat([scene.voxels, gap_voxels], 1)
        return scene
