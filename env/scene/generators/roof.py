from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class RoofGenerator(ObstacleGenerator):
    name: str = "roof"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import torch

        env = ctx.env
        B = ctx.batch_size
        scale = ctx.scale
        balls = scene.balls
        voxels = scene.voxels
        cyl = scene.cyl
        cyl_h = scene.cyl_h

        roof = torch.rand((B,), device=ctx.device) < 0.5
        ball_roof_count = min(15, balls.size(1), cyl.size(1))
        voxel_roof_count = min(15, voxels.size(1), max(0, cyl.size(1) - ball_roof_count))
        if ball_roof_count > 0:
            balls[~roof, :ball_roof_count, :2] = cyl[~roof, :ball_roof_count, :2]
            balls[~roof, :ball_roof_count] = balls[~roof, :ball_roof_count] + env.roof_add[:4]
        if voxel_roof_count > 0:
            cyl_start = ball_roof_count
            cyl_end = cyl_start + voxel_roof_count
            voxels[~roof, :voxel_roof_count, :2] = cyl[~roof, cyl_start:cyl_end, :2]
            voxels[~roof, :voxel_roof_count] = voxels[~roof, :voxel_roof_count] + env.roof_add

        if balls.size(1) > 0:
            balls[..., 0] = torch.minimum(torch.maximum(balls[..., 0], balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - balls[..., 3])
        if voxels.size(1) > 0:
            voxels[..., 0] = torch.minimum(torch.maximum(voxels[..., 0], voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - voxels[..., 3])
            voxels[roof, 0, 2] = voxels[roof, 0, 2] * 0.5 + 201
            voxels[roof, 0, 3:] = 200
        if cyl.size(1) > 0:
            cyl[..., 0] = torch.minimum(torch.maximum(cyl[..., 0], cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - cyl[..., 2])
        if cyl_h.size(1) > 0:
            cyl_h[..., 0] = torch.minimum(torch.maximum(cyl_h[..., 0], cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - cyl_h[..., 2])

        return scene
