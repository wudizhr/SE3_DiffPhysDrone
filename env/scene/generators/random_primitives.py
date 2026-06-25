from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class RandomPrimitiveGenerator(ObstacleGenerator):
    n_balls: int = 30
    n_voxels: int = 30
    n_cyl: int = 30
    n_cyl_h: int = 2
    name: str = "random_primitives"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import torch

        env = ctx.env
        B = ctx.batch_size
        device = ctx.device
        balls = torch.rand((B, self.n_balls, 4), device=device) * env.ball_w + env.ball_b
        voxels = torch.rand((B, self.n_voxels, 6), device=device) * env.voxel_w + env.voxel_b
        cyl = torch.rand((B, self.n_cyl, 3), device=device) * env.cyl_w + env.cyl_b
        cyl_h = torch.rand((B, self.n_cyl_h, 3), device=device) * env.cyl_h_w + env.cyl_h_b
        return ScenePrimitives(balls=balls, voxels=voxels, cyl=cyl, cyl_h=cyl_h)
