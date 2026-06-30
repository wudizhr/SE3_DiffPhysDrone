from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class GroundObstacleGenerator(ObstacleGenerator):
    n_ground_voxels: int = 10
    name: str = "ground"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import torch

        env = ctx.env
        B = ctx.batch_size
        device = ctx.device
        balls = scene.balls
        voxels = scene.voxels

        ground_ball_count = min(2, balls.size(1))
        ground_balls_r_ground = None
        if ground_ball_count > 0:
            ground_balls_r = 8 + torch.rand((B, ground_ball_count), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, ground_ball_count), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            balls[:, :ground_ball_count, 3] = ground_balls_r
            balls[:, :ground_ball_count, 2] = ground_balls_h - ground_balls_r - 1

        n_ground_voxels = ctx.obstacle_count("n_ground_voxels", self.n_ground_voxels)
        if n_ground_voxels > 0:
            ground_voxels = torch.rand((B, n_ground_voxels, 6), device=device) * env.ground_voxel_w + env.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            voxels = torch.cat([voxels, ground_voxels], 1)
            scene.voxels = voxels

        env._ground_ball_count = ground_ball_count
        env._ground_balls_r_ground = ground_balls_r_ground
        return scene
