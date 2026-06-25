from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class SceneGenerationPipeline:
    generators: Sequence[ObstacleGenerator]
    post_scale_generators: Sequence[ObstacleGenerator]

    def generate(self, ctx: SceneContext) -> ScenePrimitives:
        scene = ScenePrimitives.empty(batch_size=ctx.batch_size, device=ctx.device)
        for generator in self.generators:
            scene = generator.generate(ctx, scene)
        if ctx.env.is_scale:
            self._apply_speed_scaling(ctx, scene)
        for generator in self.post_scale_generators:
            scene = generator.generate(ctx, scene)
        self._apply_random_rotation(ctx, scene)
        return scene

    def _apply_speed_scaling(self, ctx: SceneContext, scene: ScenePrimitives) -> None:
        import torch

        env = ctx.env
        scale = ctx.scale
        y_scale = (ctx.max_speed + 4) / scale
        if scene.voxels.size(1) > 0:
            scene.voxels[:, :, 1] *= y_scale
        if scene.balls.size(1) > 0:
            scene.balls[:, :, 1] *= y_scale
        if scene.cyl.size(1) > 0:
            scene.cyl[:, :, 1] *= y_scale

        if scene.voxels.size(1) > 0:
            scene.voxels[..., 0] *= scale
        if scene.balls.size(1) > 0:
            scene.balls[..., 0] *= scale
        if scene.cyl.size(1) > 0:
            scene.cyl[..., 0] *= scale
        if scene.cyl_h.size(1) > 0:
            scene.cyl_h[..., 0] *= scale

        ground_ball_count = getattr(env, "_ground_ball_count", 0)
        ground_balls_r_ground = getattr(env, "_ground_balls_r_ground", None)
        if ground_ball_count > 0 and ground_balls_r_ground is not None:
            scene.balls[:, :ground_ball_count, 0] = torch.minimum(
                torch.maximum(scene.balls[:, :ground_ball_count, 0], ground_balls_r_ground + 0.3),
                scale * 8 - 0.3 - ground_balls_r_ground,
            )

    def _apply_random_rotation(self, ctx: SceneContext, scene: ScenePrimitives) -> None:
        import torch

        env = ctx.env
        if not env.random_rotation:
            return
        B = ctx.batch_size
        yaw_bias = torch.rand(B // ctx.n_drones_per_group, device=ctx.device).repeat_interleave(ctx.n_drones_per_group, 0) * 1.5 - 0.75
        c = torch.cos(yaw_bias)
        s = torch.sin(yaw_bias)
        l = torch.ones_like(yaw_bias)
        o = torch.zeros_like(yaw_bias)
        R = torch.stack([c, -s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
        env.p = torch.squeeze(R @ env.p[..., None], -1)
        env.p_target = torch.squeeze(R @ env.p_target[..., None], -1)
        if scene.voxels.size(1) > 0:
            scene.voxels[..., :3] = (R @ scene.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
        if scene.balls.size(1) > 0:
            scene.balls[..., :3] = (R @ scene.balls[..., :3].transpose(1, 2)).transpose(1, 2)
        if scene.cyl.size(1) > 0:
            scene.cyl[..., :3] = (R @ scene.cyl[..., :3].transpose(1, 2)).transpose(1, 2)
