from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class ScaffoldGenerator(ObstacleGenerator):
    prob: float = 0.5
    name: str = "scaffold"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import random
        import torch

        if random.random() >= self.prob:
            return scene

        B = ctx.batch_size
        device = ctx.device
        x = torch.arange(1, 6, dtype=torch.float, device=device)
        y = torch.arange(-3, 4, dtype=torch.float, device=device)
        z = torch.arange(1, 4, dtype=torch.float, device=device)
        _x, _y = torch.meshgrid(x, y)
        scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
        x_bias = torch.rand_like(ctx.max_speed) * ctx.max_speed
        scale = 1 + torch.rand((B, 1, 1), device=device)
        scaf_v = scaf_v * scale + torch.stack([
            x_bias,
            torch.randn_like(ctx.max_speed),
            torch.rand_like(ctx.max_speed) * 0.01,
        ], -1)
        scene.cyl = torch.cat([scene.cyl, scaf_v], 1)

        _x, _z = torch.meshgrid(x, z)
        scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
        scaf_h = scaf_h * scale + torch.stack([
            x_bias,
            torch.randn_like(ctx.max_speed) * 0.1,
            torch.rand_like(ctx.max_speed) * 0.01,
        ], -1)
        scene.cyl_h = torch.cat([scene.cyl_h, scaf_h], 1)
        return scene
