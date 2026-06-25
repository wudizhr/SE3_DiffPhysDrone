from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.generators.base import ObstacleGenerator
from env.scene.primitives import ScenePrimitives


@dataclass
class GateGenerator(ObstacleGenerator):
    name: str = "gate"

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        import torch
        import quadsim_cuda

        env = ctx.env
        B = ctx.batch_size
        device = ctx.device
        gate = torch.rand((B, 4), device=device) * env.gate_w + env.gate_b
        p = gate[None, :, :3]
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, scene.balls, scene.cyl, scene.cyl_h, scene.voxels, p, env.drone_radius, 1)
        gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
        gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
        ones = torch.ones_like(gate_x)
        gate_voxels = torch.stack([
            torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
            torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
            torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
            torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
        ], 1)
        scene.voxels = torch.cat([scene.voxels, gate_voxels], 1)
        return scene
