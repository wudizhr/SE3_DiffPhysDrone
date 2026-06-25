from __future__ import annotations

from dataclasses import dataclass

from env.scene.context import SceneContext
from env.scene.primitives import ScenePrimitives


@dataclass
class ObstacleGenerator:
    name: str

    def generate(self, ctx: SceneContext, scene: ScenePrimitives) -> ScenePrimitives:
        raise NotImplementedError
