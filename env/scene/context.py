from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SceneContext:
    env: Any
    batch_size: int
    device: Any
    max_speed: Any
    scale: Any
    n_drones_per_group: int

    @classmethod
    def from_env(cls, env: Any, *, max_speed: Any, scale: Any) -> "SceneContext":
        return cls(
            env=env,
            batch_size=env.batch_size,
            device=env.device,
            max_speed=max_speed,
            scale=scale,
            n_drones_per_group=int(env.n_drones_per_group),
        )
