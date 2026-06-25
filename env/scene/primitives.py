from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ScenePrimitives:
    balls: Any | None = None
    voxels: Any | None = None
    cyl: Any | None = None
    cyl_h: Any | None = None

    @staticmethod
    def empty(*, batch_size: int, device: Any, dtype: Any | None = None) -> "ScenePrimitives":
        import torch

        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ScenePrimitives(
            balls=torch.empty((batch_size, 0, 4), **kwargs),
            voxels=torch.empty((batch_size, 0, 6), **kwargs),
            cyl=torch.empty((batch_size, 0, 3), **kwargs),
            cyl_h=torch.empty((batch_size, 0, 3), **kwargs),
        )

    @classmethod
    def cat(cls, scenes: Iterable["ScenePrimitives"], *, batch_size: int, device: Any, dtype: Any | None = None) -> "ScenePrimitives":
        import torch

        scenes = list(scenes)
        empty = cls.empty(batch_size=batch_size, device=device, dtype=dtype)

        def cat_attr(name: str):
            tensors = [getattr(scene, name) for scene in scenes if getattr(scene, name) is not None]
            if not tensors:
                return getattr(empty, name)
            return torch.cat(tensors, dim=1)

        return cls(
            balls=cat_attr("balls"),
            voxels=cat_attr("voxels"),
            cyl=cat_attr("cyl"),
            cyl_h=cat_attr("cyl_h"),
        )

    def ensure_all(self, *, batch_size: int, device: Any, dtype: Any | None = None) -> "ScenePrimitives":
        empty = self.empty(batch_size=batch_size, device=device, dtype=dtype)
        return ScenePrimitives(
            balls=self.balls if self.balls is not None else empty.balls,
            voxels=self.voxels if self.voxels is not None else empty.voxels,
            cyl=self.cyl if self.cyl is not None else empty.cyl,
            cyl_h=self.cyl_h if self.cyl_h is not None else empty.cyl_h,
        )

    def to_env(self, env: Any) -> None:
        scene = self.ensure_all(batch_size=env.batch_size, device=env.device)
        env.balls = scene.balls
        env.voxels = scene.voxels
        env.cyl = scene.cyl
        env.cyl_h = scene.cyl_h
