from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


OBSTACLE_COUNT_KEYS = (
    "n_balls",
    "n_voxels",
    "n_cyl",
    "n_cyl_h",
    "n_ground_voxels",
)


@dataclass
class ObstacleCountCurriculum:
    """Linear schedule for scene obstacle counts."""

    enabled: bool = False
    start_iter: int = 0
    end_iter: int = 0
    start_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ObstacleCountCurriculum":
        if not config:
            return cls()
        raw_start_counts = config.get("start_counts") or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            start_iter=int(config.get("start_iter", 0)),
            end_iter=int(config.get("end_iter", 0)),
            start_counts={
                key: max(0, int(raw_start_counts[key]))
                for key in OBSTACLE_COUNT_KEYS
                if key in raw_start_counts
            },
        )

    def counts_at(self, *, step: int | None, final_counts: Mapping[str, int]) -> dict[str, int]:
        final = {key: max(0, int(final_counts.get(key, 0))) for key in OBSTACLE_COUNT_KEYS}
        if not self.enabled or step is None:
            return final

        start = {
            key: min(max(0, int(self.start_counts.get(key, final[key]))), final[key])
            for key in OBSTACLE_COUNT_KEYS
        }
        if self.end_iter <= self.start_iter:
            progress = 1.0 if step >= self.end_iter else 0.0
        elif step <= self.start_iter:
            progress = 0.0
        elif step >= self.end_iter:
            progress = 1.0
        else:
            progress = (step - self.start_iter) / (self.end_iter - self.start_iter)

        return {
            key: int(round(start[key] + progress * (final[key] - start[key])))
            for key in OBSTACLE_COUNT_KEYS
        }
