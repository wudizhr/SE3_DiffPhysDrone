from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PointMassDynamics:
    """Adapter around the existing Env.run point-mass dynamics."""

    name: str = "point_mass"

    def step(self, env: Any, control: Any, ctl_dt: float, yaw_correction_vec: Any) -> None:
        env.run(control, ctl_dt, yaw_correction_vec)
