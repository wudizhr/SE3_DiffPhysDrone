from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CtbrDynamics:
    """PyTorch reference CTBR rigid-body dynamics backend."""

    name: str = "ctbr"

    def step(self, env: Any, control: Any, ctl_dt: float, yaw_correction_vec: Any) -> None:
        env.run_ctbr(control, ctl_dt, yaw_correction_vec)
