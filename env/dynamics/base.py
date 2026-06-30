from __future__ import annotations

from typing import Any, Protocol


class DynamicsBackend(Protocol):
    name: str

    def step(self, env: Any, control: Any, ctl_dt: float, yaw_correction_vec: Any) -> None:
        """Advance env dynamics by one control step."""
