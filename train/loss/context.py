from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class LossContext:
    """Tensor view of one rollout, shared by all loss terms."""

    v: torch.Tensor
    R: torch.Tensor | None
    target_v: torch.Tensor
    action: torch.Tensor
    clearance_vec: torch.Tensor
    margin: torch.Tensor
    mass: torch.Tensor | None = None
    action_mode: str = "accel_velocity"
    v_pred: torch.Tensor | None = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_rollout(
        cls,
        *,
        v_history: List[torch.Tensor],
        target_v_history: List[torch.Tensor],
        v_preds: List[torch.Tensor],
        act_buffer: List[torch.Tensor],
        vec_to_pt_history: List[torch.Tensor],
        margin: torch.Tensor,
        R_history: List[torch.Tensor] | None = None,
        mass: torch.Tensor | None = None,
        action_mode: str = "accel_velocity",
        extras: Dict[str, List[torch.Tensor] | torch.Tensor] | None = None,
    ) -> "LossContext":
        import torch

        stacked_extras = {}
        for key, value in (extras or {}).items():
            stacked_extras[key] = torch.stack(value) if isinstance(value, list) else value
        v = torch.stack(v_history)
        v_pred = torch.stack(v_preds) if v_preds else None
        return cls(
            v=v,
            R=torch.stack(R_history) if R_history is not None else None,
            target_v=torch.stack(target_v_history),
            action=torch.stack(act_buffer),
            clearance_vec=torch.stack(vec_to_pt_history),
            margin=margin,
            mass=mass,
            action_mode=action_mode,
            v_pred=v_pred,
            extras=stacked_extras,
        )

    @property
    def speed_history(self) -> torch.Tensor:
        return self.v.norm(2, -1)

    @property
    def distance(self) -> torch.Tensor:
        return self.clearance_vec.norm(2, -1) - self.margin

    def get_extra(self, name: str) -> torch.Tensor:
        try:
            return self.extras[name]
        except KeyError as exc:
            raise KeyError(f"LossContext is missing required history '{name}'") from exc
