from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class TrainingLossResult:
    """Aggregated training loss and logging tensors."""

    total: torch.Tensor
    terms: Dict[str, torch.Tensor]
    weighted_terms: Dict[str, torch.Tensor]
    weights: Dict[str, float]
    distance: torch.Tensor
    speed_history: torch.Tensor
    v_history: torch.Tensor
    act_buffer: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return self.total

    def __getattr__(self, name: str):
        if name.startswith("loss_") and name in self.terms:
            return self.terms[name]
        raise AttributeError(name)

    def tensorboard_scalars(self, success: torch.Tensor, avg_speed: torch.Tensor) -> Dict[str, torch.Tensor]:
        scalars = {"loss": self.loss}
        scalars.update(self.terms)
        scalars.update({f"{name}_weight": value for name, value in self.weights.items()})
        scalars.update({
            "success": success,
            "max_speed": self.speed_history.max(0).values.mean(),
            "avg_speed": avg_speed.mean(),
            "ar": (success * avg_speed).mean(),
        })
        return scalars
