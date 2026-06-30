from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, FrozenSet, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from train.loss.context import LossContext


@dataclass
class LossTerm:
    """Base class for one weighted training loss term."""

    weight: float = 1.0
    enabled: bool = True

    name: ClassVar[str] = ""
    log_name: ClassVar[str] = ""
    required_history: ClassVar[FrozenSet[str]] = frozenset()

    def compute(self, context: LossContext) -> torch.Tensor:
        raise NotImplementedError
