from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from se3diff_config.schema import LossConfig

from .context import LossContext
from .registry import build_loss_registry
from .result import TrainingLossResult
from .terms.obstacle import barrier

if TYPE_CHECKING:
    import torch


def compute_training_loss(
    *,
    loss_config: LossConfig,
    v_history: List[torch.Tensor] | None = None,
    target_v_history: List[torch.Tensor] | None = None,
    v_preds: List[torch.Tensor] | None = None,
    act_buffer: List[torch.Tensor] | None = None,
    vec_to_pt_history: List[torch.Tensor] | None = None,
    margin: torch.Tensor | None = None,
    context: LossContext | None = None,
    extras: Dict[str, List[torch.Tensor] | torch.Tensor] | None = None,
) -> TrainingLossResult:
    """Compute weighted training loss through the extensible loss registry.

    The old list-based signature is kept for current training code. New callers
    may pass a prebuilt LossContext directly.
    """

    if context is None:
        if any(value is None for value in (v_history, target_v_history, v_preds, act_buffer, vec_to_pt_history, margin)):
            raise ValueError("Either context or all rollout history inputs must be provided")
        context = LossContext.from_rollout(
            v_history=v_history,
            target_v_history=target_v_history,
            v_preds=v_preds,
            act_buffer=act_buffer,
            vec_to_pt_history=vec_to_pt_history,
            margin=margin,
            extras=extras,
        )

    return build_loss_registry(loss_config).compute(context)
