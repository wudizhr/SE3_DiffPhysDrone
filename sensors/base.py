from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class ObservationBatch:
    depth: Any
    state: Any
    flow: Any = None
    point_cloud: Any = None


class ObservationBuilder:
    name: str
    dim_state: int
    depth_pool_kernel: int
    model_input_keys: Tuple[str, ...]
    requires_depth: bool = False
    requires_flow: bool = False

    def render_inputs(self, env, ctl_dt, *, include_debug_outputs: bool = False) -> dict[str, Any]:
        return {}

    def build(self, *args, **kwargs) -> ObservationBatch:
        raise NotImplementedError
