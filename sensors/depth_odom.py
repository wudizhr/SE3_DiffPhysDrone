from __future__ import annotations

from .base import ObservationBuilder
from se3diff_config.schema import SensorConfig


class DepthOdomObservationBuilder(ObservationBuilder):
    name = "depth_odom"
    model_input_keys = ("depth", "state")
    requires_depth = True

    def __init__(self, config: SensorConfig):
        self.config = config
        self.depth_pool_kernel = int(config.depth_pool_kernel)
        self.dim_state = 10 if bool(config.use_odom) else 7

    def render_inputs(self, env, ctl_dt, *, include_debug_outputs: bool = False):
        depth, flow = env.render(ctl_dt)
        return {"depth": depth, "flow": flow}

    def build(self, *, sensor_inputs, state=None):
        return {"depth": sensor_inputs["depth"], "state": state}
