from __future__ import annotations

from se3diff_config.schema import SensorConfig

from .depth_odom import DepthOdomObservationBuilder
from .mid360 import Mid360ObservationBuilder


OBSERVATION_BUILDERS = {
    "depth": DepthOdomObservationBuilder,
    "depth_odom": DepthOdomObservationBuilder,
    "mid360": Mid360ObservationBuilder,
}


def create_observation_builder(config: SensorConfig):
    try:
        builder_cls = OBSERVATION_BUILDERS[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(OBSERVATION_BUILDERS))
        raise ValueError(f"Unknown sensor '{config.name}'. Available sensors: {available}") from exc
    return builder_cls(config)
