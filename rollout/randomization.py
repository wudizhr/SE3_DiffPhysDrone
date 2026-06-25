from __future__ import annotations

from typing import Any


def _zero_attr(env: Any, name: str) -> None:
    value = getattr(env, name, None)
    if hasattr(value, "zero_"):
        value.zero_()


def _fill_attr(env: Any, name: str, value: float) -> None:
    tensor = getattr(env, name, None)
    if hasattr(tensor, "fill_"):
        tensor.fill_(value)


def disable_visualization_randomization(env: Any) -> None:
    """Normalize runtime noise fields for repeatable policy playback."""

    _zero_attr(env, "v_wind")
    _zero_attr(env, "dg")
    _zero_attr(env, "drag_2")
    _fill_attr(env, "thr_est_error", 1.0)
    _fill_attr(env, "pitch_ctl_delay", 12.0)
    _fill_attr(env, "yaw_ctl_delay", 6.0)
    if hasattr(env, "fov_x_half_tan"):
        env._fov_x_half_tan = env.fov_x_half_tan
    env.drone_radius = 0.15
    env.n_drones_per_group = 1
