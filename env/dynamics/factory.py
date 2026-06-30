from __future__ import annotations

from .ctbr import CtbrDynamics
from .point_mass import PointMassDynamics

POINT_MASS_BACKENDS = {"point_mass"}
CTBR_BACKENDS = {"ctbr"}


def create_dynamics_backend(backend_name: str):
    if backend_name in POINT_MASS_BACKENDS:
        return PointMassDynamics()
    if backend_name in CTBR_BACKENDS:
        return CtbrDynamics()
    raise ValueError(f"Unknown dynamics backend '{backend_name}'")
