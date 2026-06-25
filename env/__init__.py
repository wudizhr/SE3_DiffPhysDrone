__all__ = ["Env"]


def __getattr__(name):
    if name == "Env":
        from .env_cuda import Env

        return Env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
