from __future__ import annotations

from importlib import import_module

from se3diff_config.schema import ModelConfig


MODEL_REGISTRY = {
    "pm_model": "model.pm_model.Model",
    "mid360_cnn_model": "model.mid360_cnn_model.Model",
    "se3_model": None,
}


def resolve_model_class(config: ModelConfig):
    target = MODEL_REGISTRY.get(config.name)
    if config.name == "se3_model" and target is None:
        raise NotImplementedError(
            "se3_model is registered as a future model option, but CTBR control integration is not ready."
        )
    if target is None:
        raise ValueError(f"Unknown model '{config.name}'")
    module_name, class_name = target.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)


def create_model(config: ModelConfig):
    if config.name == "pm_model":
        Model = resolve_model_class(config)
        dim_obs = 10 if config.dim_obs is None else int(config.dim_obs)
        return Model(dim_obs, int(config.dim_action))
    if config.name == "mid360_cnn_model":
        Model = resolve_model_class(config)
        dim_obs = 10 if config.dim_obs is None else int(config.dim_obs)
        return Model(dim_obs=dim_obs, dim_action=int(config.dim_action), hidden_dim=int(config.hidden_dim))
    resolve_model_class(config)
