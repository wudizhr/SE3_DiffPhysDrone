from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .schema import (
    EnvConfig,
    ExperimentConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    PathsConfig,
    PlaybackConfig,
    SensorConfig,
    TrainConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

SECTION_TYPES = {
    "env": EnvConfig,
    "train": TrainConfig,
    "loss": LossConfig,
    "inference": InferenceConfig,
    "model": ModelConfig,
    "sensor": SensorConfig,
    "paths": PathsConfig,
    "playback": PlaybackConfig,
}
FLAT_KEY_SECTIONS = {
    **{field.name: "env" for field in fields(EnvConfig)},
    **{field.name: "train" for field in fields(TrainConfig)},
    **{field.name: "loss" for field in fields(LossConfig)},
    **{field.name: "inference" for field in fields(InferenceConfig)},
    **{field.name: "model" for field in fields(ModelConfig)},
    **{field.name: "sensor" for field in fields(SensorConfig)},
    **{field.name: "paths" for field in fields(PathsConfig)},
}
CHECKPOINT_FLAT_KEYS = {
    field.name
    for section_type in (EnvConfig, InferenceConfig, ModelConfig, SensorConfig)
    for field in fields(section_type)
}


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _dataclass_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _dataclass_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataclass_dict(item) for item in value]
    return value


def config_to_yaml_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return _dataclass_dict(config)


def _validate_train_parallel_fields(data: Mapping[str, Any]) -> None:
    train_values = data.get("train")
    if not isinstance(train_values, Mapping):
        return
    num_envs = train_values.get("num_envs")
    batch_size = train_values.get("batch_size")
    if num_envs is not None and batch_size is not None and int(num_envs) != int(batch_size):
        raise ValueError(
            "train.num_envs and train.batch_size both set different values; "
            "use num_envs as the parallel environment count"
        )


def _normalize_train_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(values)
    if normalized.get("num_envs") is None:
        normalized["num_envs"] = normalized.get("batch_size", TrainConfig.batch_size)
    normalized["batch_size"] = normalized["num_envs"]
    return normalized


def _coerce_section(section_name: str, values: Mapping[str, Any]) -> Any:
    section_type = SECTION_TYPES[section_name]
    if section_name == "train":
        values = _normalize_train_values(values)
    valid = {field.name for field in fields(section_type)}
    return section_type(**{key: value for key, value in values.items() if key in valid})


def dict_to_config(data: Mapping[str, Any]) -> ExperimentConfig:
    normalized: Dict[str, Any] = {key: value for key, value in data.items() if key in SECTION_TYPES or key in {"scene", "rollout"}}

    for key, value in data.items():
        section = FLAT_KEY_SECTIONS.get(key)
        if section is not None:
            normalized.setdefault(section, {})
            if isinstance(normalized[section], dict):
                normalized[section][key] = value

    kwargs: Dict[str, Any] = {}
    for section_name in SECTION_TYPES:
        section_value = normalized.get(section_name, {})
        if section_value is None:
            section_value = {}
        if not isinstance(section_value, Mapping):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        kwargs[section_name] = _coerce_section(section_name, section_value)
    for passthrough in ("scene", "rollout"):
        value = normalized.get(passthrough, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{passthrough}' must be a mapping")
        kwargs[passthrough] = dict(value)
    return ExperimentConfig(**kwargs)


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    data = config_to_yaml_dict(ExperimentConfig())
    if DEFAULT_CONFIG_PATH.exists():
        default_data = load_yaml_file(DEFAULT_CONFIG_PATH)
        _validate_train_parallel_fields(default_data)
        data = deep_merge(data, default_data)
    if path is not None:
        user_data = load_yaml_file(path)
        _validate_train_parallel_fields(user_data)
        data = deep_merge(data, user_data)
    return dict_to_config(data)


def config_to_flat_args(config: ExperimentConfig) -> Dict[str, Any]:
    data = config_to_yaml_dict(config)
    flat: Dict[str, Any] = {}
    for section in ("train", "loss", "env", "inference", "model", "sensor", "paths"):
        flat.update(data.get(section, {}))
    return flat


def config_to_namespace(config: ExperimentConfig) -> argparse.Namespace:
    return argparse.Namespace(**config_to_flat_args(config))


def checkpoint_config_to_flat(config: ExperimentConfig) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    data = config_to_yaml_dict(config)
    for section in ("inference", "env", "model", "sensor"):
        for key, value in data.get(section, {}).items():
            if key == "name" or value is None:
                continue
            flat[key] = value
    flat["model_name"] = config.model.name
    flat["sensor_name"] = config.sensor.name
    return flat


def flatten_user_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section_name in SECTION_TYPES:
        section_value = data.get(section_name)
        if isinstance(section_value, Mapping):
            flat.update({key: value for key, value in section_value.items() if value is not None})
    for key, value in data.items():
        if key not in SECTION_TYPES and key not in {"scene", "rollout"}:
            flat[key] = value
    if "rollout" in data and isinstance(data["rollout"], Mapping):
        flat.update(data["rollout"])
    if "playback" in data and isinstance(data["playback"], Mapping):
        flat.update(data["playback"])
    return {key: value for key, value in flat.items() if value is not None}


def load_checkpoint_training_config(checkpoint_path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    structured_path = checkpoint_dir / "config.yaml"
    if structured_path.exists():
        return checkpoint_config_to_flat(load_experiment_config(structured_path))

    training_config: Dict[str, Any] = {}
    for filename in ("args.yaml", "model_info.yaml"):
        path = checkpoint_dir / filename
        if not path.exists():
            continue
        data = load_yaml_file(path)
        training_config.update(data)

    return {key: training_config[key] for key in CHECKPOINT_FLAT_KEYS if key in training_config}


def merge_checkpoint_with_user_config(user_config: Mapping[str, Any], checkpoint_path: str | Path) -> Dict[str, Any]:
    training_config = load_checkpoint_training_config(checkpoint_path)
    user_flat = flatten_user_config(user_config)
    return {**training_config, **user_flat}


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path
