from __future__ import annotations

from typing import Any

from env.scene.generators.gap import GapWallGenerator
from env.scene.generators.gate import GateGenerator
from env.scene.generators.ground import GroundObstacleGenerator
from env.scene.generators.random_primitives import RandomPrimitiveGenerator
from env.scene.generators.roof import RoofGenerator
from env.scene.generators.scaffold import ScaffoldGenerator
from env.scene.pipeline import SceneGenerationPipeline


SCENE_GENERATORS = {
    "random_primitives": RandomPrimitiveGenerator,
    "roof": RoofGenerator,
    "ground": GroundObstacleGenerator,
    "gate": GateGenerator,
    "gap_wall": GapWallGenerator,
    "scaffold": ScaffoldGenerator,
}


def create_scene_generator(config: dict[str, Any]):
    name = str(config["name"])
    try:
        generator_cls = SCENE_GENERATORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENE_GENERATORS))
        raise ValueError(f"Unknown scene generator '{name}'. Available generators: {available}") from exc
    kwargs = {key: value for key, value in config.items() if key not in {"name", "enabled"}}
    return generator_cls(**kwargs)


def build_scene_pipeline(
    generator_configs: list[dict[str, Any]],
    post_scale_generator_configs: list[dict[str, Any]] | None = None,
) -> SceneGenerationPipeline:
    generators = [
        create_scene_generator(config)
        for config in generator_configs
        if bool(config.get("enabled", True))
    ]
    post_scale_generators = [
        create_scene_generator(config)
        for config in (post_scale_generator_configs or [])
        if bool(config.get("enabled", True))
    ]
    return SceneGenerationPipeline(generators, post_scale_generators)


def build_scene_pipeline_from_legacy_env(env_config: Any) -> SceneGenerationPipeline:
    generator_configs: list[dict[str, Any]] = [
        {
            "name": "random_primitives",
            "n_balls": int(env_config.n_balls),
            "n_voxels": int(env_config.n_voxels),
            "n_cyl": int(env_config.n_cyl),
            "n_cyl_h": int(env_config.n_cyl_h),
        },
        {"name": "roof"},
    ]
    if bool(env_config.ground_voxels):
        generator_configs.append(
            {
                "name": "ground",
                "n_ground_voxels": int(env_config.n_ground_voxels),
            }
        )
    if bool(env_config.gate):
        generator_configs.append({"name": "gate"})
    post_scale_generator_configs: list[dict[str, Any]] = []    
    if bool(env_config.gap):
        post_scale_generator_configs.append(
            {
                "name": "gap_wall",
                "prob": float(env_config.gap_prob),
            }
        )
    if bool(env_config.scaffold):
        generator_configs.append({"name": "scaffold", "prob": 0.5})
    return build_scene_pipeline(generator_configs, post_scale_generator_configs)
