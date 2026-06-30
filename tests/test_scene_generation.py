from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_scene_primitives_concatenate_and_apply_to_env():
    import pytest

    torch = pytest.importorskip("torch")

    from env.scene import ScenePrimitives

    first = ScenePrimitives(
        balls=torch.ones((2, 1, 4)),
        voxels=torch.ones((2, 2, 6)),
    )
    second = ScenePrimitives(
        balls=torch.full((2, 3, 4), 2.0),
        cyl=torch.full((2, 1, 3), 3.0),
        cyl_h=torch.full((2, 2, 3), 4.0),
    )

    scene = ScenePrimitives.cat([first, second], batch_size=2, device=torch.device("cpu"))

    class FakeEnv:
        pass

    env = FakeEnv()
    scene.to_env(env)

    assert env.balls.shape == (2, 4, 4)
    assert env.voxels.shape == (2, 2, 6)
    assert env.cyl.shape == (2, 1, 3)
    assert env.cyl_h.shape == (2, 2, 3)
    assert torch.all(env.balls[:, :1] == 1)
    assert torch.all(env.balls[:, 1:] == 2)


def test_legacy_env_config_builds_scene_pipeline_generators():
    from env.scene import build_scene_pipeline_from_legacy_env
    from se3diff_config.schema import EnvConfig

    config = EnvConfig(
        gate=True,
        ground_voxels=True,
        scaffold=True,
        gap=True,
        gap_prob=0.7,
        n_balls=3,
        n_voxels=4,
        n_cyl=5,
        n_cyl_h=6,
        n_ground_voxels=7,
    )

    pipeline = build_scene_pipeline_from_legacy_env(config)
    names = [generator.name for generator in pipeline.generators]
    post_scale_names = [generator.name for generator in pipeline.post_scale_generators]

    assert names == ["random_primitives", "roof", "ground", "gate", "scaffold"]
    assert post_scale_names == ["gap_wall"]
    assert pipeline.generators[0].n_balls == 3
    assert pipeline.generators[2].n_ground_voxels == 7
    assert pipeline.post_scale_generators[0].prob == 0.7


def test_env_accepts_fixed_physical_params_for_repeatable_scenes():
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("quadsim_cuda")

    from env import Env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("quadsim_cuda environment smoke requires CUDA tensors")

    env = Env(
        1,
        64,
        48,
        0.4,
        device,
        single=True,
        random_rotation=False,
        max_speed=4.0,
        margin=0.2,
        quad_mass=1.3,
        quad_mass_randomization=0.0,
    )

    assert torch.allclose(env.max_speed, torch.full((1, 1), 4.0, device=device))
    assert torch.allclose(env.margin, torch.full((1,), 0.2, device=device))
    assert torch.allclose(env.mass, torch.full((1, 1), 1.3, device=device))
    env.reset()
    assert torch.allclose(env.max_speed, torch.full((1, 1), 4.0, device=device))
    assert torch.allclose(env.margin, torch.full((1,), 0.2, device=device))
    assert torch.allclose(env.mass, torch.full((1, 1), 1.3, device=device))


def test_env_randomizes_quad_mass_within_ratio():
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("quadsim_cuda")

    from env import Env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("quadsim_cuda environment smoke requires CUDA tensors")

    env = Env(
        8,
        64,
        48,
        0.4,
        device,
        single=True,
        quad_mass=2.0,
        quad_mass_randomization=0.1,
    )

    assert env.mass.shape == (8, 1)
    assert torch.all(env.mass >= 1.8)
    assert torch.all(env.mass <= 2.2)


def test_fixed_max_speed_and_margin_are_wired_through_config_entry_points():
    schema_source = (ROOT / "se3diff_config" / "schema.py").read_text()
    factory_source = (ROOT / "se3diff_config" / "env_factory.py").read_text()
    env_source = (ROOT / "env" / "env_cuda.py").read_text()
    export_source = (ROOT / "visualization" / "export_env_snapshot.py").read_text()

    assert "max_speed: float | None = None" in schema_source
    assert "margin: float | None = None" in schema_source
    assert "quad_mass: float = 1.0" in schema_source
    assert "quad_mass_randomization: float = 0.0" in schema_source
    assert "max_speed=env_config.max_speed" in factory_source
    assert "margin=env_config.margin" in factory_source
    assert "quad_mass=float(env_config.quad_mass)" in factory_source
    assert "quad_mass_randomization=float(env_config.quad_mass_randomization)" in factory_source
    assert "fixed_max_speed" in env_source
    assert "fixed_margin" in env_source
    assert "self.mass" in env_source
    assert 'max_speed=choose(args.max_speed, env_config, "max_speed", None)' in export_source
    assert 'margin=choose(args.margin, env_config, "margin", None)' in export_source
    assert 'quad_mass=float(choose(args.quad_mass, env_config, "quad_mass", 1.0))' in export_source
    assert '"mass": cpu_clone(env.mass[0])' in export_source
    assert '"ctbr_body_rate_limit": env.ctbr_body_rate_limit' in export_source
    assert '"collective_thrust": cpu_clone(env.collective_thrust[0])' in export_source


def test_gap_wall_is_generated_after_speed_scaling():
    source = (ROOT / "env" / "scene" / "pipeline.py").read_text()
    registry_source = (ROOT / "env" / "scene" / "registry.py").read_text()

    assert "post_scale_generators: Sequence[ObstacleGenerator]" in source
    assert "self._apply_speed_scaling(ctx, scene)" in source
    assert "for generator in self.post_scale_generators:" in source
    assert source.index("self._apply_speed_scaling(ctx, scene)") < source.index("for generator in self.post_scale_generators:")
    assert "post_scale_generator_configs.append" in registry_source


def test_ceiling_plane_config_is_passed_through_env_entry_points():
    schema_source = (ROOT / "se3diff_config" / "schema.py").read_text()
    factory_source = (ROOT / "se3diff_config" / "env_factory.py").read_text()
    env_source = (ROOT / "env" / "env_cuda.py").read_text()
    export_source = (ROOT / "visualization" / "export_env_snapshot.py").read_text()
    kernel_source = (ROOT / "src" / "quadsim_kernel.cu").read_text()

    assert "ceiling: bool = False" in schema_source
    assert "ceiling_height: float = 3.0" in schema_source
    assert "ceiling=bool(env_config.ceiling)" in factory_source
    assert "ceiling_height=float(env_config.ceiling_height)" in factory_source
    assert "ceiling=False" in env_source
    assert "self.ceiling = ceiling" in env_source
    assert "self.ceiling_height = float(ceiling_height)" in env_source
    assert '"ceiling": env.ceiling' in export_source
    assert '"ceiling_height": env.ceiling_height' in export_source
    assert "bool has_ceiling" in kernel_source
    assert "ceiling_height" in kernel_source


def test_box_ceiling_generator_is_removed():
    assert not (ROOT / "env" / "scene" / "generators" / "ceiling.py").exists()
    registry_source = (ROOT / "env" / "scene" / "registry.py").read_text()

    assert "CeilingObstacleGenerator" not in registry_source
    assert '"ceiling"' not in registry_source


def test_env_reset_delegates_scene_generation_to_pipeline():
    source = (ROOT / "env" / "env_cuda.py").read_text()

    assert "self.scene_pipeline = build_scene_pipeline_from_legacy_env" in source
    assert "scene = self.scene_pipeline.generate(ctx)" in source
    assert "scene.to_env(self)" in source


def test_obstacle_curriculum_counts_are_linearly_scheduled():
    from env.scene.curriculum import ObstacleCountCurriculum

    curriculum = ObstacleCountCurriculum(
        enabled=True,
        start_iter=10,
        end_iter=30,
        start_counts={
            "n_balls": 0,
            "n_voxels": 2,
            "n_cyl": 4,
            "n_cyl_h": 0,
            "n_ground_voxels": 1,
        },
    )
    final_counts = {
        "n_balls": 10,
        "n_voxels": 12,
        "n_cyl": 14,
        "n_cyl_h": 2,
        "n_ground_voxels": 5,
    }

    assert curriculum.counts_at(step=0, final_counts=final_counts)["n_balls"] == 0
    assert curriculum.counts_at(step=20, final_counts=final_counts) == {
        "n_balls": 5,
        "n_voxels": 7,
        "n_cyl": 9,
        "n_cyl_h": 1,
        "n_ground_voxels": 3,
    }
    assert curriculum.counts_at(step=30, final_counts=final_counts) == final_counts


def test_obstacle_curriculum_is_wired_to_env_and_training_loop():
    schema_source = (ROOT / "se3diff_config" / "schema.py").read_text()
    factory_source = (ROOT / "se3diff_config" / "env_factory.py").read_text()
    env_source = (ROOT / "env" / "env_cuda.py").read_text()
    train_source = (ROOT / "train" / "main_cuda.py").read_text()
    random_source = (ROOT / "env" / "scene" / "generators" / "random_primitives.py").read_text()
    ground_source = (ROOT / "env" / "scene" / "generators" / "ground.py").read_text()

    assert "obstacle_curriculum: Dict[str, Any]" in schema_source
    assert "obstacle_curriculum=env_config.obstacle_curriculum" in factory_source
    assert "def set_obstacle_curriculum_step" in env_source
    assert "env.set_obstacle_curriculum_step(i)" in train_source
    assert 'ctx.obstacle_count("n_balls"' in random_source
    assert 'ctx.obstacle_count("n_ground_voxels"' in ground_source
