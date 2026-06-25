from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_loads_sensor_and_model_extension_settings():
    from se3diff_config.io import load_experiment_config

    config = load_experiment_config(ROOT / "configs" / "single_agent.yaml")

    assert config.sensor.name == "depth_odom"
    assert config.sensor.depth_pool_kernel == 4
    assert config.sensor.use_odom is True
    assert config.sensor.use_mid360 is False
    assert config.model.name == "pm_model"
    assert config.model.action_mode == "accel_velocity"


def test_mid360_cnn_config_declares_sensor_and_model_pair():
    from se3diff_config.io import load_experiment_config

    config = load_experiment_config(ROOT / "configs" / "mid360_cnn.yaml")

    assert config.sensor.name == "mid360"
    assert config.model.name == "mid360_cnn_model"
    assert config.model.dim_obs == 16
    assert config.sensor.mid360_phi_max_deg - config.sensor.mid360_phi_min_deg == 59.0
    assert (
        config.sensor.mid360_theta_max_deg - config.sensor.mid360_theta_min_deg
    ) / config.sensor.mid360_theta_resolution_deg == 60.0


def test_depth_odom_sensor_builder_declares_observation_shape():
    from sensors import create_observation_builder
    from se3diff_config.schema import SensorConfig

    builder = create_observation_builder(SensorConfig(name="depth_odom", use_odom=True))

    assert builder.dim_state == 10
    assert builder.depth_pool_kernel == 4
    assert builder.model_input_keys == ("depth", "state")
    assert builder.requires_depth is True


def test_mid360_sensor_builder_builds_nearest_range_pseudo_image():
    from sensors import create_observation_builder
    from se3diff_config.schema import SensorConfig
    import numpy as np

    config = SensorConfig(
        name="mid360",
        use_odom=True,
        mid360_theta_min_deg=-180.0,
        mid360_theta_max_deg=180.0,
        mid360_phi_min_deg=0.0,
        mid360_phi_max_deg=180.0,
        mid360_theta_resolution_deg=90.0,
        mid360_phi_resolution_deg=90.0,
        mid360_max_range=10.0,
    )
    builder = create_observation_builder(config)
    points = np.array(
        [[[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    ranges = np.array([[2.0, 1.0, 3.0, 4.0, 0.0]], dtype=np.float32)

    pseudo = builder.points_to_pseudo_image(points, ranges)

    assert builder.model_input_keys == ("mid360_pseudo_image", "state")
    assert builder.requires_depth is False
    assert builder.dim_state == 16
    assert pseudo.shape == (1, 2, 4, 1)
    assert np.isclose(pseudo[0, 1, 2, 0], 1.0)
    assert np.isclose(pseudo[0, 1, 3, 0], 3.0)
    assert np.isclose(pseudo[0, 0, 2, 0], 4.0)
    assert np.isclose(pseudo[0, 0, 0, 0], 10.0)


def test_quadsim_cuda_exports_mid360_renderer_interface():
    quadsim_cpp = (ROOT / "src" / "quadsim.cpp").read_text()
    quadsim_kernel = (ROOT / "src" / "quadsim_kernel.cu").read_text()

    assert "void render_mid360_cuda(" in quadsim_cpp
    assert 'm.def("render_mid360"' in quadsim_cpp
    assert "__global__ void render_mid360_cuda_kernel" in quadsim_kernel
    assert "vertical_min_deg" in quadsim_kernel
    assert "vertical_max_deg" in quadsim_kernel
    assert "min_range" in quadsim_kernel
    assert "max_range" in quadsim_kernel
    assert "points.packed_accessor" in quadsim_kernel
    assert "ranges.packed_accessor" in quadsim_kernel
    assert "void points_to_pseudo_image_mid360_cuda(" in quadsim_cpp
    assert 'm.def("points_to_pseudo_image_mid360"' in quadsim_cpp
    assert "__global__ void points_to_pseudo_image_mid360_cuda_kernel" in quadsim_kernel


def test_mid360_cuda_pseudo_image_matches_torch_reference():
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for quadsim_cuda pseudo-image test")
    try:
        import quadsim_cuda
    except ImportError:
        pytest.skip("quadsim_cuda extension is not built")

    from sensors import create_observation_builder
    from se3diff_config.schema import SensorConfig

    config = SensorConfig(
        name="mid360",
        use_odom=True,
        mid360_theta_min_deg=-180.0,
        mid360_theta_max_deg=180.0,
        mid360_phi_min_deg=0.0,
        mid360_phi_max_deg=180.0,
        mid360_theta_resolution_deg=90.0,
        mid360_phi_resolution_deg=90.0,
        mid360_max_range=10.0,
    )
    builder = create_observation_builder(config)
    points = torch.tensor(
        [[[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]]],
        device="cuda",
    )
    ranges = torch.tensor([[2.0, 1.0, 3.0, 4.0, 0.0]], device="cuda")

    reference = builder._points_to_pseudo_image_torch(points, ranges).permute(0, 3, 1, 2).contiguous()
    cuda_image = torch.full(
        (points.shape[0], 1, builder.n_phi, builder.n_theta),
        builder.max_range,
        device=points.device,
        dtype=points.dtype,
    )
    quadsim_cuda.points_to_pseudo_image_mid360(
        cuda_image,
        points,
        ranges,
        builder.min_range,
        builder.max_range,
        builder.theta_min,
        builder.phi_min,
        builder.theta_resolution,
        builder.phi_resolution,
    )
    torch.cuda.synchronize()

    assert torch.allclose(cuda_image, reference)


def test_model_factory_registers_pm_model_and_rejects_unwired_se3_model():
    from model.factory import MODEL_REGISTRY, create_model
    from se3diff_config.schema import ModelConfig

    assert MODEL_REGISTRY["pm_model"] == "model.pm_model.Model"
    assert MODEL_REGISTRY["mid360_cnn_model"] == "model.mid360_cnn_model.Model"

    try:
        create_model(ModelConfig(name="se3_model", dim_obs=10, dim_action=4))
    except NotImplementedError as exc:
        assert "se3_model" in str(exc)
        assert "CTBR" in str(exc)
    else:
        raise AssertionError("se3_model should stay unwired until CTBR control is ready")


def test_mid360_cnn_model_declares_expected_dict_interface():
    model_source = (ROOT / "model" / "mid360_cnn_model.py").read_text()
    factory_source = (ROOT / "model" / "factory.py").read_text()

    assert "class Model" in model_source
    assert "mid360_pseudo_image" in model_source
    assert "GRUCell" in model_source
    assert "action_fc" in model_source
    assert '"mid360_cnn_model": "model.mid360_cnn_model.Model"' in factory_source


def test_mid360_observation_builder_can_render_env_observation_dict():
    source = (ROOT / "sensors" / "mid360.py").read_text()

    assert "def render_inputs(self, env, ctl_dt, *, include_debug_outputs: bool = False)" in source
    assert "def build(self, *, sensor_inputs, state=None)" in source
    assert "render_mid360" in source
    assert "world_points_to_body" in source
    assert "full_attitude_state" in source
    assert '"mid360_pseudo_image"' in source
    assert '"state"' in source


def test_mid360_build_defaults_to_model_inputs_only():
    source = (ROOT / "sensors" / "mid360.py").read_text()

    assert "include_debug_outputs: bool = False" in source
    assert "if include_debug_outputs:" in source
    assert 'sensor_inputs["mid360_points"] = body_points' in source
    assert 'sensor_inputs["mid360_world_points"] = points' in source
    assert 'sensor_inputs["mid360_ranges"] = ranges' in source
    assert 'return {"mid360_pseudo_image": sensor_inputs["mid360_pseudo_image"], "state": state}' in source


def test_mid360_observation_generation_is_detached_from_autograd_graph():
    source = (ROOT / "sensors" / "mid360.py").read_text()

    assert "with torch.no_grad():" in source
    assert "body_points = self.world_points_to_body" in source
    assert "pseudo_image = self._points_to_pseudo_image_cuda_nchw" in source


def test_mid360_build_keeps_configured_pseudo_image_width():
    source = (ROOT / "sensors" / "mid360.py").read_text()

    assert "_resize_theta_torch" not in source
    assert "torch.empty((env.batch_size, 1, self.n_phi, self.n_theta)" in source



def test_mid360_configured_pseudo_image_shape_is_not_forced_to_180_width():
    from sensors import create_observation_builder
    from se3diff_config.io import load_experiment_config
    import numpy as np

    config = load_experiment_config(ROOT / "configs" / "mid360_cnn.yaml")
    builder = create_observation_builder(config.sensor)
    points = np.zeros((1, 1, 3), dtype=np.float32)

    pseudo = builder.points_to_pseudo_image(points)

    assert builder.n_phi == 12
    assert builder.n_theta == 60
    assert pseudo.shape == (1, 12, 60, 1)



def test_training_loop_asks_observation_builder_for_render_inputs():
    source = (ROOT / "train" / "main_cuda.py").read_text()

    assert "sensor_inputs = observation_builder.render_inputs(env, ctl_dt)" in source
    assert "obs = observation_builder.build(sensor_inputs=sensor_inputs, state=state)" in source
    assert "depth, flow = env.render(ctl_dt)" not in source



def test_mid360_builder_does_not_request_depth_rendering():
    from sensors import create_observation_builder
    from se3diff_config.schema import SensorConfig

    mid360 = create_observation_builder(SensorConfig(name="mid360"))
    depth = create_observation_builder(SensorConfig(name="depth_odom"))

    assert mid360.requires_depth is False
    assert depth.requires_depth is True
