from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def zero_(self):
        self.calls.append("zero_")
        return self

    def fill_(self, value):
        self.calls.append(("fill_", value))
        return self


class FakeEnv:
    def __init__(self):
        self.v_wind = FakeTensor("v_wind")
        self.dg = FakeTensor("dg")
        self.thr_est_error = FakeTensor("thr_est_error")
        self.drag_2 = FakeTensor("drag_2")
        self.pitch_ctl_delay = FakeTensor("pitch_ctl_delay")
        self.yaw_ctl_delay = FakeTensor("yaw_ctl_delay")
        self.margin = FakeTensor("margin")
        self.mass = FakeTensor("mass")
        self.drone_radius = 0.11
        self.n_drones_per_group = 8
        self.fov_x_half_tan = 0.53
        self._fov_x_half_tan = 0.49


def test_inference_config_defaults_to_deterministic_visualization():
    from se3diff_config.io import load_experiment_config

    config = load_experiment_config(ROOT / "configs" / "single_agent.yaml")

    assert config.inference.deterministic_visualization is True


def test_policy_runner_config_is_lightweight_and_defaults_to_deterministic():
    from rollout import PolicyRunnerConfig

    config = PolicyRunnerConfig(ctl_dt=0.05, max_steps=12)

    assert config.ctl_dt == 0.05
    assert config.max_steps == 12
    assert config.deterministic_visualization is True
    assert config.backend_name == "point_mass"


def test_disable_visualization_randomization_normalizes_runtime_noise_fields():
    from rollout.randomization import disable_visualization_randomization

    env = FakeEnv()

    disable_visualization_randomization(env)

    assert env.v_wind.calls == ["zero_"]
    assert env.dg.calls == ["zero_"]
    assert env.thr_est_error.calls == [("fill_", 1.0)]
    assert env.drag_2.calls == ["zero_"]
    assert env.pitch_ctl_delay.calls == [("fill_", 12.0)]
    assert env.yaw_ctl_delay.calls == [("fill_", 6.0)]
    assert env.margin.calls == []
    assert env.drone_radius == 0.15
    assert env.n_drones_per_group == 1
    assert env._fov_x_half_tan == env.fov_x_half_tan


def test_policy_runner_has_mid360_observation_builder_path():
    source = (ROOT / "rollout" / "policy_runner.py").read_text()

    assert "observation_builder=None" in source
    assert 'self.config.sensor_name == "mid360"' in source
    assert "sensor_inputs = self.observation_builder.render_inputs(" in source
    assert "include_debug_outputs=True" in source
    assert "obs = self.observation_builder.build(sensor_inputs=sensor_inputs, state=state)" in source
    assert "self.model(obs, hx=self.hidden_state)" in source


def test_policy_runner_exports_mid360_world_points_for_rviz():
    source = (ROOT / "rollout" / "policy_runner.py").read_text()

    assert 'mid360_world_points = sensor_inputs.get("mid360_world_points")' in source
    assert 'result["mid360_world_points"] = mid360_world_points' in source


def test_policy_rollout_saves_mid360_world_points_when_available():
    source = (ROOT / "visualization" / "export_policy_rollout.py").read_text()

    assert 'step.get("mid360_world_points", step["mid360_points"])' in source
