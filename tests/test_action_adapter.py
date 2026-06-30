from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_accel_velocity_adapter_matches_legacy_formula():
    import pytest

    torch = pytest.importorskip("torch")

    from control import AccelVelocityActionAdapter

    class FakeEnv:
        batch_size = 2

    env = FakeEnv()
    env.g_std = torch.tensor([0.0, 0.0, -9.80665])
    env.thr_est_error = torch.tensor([1.0, 1.1])
    policy_frame = torch.eye(3).repeat(2, 1, 1)
    raw_action = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [-1.0, 0.5, 2.0, -0.2, 0.4, 0.8],
        ]
    )

    legacy_a_pred, legacy_v_pred, *_ = (policy_frame @ raw_action.reshape(2, 3, -1)).unbind(-1)
    legacy_control = (legacy_a_pred - legacy_v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std

    result = AccelVelocityActionAdapter().to_control(raw_action, env, policy_frame)

    assert torch.allclose(result.a_pred, legacy_a_pred)
    assert torch.allclose(result.v_pred, legacy_v_pred)
    assert torch.allclose(result.control, legacy_control)


def test_ctbr_adapter_scales_raw_action_to_thrust_and_body_rates():
    import pytest

    torch = pytest.importorskip("torch")

    from control import CtbrActionAdapter

    class FakeEnv:
        batch_size = 2

    env = FakeEnv()
    env.ctbr_thrust_min = 2.0
    env.ctbr_thrust_max = 10.0
    env.ctbr_body_rate_limit = 4.0
    env.collective_thrust = torch.full((2, 1), 6.0)
    env.omega = torch.zeros((2, 3))

    adapter = CtbrActionAdapter()
    raw_action = torch.tensor([[0.0, 0.0, 1.0, -1.0], [10.0, -10.0, 0.5, -0.5]])

    initial = adapter.initial_control(env)
    result = adapter.to_control(raw_action, env, torch.eye(3).repeat(2, 1, 1))

    assert torch.allclose(initial, torch.tensor([[6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]]))
    assert result.control.shape == (2, 4)
    assert torch.all(result.thrust_cmd >= 2.0)
    assert torch.all(result.thrust_cmd <= 10.0)
    assert torch.all(result.omega_cmd >= -4.0)
    assert torch.all(result.omega_cmd <= 4.0)
    assert torch.allclose(result.control[:, :1], result.thrust_cmd)
    assert torch.allclose(result.control[:, 1:], result.omega_cmd)


def test_training_and_rollout_use_action_adapter_instead_of_inline_formula():
    train_source = (ROOT / "train" / "main_cuda.py").read_text()
    runner_source = (ROOT / "rollout" / "policy_runner.py").read_text()

    assert "create_action_adapter" in train_source
    assert "action_adapter.to_control" in train_source
    assert "create_action_adapter" in runner_source
    assert "self.action_adapter.to_control" in runner_source
    assert "a_pred - v_pred - env.g_std" not in train_source
    assert "a_pred - v_pred - env.g_std" not in runner_source
