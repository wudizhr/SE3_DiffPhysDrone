from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_point_mass_dynamics_delegates_to_env_run():
    from env.dynamics import create_dynamics_backend

    class FakeEnv:
        def __init__(self):
            self.calls = []

        def run(self, control, ctl_dt, v_pred):
            self.calls.append((control, ctl_dt, v_pred))

    env = FakeEnv()
    backend = create_dynamics_backend("point_mass")

    backend.step(env, "control", 0.05, "yaw_vec")

    assert backend.name == "point_mass"
    assert env.calls == [("control", 0.05, "yaw_vec")]


def test_ctbr_dynamics_backend_delegates_to_env_run_ctbr():
    from env.dynamics import create_dynamics_backend

    class FakeEnv:
        def __init__(self):
            self.calls = []

        def run_ctbr(self, control, ctl_dt, yaw_vec):
            self.calls.append((control, ctl_dt, yaw_vec))

    backend = create_dynamics_backend("ctbr")
    env = FakeEnv()

    backend.step(env, "control", 0.05, "yaw_vec")

    assert backend.name == "ctbr"
    assert env.calls == [("control", 0.05, "yaw_vec")]


def test_env_run_ctbr_is_differentiable_reference_dynamics():
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("quadsim_cuda")
    from env import Env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("Env initialization currently requires CUDA tensors for quadsim_cuda")

    env = Env(
        2,
        64,
        48,
        0.4,
        device,
        single=True,
        quad_mass=1.0,
        ctbr_linear_drag=0.05,
        ctbr_thrust_max=30.0,
    )
    control = torch.tensor(
        [[9.80665, 0.1, -0.2, 0.3], [11.0, -0.1, 0.2, -0.3]],
        device=device,
        requires_grad=True,
    )

    env.run_ctbr(control, 0.02, None)
    loss = env.p.sum() + env.v.sum() + env.R.sum()
    loss.backward()

    assert control.grad is not None
    assert torch.isfinite(control.grad).all()
    assert env.R.shape == (2, 3, 3)
    assert env.v.shape == (2, 3)


def test_ctbr_cuda_extension_symbols_are_declared_and_built():
    quadsim_source = (ROOT / "src" / "quadsim.cpp").read_text()
    setup_source = (ROOT / "src" / "setup.py").read_text()
    env_source = (ROOT / "env" / "env_cuda.py").read_text()

    assert "run_ctbr_forward_cuda" in quadsim_source
    assert "run_ctbr_backward_cuda" in quadsim_source
    assert 'm.def("run_ctbr_forward"' in quadsim_source
    assert 'm.def("run_ctbr_backward"' in quadsim_source
    assert "'dynamics_ctbr_kernel.cu'" in setup_source
    assert "class RunCtbrFunction" in env_source
    assert "run_ctbr_cuda = RunCtbrFunction.apply" in env_source


def test_run_ctbr_cuda_smoke_returns_gradients_for_state_and_commands():
    import pytest

    torch = pytest.importorskip("torch")
    quadsim_cuda = pytest.importorskip("quadsim_cuda")
    if not torch.cuda.is_available():
        pytest.skip("CTBR CUDA smoke requires CUDA")
    if not hasattr(quadsim_cuda, "run_ctbr_forward"):
        pytest.skip("quadsim_cuda was built before CTBR symbols were added")

    from env.env_cuda import run_ctbr_cuda

    device = torch.device("cuda")
    batch = 2
    R = torch.eye(3, device=device).repeat(batch, 1, 1).requires_grad_(True)
    omega = torch.zeros(batch, 3, device=device, requires_grad=True)
    collective_thrust = torch.full((batch, 1), 9.80665, device=device, requires_grad=True)
    thrust_cmd = torch.tensor([[10.0], [11.0]], device=device, requires_grad=True)
    omega_cmd = torch.tensor([[0.1, -0.2, 0.3], [-0.2, 0.1, -0.1]], device=device, requires_grad=True)
    mass = torch.ones(batch, 1, device=device)
    dg = torch.zeros(batch, 3, device=device)
    p = torch.zeros(batch, 3, device=device, requires_grad=True)
    v = torch.zeros(batch, 3, device=device, requires_grad=True)
    v_wind = torch.zeros(batch, 3, device=device)
    a = torch.zeros(batch, 3, device=device, requires_grad=True)

    R_next, omega_next, thrust_next, p_next, v_next, a_next = run_ctbr_cuda(
        R,
        omega,
        collective_thrust,
        thrust_cmd,
        omega_cmd,
        mass,
        dg,
        p,
        v,
        v_wind,
        a,
        0.4,
        0.02,
        0.03,
        0.05,
        0.02,
    )
    loss = (
        R_next.sum()
        + omega_next.sum()
        + thrust_next.sum()
        + p_next.sum()
        + v_next.sum()
        + a_next.sum()
    )
    loss.backward()

    for tensor in (R, omega, collective_thrust, thrust_cmd, omega_cmd, p, v, a):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_mid360_se3_training_components_share_ctbr_data_path():
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("quadsim_cuda")
    if not torch.cuda.is_available():
        pytest.skip("MID360 CTBR data path smoke requires CUDA")

    from control import create_action_adapter
    from env import Env
    from env.dynamics import create_dynamics_backend
    from model import create_model
    from sensors import create_observation_builder
    from se3diff_config.io import load_experiment_config

    config = load_experiment_config(ROOT / "configs" / "mid360_se3.yaml")
    config.train.num_envs = 2
    config.train.batch_size = 2
    env = Env(
        2,
        config.env.width,
        config.env.height,
        config.env.grad_decay,
        torch.device("cuda"),
        single=True,
        quad_mass=config.env.quad_mass,
        quad_mass_randomization=0.0,
        ctbr_thrust_max=config.env.ctbr_thrust_max,
    )
    observation_builder = create_observation_builder(config.sensor)
    model = create_model(config.model).to(env.device)
    action_adapter = create_action_adapter(config.model.action_mode)
    dynamics_backend = create_dynamics_backend(config.model.backend_name)

    target_v_raw = env.p_target - env.p.detach()
    fwd = env.R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = torch.nn.functional.normalize(fwd, 2, -1)
    policy_frame = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)
    target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True).clamp_min(1e-6)
    target_v = target_v_raw / target_v_norm * torch.minimum(target_v_norm, env.max_speed)
    local_v = torch.squeeze(env.v[:, None] @ policy_frame, 1)
    target_v_local = torch.squeeze(target_v[:, None] @ policy_frame, 1)
    state = torch.cat([local_v, target_v_local, env.R.reshape(env.batch_size, 9), env.margin[:, None]], -1)
    sensor_inputs = observation_builder.render_inputs(env, 0.1)
    obs = observation_builder.build(sensor_inputs=sensor_inputs, state=state)
    raw_action, _, hidden = model(obs)
    adapted = action_adapter.to_control(raw_action, env, policy_frame)
    dynamics_backend.step(env, adapted.control, 0.02, target_v_raw)

    assert raw_action.shape == (2, 4)
    assert adapted.control.shape == (2, 4)
    assert hidden.shape[0] == 2
    assert env.omega.shape == (2, 3)
    assert env.collective_thrust.shape == (2, 1)


def test_training_and_rollout_use_dynamics_backend_instead_of_env_run():
    train_source = (ROOT / "train" / "main_cuda.py").read_text()
    runner_source = (ROOT / "rollout" / "policy_runner.py").read_text()

    assert "create_dynamics_backend" in train_source
    assert "create_dynamics_backend(config.model.backend_name)" in train_source
    assert "dynamics_backend.step" in train_source
    assert "create_dynamics_backend" in runner_source
    assert "create_dynamics_backend(config.backend_name)" in runner_source
    assert "self.dynamics_backend.step" in runner_source
    assert "env.run(act_buffer[t]" not in train_source
    assert "env.run(self.act_buffer[step_idx]" not in runner_source
    assert "create_dynamics_backend(config.model_name)" not in runner_source
