from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_loss_package_exposes_registry_extension_points():
    from train.loss import LossContext, LossRegistry, LossTerm, build_loss_registry

    assert LossContext.__name__ == "LossContext"
    assert LossRegistry.__name__ == "LossRegistry"
    assert LossTerm.__name__ == "LossTerm"
    assert callable(build_loss_registry)


def test_yaw_alignment_loss_is_registered_as_extension_term():
    registry_source = (ROOT / "train" / "loss" / "registry.py").read_text()
    yaw_source = (ROOT / "train" / "loss" / "terms" / "attitude.py").read_text()
    main_source = (ROOT / "train" / "main_cuda.py").read_text()

    assert '"yaw_alignment": YawAlignmentLoss' in registry_source
    assert '"target_yaw_alignment": TargetYawAlignmentLoss' in registry_source
    assert "class YawAlignmentLoss" in yaw_source
    assert "class TargetYawAlignmentLoss" in yaw_source
    assert "R_history=R_history" in main_source


def test_ctbr_control_losses_are_registered_as_extension_terms():
    registry_source = (ROOT / "train" / "loss" / "registry.py").read_text()
    control_source = (ROOT / "train" / "loss" / "terms" / "control.py").read_text()
    context_source = (ROOT / "train" / "loss" / "context.py").read_text()
    main_source = (ROOT / "train" / "main_cuda.py").read_text()

    assert '"thrust_regularization": ThrustRegularizationLoss' in registry_source
    assert '"ctbr_smoothness": CtbrSmoothnessLoss' in registry_source
    assert "class ThrustRegularizationLoss" in control_source
    assert "class CtbrSmoothnessLoss" in control_source
    assert "mass:" in context_source
    assert "action_mode:" in context_source
    assert "mass=env.mass" in main_source
    assert "action_mode=config.model.action_mode" in main_source


def test_se3_configs_enable_target_yaw_alignment_loss():
    from se3diff_config.io import load_experiment_config
    from train.loss.registry import build_loss_registry

    for relative_path in ("configs/depth_se3.yaml", "configs/mid360_se3.yaml"):
        config = load_experiment_config(ROOT / relative_path)
        enabled_names = {term.name for term in build_loss_registry(config.loss).terms if term.enabled}

        assert "target_yaw_alignment" in enabled_names


def test_loss_curriculum_scheduler_is_exposed_in_registry_and_training_loop():
    registry_source = (ROOT / "train" / "loss" / "registry.py").read_text()
    main_source = (ROOT / "train" / "main_cuda.py").read_text()

    assert "class LossWeightSchedule" in registry_source
    assert "def weight_at" in registry_source
    assert "step: int | None" in registry_source
    assert "loss_registry.compute(loss_context, step=i)" in main_source


def _make_context(torch):
    from train.loss.context import LossContext

    timesteps = 40
    batch_size = 2
    v_history = [torch.zeros(batch_size, 3) for _ in range(timesteps)]
    target_v_history = [torch.ones(batch_size, 3) * 0.2 for _ in range(timesteps)]
    v_preds = [torch.zeros(batch_size, 3) for _ in range(timesteps)]
    act_buffer = [torch.zeros(batch_size, 3) for _ in range(timesteps + 2)]
    vec_to_pt_history = [torch.ones(1, batch_size, 3) for _ in range(timesteps)]
    R_history = [torch.eye(3).repeat(batch_size, 1, 1) for _ in range(timesteps)]

    return LossContext.from_rollout(
        v_history=v_history,
        target_v_history=target_v_history,
        v_preds=v_preds,
        act_buffer=act_buffer,
        vec_to_pt_history=vec_to_pt_history,
        margin=torch.zeros(batch_size),
        R_history=R_history,
        mass=torch.ones(batch_size, 1),
        action_mode="accel_velocity",
    )


def test_loss_config_builds_default_registry_from_legacy_coefficients():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss.registry import build_loss_registry

    registry = build_loss_registry(LossConfig())
    context = _make_context(torch)
    result = registry.compute(context)

    assert registry.required_history() == set()
    assert set(result.terms) == {
        "loss_v",
        "loss_v_pred",
        "loss_obj_avoidance",
        "loss_d_acc",
        "loss_d_jerk",
        "loss_collide",
    }
    assert result.loss is result.total
    scalars = result.tensorboard_scalars(torch.tensor(1.0), torch.ones(2))
    assert "loss_v" in scalars
    assert "loss_collide" in scalars


def test_loss_terms_can_be_disabled_from_structured_config():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss.registry import build_loss_registry

    loss_config = LossConfig(
        terms={
            "velocity_tracking": {"weight": 1.0, "enabled": False},
            "collision": {"weight": 7.5, "enabled": True},
        }
    )
    result = build_loss_registry(loss_config).compute(_make_context(torch))

    assert "loss_v" not in result.terms
    assert "loss_collide" in result.terms
    assert "loss_v" not in result.tensorboard_scalars(torch.tensor(1.0), torch.ones(2))


def test_loss_term_curriculum_linearly_increases_weight():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss.registry import build_loss_registry

    registry = build_loss_registry(
        LossConfig(
            terms={
                "collision": {
                    "weight": 4.0,
                    "enabled": True,
                    "curriculum": {
                        "start_weight": 0.0,
                        "end_weight": 4.0,
                        "start_iter": 10,
                        "end_iter": 30,
                    },
                }
            }
        )
    )
    context = _make_context(torch)

    early = registry.compute(context, step=0)
    middle = registry.compute(context, step=20)
    late = registry.compute(context, step=30)

    raw = early.terms["loss_collide"]
    assert torch.isclose(early.weighted_terms["loss_collide"], raw * 0.0)
    assert torch.isclose(middle.weighted_terms["loss_collide"], raw * 2.0)
    assert torch.isclose(late.weighted_terms["loss_collide"], raw * 4.0)
    assert early.weights["loss_collide"] == 0.0
    assert middle.weights["loss_collide"] == 2.0
    assert late.weights["loss_collide"] == 4.0


def test_custom_loss_terms_declare_extra_history_requirements():
    import pytest

    pytest.importorskip("torch")
    from train.loss.registry import LossRegistry
    from train.loss.terms.base import LossTerm

    class FakeImageLoss(LossTerm):
        name = "fake_image"
        log_name = "loss_fake_image"
        required_history = frozenset({"mid360_pseudo_image"})

        def compute(self, context):
            return context.v.new_tensor(2.0)

    registry = LossRegistry()
    registry.register(FakeImageLoss(weight=0.5))

    assert registry.required_history() == {"mid360_pseudo_image"}


def test_yaw_alignment_loss_matches_body_x_velocity_alignment_formula():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss import LossContext, build_loss_registry

    v_history = [
        torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
    ]
    R = torch.eye(3).repeat(3, 1, 1)
    context = LossContext.from_rollout(
        v_history=v_history,
        target_v_history=[torch.zeros(3, 3)],
        v_preds=[],
        act_buffer=[torch.zeros(3, 3), torch.zeros(3, 3)],
        vec_to_pt_history=[torch.ones(1, 3, 3)],
        margin=torch.zeros(3),
        R_history=[R],
        mass=torch.ones(3, 1),
        action_mode="accel_velocity",
    )
    result = build_loss_registry(
        LossConfig(terms={"yaw_alignment": {"weight": 1.0, "enabled": True}})
    ).compute(context)

    assert torch.isclose(result.terms["loss_yaw_alignment"], torch.tensor(1.0))
    assert torch.isclose(result.loss, torch.tensor(1.0))


def test_target_yaw_alignment_loss_matches_body_x_target_xy_formula():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss import LossContext, build_loss_registry

    R = torch.zeros(3, 3, 3)
    R[0, :, 0] = torch.tensor([1.0, 0.0, 10.0])
    R[1, :, 0] = torch.tensor([0.0, 1.0, 10.0])
    R[2, :, 0] = torch.tensor([-1.0, 0.0, 10.0])
    target_v = torch.tensor([
        [2.0, 0.0, 5.0],
        [2.0, 0.0, -5.0],
        [2.0, 0.0, 0.0],
    ])
    context = LossContext.from_rollout(
        v_history=[torch.zeros(3, 3)],
        target_v_history=[target_v],
        v_preds=[],
        act_buffer=[torch.zeros(3, 3), torch.zeros(3, 3)],
        vec_to_pt_history=[torch.ones(1, 3, 3)],
        margin=torch.zeros(3),
        R_history=[R],
        mass=torch.ones(3, 1),
        action_mode="accel_velocity",
    )
    result = build_loss_registry(
        LossConfig(terms={"target_yaw_alignment": {"weight": 1.0, "enabled": True}})
    ).compute(context)

    assert torch.isclose(result.terms["loss_target_yaw_alignment"], torch.tensor(1.0))
    assert torch.isclose(result.loss, torch.tensor(1.0))


def test_ctbr_control_losses_convert_rewards_to_minimized_losses():
    import pytest

    torch = pytest.importorskip("torch")
    from se3diff_config.schema import LossConfig
    from train.loss import LossContext, build_loss_registry

    action = torch.tensor([
        [[9.80665, 1.0, 2.0, 2.0], [11.80665, 0.0, 0.0, 1.0]],
        [[8.80665, 0.0, 0.0, 0.0], [9.80665, 0.0, 3.0, 4.0]],
    ])
    context = LossContext.from_rollout(
        v_history=[torch.zeros(2, 3), torch.zeros(2, 3)],
        R_history=[torch.eye(3).repeat(2, 1, 1), torch.eye(3).repeat(2, 1, 1)],
        target_v_history=[torch.zeros(2, 3), torch.zeros(2, 3)],
        v_preds=[],
        act_buffer=[action[0], action[1]],
        vec_to_pt_history=[torch.ones(1, 2, 3), torch.ones(1, 2, 3)],
        margin=torch.zeros(2),
        mass=torch.ones(2, 1),
        action_mode="ctbr",
    )
    result = build_loss_registry(
        LossConfig(
            terms={
                "thrust_regularization": {"weight": 1.0, "enabled": True},
                "ctbr_smoothness": {"weight": 1.0, "enabled": True},
            }
        )
    ).compute(context)

    expected_thrust = (torch.abs(action[..., :1] - 9.80665)).mean()
    expected_smoothness = torch.norm(action[..., 1:4], 2, -1).mean()
    expected_smoothness = expected_smoothness + torch.norm(action.diff(1, 0), 2, -1).mean()

    assert torch.isclose(result.terms["loss_thrust_regularization"], expected_thrust)
    assert torch.isclose(result.terms["loss_ctbr_smoothness"], expected_smoothness)
