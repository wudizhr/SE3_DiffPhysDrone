import importlib
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_main_with_args(monkeypatch, argv):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SE3_DIFF_SKIP_TRAIN", "1")
    monkeypatch.setattr(sys, "argv", ["train/main_cuda.py", *argv])
    return runpy.run_path(str(ROOT / "train" / "main_cuda.py"), run_name="__main__")


def test_env_package_exports_env():
    sys.path.insert(0, str(ROOT))
    try:
        env = importlib.import_module("env")
        assert "Env" in env.__all__
    finally:
        sys.path.remove(str(ROOT))


def test_main_accepts_yaml_config(monkeypatch):
    globals_ = load_main_with_args(monkeypatch, ["--config", "configs/single_agent.yaml"])

    config = globals_["config"]
    assert config.env.single is True
    assert config.env.speed_mtp == 4
    assert config.env.ground_voxels is True
    assert config.train.num_iters == 50000
    assert config.train.timesteps == 150
    assert config.inference.yaw_target_correction is True
    assert config.inference.ctl_freq == 15
    assert config.loss.coef_collide == 7.5


def test_legacy_args_file_is_removed():
    assert not (ROOT / "configs" / "single_agent.args").exists()


def test_yaml_config_contains_single_agent_training_fields():
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import config_to_flat_args, load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    yaml_config = load_experiment_config(ROOT / "configs" / "single_agent.yaml")
    yaml_args = config_to_flat_args(yaml_config)

    assert yaml_args["single"] is True
    assert yaml_args["speed_mtp"] == 4
    assert yaml_args["coef_d_acc"] == 0.01
    assert yaml_args["coef_d_jerk"] == 0.001
    assert yaml_args["ground_voxels"] is True
    assert yaml_args["random_rotation"] is True
    assert yaml_args["yaw_drift"] is True
    assert yaml_args["yaw_target_correction"] is True
    assert yaml_args["coef_collide"] == 7.5
    assert yaml_args["coef_obj_avoidance"] == 3.0
    assert yaml_args["cam_angle"] == 20
    assert yaml_args["fov_x_half_tan"] == 0.82
    assert yaml_args["num_iters"] == 50000
    assert yaml_args["timesteps"] == 150
    assert yaml_args["ctl_freq"] == 15


def test_num_envs_is_training_parallel_environment_count(tmp_path):
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import config_to_flat_args, load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    config_path = tmp_path / "num_envs.yaml"
    config_path.write_text(
        """
train:
  num_envs: 12
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    flat = config_to_flat_args(config)

    assert config.train.num_envs == 12
    assert config.train.batch_size == 12
    assert flat["num_envs"] == 12
    assert flat["batch_size"] == 12


def test_num_envs_rejects_conflicting_legacy_batch_size(tmp_path):
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    config_path = tmp_path / "conflict.yaml"
    config_path.write_text(
        """
train:
  num_envs: 12
  batch_size: 8
""",
        encoding="utf-8",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "num_envs" in str(exc)
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("conflicting num_envs and batch_size should be rejected")


def test_future_collision_sampling_defaults_to_enabled():
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import config_to_flat_args, load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    config = load_experiment_config(ROOT / "configs" / "single_agent.yaml")
    flat = config_to_flat_args(config)

    assert config.train.use_future_collision_samples is True
    assert flat["use_future_collision_samples"] is True


def test_future_collision_sampling_can_be_disabled(tmp_path):
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    config_path = tmp_path / "current_position_collision.yaml"
    config_path.write_text(
        """
train:
  use_future_collision_samples: false
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.train.use_future_collision_samples is False


def test_training_passes_collision_sampling_config_to_env():
    source = (ROOT / "train" / "main_cuda.py").read_text()

    assert "use_future_samples=config.train.use_future_collision_samples" in source


def test_env_can_query_nearest_point_from_current_position_only():
    source = (ROOT / "env" / "env_cuda.py").read_text()

    assert "def find_vec_to_nearest_pt(self, *, use_future_samples=True)" in source
    assert "query_offsets = self.sub_div if use_future_samples else self.current_pos_div" in source


def test_training_loss_supports_current_position_collision_samples():
    import pytest

    torch = pytest.importorskip("torch")

    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.schema import LossConfig
        from train.loss.training_loss import compute_training_loss
    finally:
        sys.path.remove(str(ROOT))

    timesteps = 35
    batch_size = 2
    zeros = [torch.zeros(batch_size, 3) for _ in range(timesteps)]
    act_buffer = [torch.zeros(batch_size, 3) for _ in range(timesteps + 2)]
    vec_to_pt_history = [torch.ones(1, batch_size, 3) for _ in range(timesteps)]

    result = compute_training_loss(
        loss_config=LossConfig(),
        v_history=zeros,
        target_v_history=zeros,
        v_preds=zeros,
        act_buffer=act_buffer,
        vec_to_pt_history=vec_to_pt_history,
        margin=torch.zeros(batch_size),
    )

    assert result.distance.shape == (timesteps, 1, batch_size)
    assert torch.isfinite(result.loss_obj_avoidance)
    assert torch.isfinite(result.loss_collide)
    assert torch.isfinite(result.loss)


def test_training_rejects_legacy_cli_training_overrides(monkeypatch):
    for argv in (
        ["--num_iters", "7"],
        ["--config", "configs/single_agent.yaml", "--ctl_freq", "20"],
        ["--config", "configs/single_agent.yaml", "--resume", "checkpoints/example.pth"],
        ["--single"],
    ):
        try:
            load_main_with_args(monkeypatch, argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"legacy CLI override should be rejected: {argv}")


def test_depth_guidance_and_recovery_wall_options_are_removed(monkeypatch):
    removed_options = [
        "--coef_depth_guidance",
        "--depth_guidance_blocked_depth",
        "--depth_guidance_sharpness",
        "--depth_guidance_temperature",
        "--recovery_avf_front_dist",
        "--recovery_avf_pass_dist",
        "--recovery_avf_wall_half_width",
        "--recovery_wall_prob",
    ]
    for option in removed_options:
        try:
            load_main_with_args(monkeypatch, [option, "1"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{option} should not be accepted")


def test_depth_guidance_and_recovery_wall_code_is_removed():
    removed_terms = [
        "compute_depth_guidance",
        "compute_recovery_avf_guidance",
        "depth_guidance",
        "recovery_avf",
        "recovery_wall",
        "_append_recovery_wall_voxels",
    ]
    checked_paths = [
        "configs/single_agent.yaml",
        "env/env_cuda.py",
        "train/config_args.py",
        "train/main_cuda.py",
        "visualization/export_env_snapshot.py",
        "visualization/config/scene_example.yaml",
        "visualization/config/scene_gap.yaml",
    ]
    for relative_path in checked_paths:
        source = (ROOT / relative_path).read_text()
        for term in removed_terms:
            assert term not in source


def test_legacy_loss_options_are_removed(monkeypatch):
    removed_options = [
        "--coef_speed",
        "--coef_d_snap",
        "--coef_ground_affinity",
        "--coef_bias",
    ]
    for option in removed_options:
        try:
            load_main_with_args(monkeypatch, [option, "1"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{option} should not be accepted")


def test_legacy_loss_code_is_removed():
    removed_terms = [
        "legacy",
        "coef_speed",
        "coef_d_snap",
        "coef_ground_affinity",
        "coef_bias",
        "loss_speed",
        "loss_d_snap",
        "loss_ground_affinity",
        "loss_bias",
        "snap_history",
    ]
    checked_paths = [
        "configs/single_agent.yaml",
        "train/config_args.py",
        "train/main_cuda.py",
    ]
    for relative_path in checked_paths:
        source = (ROOT / relative_path).read_text()
        for term in removed_terms:
            assert term not in source


def test_visualization_helpers_are_synced_with_training_interface():
    import importlib.util

    module_path = ROOT / "visualization" / "visualization_common.py"
    spec = importlib.util.spec_from_file_location("visualization_common", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    removed_terms = [
        "Model_gap",
        "recovery_wall_prob",
        "coef_speed",
        "coef_d_snap",
        "coef_ground_affinity",
        "coef_bias",
        "depth_guidance",
        "legacy",
    ]
    checked_paths = [
        "visualization/export_env_snapshot.py",
        "visualization/export_policy_rollout.py",
        "visualization/policy_rollout_common.py",
        "visualization/config/offline_example.yaml",
        "visualization/config/scene_example.yaml",
        "visualization/config/scene_gap.yaml",
    ]
    for relative_path in checked_paths:
        source = (ROOT / relative_path).read_text()
        for term in removed_terms:
            assert term not in source


def test_model_name_is_primary_model_selection_field():
    config_text = (ROOT / "configs" / "single_agent.yaml").read_text()
    assert "name: pm_model" in config_text


def test_ros2_online_inference_visualization_files_are_removed():
    removed_paths = [
        "visualization/config/example.yaml",
    ]
    for relative_path in removed_paths:
        assert not (ROOT / relative_path).exists()

    removed_terms = [
        "export_policy_server",
        "online_policy",
        "cmd_vel",
        "checkpoint_topic",
    ]
    for relative_path in [
        "visualization/export_env_snapshot.py",
        "visualization/export_policy_rollout.py",
        "visualization/policy_rollout_common.py",
        "visualization/visualization_common.py",
        "visualization/config/offline_example.yaml",
        "visualization/config/scene_example.yaml",
        "visualization/config/scene_gap.yaml",
    ]:
        source = (ROOT / relative_path).read_text()
        for term in removed_terms:
            assert term not in source


def test_offline_rviz2_playback_helpers_exist_without_online_inference():
    import importlib.util

    common_path = ROOT / "visualization" / "rviz2_common.py"
    player_path = ROOT / "visualization" / "rviz2_play_rollout.py"
    rviz_path = ROOT / "visualization" / "diffphysdrone.rviz"

    assert common_path.exists()
    assert player_path.exists()
    assert rviz_path.exists()

    spec = importlib.util.spec_from_file_location("rviz2_common", common_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("load_yaml", "resolve_path", "load_rollout_npz"):
        assert hasattr(module, name)

    source = player_path.read_text()
    assert "from rviz2_common import" in source
    assert "from model import" not in source
    assert "from env import" not in source
    assert "checkpoint_path" not in source


def test_rviz2_playback_config_accepts_nested_rollout_settings():
    import importlib.util

    player_path = ROOT / "visualization" / "rviz2_play_rollout.py"
    spec = importlib.util.spec_from_file_location("rviz2_play_rollout", player_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {
        "rollout_path": "rollouts/example_rollout.npz",
        "rollout": {
            "rate_hz": 15,
            "frame_id": "map",
            "playback_speed": 2.0,
        },
    }
    playback = module.build_playback_config(config, rollout_rate_hz=10.0, rollout_frame_id="odom")

    assert playback["rate_hz"] == 15
    assert playback["frame_id"] == "map"
    assert playback["playback_speed"] == 2.0
    assert playback["timer_period"] == 1.0 / 30.0

    ctl_dt_playback = module.build_playback_config(
        {"rollout": {"ctl_freq": 20.0, "playback_speed": 2.0}},
        rollout_rate_hz=10.0,
        rollout_frame_id="odom",
    )
    assert ctl_dt_playback["rate_hz"] == 20.0
    assert ctl_dt_playback["timer_period"] == 1.0 / 40.0


def test_rviz2_playback_publishes_drone_pose_as_odometry_not_markers():
    player_source = (ROOT / "visualization" / "rviz2_play_rollout.py").read_text()

    assert "from nav_msgs.msg import Odometry" in player_source
    assert "self.odom_pub = self.create_publisher(Odometry" in player_source
    assert "self.odom_pub.publish(" in player_source
    assert "def make_odometry(" in player_source
    assert "make_body_axis_markers" not in player_source
    assert "drone_body" not in player_source


def test_policy_rollout_exports_and_replays_mid360_pointcloud_when_enabled():
    exporter_source = (ROOT / "visualization" / "export_policy_rollout.py").read_text()
    player_source = (ROOT / "visualization" / "rviz2_play_rollout.py").read_text()
    common_source = (ROOT / "visualization" / "rviz2_common.py").read_text()

    assert "render_mid360" in exporter_source
    assert "record_mid360" in exporter_source
    assert '"mid360_points"' in exporter_source
    assert '"mid360_ranges"' in exporter_source
    assert "from sensor_msgs.msg import Image, PointCloud2" in player_source
    assert "self.mid360_pub = self.create_publisher(PointCloud2" in player_source
    assert "make_pointcloud2(" in player_source
    assert "sensor_msgs_py" not in common_source


def test_policy_rollout_exports_and_replays_mid360_pseudo_image_when_enabled():
    exporter_source = (ROOT / "visualization" / "export_policy_rollout.py").read_text()
    player_source = (ROOT / "visualization" / "rviz2_play_rollout.py").read_text()

    assert "record_mid360_pseudo_image" in exporter_source
    assert "mid360_pseudo_image" in exporter_source
    assert "mid360_pseudo_image_shape" in exporter_source
    assert "self.mid360_pseudo_image_pub = self.create_publisher(Image" in player_source
    assert "self.mid360_pseudo_image_viz_pub = self.create_publisher(Image" in player_source
    assert "make_float_image(mid360_pseudo_image" in player_source
    assert "make_mono8_image(mid360_pseudo_image" in player_source


def test_policy_rollout_exports_training_style_success_statistics():
    exporter_source = (ROOT / "visualization" / "export_policy_rollout.py").read_text()

    assert "clearance_distance_history" in exporter_source
    assert "env.find_vec_to_nearest_pt(" in exporter_source
    assert "success = torch.all(clearance_distance.flatten(0, 1) > 0, 0)" in exporter_source
    assert '"success"' in exporter_source
    assert '"success_rate"' in exporter_source
    assert '"clearance_distance"' in exporter_source


def test_visualization_rollout_config_accepts_ctl_dt():
    import importlib.util

    module_path = ROOT / "visualization" / "export_policy_rollout.py"
    spec = importlib.util.spec_from_file_location("export_policy_rollout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.get_ctl_dt({"ctl_freq": 20.0, "rate_hz": 15.0}) == 0.05
    assert module.get_ctl_dt({"ctl_dt": 0.05, "rate_hz": 15.0}) == 0.05
    assert module.get_ctl_dt({"rate_hz": 20.0}) == 0.05
    assert module.get_rate_hz({"ctl_freq": 20.0}) == 20.0
    assert module.get_rate_hz({"ctl_dt": 0.05}) == 20.0


def test_legacy_checkpoint_inference_config_includes_control_frequency(tmp_path):
    import importlib.util

    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.write_bytes(b"not a real checkpoint")
    (tmp_path / "args.yaml").write_text(
        "ctl_freq: 30.0\nno_odom: false\nyaw_target_correction: true\n",
        encoding="utf-8",
    )
    (tmp_path / "model_info.yaml").write_text(
        "dim_obs: 10\ndim_action: 6\ndepth_pool_kernel: 4\n",
        encoding="utf-8",
    )

    module_path = ROOT / "visualization" / "policy_rollout_common.py"
    spec = importlib.util.spec_from_file_location("policy_rollout_common", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    training_config = module.load_training_inference_config(checkpoint)
    assert training_config["ctl_freq"] == 30.0
    assert training_config["yaw_target_correction"] is True
    assert training_config["dim_obs"] == 10


def test_checkpoint_structured_config_overrides_legacy_flat_files(tmp_path):
    import importlib.util
    import yaml

    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.write_bytes(b"not a real checkpoint")
    (tmp_path / "args.yaml").write_text("ctl_freq: 15\nno_odom: false\n", encoding="utf-8")
    (tmp_path / "model_info.yaml").write_text("dim_obs: 10\ndim_action: 6\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "inference": {"ctl_freq": 60, "no_odom": True, "yaw_target_correction": True},
                "model": {"dim_obs": 7, "dim_action": 6, "depth_pool_kernel": 4},
                "env": {"fov_x_half_tan": 0.82, "cam_angle": 20, "single": True},
            }
        ),
        encoding="utf-8",
    )

    module_path = ROOT / "visualization" / "policy_rollout_common.py"
    spec = importlib.util.spec_from_file_location("policy_rollout_common", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    training_config = module.load_training_inference_config(checkpoint)
    assert training_config["ctl_freq"] == 60
    assert training_config["no_odom"] is True
    assert training_config["dim_obs"] == 7
    assert training_config["fov_x_half_tan"] == 0.82
    assert training_config["model_name"] == "pm_model"
    assert training_config["sensor_name"] == "depth_odom"


def test_model_loader_does_not_treat_sensor_name_as_model_name():
    source = (ROOT / "visualization" / "policy_rollout_common.py").read_text()

    assert 'config.get("name"' not in source
    assert 'config.get("model_type", "pm_model")' in source


def test_checkpoint_flat_config_does_not_expose_ambiguous_name_key():
    sys.path.insert(0, str(ROOT))
    try:
        from se3diff_config.io import checkpoint_config_to_flat, load_experiment_config
    finally:
        sys.path.remove(str(ROOT))

    config = load_experiment_config(ROOT / "configs" / "mid360_cnn.yaml")
    flat = checkpoint_config_to_flat(config)

    assert "name" not in flat
    assert flat["model_name"] == "mid360_cnn_model"
    assert flat["sensor_name"] == "mid360"


def test_latest_final_checkpoint_is_used_when_rollout_config_omits_checkpoint_path(tmp_path):
    import importlib.util
    import os
    import yaml

    older_dir = tmp_path / "checkpoints" / "20260618_010101"
    newer_dir = tmp_path / "checkpoints" / "20260619_010101"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    older_checkpoint = older_dir / "checkpoint_final.pth"
    newer_checkpoint = newer_dir / "checkpoint_final.pth"
    older_checkpoint.write_bytes(b"older")
    newer_checkpoint.write_bytes(b"newer")
    (older_dir / "config.yaml").write_text(
        yaml.safe_dump({"inference": {"ctl_freq": 15}, "model": {"dim_obs": 10}}),
        encoding="utf-8",
    )
    (newer_dir / "config.yaml").write_text(
        yaml.safe_dump({"inference": {"ctl_freq": 40}, "model": {"dim_obs": 10}}),
        encoding="utf-8",
    )
    os.utime(older_checkpoint, (100, 100))
    os.utime(newer_checkpoint, (200, 200))

    module_path = ROOT / "visualization" / "export_policy_rollout.py"
    spec = importlib.util.spec_from_file_location("export_policy_rollout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_path = tmp_path / "visualization" / "config" / "offline.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("scene_path: scene.pt\nrollout_path: rollout.npz\n", encoding="utf-8")

    checkpoint = module.resolve_checkpoint_path({}, config_path)
    assert checkpoint == newer_checkpoint.resolve()


def test_one_shot_scene_rollout_allows_omitted_checkpoint_path():
    import importlib.util

    module_path = ROOT / "visualization" / "export_scene_rollout.py"
    spec = importlib.util.spec_from_file_location("export_scene_rollout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {
        "paths": {
            "scene_path": "scenes/generated.pt",
            "rollout_path": "rollouts/generated.npz",
        },
        "scene": {"mission": {"target_reached_radius": 2.0}},
        "inference": {"ctl_freq": 20.0},
    }
    config_path = ROOT / "visualization" / "config" / "scene_rollout_example.yaml"
    rollout_config = module.build_rollout_config(
        config,
        config_path,
        ROOT / "visualization" / "scenes" / "generated.pt",
    )

    assert "checkpoint_path" not in rollout_config
    assert rollout_config["ctl_freq"] == 20.0


def test_training_writes_structured_config_after_model_metadata_is_known():
    main_source = (ROOT / "train" / "main_cuda.py").read_text()
    dim_obs_assignment = main_source.index("config.model.dim_obs = dim_obs")
    config_yaml_dump = main_source.index("config_to_yaml_dict(config)")

    assert dim_obs_assignment < config_yaml_dump


def test_one_shot_scene_rollout_script_and_config_exist():
    script = ROOT / "visualization" / "export_scene_rollout.py"
    config = ROOT / "visualization" / "config" / "scene_rollout_example.yaml"

    assert script.exists()
    assert config.exists()

    config_text = config.read_text()
    required_terms = [
        "scene_path:",
        "rollout_path:",
        "checkpoint_path:",
        "paths:",
        "inference:",
        "playback:",
        "yaw_target_correction:",
        "ctl_freq:",
        "scene:",
        "rollout:",
    ]
    for term in required_terms:
        assert term in config_text


def test_one_shot_scene_rollout_help_does_not_require_torch():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "visualization" / "export_scene_rollout.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout


def test_one_shot_scene_rollout_stays_offline_only():
    removed_terms = [
        "publish_prefix",
        "body_axis",
        "loop:",
        "exit_on_finish",
    ]
    checked_paths = [
        "visualization/export_scene_rollout.py",
        "visualization/config/scene_rollout_example.yaml",
    ]
    for relative_path in checked_paths:
        source = (ROOT / relative_path).read_text()
        for term in removed_terms:
            assert term not in source


def test_one_shot_scene_rollout_splits_combined_config():
    import importlib.util

    module_path = ROOT / "visualization" / "export_scene_rollout.py"
    spec = importlib.util.spec_from_file_location("export_scene_rollout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {
        "paths": {
            "scene_path": "scenes/generated.pt",
            "rollout_path": "rollouts/generated.npz",
            "checkpoint_path": "checkpoints/policy.pth",
        },
        "scene": {
            "seed": 7,
            "device": "cuda",
            "mission": {"start": [0, 0, 1], "target": [1, 0, 1], "target_reached_radius": 2.0},
            "env": {"single": True},
        },
        "inference": {
            "device": "cpu",
            "ctl_freq": 20.0,
            "max_steps": 12,
        },
    }
    config_path = ROOT / "visualization" / "config" / "scene_rollout_example.yaml"

    scene_config = module.build_scene_config(config, config_path)
    rollout_config = module.build_rollout_config(config, config_path, ROOT / "visualization" / "scenes" / "generated.pt")

    assert scene_config["output"] == "scenes/generated.pt"
    assert scene_config["seed"] == 7
    assert scene_config["mission"]["target_reached_radius"] == 2.0
    assert scene_config["env"]["single"] is True

    assert rollout_config["scene_path"] == str(ROOT / "visualization" / "scenes" / "generated.pt")
    assert rollout_config["rollout_path"] == "rollouts/generated.npz"
    assert rollout_config["checkpoint_path"] == "checkpoints/policy.pth"
    assert rollout_config["device"] == "cpu"
    assert rollout_config["ctl_freq"] == 20.0
    assert rollout_config["target_reached_radius"] == 2.0


def test_one_shot_scene_rollout_uses_config_relative_rollout_path(monkeypatch):
    import importlib.util
    import types

    module_path = ROOT / "visualization" / "export_scene_rollout.py"
    spec = importlib.util.spec_from_file_location("export_scene_rollout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = {}
    config_path = ROOT / "visualization" / "config" / "scene_rollout_example.yaml"
    scene_path = ROOT / "visualization" / "scenes" / "generated.pt"

    export_env_snapshot = types.SimpleNamespace(
        export_scene_snapshot=lambda config, config_path, output_override=None: scene_path
    )

    def fake_export_policy_rollout(config, config_path, output_override=None):
        calls["config"] = config
        calls["config_path"] = config_path
        calls["output_override"] = output_override
        return ROOT / "visualization" / "config" / "rollouts" / "generated.npz"

    export_policy_rollout = types.SimpleNamespace(export_policy_rollout=fake_export_policy_rollout)
    monkeypatch.setitem(sys.modules, "export_env_snapshot", export_env_snapshot)
    monkeypatch.setitem(sys.modules, "export_policy_rollout", export_policy_rollout)

    config = {
        "paths": {
            "scene_path": "scenes/generated.pt",
            "rollout_path": "rollouts/generated.npz",
            "checkpoint_path": "checkpoints/policy.pth",
        },
        "scene": {"mission": {}},
        "inference": {},
    }

    module.export_scene_rollout(config, config_path)

    assert calls["output_override"] is None
    assert calls["config"]["rollout_path"] == "rollouts/generated.npz"


def test_training_loss_calculation_is_split_into_loss_package():
    sys.path.insert(0, str(ROOT))
    try:
        training_loss = importlib.import_module("train.loss.training_loss")
        loss_pkg = importlib.import_module("train.loss")
    finally:
        sys.path.remove(str(ROOT))

    assert hasattr(training_loss, "compute_training_loss")
    assert loss_pkg.compute_training_loss is training_loss.compute_training_loss
    assert not (ROOT / "loss").exists()

    main_source = (ROOT / "train" / "main_cuda.py").read_text()
    loss_source = (ROOT / "train" / "loss" / "training_loss.py").read_text()
    registry_source = (ROOT / "train" / "loss" / "registry.py").read_text()
    obstacle_source = (ROOT / "train" / "loss" / "terms" / "obstacle.py").read_text()
    velocity_source = (ROOT / "train" / "loss" / "terms" / "velocity.py").read_text()

    assert "from train.loss import LossContext, build_loss_registry" in main_source
    assert "loss_registry = build_loss_registry(config.loss)" in main_source
    assert "loss_registry.compute(" in main_source
    assert "def barrier" not in main_source
    assert "loss_v = F.smooth_l1_loss" not in main_source
    assert "loss_collide = F.softplus" not in main_source

    assert "LossContext.from_rollout" in loss_source
    assert "build_loss_registry" in loss_source
    assert "class LossRegistry" in registry_source
    assert "def barrier" in obstacle_source
    assert "F.smooth_l1_loss" in velocity_source
    assert "F.softplus" in obstacle_source
