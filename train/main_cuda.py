from collections import defaultdict
import math
import os
import sys
from random import normalvariate
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datetime import datetime
import yaml
from train.loss import compute_training_loss
from train.config_args import parse_train_args
from se3diff_config.env_factory import create_env
from se3diff_config.io import config_to_flat_args, config_to_yaml_dict
from sensors import create_observation_builder
from control import create_action_adapter


# 主训练脚本：在 CUDA 批量环境中 rollout 无人机轨迹，
# 用速度跟踪、避障、碰撞和控制平滑等 loss 训练视觉控制网络。
args = parse_train_args()
config = args.structured_config


def normalize_checkpoint_keys(state_dict):
    """兼容旧 checkpoint 的参数名，同时保持新模型命名。"""

    key_map = {
        'v_proj.weight': 'observation_fc.weight',
        'v_proj.bias': 'observation_fc.bias',
        'fc.weight': 'action_fc.weight',
    }
    return {key_map.get(k, k): v for k, v in state_dict.items()}


def depth_pool_kernel_for_model():
    """pm_model 沿用 4x4 降采样。"""

    return 4


def build_policy_state(env, target_v_raw, use_odom, *, full_attitude: bool = False):
    import torch
    from torch.nn import functional as F

    fwd = env.R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

    target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
    target_v_unit = target_v_raw / target_v_norm
    target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

    target_v_local = torch.squeeze(target_v[:, None] @ R, 1)
    attitude = env.R.reshape(env.batch_size, 9) if full_attitude else env.R[:, 2]
    state = [target_v_local, attitude, env.margin[:, None]]
    local_v = torch.squeeze(env.v[:, None] @ R, 1)
    if use_odom:
        state.insert(0, local_v)
    state = torch.cat(state, -1)
    return R, state, local_v, target_v



def main():
    from matplotlib import pyplot as plt
    import torch
    from torch.nn import functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from env import Env
    from model import create_model

    checkpoint_root = Path(config.train.save_dir).expanduser() / 'checkpoints'
    training_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoints_dir = checkpoint_root / training_start_time
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(checkpoints_dir / 'runs')

    device = torch.device('cuda')

    # Env 负责随机生成场景、渲染深度图、查询最近障碍物和推进可微动力学。
    env = create_env(Env, config.env, batch_size=config.train.num_envs, device=device)
    observation_builder = create_observation_builder(config.sensor)
    dim_obs = observation_builder.dim_state
    dim_action = config.model.dim_action
    config.inference.no_odom = not config.sensor.use_odom
    config.model.dim_obs = dim_obs
    model = create_model(config.model)
    model = model.to(device)
    action_adapter = create_action_adapter(config.model.action_mode)
    depth_pool_kernel = observation_builder.depth_pool_kernel
    config.model.model_class = model.__class__.__name__
    config.model.dim_obs = dim_obs
    config.model.dim_action = dim_action
    config.model.depth_pool_kernel = depth_pool_kernel
    config.model.model_type = config.model.name
    model_info_path = checkpoints_dir / 'model_info.yaml'
    with model_info_path.open('w') as f:
        yaml.safe_dump({
            'model_type': config.model.model_type,
            'model_class': model.__class__.__name__,
            'dim_obs': dim_obs,
            'dim_action': dim_action,
            'depth_pool_kernel': depth_pool_kernel,
        }, f, sort_keys=True)
    args_path = checkpoints_dir / 'args.yaml'
    with args_path.open('w') as f:
        yaml.safe_dump(config_to_flat_args(config), f, sort_keys=True)
    config_path = checkpoints_dir / 'config.yaml'
    with config_path.open('w') as f:
        yaml.safe_dump(config_to_yaml_dict(config), f, sort_keys=True)
    print(yaml.safe_dump(config_to_yaml_dict(config), sort_keys=True))
    print(f"Saving checkpoints and logs to: {checkpoints_dir.resolve()}")
    print(f"Saved training args to: {args_path.resolve()}")

    if config.train.resume:
        state_dict = torch.load(config.train.resume, map_location=device)
        state_dict = normalize_checkpoint_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        if missing_keys:
            print("missing_keys:", missing_keys)
        if unexpected_keys:
            print("unexpected_keys:", unexpected_keys)
    optim = AdamW(model.parameters(), config.train.lr)
    sched = CosineAnnealingLR(optim, config.train.num_iters, config.train.lr * 0.01)

    scaler_q = defaultdict(list)
    def smooth_dict(ori_dict):
        """先缓存若干 step 的标量，后面按平均值写入 TensorBoard。"""

        for k, v in ori_dict.items():
            scaler_q[k].append(float(v))

    def is_save_iter(i):
        """前期更频繁保存可视化，后期降低频率。"""

        if i < 2000:
            return (i + 1) % 250 == 0
        return (i + 1) % 1000 == 0

    pbar = tqdm(range(config.train.num_iters), ncols=80)
    # depths = []
    # states = []保证
    B = config.train.num_envs
    for i in pbar:
        # 每个 iteration 都重新随机生成一批环境，并清空模型的循环隐藏状态。
        env.reset()
        model.reset()

        # 这些 history 用来在 rollout 结束后统一计算轨迹级 loss。
        p_history = []
        v_history = []
        target_v_history = []
        vec_to_pt_history = []
        v_preds = []
        v_net_feats = []
        h = None

        # 模拟一拍控制延迟：当前环境执行 act_buffer[t]，而网络输出 append 到队尾。
        act_lag = 1
        act_buffer = [env.act] * (act_lag + 1)
        target_v_raw = env.p_target - env.p
        if config.train.yaw_drift:
            # 可选目标方向漂移，用来训练策略对 yaw 偏差更鲁棒。
            drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            R_drift = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av) , zeros,
                zeros              , zeros               , ones ,
            ], -1).reshape(B, 3, 3)


        for t in range(config.train.timesteps):
            # 控制周期带一点随机抖动，避免策略只适配固定 dt。
            ctl_dt = normalvariate(1.0 / config.inference.ctl_freq, 0.1 / config.inference.ctl_freq)

            p_history.append(env.p)
            vec_to_pt_history.append(
                env.find_vec_to_nearest_pt(
                    use_future_samples=config.train.use_future_collision_samples
                )
            )


            if config.train.yaw_drift:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            else:
                target_v_raw = env.p_target - env.p.detach()

            # 先用上一拍控制推进环境；当前时刻新动作在后面由模型产生。
            # 默认用真实速度方向修正 yaw；开启参数后改用目标方向修正 yaw。
            yaw_correction_vec = target_v_raw if config.inference.yaw_target_correction else env.v
            env.run(act_buffer[t], ctl_dt, yaw_correction_vec)

            # 状态输入包含目标速度局部表达、机体 up 向量和安全距离 margin；
            # 如果启用 odom，还加入当前速度的局部表达。
            R, state, local_v, target_v = build_policy_state(
                env,
                target_v_raw,
                config.sensor.use_odom,
                full_attitude=config.sensor.name == "mid360",
            ) 

            sensor_inputs = observation_builder.render_inputs(env, ctl_dt)
            obs = observation_builder.build(sensor_inputs=sensor_inputs, state=state)
            if config.sensor.name == "mid360":
                x = obs["mid360_pseudo_image"]
                act, _, h = model(obs, hx=h)
            else:
                depth = obs["depth"]
                # 深度图预处理：近处更亮，裁剪极近/极远深度，加入噪声，再池化降采样。
                x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
                x = F.max_pool2d(x[:, None], depth_pool_kernel, depth_pool_kernel)
                act, _, h = model(x, state, h)

            # 动作语义由 adapter 统一处理；accel_velocity 保持旧的 6D 动作公式。
            adapted_action = action_adapter.to_control(act, env, R)
            v_preds.append(adapted_action.v_pred)
            act_buffer.append(adapted_action.control)
            v_net_feats.append(torch.cat([adapted_action.control, local_v, h], -1))

            v_history.append(env.v)
            target_v_history.append(target_v)

        p_history = torch.stack(p_history)

        loss_result = compute_training_loss(
            loss_config=config.loss,
            v_history=v_history,
            target_v_history=target_v_history,
            v_preds=v_preds,
            act_buffer=act_buffer,
            vec_to_pt_history=vec_to_pt_history,
            margin=env.margin,
        )

        if torch.isnan(loss_result.loss):
            print("loss is nan, exiting...")
            exit(1)

        pbar.set_description_str(f'loss: {loss_result.loss:.3f}')
        optim.zero_grad()
        # 反向会穿过模型和 Env.run 的自定义 CUDA backward。
        loss_result.loss.backward()
        optim.step()
        sched.step()


        with torch.no_grad():
            # 统计成功率：整条轨迹所有采样距离都大于 0 视为未碰撞。
            avg_speed = loss_result.speed_history.mean(0)
            success = torch.all(loss_result.distance.flatten(0, 1) > 0, 0)
            _success = success.sum() / B
            smooth_dict(loss_result.tensorboard_scalars(_success, avg_speed))
            log_dict = {}
            if is_save_iter(i):
                # 可视化 batch 中第 4 个样本的位置、速度和控制历史。
                # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
                fig_p, ax = plt.subplots()
                p_history = p_history[:, 4].cpu()
                ax.plot(p_history[:, 0], label='x')
                ax.plot(p_history[:, 1], label='y')
                ax.plot(p_history[:, 2], label='z')
                ax.legend()
                fig_v, ax = plt.subplots()
                v_history_plot = loss_result.v_history[:, 4].cpu()
                ax.plot(v_history_plot[:, 0], label='x')
                ax.plot(v_history_plot[:, 1], label='y')
                ax.plot(v_history_plot[:, 2], label='z')
                ax.legend()
                fig_a, ax = plt.subplots()
                act_buffer_plot = loss_result.act_buffer[:, 4].cpu()
                ax.plot(act_buffer_plot[:, 0], label='x')
                ax.plot(act_buffer_plot[:, 1], label='y')
                ax.plot(act_buffer_plot[:, 2], label='z')
                ax.legend()
                # writer.add_video('demo', vid, i + 1, 15)
                writer.add_figure('p_history', fig_p, i + 1)
                writer.add_figure('v_history', fig_v, i + 1)
                writer.add_figure('a_reals', fig_a, i + 1)
            if (i + 1) % 10000 == 0:
                # 周期性保存模型权重到本次训练的时间戳目录。
                torch.save(model.state_dict(), checkpoints_dir / f'checkpoint{i//10000:04d}.pth')
            if (i + 1) % 25 == 0:
                # 将累计的标量均值写入 TensorBoard。
                for k, v in scaler_q.items():
                    writer.add_scalar(k, sum(v) / len(v), i + 1)
                scaler_q.clear()

    final_checkpoint_path = checkpoints_dir / "checkpoint_final.pth"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Saved final checkpoint to: {final_checkpoint_path.resolve()}")


if __name__ == '__main__' and os.environ.get('SE3_DIFF_SKIP_TRAIN') != '1':
    main()
