# SE3_Diff 项目说明

SE3_Diff 是一个基于 CUDA 批量仿真环境的无人机视觉避障训练与离线可视化项目。训练脚本在随机障碍物场景中 rollout 无人机轨迹，用速度跟踪、避障、碰撞和控制平滑等损失训练深度策略网络；可视化脚本负责导出场景、导出策略 rollout，并用 RViz2 离线播放。

## 目录结构

```text
SE3_Diff/
  configs/                 # 训练配置，推荐使用 YAML
  env/                     # CUDA 批量无人机环境封装
  model/                   # 策略网络定义
  sensors/                 # 传感器输入与 observation 构造扩展点
  env/dynamics/            # 动力学后端抽象，当前 point-mass 行为由这里统一封装
  se3diff_config/          # 训练与可视化共享配置系统
  src/                     # quadsim_cuda 扩展源码
  tests/                   # 回归测试
  train/                   # 训练入口和 loss
  util/                    # 跨模块轻量工具预留目录
  visualization/           # 场景导出、rollout 导出、RViz2 播放
  checkpoints/             # 训练输出目录
```

## 环境准备

本项目依赖 PyTorch、CUDA 编译工具链、PyYAML、tqdm、matplotlib、TensorBoard 等 Python 包。当前仓库没有固定依赖清单，建议在已安装 PyTorch CUDA 环境中运行。

当前机器推荐使用 `diffphysdrone` conda 环境：

```bash
conda activate diffphysdrone
cd /home/zhr/SE3_Diff
```

先编译并以 editable 方式安装 CUDA 扩展：

```bash
cd /home/zhr/SE3_Diff/src

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

python3 setup.py build_ext --inplace
python3 -m pip install -e . --no-build-isolation

cd /home/zhr/SE3_Diff
```

这里显式指定 `/usr/bin/gcc` 和 `/usr/bin/g++`，是为了避开 `diffphysdrone` 环境中 conda 自带的 `x86_64-conda-linux-gnu-c++ 14.3.0`。PyTorch CUDA 12.8 编译扩展要求 host C++ compiler 版本 `<14.0`，系统 `g++ 13.3` 可以通过检查。

验证扩展是否安装到当前环境：

```bash
python3 -c "import quadsim_cuda; print(quadsim_cuda.__file__); print(hasattr(quadsim_cuda, 'render_mid360')); print(hasattr(quadsim_cuda, 'run_ctbr_forward'))"
```

期望输出路径指向 `/home/zhr/SE3_Diff/src/quadsim_cuda...so`，并且后两行都是 `True`。

如果 Python 环境中没有 `python` 命令，请使用 `python3`。

## 训练

推荐使用统一 YAML 配置启动训练：

```bash
cd /home/zhr/SE3_Diff
python3 train/main_cuda.py --config configs/single_agent.yaml
```

训练参数只从 YAML 读取。需要修改迭代次数、控制频率、损失权重或恢复训练时，请编辑 `configs/single_agent.yaml`。

恢复训练时，在 YAML 中设置：

```yaml
train:
  resume: checkpoints/某次训练/checkpoint_final.pth
```

如果希望避障/碰撞 loss 只根据当前真实位置计算最近距离，而不是默认沿当前速度外推 10 个短时刻采样，可以设置：

```yaml
train:
  use_future_collision_samples: false
```

如果希望训练早期先用少量障碍物，随后逐步增加到最终场景复杂度，可以在 `env` 下启用障碍物课程学习。`n_balls/n_voxels/n_cyl/n_cyl_h/n_ground_voxels` 仍表示最终数量，`start_counts` 表示训练初期数量：

```yaml
env:
  n_balls: 20
  n_voxels: 20
  n_cyl: 0
  n_cyl_h: 2
  n_ground_voxels: 10
  obstacle_curriculum:
    enabled: true
    start_iter: 0
    end_iter: 12000
    start_counts:
      n_balls: 2
      n_voxels: 2
      n_cyl: 0
      n_cyl_h: 0
      n_ground_voxels: 1
```

训练循环会在每次 `env.reset()` 前按当前 iteration 计算有效障碍物数量；可视化和离线场景默认不传训练 step，因此使用最终数量，方便复现。

### Loss 扩展

训练 loss 使用 `train/loss/` 下的扩展结构：

- `LossContext`：把一次 rollout 的 history 整理成 tensor。
- `LossRegistry`：根据配置启用 loss term，并统一加权求和。
- `train/loss/terms/`：每个文件放一类 loss，例如速度、避障、控制平滑。

旧配置中的 `coef_*` 字段仍然可用。推荐新实验逐步改成 `terms` 写法：

```yaml
loss:
  terms:
    velocity_tracking:
      weight: 1.0
      enabled: true
    collision:
      weight: 7.5
      enabled: true
    action_l2:
      weight: 0.01
      enabled: true
    yaw_alignment:
      weight: 0.1
      enabled: true
    thrust_regularization:
      weight: 0.01
      enabled: true
      curriculum:
        start_weight: 0.0
        end_weight: 0.01
        start_iter: 1000
        end_iter: 10000
    ctbr_smoothness:
      weight: 0.01
      enabled: true
```

`curriculum` 是可选字段，用来让指定 loss 的实际权重随训练迭代线性变化；没有该字段时直接使用 `weight`。

`yaw_alignment` 对应 `loss_yaw_alignment = mean(1 - x_body · v_unit)`，用于鼓励机体 x 轴和实际速度方向对齐。`thrust_regularization` 对应 `mean(|T / mass - g|)`，用于惩罚 CTBR 总推力偏离悬停加速度；`ctbr_smoothness` 对应 `mean(||omega||) + mean(||u_t - u_{t-1}||)`，用于抑制角速度和控制跳变。新增 loss 时，在 `train/loss/terms/` 中继承 `LossTerm`，实现 `compute(context)`，再在 `train/loss/registry.py` 的 `TERM_CLASSES` 注册名字。普通状态 history 由训练循环收集；深度图、MID360 伪图像、隐藏状态等大 tensor 由 loss term 的 `required_history` 显式声明后按需收集。

训练输出会写入：

```text
checkpoints/YYYYMMDD_HHMMSS/
  args.yaml              # 扁平兼容参数
  config.yaml            # 完整结构化配置
  model_info.yaml        # 模型输入输出维度等信息
  checkpoint*.pth        # 训练过程权重
  checkpoint_final.pth   # 最终权重
  runs/                  # TensorBoard 日志
```

查看 TensorBoard：

```bash
tensorboard --logdir checkpoints
```

## 配置系统

训练和可视化共用 `se3diff_config/` 中的配置结构。推荐配置入口：

- `configs/default.yaml`：全局默认值。
- `configs/single_agent.yaml`：单机训练配置，也是训练入口参数的唯一来源。
- `visualization/config/scene_rollout_example.yaml`：一键生成场景并导出 rollout 的示例。

配置分组含义见 [配置说明](docs/配置说明.md)。

## 传感器和模型扩展

项目已经预留传感器和模型工厂骨架：

- `sensors/`：负责把不同传感器输入整理成模型 observation。当前接入 `depth_odom`，`mid360` 已实现点云到单通道距离伪图像的预处理方法，后续可接入专用点云/伪图像策略模型。
- `model/factory.py`：负责根据 YAML 中的 `model.name` 创建模型。当前接入 `pm_model`、`depth_se3_model`、`mid360_cnn_model`、`mid360_se3_model`；`depth_se3_model` 使用 `depth_odom` 深度图和里程计状态输出 CTBR 控制，配置入口是 `configs/depth_se3.yaml`。
- `control/`：负责把模型输出的原始 action 转成环境可执行控制量。当前 `accel_velocity` adapter 保持原 point-mass 策略行为不变，后续 CTBR 控制应从这里接入。
- `env/dynamics/`：负责把环境推进逻辑封装成统一后端。当前 `PointMassDynamics` 是对 `Env.run(...)` 的薄包装，`CtbrDynamics` 是对 `Env.run_ctbr(...)` 的薄包装；训练和可视化都会从 `model.backend_name` 选择后端并通过 `DynamicsBackend.step(...)` 调用。
- `rollout/PolicyRunner`：集中策略播放时的 observation 构造、模型调用、action adapter 调用和 dynamics backend 调用，避免训练与可视化各写一套 rollout。

无人机质量属于环境物理参数，配置在 `env.quad_mass`；训练时可用 `env.quad_mass_randomization` 做质量随机化，CTBR 后端会从 `env.mass` 读取实际 batch 质量。

CTBR 刚体动力学已经接入 CUDA 可微核：

- CUDA 文件：`src/dynamics_ctbr_kernel.cu`。
- Python autograd 包装：`env.env_cuda.RunCtbrFunction`。
- 扩展符号：`quadsim_cuda.run_ctbr_forward` 和 `quadsim_cuda.run_ctbr_backward`。
- 当前前向包括一阶角速度/推力响应、Rodrigues 姿态积分、推力加速度、重力、外部扰动和线性阻力；暂不包含 `airmode_av2a`。
- 如果扩展尚未重编译或不在 CUDA tensor 上运行，`Env.run_ctbr()` 会回退到 PyTorch 参考实现。

新增实验时优先复制 YAML，并修改 `sensor` 和 `model` 分组，而不是复制训练脚本。

## 可视化流程

生成一个场景快照：

```bash
python3 visualization/export_env_snapshot.py --config visualization/config/scene_example.yaml
```

使用 checkpoint 在场景中导出离线 rollout：

```bash
python3 visualization/export_policy_rollout.py --config visualization/config/offline_example.yaml
```

如果可视化配置中没有写 `checkpoint_path`，脚本会自动使用 `checkpoints/` 下修改时间最新的 `checkpoint_final.pth`，并读取该 checkpoint 目录中的 `config.yaml`、`args.yaml` 和 `model_info.yaml`。

一键生成场景并导出 rollout：

```bash
python3 visualization/export_scene_rollout.py --config visualization/config/scene_rollout_example.yaml
```

播放 rollout 到 RViz2：

```bash
python3 visualization/rviz2_play_rollout.py --config visualization/config/offline_example.yaml
```

RViz2 播放脚本不导入 PyTorch、Env 或模型，只读取 `.npz` rollout 数据。当前离线 rollout 可同时保存并播放 MID360 点云：

- 导出时设置 `record_mid360: true`，点云会写入 `.npz` 的 `mid360_points` 和 `mid360_ranges`。
- 播放时发布 `/diffphys/mid360_points`，消息类型为 `sensor_msgs/PointCloud2`。
- MID360 策略导出时默认保存网络实际输入的伪图像，播放时发布 `/diffphys/mid360_pseudo_image` 和 `/diffphys/mid360_pseudo_image_viz`。
- 无人机位姿发布 `/diffphys/odom`，消息类型为 `nav_msgs/Odometry`。

## 测试与检查

运行现有回归测试：

```bash
pytest tests/test_train_launcher.py -v
```

只验证训练配置解析，不进入训练：

```bash
SE3_DIFF_SKIP_TRAIN=1 python3 train/main_cuda.py --config configs/single_agent.yaml
```

检查一体化可视化脚本入口：

```bash
python3 visualization/export_scene_rollout.py --help
```

## 常见问题

### 找不到 `quadsim_cuda`

先确认已经激活 `diffphysdrone`，并在 `src/` 下编译、安装 CUDA 扩展：

```bash
conda activate diffphysdrone
cd /home/zhr/SE3_Diff/src

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

python3 setup.py build_ext --inplace
python3 -m pip install -e . --no-build-isolation
```

如果 `import quadsim_cuda` 指向了其他目录，例如旧 worktree，重新执行上面的 editable install。可以用下面命令检查：

```bash
python3 -c "import quadsim_cuda; print(quadsim_cuda.__file__)"
```

### 编译时报 `x86_64-conda-linux-gnu-c++ 14.3.0`

这是 conda 环境默认 C++ 编译器版本过高导致的。CUDA 12.8 要求 host compiler `<14.0`。使用系统编译器重新编译：

```bash
cd /home/zhr/SE3_Diff/src

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

python3 setup.py build_ext --inplace
python3 -m pip install -e . --no-build-isolation
```

### `python` 命令不存在

使用 `python3`。

### 训练和可视化参数对不上

优先检查实际使用的 checkpoint 目录中的 `config.yaml`。如果可视化 YAML 没有指定 `checkpoint_path`，脚本会自动选择最新的 `checkpoints/*/checkpoint_final.pth`。

### 可视化每次播放不一致

可视化 policy rollout 默认启用：

```yaml
inference:
  deterministic_visualization: true
```

该开关会关闭运行时风、外部扰动、推力误差、阻力扰动等随机化，让同一场景和同一 checkpoint 的播放更一致。训练不受这个开关影响。

### 不知道参数应该改哪里

训练优先改 `configs/single_agent.yaml`。可视化场景优先改 `visualization/config/scene_rollout_example.yaml` 中的 `scene` 和 `paths`。
