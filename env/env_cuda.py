import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda
from env.scene import SceneContext, build_scene_pipeline_from_legacy_env
from env.scene.curriculum import ObstacleCountCurriculum
from se3diff_config.schema import EnvConfig


# 这个文件实现训练用的批量 CUDA 无人机环境。
# Env 负责随机生成障碍物场景、维护无人机状态、调用 CUDA 扩展渲染深度图、
# 查询最近障碍物，并用可微动力学推进一小步仿真。

class GDecay(torch.autograd.Function):
    """前向保持输入不变，反向把梯度乘上 alpha 的小工具。"""

    @staticmethod
    def forward(ctx, x, alpha):
        # ctx 是 autograd 上下文，用来把 backward 需要的普通值保存起来。
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    """把 quadsim_cuda.run_forward/run_backward 包装成 PyTorch 可微函数。"""

    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        # 前向调用 CUDA 动力学，输出下一时刻控制、位置、速度和加速度。
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        # 保存反向传播需要复用的中间量；不需要求梯度的标量直接挂在 ctx 上。
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        # d_*_next 是 loss 对 forward 输出的上游梯度。
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        # 返回值必须和 forward 参数一一对应；None 表示该输入不需要梯度。
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class RunCtbrFunction(torch.autograd.Function):
    """把 quadsim_cuda.run_ctbr_forward/run_ctbr_backward 包装成 PyTorch 可微函数。"""

    @staticmethod
    def forward(
        ctx,
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
        grad_decay,
        ctl_dt,
        omega_time_constant,
        thrust_time_constant,
        linear_drag,
    ):
        R_next, omega_next, collective_thrust_next, p_next, v_next, a_next = quadsim_cuda.run_ctbr_forward(
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
            ctl_dt,
            omega_time_constant,
            thrust_time_constant,
            linear_drag,
        )
        ctx.save_for_backward(R, omega_next, collective_thrust_next, mass, v)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        ctx.omega_time_constant = omega_time_constant
        ctx.thrust_time_constant = thrust_time_constant
        ctx.linear_drag = linear_drag
        return R_next, omega_next, collective_thrust_next, p_next, v_next, a_next

    @staticmethod
    def backward(
        ctx,
        d_R_next,
        d_omega_next,
        d_collective_thrust_next,
        d_p_next,
        d_v_next,
        d_a_next,
    ):
        R, omega_next, collective_thrust_next, mass, v = ctx.saved_tensors
        if d_R_next is None:
            d_R_next = torch.zeros_like(R)
        if d_omega_next is None:
            d_omega_next = torch.zeros_like(omega_next)
        if d_collective_thrust_next is None:
            d_collective_thrust_next = torch.zeros_like(collective_thrust_next)
        if d_p_next is None:
            d_p_next = torch.zeros_like(v)
        if d_v_next is None:
            d_v_next = torch.zeros_like(v)
        if d_a_next is None:
            d_a_next = torch.zeros_like(v)

        d_R, d_omega, d_collective_thrust, d_thrust_cmd, d_omega_cmd, d_p, d_v, d_a = quadsim_cuda.run_ctbr_backward(
            R,
            omega_next,
            collective_thrust_next,
            mass,
            v,
            d_R_next.contiguous(),
            d_omega_next.contiguous(),
            d_collective_thrust_next.contiguous(),
            d_p_next.contiguous(),
            d_v_next.contiguous(),
            d_a_next.contiguous(),
            ctx.grad_decay,
            ctx.ctl_dt,
            ctx.omega_time_constant,
            ctx.thrust_time_constant,
            ctx.linear_drag,
        )
        return (
            d_R,
            d_omega,
            d_collective_thrust,
            d_thrust_cmd,
            d_omega_cmd,
            None,
            None,
            d_p,
            d_v,
            None,
            d_a,
            None,
            None,
            None,
            None,
            None,
        )


run_ctbr_cuda = RunCtbrFunction.apply


class Env:
    """批量无人机仿真环境。

    每个 batch 元素对应一架无人机/一个随机场景。训练脚本会反复调用
    reset -> render -> run，并用 find_vec_to_nearest_pt 构造避障损失。
    """

    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, ceiling=False, ceiling_height=3.0, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10, start=None, target=None, max_speed=None, margin=None,
                 quad_mass=1.0, quad_mass_randomization=0.0,
                 ctbr_body_rate_limit=8.0, ctbr_thrust_min=0.0, ctbr_thrust_max=30.0,
                 ctbr_omega_time_constant=0.03, ctbr_thrust_time_constant=0.05, ctbr_linear_drag=0.0,
                 gap=False, gap_prob=0.0,
                 n_balls=30, n_voxels=30, n_cyl=30, n_cyl_h=2, n_ground_voxels=10,
                 obstacle_curriculum=None, is_scale=True) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay

        # 各类障碍物的随机采样范围：rand * *_w + *_b。
        # balls: [x, y, z, r]；voxels: [cx, cy, cz, rx, ry, rz]；
        # cyl: [x, y, r]；cyl_h: [x, z, r] 沿 y 轴。
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        # 沿当前速度外推 10 个短时刻，用于查询未来一小段轨迹上的最近障碍物。
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.current_pos_div = torch.zeros((1, 1, 1), device=device)

        # 起点和终点模板；batch 大于 8 时循环复用，再在 reset 中随机缩放和扰动。
        self.p_init = torch.as_tensor([
            [-10.5, -3.,  1],
            [ 19.5, -3.,  1],
            [-10.5,  1.,  1],
            [ 18.5,  1.,  1],
            [ 10.0,  3.,  1],
            [ 10.0,  3.,  1],
            [-10.0, -1.,  1],
            [ 10.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [18.,  3.,  1],
            [-10.,  3.,  1],
            [-18., -1.,  1],
            [-10., -1.,  1],
            [-10., -3.,  1],
            [-10., -3.,  1],
            [10.,  1.,  1],
            [-10.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.ceiling = ceiling
        self.ceiling_height = float(ceiling_height)
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.gap = gap
        self.gap_prob = gap_prob
        self.n_balls = int(n_balls)
        self.n_voxels = int(n_voxels)
        self.n_cyl = int(n_cyl)
        self.n_cyl_h = int(n_cyl_h)
        self.n_ground_voxels = int(n_ground_voxels)
        self.final_obstacle_counts = {
            "n_balls": self.n_balls,
            "n_voxels": self.n_voxels,
            "n_cyl": self.n_cyl,
            "n_cyl_h": self.n_cyl_h,
            "n_ground_voxels": self.n_ground_voxels,
        }
        self.obstacle_curriculum = ObstacleCountCurriculum.from_config(obstacle_curriculum)
        self.obstacle_curriculum_step = None
        self.current_obstacle_counts = dict(self.final_obstacle_counts)
        self.is_scale = bool(is_scale)
        self.fixed_max_speed = None if max_speed is None else float(max_speed)
        self.fixed_margin = None if margin is None else float(margin)
        self.quad_mass = float(quad_mass)
        self.quad_mass_randomization = float(quad_mass_randomization)
        if self.quad_mass <= 0:
            raise ValueError(f"quad_mass must be positive, got {self.quad_mass}")
        if self.quad_mass_randomization < 0:
            raise ValueError(f"quad_mass_randomization must be non-negative, got {self.quad_mass_randomization}")
        self.ctbr_body_rate_limit = float(ctbr_body_rate_limit)
        self.ctbr_thrust_min = float(ctbr_thrust_min)
        self.ctbr_thrust_max = float(ctbr_thrust_max)
        self.ctbr_omega_time_constant = float(ctbr_omega_time_constant)
        self.ctbr_thrust_time_constant = float(ctbr_thrust_time_constant)
        self.ctbr_linear_drag = float(ctbr_linear_drag)
        if self.ctbr_body_rate_limit <= 0:
            raise ValueError(f"ctbr_body_rate_limit must be positive, got {self.ctbr_body_rate_limit}")
        if self.ctbr_thrust_max <= self.ctbr_thrust_min:
            raise ValueError("ctbr_thrust_max must be greater than ctbr_thrust_min")
        if self.ctbr_omega_time_constant <= 0 or self.ctbr_thrust_time_constant <= 0:
            raise ValueError("CTBR response time constants must be positive")
        if self.ctbr_linear_drag < 0:
            raise ValueError(f"ctbr_linear_drag must be non-negative, got {self.ctbr_linear_drag}")
        self.scene_pipeline = build_scene_pipeline_from_legacy_env(
            EnvConfig(
                width=width,
                height=height,
                grad_decay=grad_decay,
                fov_x_half_tan=fov_x_half_tan,
                single=single,
                gate=gate,
                ground_voxels=ground_voxels,
                ceiling=ceiling,
                ceiling_height=ceiling_height,
                scaffold=scaffold,
                speed_mtp=speed_mtp,
                random_rotation=random_rotation,
                cam_angle=cam_angle,
                gap=gap,
                gap_prob=gap_prob,
                n_balls=self.n_balls,
                n_voxels=self.n_voxels,
                n_cyl=self.n_cyl,
                n_cyl_h=self.n_cyl_h,
                n_ground_voxels=self.n_ground_voxels,
                obstacle_curriculum=obstacle_curriculum or {},
                max_speed=self.fixed_max_speed,
                margin=self.fixed_margin,
                quad_mass=self.quad_mass,
                quad_mass_randomization=self.quad_mass_randomization,
                ctbr_body_rate_limit=self.ctbr_body_rate_limit,
                ctbr_thrust_min=self.ctbr_thrust_min,
                ctbr_thrust_max=self.ctbr_thrust_max,
                ctbr_omega_time_constant=self.ctbr_omega_time_constant,
                ctbr_thrust_time_constant=self.ctbr_thrust_time_constant,
                ctbr_linear_drag=self.ctbr_linear_drag,
                is_scale=self.is_scale,
            )
        )
        self.external_start = self._make_external_point_tensor(start, "start")
        self.external_target = self._make_external_point_tensor(target, "target")
        # 构造对象时立即生成第一批随机场景和无人机状态。
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def _make_external_point_tensor(self, value, name):
        """把外部起点/终点整理成 [1, 3] 或 [B, 3]，未传则返回 None。"""

        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.numel() == 3:
            return tensor.reshape(1, 3)
        if tensor.shape == (self.batch_size, 3):
            return tensor
        raise ValueError(f"{name} must have shape [3] or [{self.batch_size}, 3], got {tuple(tensor.shape)}")

    def set_obstacle_curriculum_step(self, step):
        self.obstacle_curriculum_step = None if step is None else int(step)
        self.current_obstacle_counts = self.obstacle_curriculum.counts_at(
            step=self.obstacle_curriculum_step,
            final_counts=self.final_obstacle_counts,
        )

    def _refresh_obstacle_curriculum_counts(self):
        self.current_obstacle_counts = self.obstacle_curriculum.counts_at(
            step=self.obstacle_curriculum_step,
            final_counts=self.final_obstacle_counts,
        )

    def reset(self):
        """重新随机生成一批场景，并初始化所有无人机状态。"""

        B = self.batch_size
        device = self.device
        self._refresh_obstacle_curriculum_counts()

        # 相机相对机体的俯仰安装角，训练时为每个样本加一点随机扰动。
        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # 每次 reset 轻微随机化视场、多机分组和无人机半径，增加训练扰动。
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        # 同一组无人机共享速度尺度，便于形成多机同场景训练。
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        
        if self.fixed_max_speed is not None:
            self.max_speed = torch.full((B, 1), self.fixed_max_speed, device=device)
        else:
            self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp

        scale = (self.max_speed - 0.5).clamp_min(1)
        # scale = torch.tensor(1.0, device=device).reshape(B, 1)
        if self.is_scale:
            # 起终点在模板基础上按组缩放，并加入小噪声。
            rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
            scale = torch.cat([
                scale,
                rd + 0.5,
                torch.rand_like(scale)], -1)
            self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
            self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        # 模拟推力估计误差，训练策略对轻微模型不准更鲁棒。
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        if self.quad_mass_randomization > 0:
            mass_noise = 1 + (torch.rand((B, 1), device=device) * 2 - 1) * self.quad_mass_randomization
            self.mass = self.quad_mass * mass_noise
        else:
            self.mass = torch.full((B, 1), self.quad_mass, device=device)
        self.omega = torch.zeros((B, 3), device=device)
        hover_thrust = self.mass * -self.g_std[2]
        self.collective_thrust = hover_thrust.clamp(self.ctbr_thrust_min, self.ctbr_thrust_max)

        # 无人机动力学随机化：控制延迟、起终点、风、扰动、阻力等。
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        if self.external_start is not None:
            self.p = self.external_start.repeat(B, 1) if self.external_start.size(0) == 1 else self.external_start.clone()
        if self.external_target is not None:
            self.p_target = self.external_target.repeat(B, 1) if self.external_target.size(0) == 1 else self.external_target.clone()
        # 场景生成由 pipeline 负责：所有高级障碍物最终展开成 balls/voxels/cyl/cyl_h。
        ctx = SceneContext.from_env(self, max_speed=self.max_speed, scale=(self.max_speed - 0.5).clamp_min(1))
        scene = self.scene_pipeline.generate(ctx)
        scene.to_env(self)

        # 初始化状态：速度、风速、当前控制、当前加速度和随机外部扰动。
        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        # 根据目标方向和当前控制初始化姿态矩阵 R=[forward,left,up]。
        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p

        # margin 是每架无人机的安全距离，用于训练中的避障/碰撞损失。
        if self.fixed_margin is not None:
            self.margin = torch.full((B,), self.fixed_margin, device=device)
        else:
            self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # 空气阻力系数；当前默认关闭二次阻力项 drag_2[:, 0]。
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @staticmethod
    def _skew(vec):
        """Return a batch of skew-symmetric matrices for vectors shaped [B, 3]."""

        zeros = torch.zeros_like(vec[:, 0])
        wx, wy, wz = vec.unbind(-1)
        return torch.stack([
            zeros, -wz, wy,
            wz, zeros, -wx,
            -wy, wx, zeros,
        ], -1).reshape(vec.shape[0], 3, 3)

    @classmethod
    def _rotation_increment(cls, omega, dt):
        """Rodrigues update for body-rate integration, differentiable in PyTorch."""

        rot_vec = omega * dt
        theta = torch.linalg.norm(rot_vec, dim=-1, keepdim=True)
        K = cls._skew(rot_vec)
        K2 = K @ K
        eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
        theta2 = theta.square()
        sin_over_theta = torch.where(
            theta > 1e-6,
            torch.sin(theta) / theta,
            1 - theta2 / 6 + theta2 * theta2 / 120,
        )
        one_minus_cos_over_theta2 = torch.where(
            theta > 1e-6,
            (1 - torch.cos(theta)) / theta2.clamp_min(1e-12),
            0.5 - theta2 / 24 + theta2 * theta2 / 720,
        )
        return eye + sin_over_theta[:, None] * K + one_minus_cos_over_theta2[:, None] * K2

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        """Python 版姿态更新参考实现；实际训练主要使用 CUDA 版本。"""

        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        # 推力方向 = 期望净加速度 - 重力；因为重力 z 为负，所以 z 分量会加 9.80665。
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        # forward 方向由旧朝向惯性和预测速度方向共同决定，再用 alpha 平滑。
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        # 调整 z 分量，强制 forward 与 up 正交。
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        # 右手叉乘：up x forward = left。
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        """渲染当前 batch 的深度图，输出形状为 [B, height, width]。"""

        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        # R @ R_cam 将机体姿态和相机安装角合成相机朝向。
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan, self.ceiling, self.ceiling_height)
        return canvas, None

    def find_vec_to_nearest_pt(self, *, use_future_samples=True):
        """查询当前位置或未来一小段轨迹采样点到最近障碍物点的向量。"""

        query_offsets = self.sub_div if use_future_samples else self.current_pos_div
        p = self.p + self.v * query_offsets
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(
            nearest_pt,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            p,
            self.drone_radius,
            self.n_drones_per_group,
            self.ceiling,
            self.ceiling_height,
        )
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """用可微 CUDA 动力学推进一小步，并更新姿态。"""

        # 外部扰动 dg 采用带噪声的缓慢随机游走。
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        # RunFunction.apply 会调用 CUDA forward，并在 backward 时调用配套 CUDA backward。
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # 用 yaw 延迟平滑姿态更新；v_pred 通常来自网络预测速度方向。
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def _run_ctbr_torch_reference(self, ctbr_cmd, ctl_dt=1/15, _yaw_correction_vec=None):
        """Pure PyTorch CTBR reference dynamics.

        ctbr_cmd is [collective_thrust_N, wx, wy, wz] with body rates in rad/s.
        """

        thrust_cmd = ctbr_cmd[:, :1].clamp(self.ctbr_thrust_min, self.ctbr_thrust_max)
        omega_cmd = ctbr_cmd[:, 1:4].clamp(-self.ctbr_body_rate_limit, self.ctbr_body_rate_limit)

        alpha_w = math.exp(-ctl_dt / self.ctbr_omega_time_constant)
        alpha_c = math.exp(-ctl_dt / self.ctbr_thrust_time_constant)
        self.omega = self.omega * alpha_w + omega_cmd * (1 - alpha_w)
        self.collective_thrust = self.collective_thrust * alpha_c + thrust_cmd * (1 - alpha_c)

        self.p_old = self.p
        self.R_old = self.R.clone()
        dR_body = self._rotation_increment(self.omega, ctl_dt)
        self.R = self.R @ dR_body

        thrust_acc = self.collective_thrust / self.mass * self.R[:, :, 2]
        v_rel_wind = self.v - self.v_wind
        linear_drag = self.ctbr_linear_drag * v_rel_wind
        a_next = thrust_acc + self.g_std + self.dg - linear_drag

        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + 0.5 * (self.a + a_next) * ctl_dt
        self.a = a_next
        self.act = ctbr_cmd

    def run_ctbr(self, ctbr_cmd, ctl_dt=1/15, _yaw_correction_vec=None):
        """用 CTBR 刚体动力学推进一小步，CUDA 可用时使用自定义前反向核。"""

        if self.R.is_cuda and hasattr(quadsim_cuda, "run_ctbr_forward") and hasattr(quadsim_cuda, "run_ctbr_backward"):
            thrust_cmd = ctbr_cmd[:, :1].clamp(self.ctbr_thrust_min, self.ctbr_thrust_max)
            omega_cmd = ctbr_cmd[:, 1:4].clamp(-self.ctbr_body_rate_limit, self.ctbr_body_rate_limit)
            self.p_old = self.p
            self.R_old = self.R.clone()
            self.R, self.omega, self.collective_thrust, self.p, self.v, self.a = run_ctbr_cuda(
                self.R,
                self.omega,
                self.collective_thrust,
                thrust_cmd,
                omega_cmd,
                self.mass,
                self.dg,
                self.p,
                self.v,
                self.v_wind,
                self.a,
                self.grad_decay,
                ctl_dt,
                self.ctbr_omega_time_constant,
                self.ctbr_thrust_time_constant,
                self.ctbr_linear_drag,
            )
            self.act = ctbr_cmd
            return

        self._run_ctbr_torch_reference(ctbr_cmd, ctl_dt, _yaw_correction_vec)

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """旧的 Python 动力学实现/参考实现，主训练路径使用 run()。"""

        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            # 旧实现里用 up 方向速度和电机速度近似 z 方向阻力。
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        # 位置用匀加速度公式，速度用当前/下一加速度的梯形积分。
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
