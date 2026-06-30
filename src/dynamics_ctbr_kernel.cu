#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__device__ void make_skew(const scalar_t rx, const scalar_t ry, const scalar_t rz, scalar_t K[3][3]) {
    K[0][0] = 0;
    K[0][1] = -rz;
    K[0][2] = ry;
    K[1][0] = rz;
    K[1][1] = 0;
    K[1][2] = -rx;
    K[2][0] = -ry;
    K[2][1] = rx;
    K[2][2] = 0;
}

template <typename scalar_t>
__device__ void make_rotation_increment(
    const scalar_t wx,
    const scalar_t wy,
    const scalar_t wz,
    const scalar_t ctl_dt,
    scalar_t M[3][3]) {
    const scalar_t rx = wx * ctl_dt;
    const scalar_t ry = wy * ctl_dt;
    const scalar_t rz = wz * ctl_dt;
    const scalar_t theta2 = rx * rx + ry * ry + rz * rz;
    const scalar_t theta = sqrt(theta2);

    scalar_t A;
    scalar_t B;
    if (theta > static_cast<scalar_t>(1e-6)) {
        A = sin(theta) / theta;
        B = (static_cast<scalar_t>(1) - cos(theta)) / theta2;
    } else {
        const scalar_t theta4 = theta2 * theta2;
        A = static_cast<scalar_t>(1) - theta2 / static_cast<scalar_t>(6) + theta4 / static_cast<scalar_t>(120);
        B = static_cast<scalar_t>(0.5) - theta2 / static_cast<scalar_t>(24) + theta4 / static_cast<scalar_t>(720);
    }

    scalar_t K[3][3];
    scalar_t K2[3][3];
    make_skew(rx, ry, rz, K);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            K2[r][c] = 0;
            for (int k = 0; k < 3; k++) {
                K2[r][c] += K[r][k] * K[k][c];
            }
            M[r][c] = (r == c ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0))
                    + A * K[r][c]
                    + B * K2[r][c];
        }
    }
}

template <typename scalar_t>
__global__ void run_ctbr_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> collective_thrust,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> thrust_cmd,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega_cmd,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mass,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> collective_thrust_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,
    float ctl_dt,
    float omega_time_constant,
    float thrust_time_constant,
    float linear_drag) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;

    const scalar_t dt = static_cast<scalar_t>(ctl_dt);
    const scalar_t alpha_w = exp(-dt / static_cast<scalar_t>(omega_time_constant));
    const scalar_t alpha_c = exp(-dt / static_cast<scalar_t>(thrust_time_constant));
    const scalar_t beta_w = static_cast<scalar_t>(1) - alpha_w;
    const scalar_t beta_c = static_cast<scalar_t>(1) - alpha_c;

    for (int j = 0; j < 3; j++) {
        omega_next[b][j] = omega[b][j] * alpha_w + omega_cmd[b][j] * beta_w;
    }
    collective_thrust_next[b][0] = collective_thrust[b][0] * alpha_c + thrust_cmd[b][0] * beta_c;

    scalar_t M[3][3];
    make_rotation_increment(
        omega_next[b][0],
        omega_next[b][1],
        omega_next[b][2],
        dt,
        M);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            scalar_t value = 0;
            for (int k = 0; k < 3; k++) {
                value += R[b][r][k] * M[k][c];
            }
            R_next[b][r][c] = value;
        }
    }

    const scalar_t thrust_acc_scale = collective_thrust_next[b][0] / mass[b][0];
    for (int j = 0; j < 3; j++) {
        const scalar_t gravity = (j == 2) ? static_cast<scalar_t>(-9.80665) : static_cast<scalar_t>(0);
        const scalar_t v_rel_wind = v[b][j] - v_wind[b][j];
        a_next[b][j] = thrust_acc_scale * R_next[b][j][2]
                     + gravity
                     + dg[b][j]
                     - static_cast<scalar_t>(linear_drag) * v_rel_wind;
    }

    for (int j = 0; j < 3; j++) {
        p_next[b][j] = p[b][j] + v[b][j] * dt + static_cast<scalar_t>(0.5) * a[b][j] * dt * dt;
        v_next[b][j] = v[b][j] + static_cast<scalar_t>(0.5) * (a[b][j] + a_next[b][j]) * dt;
    }
}

template <typename scalar_t>
__global__ void run_ctbr_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> collective_thrust_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mass,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> _d_R_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_omega_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_collective_thrust_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_omega,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_collective_thrust,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_thrust_cmd,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_omega_cmd,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,
    float grad_decay,
    float ctl_dt,
    float omega_time_constant,
    float thrust_time_constant,
    float linear_drag) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;

    const scalar_t dt = static_cast<scalar_t>(ctl_dt);
    const scalar_t alpha_w = exp(-dt / static_cast<scalar_t>(omega_time_constant));
    const scalar_t alpha_c = exp(-dt / static_cast<scalar_t>(thrust_time_constant));
    const scalar_t beta_w = static_cast<scalar_t>(1) - alpha_w;
    const scalar_t beta_c = static_cast<scalar_t>(1) - alpha_c;
    const scalar_t grad_alpha = pow(static_cast<scalar_t>(grad_decay), dt);

    scalar_t d_R_next[3][3];
    scalar_t d_a_next[3];
    scalar_t local_d_v[3];
    scalar_t local_d_a[3];
    scalar_t local_d_p[3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            d_R_next[r][c] = _d_R_next[b][r][c];
        }
    }
    for (int j = 0; j < 3; j++) {
        d_a_next[j] = _d_a_next[b][j];
        local_d_v[j] = d_v_next[b][j] * grad_alpha;
        local_d_a[j] = static_cast<scalar_t>(0.5) * dt * d_v_next[b][j];
        d_a_next[j] += static_cast<scalar_t>(0.5) * dt * d_v_next[b][j];
    }

    for (int j = 0; j < 3; j++) {
        local_d_p[j] = d_p_next[b][j] * grad_alpha;
        local_d_v[j] += dt * d_p_next[b][j];
        local_d_a[j] += static_cast<scalar_t>(0.5) * dt * dt * d_p_next[b][j];
    }

    scalar_t M[3][3];
    make_rotation_increment(
        omega_next[b][0],
        omega_next[b][1],
        omega_next[b][2],
        dt,
        M);

    scalar_t R_next_col2[3];
    for (int r = 0; r < 3; r++) {
        R_next_col2[r] = 0;
        for (int k = 0; k < 3; k++) {
            R_next_col2[r] += R[b][r][k] * M[k][2];
        }
    }

    scalar_t local_d_thrust_next = _d_collective_thrust_next[b][0];
    const scalar_t thrust_acc_scale = collective_thrust_next[b][0] / mass[b][0];
    const scalar_t inv_mass = static_cast<scalar_t>(1) / mass[b][0];
    for (int j = 0; j < 3; j++) {
        d_R_next[j][2] += d_a_next[j] * thrust_acc_scale;
        local_d_thrust_next += d_a_next[j] * R_next_col2[j] * inv_mass;
        local_d_v[j] -= static_cast<scalar_t>(linear_drag) * d_a_next[j];
    }

    scalar_t local_d_R[3][3];
    scalar_t d_M[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            local_d_R[r][c] = 0;
            d_M[r][c] = 0;
        }
    }

    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 3; k++) {
            for (int c = 0; c < 3; c++) {
                local_d_R[r][k] += d_R_next[r][c] * M[k][c];
                d_M[k][c] += R[b][r][k] * d_R_next[r][c];
            }
        }
    }

    const scalar_t rx = omega_next[b][0] * dt;
    const scalar_t ry = omega_next[b][1] * dt;
    const scalar_t rz = omega_next[b][2] * dt;
    const scalar_t theta2 = rx * rx + ry * ry + rz * rz;
    const scalar_t theta = sqrt(theta2);

    scalar_t K[3][3];
    scalar_t K2[3][3];
    make_skew(rx, ry, rz, K);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            K2[r][c] = 0;
            for (int k = 0; k < 3; k++) {
                K2[r][c] += K[r][k] * K[k][c];
            }
        }
    }

    scalar_t A;
    scalar_t Bcoef;
    scalar_t A_prime_over_theta;
    scalar_t B_prime_over_theta;
    if (theta > static_cast<scalar_t>(1e-6)) {
        A = sin(theta) / theta;
        Bcoef = (static_cast<scalar_t>(1) - cos(theta)) / theta2;
        A_prime_over_theta = (theta * cos(theta) - sin(theta)) / (theta2 * theta);
        B_prime_over_theta = (theta * sin(theta) - static_cast<scalar_t>(2) * (static_cast<scalar_t>(1) - cos(theta))) / (theta2 * theta2);
    } else {
        const scalar_t theta4 = theta2 * theta2;
        A = static_cast<scalar_t>(1) - theta2 / static_cast<scalar_t>(6) + theta4 / static_cast<scalar_t>(120);
        Bcoef = static_cast<scalar_t>(0.5) - theta2 / static_cast<scalar_t>(24) + theta4 / static_cast<scalar_t>(720);
        A_prime_over_theta = -static_cast<scalar_t>(1.0 / 3.0) + theta2 / static_cast<scalar_t>(30);
        B_prime_over_theta = -static_cast<scalar_t>(1.0 / 12.0) + theta2 / static_cast<scalar_t>(180);
    }

    scalar_t d_A = 0;
    scalar_t d_Bcoef = 0;
    scalar_t d_K[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            d_A += d_M[r][c] * K[r][c];
            d_Bcoef += d_M[r][c] * K2[r][c];
            d_K[r][c] = A * d_M[r][c];
        }
    }

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            scalar_t d_from_k2 = 0;
            for (int k = 0; k < 3; k++) {
                d_from_k2 += d_M[r][k] * K[c][k];
                d_from_k2 += K[k][r] * d_M[k][c];
            }
            d_K[r][c] += Bcoef * d_from_k2;
        }
    }

    scalar_t d_rx = d_K[2][1] - d_K[1][2] + (d_A * A_prime_over_theta + d_Bcoef * B_prime_over_theta) * rx;
    scalar_t d_ry = d_K[0][2] - d_K[2][0] + (d_A * A_prime_over_theta + d_Bcoef * B_prime_over_theta) * ry;
    scalar_t d_rz = d_K[1][0] - d_K[0][1] + (d_A * A_prime_over_theta + d_Bcoef * B_prime_over_theta) * rz;

    scalar_t local_d_omega_next[3] = {
        _d_omega_next[b][0] + dt * d_rx,
        _d_omega_next[b][1] + dt * d_ry,
        _d_omega_next[b][2] + dt * d_rz,
    };

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            d_R[b][r][c] = local_d_R[r][c];
        }
    }
    for (int j = 0; j < 3; j++) {
        d_omega[b][j] = alpha_w * local_d_omega_next[j];
        d_omega_cmd[b][j] = beta_w * local_d_omega_next[j];
        d_p[b][j] = local_d_p[j];
        d_v[b][j] = local_d_v[j];
        d_a[b][j] = local_d_a[j];
    }
    d_collective_thrust[b][0] = alpha_c * local_d_thrust_next;
    d_thrust_cmd[b][0] = beta_c * local_d_thrust_next;
}

} // namespace

std::vector<torch::Tensor> run_ctbr_forward_cuda(
    torch::Tensor R,
    torch::Tensor omega,
    torch::Tensor collective_thrust,
    torch::Tensor thrust_cmd,
    torch::Tensor omega_cmd,
    torch::Tensor mass,
    torch::Tensor dg,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float omega_time_constant,
    float thrust_time_constant,
    float linear_drag) {

    torch::Tensor R_next = torch::empty_like(R);
    torch::Tensor omega_next = torch::empty_like(omega);
    torch::Tensor collective_thrust_next = torch::empty_like(collective_thrust);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    const int threads = 256;
    const dim3 blocks((R.size(0) + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_ctbr_forward_cuda", ([&] {
        run_ctbr_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            omega.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            collective_thrust.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            thrust_cmd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            omega_cmd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mass.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            R_next.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            omega_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            collective_thrust_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt,
            omega_time_constant,
            thrust_time_constant,
            linear_drag);
    }));
    return {R_next, omega_next, collective_thrust_next, p_next, v_next, a_next};
}

std::vector<torch::Tensor> run_ctbr_backward_cuda(
    torch::Tensor R,
    torch::Tensor omega_next,
    torch::Tensor collective_thrust_next,
    torch::Tensor mass,
    torch::Tensor v,
    torch::Tensor _d_R_next,
    torch::Tensor _d_omega_next,
    torch::Tensor _d_collective_thrust_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt,
    float omega_time_constant,
    float thrust_time_constant,
    float linear_drag) {

    torch::Tensor d_R = torch::empty_like(R);
    torch::Tensor d_omega = torch::empty_like(omega_next);
    torch::Tensor d_collective_thrust = torch::empty_like(collective_thrust_next);
    torch::Tensor d_thrust_cmd = torch::empty_like(collective_thrust_next);
    torch::Tensor d_omega_cmd = torch::empty_like(omega_next);
    torch::Tensor d_p = torch::empty_like(v);
    torch::Tensor d_v = torch::empty_like(v);
    torch::Tensor d_a = torch::empty_like(v);

    const int threads = 256;
    const dim3 blocks((R.size(0) + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_ctbr_backward_cuda", ([&] {
        run_ctbr_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            omega_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            collective_thrust_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mass.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_R_next.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            _d_omega_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_collective_thrust_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            d_omega.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_collective_thrust.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_thrust_cmd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_omega_cmd.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay,
            ctl_dt,
            omega_time_constant,
            thrust_time_constant,
            linear_drag);
    }));
    return {d_R, d_omega, d_collective_thrust, d_thrust_cmd, d_omega_cmd, d_p, d_v, d_a};
}
