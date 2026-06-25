#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// 本文件实现 quadsim_cuda 扩展中与环境几何感知相关的 CUDA kernel。
//
// 主要包含三类功能：
// 1. render_cuda_kernel：从每架无人机的相机位姿出发，对每个像素发射一条射线，
//    计算该射线与地面、其他无人机、球体、圆柱体和体素盒子的最近交点距离，
//    输出深度图 canvas。
// 2. render_mid360_cuda_kernel：用 MID360 常用视场参数渲染 360 度点云。
// 3. nearest_pt_cuda_kernel：对每个无人机位置，查找离它最近的障碍物/地面点，
//    供避障或奖励计算使用。
// 4. rerender_backward_cuda_kernel：根据 2x2 深度块估计低分辨率的局部深度梯度方向。
//
// flow、R_old、pos_old、drone_radius 等参数保留在接口中；在当前实现里，
// render_cuda_kernel/nearest_pt_cuda_kernel 并未实际使用它们。

namespace {

__device__ float atomicMinFloat(float* address, float value) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i;
    int assumed;
    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
    return __int_as_float(old);
}

template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan,
    bool has_ceiling,
    float ceiling_height) {

    // 每个 CUDA 线程负责渲染一个 batch 中的一个像素。
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    // 将像素坐标映射到相机坐标系中的归一化视线偏移。
    // R[..., 0/1/2] 分别被当作 forward/left/up 方向使用。
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // 默认最大深度为 100；先计算与地面平面 z = -1 的交点。
    scalar_t min_dist = 100;
    scalar_t  t = (-1 - oz) / dz;
    if (t > 0) min_dist = t;
    t = (static_cast<scalar_t>(ceiling_height) - oz) / dz;
    if (has_ceiling && t > 0) min_dist = min(min_dist, t);

    // 同组无人机互相可见；这里用 z 轴缩放的椭球近似其他无人机。
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 球形障碍物：求射线与球体二次方程的最近正根。
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 竖直圆柱障碍物：只在 x-y 平面上求交，相当于无限高圆柱。
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    // 水平圆柱障碍物：在 x-z 平面上求交，相当于沿 y 方向延伸的圆柱。
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    // 体素盒子/AABB：用 slab 方法计算射线进入和离开盒子的参数范围。
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    // canvas 保存每个像素沿射线方向看到的最近距离。
    canvas[b][u][v] = min_dist;
}

template <typename scalar_t>
__global__ void render_mid360_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> points,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ranges,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group,
    int vertical_channels,
    float min_range,
    float max_range,
    float vertical_min_deg,
    float vertical_max_deg,
    bool has_ceiling,
    float ceiling_height) {

    // MID360 commonly runs as a 360-degree horizontal lidar with about -7..52 deg
    // vertical coverage. Each thread casts one deterministic lidar ray.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = points.size(0);
    const int N = points.size(1);
    if (idx >= B * N) return;

    const int b = idx / N;
    const int ray = idx % N;
    const int v_channels = max(1, vertical_channels);
    const int h_bins = max(1, (N + v_channels - 1) / v_channels);
    const int v_idx = ray % v_channels;
    const int h_idx = ray / v_channels;

    const scalar_t pi = static_cast<scalar_t>(3.14159265358979323846);
    const scalar_t deg_to_rad = pi / static_cast<scalar_t>(180.0);
    const scalar_t azimuth = static_cast<scalar_t>(2.0) * pi *
        (static_cast<scalar_t>(h_idx) + static_cast<scalar_t>(0.5)) /
        static_cast<scalar_t>(h_bins);
    const scalar_t vertical_min = static_cast<scalar_t>(vertical_min_deg) * deg_to_rad;
    const scalar_t vertical_max = static_cast<scalar_t>(vertical_max_deg) * deg_to_rad;
    const scalar_t elevation = vertical_min +
        (static_cast<scalar_t>(v_idx) + static_cast<scalar_t>(0.5)) /
        static_cast<scalar_t>(v_channels) * (vertical_max - vertical_min);

    const scalar_t ce = cos(elevation);
    const scalar_t se = sin(elevation);
    const scalar_t ca = cos(azimuth);
    const scalar_t sa = sin(azimuth);
    const scalar_t local_x = ce * ca;
    const scalar_t local_y = ce * sa;
    const scalar_t local_z = se;

    scalar_t dx = R[b][0][0] * local_x + R[b][0][1] * local_y + R[b][0][2] * local_z;
    scalar_t dy = R[b][1][0] * local_x + R[b][1][1] * local_y + R[b][1][2] * local_z;
    scalar_t dz = R[b][2][0] * local_x + R[b][2][1] * local_y + R[b][2][2] * local_z;
    const scalar_t norm = max(static_cast<scalar_t>(1e-6), sqrt(dx * dx + dy * dy + dz * dz));
    dx /= norm;
    dy /= norm;
    dz /= norm;

    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];
    const scalar_t min_t = static_cast<scalar_t>(min_range);
    const scalar_t max_t = static_cast<scalar_t>(max_range);

    scalar_t min_dist = max_t;
    bool hit = false;

    scalar_t t = (-1 - oz) / dz;
    if (t >= min_t && t <= min_dist) {
        min_dist = t;
        hit = true;
    }
    t = (static_cast<scalar_t>(ceiling_height) - oz) / dz;
    if (has_ceiling && t >= min_t && t <= min_dist) {
        min_dist = t;
        hit = true;
    }

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t qb = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = qb * qb - 4 * a * c;
        if (d >= 0) {
            r = (-qb - sqrt(d)) / (2 * a);
            if (r < min_t) r = (-qb + sqrt(d)) / (2 * a);
            if (r >= min_t && r <= min_dist) {
                min_dist = r;
                hit = true;
            }
        }
    }

    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t qb = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = qb * qb - 4 * a * c;
        if (d >= 0) {
            r = (-qb - sqrt(d)) / (2 * a);
            if (r < min_t) r = (-qb + sqrt(d)) / (2 * a);
            if (r >= min_t && r <= min_dist) {
                min_dist = r;
                hit = true;
            }
        }
    }

    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        if (a <= static_cast<scalar_t>(1e-9)) continue;
        scalar_t qb = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = qb * qb - 4 * a * c;
        if (d >= 0) {
            r = (-qb - sqrt(d)) / (2 * a);
            if (r < min_t) r = (-qb + sqrt(d)) / (2 * a);
            if (r >= min_t && r <= min_dist) {
                min_dist = r;
                hit = true;
            }
        }
    }

    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        if (a <= static_cast<scalar_t>(1e-9)) continue;
        scalar_t qb = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = qb * qb - 4 * a * c;
        if (d >= 0) {
            r = (-qb - sqrt(d)) / (2 * a);
            if (r < min_t) r = (-qb + sqrt(d)) / (2 * a);
            if (r >= min_t && r <= min_dist) {
                min_dist = r;
                hit = true;
            }
        }
    }
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < t_max) {
            t = t_min >= min_t ? t_min : t_max;
            if (t >= min_t && t <= min_dist) {
                min_dist = t;
                hit = true;
            }
        }
    }

    if (hit) {
        ranges[b][ray] = min_dist;
        points[b][ray][0] = ox + min_dist * dx;
        points[b][ray][1] = oy + min_dist * dy;
        points[b][ray][2] = oz + min_dist * dz;
    } else {
        ranges[b][ray] = static_cast<scalar_t>(0);
        points[b][ray][0] = static_cast<scalar_t>(0);
        points[b][ray][1] = static_cast<scalar_t>(0);
        points[b][ray][2] = static_cast<scalar_t>(0);
    }
}

template <typename scalar_t>
__global__ void points_to_pseudo_image_mid360_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> pseudo_image,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> points,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ranges,
    float min_range,
    float max_range,
    float theta_min,
    float phi_min,
    float theta_resolution,
    float phi_resolution) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = points.size(0);
    const int N = points.size(1);
    if (idx >= B * N) return;

    const int b = idx / N;
    const int n = idx % N;
    const scalar_t x = points[b][n][0];
    const scalar_t y = points[b][n][1];
    const scalar_t z = points[b][n][2];

    scalar_t r = ranges[b][n];
    if (r <= static_cast<scalar_t>(0)) {
        r = sqrt(x * x + y * y + z * z);
    }
    if (r < static_cast<scalar_t>(min_range) || r > static_cast<scalar_t>(max_range)) return;

    const scalar_t theta = atan2(y, x);
    const scalar_t safe_r = max(r, static_cast<scalar_t>(1e-6));
    const scalar_t zr = min(max(z / safe_r, static_cast<scalar_t>(-1)), static_cast<scalar_t>(1));
    const scalar_t phi = acos(zr);
    const int theta_idx = static_cast<int>(floor((theta - static_cast<scalar_t>(theta_min)) / static_cast<scalar_t>(theta_resolution)));
    const int phi_idx = static_cast<int>(floor((phi - static_cast<scalar_t>(phi_min)) / static_cast<scalar_t>(phi_resolution)));

    if (theta_idx < 0 || theta_idx >= pseudo_image.size(3) || phi_idx < 0 || phi_idx >= pseudo_image.size(2)) return;
    atomicMinFloat(reinterpret_cast<float*>(&pseudo_image[b][0][phi_idx][theta_idx]), static_cast<float>(r));
}

template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group,
    bool has_ceiling,
    float ceiling_height) {

    // pos 的形状通常是 [T, B, 3]，这里每个线程处理一个 (时间/子步 j, batch b)。
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;
    // assert(j < pos.size(0));
    // assert(b < pos.size(1));

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // 先把地面 z = -1 作为候选最近点，后续障碍物更近时再覆盖。
    scalar_t min_dist = max(1e-3f, oz + 1);
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);
    if (has_ceiling) {
        scalar_t ceiling_dist = max(1e-3f, abs(static_cast<scalar_t>(ceiling_height) - oz));
        if (ceiling_dist < min_dist) {
            min_dist = ceiling_dist;
            nearest_ptx = ox;
            nearest_pty = oy;
            nearest_ptz = static_cast<scalar_t>(ceiling_height);
        }
    }

    // 与 render 一致，同组其他无人机用扁椭球近似。
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 球形障碍物：用点到球心距离减半径得到近似表面距离。
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 竖直圆柱：只考虑 x-y 平面距离，z 坐标保持无人机当前位置。
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz;
        }
    }
    // 水平圆柱：只考虑 x-z 平面距离，y 坐标保持无人机当前位置。
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy;
            nearest_ptz = oz + dist * (cz - oz);
        }
    }
    // 体素盒子/AABB：把点投影/夹到盒子表面附近，得到最近候选点。
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }

    // 写回最近障碍物点坐标；Python 侧通常会再减去 pos 得到指向最近点的向量。
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    // 输出 dddp 的每个像素对应输入 depth 的一个 2x2 区块。
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    // 对 2x2 深度块做差分，估计局部 y/z 方向深度斜率。
    // d 是带视场比例的平均深度，用来把像素差分归一化到近似几何斜率。
    const scalar_t unit = fov_x_half_tan / W;
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    // if ReRender.diff_kernel is None:
    //     unit = 0.637 / depth.size(3)
    //     ReRender.diff_kernel = torch.tensor([
    //         [[1, -1], [1, -1]],
    //         [[1, 1], [-1, -1]],
    //         [[unit, unit], [unit, unit]],
    //     ], device=device).mul(0.5)[:, None]
    // ddepthdyz = F.conv2d(depth, ReRender.diff_kernel, None, 2)
    // depth = ddepthdyz[:, 2:]
    // ddepthdyz = torch.cat([
    //     torch.full_like(depth, -1.),
    //     ddepthdyz[:, :2] / depth,
    // ], 1)
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    // 输出类似法向/梯度方向的三通道结果，并用最小范数 8 限制幅值。
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
    // ddepthdyz /= ddepthdyz.norm(2, 1, True).clamp_min(8);
}

} // namespace

void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan,
    bool has_ceiling,
    float ceiling_height) {
    const int threads = 1024;
    // 渲染任务数等于 canvas 元素总数，即 B * H * W。
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    // 根据输入张量 dtype 实例化 float/double 版本的 CUDA kernel。
    AT_DISPATCH_FLOATING_TYPES(canvas.scalar_type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan,
            has_ceiling,
            ceiling_height);
    }));
}

void render_mid360_cuda(
    torch::Tensor points,
    torch::Tensor ranges,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    int vertical_channels,
    float min_range,
    float max_range,
    float vertical_min_deg,
    float vertical_max_deg,
    bool has_ceiling,
    float ceiling_height) {
    const int threads = 1024;
    // points shape: [B, N, 3], ranges shape: [B, N].
    size_t state_size = ranges.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "render_mid360_cuda", ([&] {
        render_mid360_cuda_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            ranges.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            n_drones_per_group,
            vertical_channels,
            min_range,
            max_range,
            vertical_min_deg,
            vertical_max_deg,
            has_ceiling,
            ceiling_height);
    }));
}

void points_to_pseudo_image_mid360_cuda(
    torch::Tensor pseudo_image,
    torch::Tensor points,
    torch::Tensor ranges,
    float min_range,
    float max_range,
    float theta_min,
    float phi_min,
    float theta_resolution,
    float phi_resolution) {
    const int threads = 1024;
    size_t state_size = points.size(0) * points.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "points_to_pseudo_image_mid360_cuda", ([&] {
        points_to_pseudo_image_mid360_cuda_kernel<scalar_t><<<blocks, threads>>>(
            pseudo_image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            points.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            ranges.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            min_range,
            max_range,
            theta_min,
            phi_min,
            theta_resolution,
            phi_resolution);
    }));
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    const int threads = 1024;
    // dddp 是输出张量；每个输出像素由一个线程计算。
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group,
    bool has_ceiling,
    float ceiling_height) {
    const int threads = 1024;
    // pos 的前两维展开成一维任务队列：每个线程处理一个位置点。
    size_t state_size = pos.size(0) * pos.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            has_ceiling,
            ceiling_height);
    }));
}
