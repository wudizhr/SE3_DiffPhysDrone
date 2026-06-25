#include <torch/extension.h>

#include <vector>

namespace py = pybind11;

// CUDA forward declarations

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
    float ceiling_height);

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
    float ceiling_height);

void points_to_pseudo_image_mid360_cuda(
    torch::Tensor pseudo_image,
    torch::Tensor points,
    torch::Tensor ranges,
    float min_range,
    float max_range,
    float theta_min,
    float phi_min,
    float theta_resolution,
    float phi_resolution);

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

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
    float ceiling_height);

torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a);

std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt);

// C++ interface

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render",
      &render_cuda,
      "render (CUDA)",
      py::arg("canvas"),
      py::arg("flow"),
      py::arg("balls"),
      py::arg("cylinders"),
      py::arg("cylinders_h"),
      py::arg("voxels"),
      py::arg("R"),
      py::arg("R_old"),
      py::arg("pos"),
      py::arg("pos_old"),
      py::arg("drone_radius"),
      py::arg("n_drones_per_group"),
      py::arg("fov_x_half_tan"),
      py::arg("has_ceiling") = false,
      py::arg("ceiling_height") = 3.0f);
  m.def("render_mid360",
      &render_mid360_cuda,
      "render_mid360 (CUDA)",
      py::arg("points"),
      py::arg("ranges"),
      py::arg("balls"),
      py::arg("cylinders"),
      py::arg("cylinders_h"),
      py::arg("voxels"),
      py::arg("R"),
      py::arg("pos"),
      py::arg("n_drones_per_group") = 1,
      py::arg("vertical_channels") = 64,
      py::arg("min_range") = 0.1f,
      py::arg("max_range") = 70.0f,
      py::arg("vertical_min_deg") = -7.0f,
      py::arg("vertical_max_deg") = 52.0f,
      py::arg("has_ceiling") = false,
      py::arg("ceiling_height") = 3.0f);
  m.def("points_to_pseudo_image_mid360",
      &points_to_pseudo_image_mid360_cuda,
      "points_to_pseudo_image_mid360 (CUDA)",
      py::arg("pseudo_image"),
      py::arg("points"),
      py::arg("ranges"),
      py::arg("min_range"),
      py::arg("max_range"),
      py::arg("theta_min"),
      py::arg("phi_min"),
      py::arg("theta_resolution"),
      py::arg("phi_resolution"));
  m.def("find_nearest_pt",
      &find_nearest_pt_cuda,
      "find_nearest_pt (CUDA)",
      py::arg("nearest_pt"),
      py::arg("balls"),
      py::arg("cylinders"),
      py::arg("cylinders_h"),
      py::arg("voxels"),
      py::arg("pos"),
      py::arg("drone_radius"),
      py::arg("n_drones_per_group"),
      py::arg("has_ceiling") = false,
      py::arg("ceiling_height") = 3.0f);
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
}
