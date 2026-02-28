// BEV Pool v3 — PyTorch C++ bindings
// Dispatches to CUDA kernels in bev_pool_v3_cuda.cu

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// ======================== CUDA declarations ========================

void bev_pool_v3_fwd_f32(
    int c, int n_intervals,
    const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths,
    float* out);

void bev_pool_v3_fwd_f16(
    int c, int n_intervals,
    const void* depth, const void* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths,
    void* out);

void bev_pool_v3_fwd_bf16(
    int c, int n_intervals,
    const void* depth, const void* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths,
    void* out);

void bev_pool_v3_bwd_depth_f32(
    int c, int n_points,
    const float* out_grad, const float* feat,
    const int* ranks_bev, const int* ranks_feat, const int* ranks_depth,
    float* depth_grad);

void bev_pool_v3_bwd_depth_f16(
    int c, int n_points,
    const void* out_grad, const void* feat,
    const int* ranks_bev, const int* ranks_feat, const int* ranks_depth,
    float* depth_grad);

void bev_pool_v3_bwd_depth_bf16(
    int c, int n_points,
    const void* out_grad, const void* feat,
    const int* ranks_bev, const int* ranks_feat, const int* ranks_depth,
    float* depth_grad);

void bev_pool_v3_bwd_feat_f32(
    int c, int n_intervals,
    const float* depth, const float* out_grad,
    const int* ranks_depth, const int* ranks_bev, const int* ranks_feat,
    const int* interval_starts, const int* interval_lengths,
    float* feat_grad);

void bev_pool_v3_bwd_feat_f16(
    int c, int n_intervals,
    const void* depth, const void* out_grad,
    const int* ranks_depth, const int* ranks_bev, const int* ranks_feat,
    const int* interval_starts, const int* interval_lengths,
    void* feat_grad);

void bev_pool_v3_bwd_feat_bf16(
    int c, int n_intervals,
    const void* depth, const void* out_grad,
    const int* ranks_depth, const int* ranks_bev, const int* ranks_feat,
    const int* interval_starts, const int* interval_lengths,
    void* feat_grad);

void voxel_prepare_fused(
    const float* coor, int* out_ranks_bev, int* out_ranks_depth, int* out_ranks_feat,
    int* counter,
    float lower_x, float lower_y, float lower_z,
    float dx, float dy, float dz,
    int Dx, int Dy, int Dz,
    int B, int N, int D, int H, int W);

void compute_intervals_flags(
    const int* sorted_keys, int* is_start_flag, int n);

// ======================== PyTorch wrappers ========================

void bev_pool_v3_forward(
    const at::Tensor _depth,
    const at::Tensor _feat,
    at::Tensor _out,
    const at::Tensor _ranks_depth,
    const at::Tensor _ranks_feat,
    const at::Tensor _ranks_bev,
    const at::Tensor _interval_lengths,
    const at::Tensor _interval_starts)
{
    int c = _feat.size(-1);  // last dim is channels
    int n_intervals = _interval_lengths.size(0);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));

    if (_depth.scalar_type() == at::kFloat) {
        bev_pool_v3_fwd_f32(
            c, n_intervals,
            _depth.data_ptr<float>(), _feat.data_ptr<float>(),
            _ranks_depth.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_bev.data_ptr<int>(),
            _interval_starts.data_ptr<int>(), _interval_lengths.data_ptr<int>(),
            _out.data_ptr<float>());
    } else if (_depth.scalar_type() == at::kHalf) {
        bev_pool_v3_fwd_f16(
            c, n_intervals,
            _depth.data_ptr<at::Half>(), _feat.data_ptr<at::Half>(),
            _ranks_depth.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_bev.data_ptr<int>(),
            _interval_starts.data_ptr<int>(), _interval_lengths.data_ptr<int>(),
            _out.data_ptr<at::Half>());
    } else if (_depth.scalar_type() == at::kBFloat16) {
        bev_pool_v3_fwd_bf16(
            c, n_intervals,
            _depth.data_ptr<at::BFloat16>(), _feat.data_ptr<at::BFloat16>(),
            _ranks_depth.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_bev.data_ptr<int>(),
            _interval_starts.data_ptr<int>(), _interval_lengths.data_ptr<int>(),
            _out.data_ptr<at::BFloat16>());
    } else {
        AT_ERROR("bev_pool_v3_forward: unsupported dtype ", _depth.scalar_type());
    }
}

void bev_pool_v3_backward(
    const at::Tensor _out_grad,
    at::Tensor _depth_grad,          // always float32
    at::Tensor _feat_grad,
    const at::Tensor _depth,
    const at::Tensor _feat,
    // --- bev-sorted indices (for depth grad) ---
    const at::Tensor _ranks_depth,
    const at::Tensor _ranks_feat,
    const at::Tensor _ranks_bev,
    int n_points,
    // --- feat-sorted indices (for feat grad) ---
    const at::Tensor _ranks_depth_fs,
    const at::Tensor _ranks_feat_fs,
    const at::Tensor _ranks_bev_fs,
    const at::Tensor _interval_starts_fs,
    const at::Tensor _interval_lengths_fs)
{
    int c = _feat.size(-1);
    int n_feat_intervals = _interval_lengths_fs.size(0);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));

    if (_depth.scalar_type() == at::kFloat) {
        // depth gradient
        bev_pool_v3_bwd_depth_f32(
            c, n_points,
            _out_grad.data_ptr<float>(), _feat.data_ptr<float>(),
            _ranks_bev.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_depth.data_ptr<int>(),
            _depth_grad.data_ptr<float>());
        // feat gradient
        bev_pool_v3_bwd_feat_f32(
            c, n_feat_intervals,
            _depth.data_ptr<float>(), _out_grad.data_ptr<float>(),
            _ranks_depth_fs.data_ptr<int>(), _ranks_bev_fs.data_ptr<int>(),
            _ranks_feat_fs.data_ptr<int>(),
            _interval_starts_fs.data_ptr<int>(), _interval_lengths_fs.data_ptr<int>(),
            _feat_grad.data_ptr<float>());
    } else if (_depth.scalar_type() == at::kHalf) {
        bev_pool_v3_bwd_depth_f16(
            c, n_points,
            _out_grad.data_ptr<at::Half>(), _feat.data_ptr<at::Half>(),
            _ranks_bev.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_depth.data_ptr<int>(),
            _depth_grad.data_ptr<float>());
        bev_pool_v3_bwd_feat_f16(
            c, n_feat_intervals,
            _depth.data_ptr<at::Half>(), _out_grad.data_ptr<at::Half>(),
            _ranks_depth_fs.data_ptr<int>(), _ranks_bev_fs.data_ptr<int>(),
            _ranks_feat_fs.data_ptr<int>(),
            _interval_starts_fs.data_ptr<int>(), _interval_lengths_fs.data_ptr<int>(),
            _feat_grad.data_ptr<at::Half>());
    } else if (_depth.scalar_type() == at::kBFloat16) {
        bev_pool_v3_bwd_depth_bf16(
            c, n_points,
            _out_grad.data_ptr<at::BFloat16>(), _feat.data_ptr<at::BFloat16>(),
            _ranks_bev.data_ptr<int>(), _ranks_feat.data_ptr<int>(),
            _ranks_depth.data_ptr<int>(),
            _depth_grad.data_ptr<float>());
        bev_pool_v3_bwd_feat_bf16(
            c, n_feat_intervals,
            _depth.data_ptr<at::BFloat16>(), _out_grad.data_ptr<at::BFloat16>(),
            _ranks_depth_fs.data_ptr<int>(), _ranks_bev_fs.data_ptr<int>(),
            _ranks_feat_fs.data_ptr<int>(),
            _interval_starts_fs.data_ptr<int>(), _interval_lengths_fs.data_ptr<int>(),
            _feat_grad.data_ptr<at::BFloat16>());
    } else {
        AT_ERROR("bev_pool_v3_backward: unsupported dtype ", _depth.scalar_type());
    }
}

/*
  Fused voxel_pooling_prepare_v3:
    coor:  (B*N*D*H*W, 3) float — world coordinates (flattened from (B,N,D,H,W,3))
    Returns: (ranks_bev, ranks_depth, ranks_feat, n_valid)   via pre-allocated output tensors + counter
*/
int voxel_pooling_prepare_v3_fused(
    const at::Tensor _coor,
    at::Tensor _out_ranks_bev,
    at::Tensor _out_ranks_depth,
    at::Tensor _out_ranks_feat,
    at::Tensor _counter,               // int32 scalar tensor, initialised to 0
    float lower_x, float lower_y, float lower_z,
    float dx, float dy, float dz,
    int Dx, int Dy, int Dz,
    int B, int N, int D, int H, int W)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_coor));
    voxel_prepare_fused(
        _coor.data_ptr<float>(),
        _out_ranks_bev.data_ptr<int>(),
        _out_ranks_depth.data_ptr<int>(),
        _out_ranks_feat.data_ptr<int>(),
        _counter.data_ptr<int>(),
        lower_x, lower_y, lower_z,
        dx, dy, dz,
        Dx, Dy, Dz,
        B, N, D, H, W);
    // synchronise to read counter
    return _counter.item<int>();
}

/*
  Fused interval detection:
    sorted_keys (int tensor, sorted)  →  is_start_flag (int tensor, 0/1)
*/
void compute_intervals_v3(
    const at::Tensor _sorted_keys,
    at::Tensor _is_start_flag)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_sorted_keys));
    int n = _sorted_keys.size(0);
    compute_intervals_flags(
        _sorted_keys.data_ptr<int>(),
        _is_start_flag.data_ptr<int>(),
        n);
}

// ======================== PYBIND11 ========================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bev_pool_v3_forward",  &bev_pool_v3_forward,  "BEV Pool v3 forward");
    m.def("bev_pool_v3_backward", &bev_pool_v3_backward, "BEV Pool v3 backward");
    m.def("voxel_pooling_prepare_v3_fused", &voxel_pooling_prepare_v3_fused,
          "Fused voxel pooling prepare v3");
    m.def("compute_intervals_v3", &compute_intervals_v3, "Fused interval detection");
}
