#include <torch/extension.h>
#include <vector>

// Declaration of the actual CUDA implementation
std::vector<at::Tensor> compute_sparse_softmax_combined(
    const at::Tensor& x,
    const at::Tensor& y,
    float p_min,
    float threshold
);

// Python-facing wrapper
std::vector<at::Tensor> forward_wrapper(
    const at::Tensor& x,
    const at::Tensor& y,
    double p_min,
    double threshold
) {
    return compute_sparse_softmax_combined(
        x,
        y,
        static_cast<float>(p_min),
        static_cast<float>(threshold)
    );
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "APML Sparse Forward (Row + Col + Matching + COO)");
}