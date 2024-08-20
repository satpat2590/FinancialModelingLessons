#include <torch/extension.h>
#include <vector>

// C++ function for matrix multiplication
torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    return torch::mm(a, b);
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix multiplication (CPU)");
}
