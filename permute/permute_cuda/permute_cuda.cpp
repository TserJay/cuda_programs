/*
 * PyTorch C++/CUDA Extension
 * 封装CUDA算子供Python调用
 */

#include <torch/extension.h>
#include <vector>

// 声明CUDA函数（在.cu文件中定义）
torch::Tensor permute_1_cuda(torch::Tensor input, int A, int B, int C);
torch::Tensor permute_2_cuda(torch::Tensor input, int A, int B, int C);
torch::Tensor permute_3_cuda(torch::Tensor input, int A, int B, int C);

// C++包装函数 - 输入检查和调用CUDA函数
torch::Tensor permute_1(torch::Tensor input) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int A = input.size(0);
    int B = input.size(1);
    int C = input.size(2);

    return permute_1_cuda(input, A, B, C);
}

torch::Tensor permute_2(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int A = input.size(0);
    int B = input.size(1);
    int C = input.size(2);

    return permute_2_cuda(input, A, B, C);
}

torch::Tensor permute_3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int A = input.size(0);
    int B = input.size(1);
    int C = input.size(2);

    return permute_3_cuda(input, A, B, C);
}

// Pybind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permute_1", &permute_1, "Permute [0,1,2] -> [0,2,1] (CUDA)");
    m.def("permute_2", &permute_2, "Permute [0,1,2] -> [1,0,2] (CUDA)");
    m.def("permute_3", &permute_3, "Permute [0,1,2] -> [2,0,1] (CUDA)");
}
