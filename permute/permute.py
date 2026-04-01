
import torch
import numpy as np
import time
from torch.utils.cpp_extension import load


# loda the cuda kernels as a python module
lib = load(
    name = "permute_cuda",
    sources = ["permute/permute_naive.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"]
)

def run_benchmark(
    perf_func:callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 100

):
    for i in range(warmup):
        _ = perf_func(values)
    torch.cuda.synchronize()

    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # 转换为毫秒
    mean_time = total_time / iters 

    out_info = f"out_{tag}"
   

    if tag.startswith("i8"):
        print(f"time:{mean_time:.8f} ms")
    else:
        print(f"time:{mean_time:.8f} ms")

    return out,mean_time

def verify_result(torch_output, numpy_input, permute_order):
    """
    验证CUDA permute结果是否与numpy.transpose一致
    
    Args:
        torch_output: CUDA permute的输出张量
        numpy_input: numpy输入数组
        permute_order: permute的轴顺序，如 (0, 2, 1)
    
    Returns:
        bool: 是否匹配
    """
    # 使用numpy的transpose进行permute
    numpy_output = np.transpose(numpy_input, permute_order)
    
    # 转换为torch张量进行比较
    torch_output_cpu = torch_output.cpu().numpy()
    
    # 检查形状是否一致
    if torch_output_cpu.shape != numpy_output.shape:
        print(f"形状不匹配: torch {torch_output_cpu.shape} vs numpy {numpy_output.shape}")
        return False
    
    # 检查值是否接近
    is_close = np.allclose(torch_output_cpu, numpy_output, rtol=1e-5, atol=1e-5)
    
    return is_close



# 测试3D张量的permute
print("-" * 60)
print("测试3D张量 permute [0,1,2] -> [0,2,1]")

# 创建3D测试张量
values_3d = torch.randn((3, 4, 5)).cuda().float()
print(f"输入形状: {values_3d.shape}")

# 运行CUDA permute
cuda_output, _ = run_benchmark(lib.permute_1, values_3d, "permute: [0,1,2] -> [0,2,1]")

# 使用numpy验证
numpy_input = values_3d.cpu().numpy()
is_correct = verify_result(cuda_output, numpy_input, (0, 2, 1))

print(f"\n验证结果: {'True' if is_correct else 'False'}")


print("\n" + "-" * 60)
print("测试不同形状的3D张量")


test_shapes = [(2, 3, 4), (4, 5, 6), (8, 16, 32)]

for shape in test_shapes:
    print(f"\n测试形状: {shape}")
    values = torch.randn(shape).cuda().float()
    cuda_output, _ = run_benchmark(lib.permute_1, values, f"shape_{shape}")
    
    # numpy验证
    numpy_input = values.cpu().numpy()
    is_correct = verify_result(cuda_output, numpy_input, (0, 2, 1))
    print(f"验证: {'True' if is_correct else 'False'}")

print("\n" + "-" * 60)
print("测试完成")

