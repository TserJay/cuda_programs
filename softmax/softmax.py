import torch
from torch.utils.cpp_extension import load

lib = load(
    name="sofxmax_cuda",
    sources=["softmax.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"],
    verbose=True
)
# torch.cuda.set_device(1)


def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 20,
    iters: int = 1000,
):
    # 预热，让 CUDA kernel 编译、缓存和显存分配稳定
    for _ in range(warmup):
        out = perf_func(values)
    torch.cuda.synchronize()

    # 改用 CUDA Event 计时，而不是 time.time()。
    # 因为 kernel launch 默认是异步的，直接测 CPU 时间不准确
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = perf_func(values)
    end.record()
    torch.cuda.synchronize()

    mean_time = start.elapsed_time(end) / iters
    print(f"{tag:<30} out={out.item():<15.8f} time={mean_time:.8f} ms")
    return out, mean_time


def verify_result(cuda_output: torch.Tensor, values: torch.Tensor) -> bool:
    expected = values
    return torch.allclose(cuda_output, expected)





def main():
    values = torch.ones(25_600_000, device="cuda", dtype=torch.float32)

    kernels = [
        ("softmax_1", lib.block_all_reduce_sum_1),
      
    ]

    print(f"input numel={values.numel()} dtype={values.dtype} device={values.device}")
    for tag, kernel in kernels:
        cuda_output, _ = run_benchmark(kernel, values, tag)
        print(f"{tag:<30} correct={verify_result(cuda_output, values)}")


if __name__ == "__main__":
    main()

