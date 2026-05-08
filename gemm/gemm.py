import os
import torch
from torch.utils.cpp_extension import load


lib = load(
    name="gemm",
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"],
    verbose=True
)

torch.cuda.set_device(1)

def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 20,
    iters: int = 1000,
):
    for _ in range(warmup):
        out = perf_func(values)
    torch.cuda.synchronize()

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


def varify_result(cuda_output: torch.Tensor, values: torch.Tensor) -> bool:
    """
        验证
    """

def main():
    values = torch

    Ms = [4096, 8192, 16384]
    Ns = [4096, 8192, 16384]
    Ks = [2048, 4096, 8192]
    MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
    # pre allocate for fast profiling.
    A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
    B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
    C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
    torch.cuda.synchronize()

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    for M, N, K in MNKs: