import torch
from torch.utils.cpp_extension import load

# JIT 编译 CUDA kernel
lib = load(
    name="gemm",
    sources=["gemm.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"],
    verbose=True
)


def run_benchmark(
    func: callable,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    tag: str,
    warmup: int = 20,
    iters: int = 200,
):
    # 预热，让 CUDA kernel 编译、缓存和显存分配稳定
    for _ in range(warmup):
        C.zero_()
        func(A, B, C, M, N, K, 1.0, 0.0)
    torch.cuda.synchronize()

    # 用 CUDA Event 计时，因为 kernel launch 是异步的，直接用 time.time() 不准
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        func(A, B, C, M, N, K, 1.0, 0.0)
    end.record()
    torch.cuda.synchronize()

    mean_time = start.elapsed_time(end) / iters
    # GEMM 浮点运算量 = 2 * M * N * K（乘 + 加）
    flops = 2.0 * M * N * K
    tflops = flops / (mean_time * 1e-3) / 1e12
    print(f"{tag:<40} M={M:<6} N={N:<6} K={K:<6} time={mean_time:>8.4f} ms  {tflops:>6.2f} TFLOPS")
    return mean_time


def verify_result(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int) -> bool:
    """用 torch.mm (cuBLAS) 的结果作为参考，验证自定义 kernel 的正确性"""
    a = A[:M, :K]
    b = B[:K, :N]
    ref = a @ b  # cuBLAS 参考结果
    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    lib.gemm(A, B, C, M, N, K, 1.0, 0.0)
    c = C[:M, :N]
    # float32 tiled GEMM 累加顺序不同，误差会比较大，放宽容忍度
    return torch.allclose(ref, c, atol=1e-2, rtol=1e-2)


def main():
    Ms = [4096, 8192, 16384]
    Ns = [4096, 8192, 16384]
    Ks = [2048, 4096, 8192]
    MAX_M, MAX_N, MAX_K = 16384, 16384, 8192

    # 预分配最大尺寸的矩阵，后面用切片来跑不同大小，避免反复分配显存
    A = torch.randn((MAX_M, MAX_K), dtype=torch.float32, device="cuda")
    B = torch.randn((MAX_K, MAX_N), dtype=torch.float32, device="cuda")
    C = torch.zeros((MAX_M, MAX_N), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    # 先做正确性验证
    print("--- 正确性验证 ---")
    for M, N, K in [(256, 256, 256), (4096, 4096, 2048)]:
        ok = verify_result(A, B, M, N, K)
        print(f"  M={M:<6} N={N:<6} K={K:<6} correct={ok}")

    # 性能对比：自定义 GEMM vs cuBLAS (torch.mm)
    print("\n--- 性能对比: custom GEMM vs cuBLAS (torch.mm) ---")
    print(f"{'kernel':<40} {'shape':>20} {'time':>12} {'perf':>14}")
    print("-" * 90)

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    for M, N, K in MNKs:
        # 切片取对应大小，contiguous() 确保内存连续
        a = A[:M, :K].contiguous()
        b = B[:K, :N].contiguous()
        c = C[:M, :N].contiguous()

        # 自定义 kernel 性能测试
        run_benchmark(lib.gemm, A, B, C, M, N, K, tag="custom_gemm")

        # cuBLAS 基准测试 (torch.mm)
        torch.cuda.synchronize()
        for _ in range(20):  # 预热
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            torch.mm(a, b, out=c)
        end.record()
        torch.cuda.synchronize()

        cublas_time = start.elapsed_time(end) / 200
        flops = 2.0 * M * N * K
        cublas_tflops = flops / (cublas_time * 1e-3) / 1e12
        print(f"{'cublas (torch.mm)':<40} M={M:<6} N={N:<6} K={K:<6} time={cublas_time:>8.4f} ms  {cublas_tflops:>6.2f} TFLOPS")
        print()


if __name__ == "__main__":
    main()
