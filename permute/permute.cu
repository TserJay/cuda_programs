#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

typedef float FLOAT;

// 使用 shared memory 优化的转置 kernel
#define TILE_SIZE 32

__global__ void transpose_shared(FLOAT* A, FLOAT* B, int rows, int cols)
{
    __shared__ float tile[TILE_SIZE * TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 全局坐标
    int gx = blockIdx.x * TILE_SIZE + tx;
    int gy = blockIdx.y * TILE_SIZE + ty;
    
    // 将数据从 A 加载到 shared memory
    if (gx < cols && gy < rows) {
        tile[ty * TILE_SIZE + tx] = A[gy * cols + gx];
    }
    __syncthreads();
    
    // 计算 B 中的目标位置（行列互换）
    int gx_new = blockIdx.y * TILE_SIZE + tx;
    int gy_new = blockIdx.x * TILE_SIZE + ty;
    
    // 将数据从 shared memory 写入 B（转置）
    if (gx_new < rows && gy_new < cols) {
        B[gy_new * rows + gx_new] = tile[tx * TILE_SIZE + ty];
    }
}

// 合并访问的转置（读取连续，写入连续）
__global__ void transpose_coalesced(FLOAT* A, FLOAT* B, int rows, int cols)
{
    __shared__ float tile[TILE_SIZE * TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 读取：A 中的 (by, bx) tile，线程 (tx, ty) 读取 (by*TILE_SIZE+ty, bx*TILE_SIZE+tx)
    int in_x = bx * TILE_SIZE + tx;
    int in_y = by * TILE_SIZE + ty;
    
    if (in_x < cols && in_y < rows) {
        tile[ty * TILE_SIZE + tx] = A[in_y * cols + in_x];
    }
    __syncthreads();
    
    // 写入：B 中的 (bx, by) tile，线程 (tx, ty) 写入 (by*TILE_SIZE+ty, bx*TILE_SIZE+tx)
    // 此时 tile[tx*TILE_SIZE+ty] 实际上是原矩阵的 (bx*TILE_SIZE+ty, by*TILE_SIZE+tx)
    int out_x = by * TILE_SIZE + tx;
    int out_y = bx * TILE_SIZE + ty;
    
    if (out_x < rows && out_y < cols) {
        B[out_y * rows + out_x] = tile[tx * TILE_SIZE + ty];
    }
}

// CPU 转置函数用于验证
void transpose_cpu(FLOAT* A, FLOAT* B, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

int main()
{
    // 使用更大的矩阵来更好地测试 GPU 性能
    int rows = 4096;
    int cols = 4096;
    int nbytes_A = rows * cols * sizeof(FLOAT);
    int nbytes_B = cols * rows * sizeof(FLOAT);
    
    // Block 和 Grid 配置
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, 
              (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Matrix dimensions: %d x %d\n", rows, cols);
    printf("Block size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Grid size: %d x %d\n", grid.x, grid.y);
    printf("Total threads: %d\n", grid.x * grid.y * TILE_SIZE * TILE_SIZE);
    
    FLOAT *d_A, *d_B;
    FLOAT *h_A, *h_B, *h_B_cpu;
    
    // 使用 pinned memory 加速数据传输
    cudaMallocHost((void**)&h_A, nbytes_A);
    cudaMallocHost((void**)&h_B, nbytes_B);
    cudaMallocHost((void**)&h_B_cpu, nbytes_B);
    
    // GPU 分配内存
    cudaMalloc((void**)&d_A, nbytes_A);
    cudaMalloc((void**)&d_B, nbytes_B);
    
    // 初始化输入矩阵
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_A[i * cols + j] = (FLOAT)(i * cols + j);
        }
    }
    
    // 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, nbytes_A, cudaMemcpyHostToDevice);
    
    // 计时器
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热 GPU（运行几次 kernel）
    for (int i = 0; i < 10; i++) {
        transpose_coalesced<<<grid, block>>>(d_A, d_B, rows, cols);
    }
    cudaDeviceSynchronize();
    
    // 多次运行取平均
    const int NUM_RUNS = 100;
    float total_time = 0;
    
    for (int i = 0; i < NUM_RUNS; i++) {
        cudaEventRecord(start);
        transpose_coalesced<<<grid, block>>>(d_A, d_B, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    
    float avg_time = total_time / NUM_RUNS;
    printf("GPU transpose time (avg of %d runs): %f ms\n", NUM_RUNS, avg_time);
    
    // 拷贝结果回 CPU
    cudaMemcpy(h_B, d_B, nbytes_B, cudaMemcpyDeviceToHost);
    
    // CPU 转置用于对比（只运行一次）
    cudaEventRecord(start);
    transpose_cpu(h_A, h_B_cpu, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("CPU transpose time: %f ms\n", milliseconds);
    printf("Speedup: %.2fx\n", milliseconds / avg_time);
    
    // 验证结果（前几个元素）
    int errors = 0;
    for (int i = 0; i < cols && i < 10; i++) {
        for (int j = 0; j < rows && j < 10; j++) {
            if (fabs(h_B[i * rows + j] - h_B_cpu[i * rows + j]) > 1e-6) {
                errors++;
            }
        }
    }
    
    // 全局验证（抽样检查）
    for (int i = 0; i < cols; i += 100) {
        for (int j = 0; j < rows; j += 100) {
            if (fabs(h_B[i * rows + j] - h_B_cpu[i * rows + j]) > 1e-6) {
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Verification: PASSED!\n");
    } else {
        printf("Verification: FAILED! %d errors\n", errors);
    }
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_B_cpu);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
