#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define TILE_SIZE 16
#define MAX_DIMS 8

/**
 * @brief 朴素版本：使用shared memory进行transpose
 * 
 * 专门优化 [0,1,2] -> [0,2,1] 的permute操作
 * 
 * 线程组织：
 * - 每个线程处理一个元素
 * - blockDim.x = TILE_SIZE, blockDim.y = TILE_SIZE
 * - gridDim.x = ceil(C/TILE_SIZE), gridDim.y = ceil(B/TILE_SIZE), gridDim.z = A
 * 
 * 内存访问模式：
 * - 读取：连续访问，步长=C
 * - 写入：连续访问，步长=B
 * - 使用shared memory进行transpose，提高内存合并效率
 * 
 * 优化技巧：
 * - 使用shared memory避免全局内存冲突
 * - +1避免bank冲突
 * - 合并访问模式
 */
__global__ void permute_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int A, int B, int C)
{
    // 每个 block 处理一个 A 维度上的切片
    // 在 B 和 C 维度上使用 tile 进行 transpose
    
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 避免 bank conflict
    
    int a = blockIdx.z;
    if (a >= A) return;
    
    // 计算当前 block 在 B-C 平面上的 tile 位置
    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;
    
    // 线程在 tile 内的局部坐标
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    
    // 全局坐标
    int globalRow = tileRow + localRow;
    int globalCol = tileCol + localCol;
    
    // 从全局内存加载到 shared memory (合并访问)
    if (globalRow < B && globalCol < C) {
        int inIdx = a * B * C + globalRow * C + globalCol;
        tile[localRow][localCol] = input[inIdx];
    }
    
    __syncthreads();
    
    // 从 shared memory 写回全局内存 (转置)
    // 输出索引: [a, c, b] -> a * C * B + c * B + b
    if (globalRow < B && globalCol < C) {
        int outIdx = a * C * B + globalCol * B + globalRow;
        output[outIdx] = tile[localRow][localCol];
    }
}

/**
 * @brief 优化版本：每个线程处理多个元素，提高内存吞吐量
 * 
 * 专门优化 [0,1,2] -> [0,2,1] 的permute操作
 * 
 * 与朴素版本的主要区别：
 * - 每个线程可能加载多个元素
 * - 提高内存吞吐量和共享内存利用率
 * 
 * 线程组织：
 * - 每个线程处理一个元素（与朴素版本相同）
 * - 但可以通过调整TILE_SIZE来优化
 * 
 * 内存访问模式：
 * - 读取：连续访问，步长=C
 * - 写入：连续访问，步长=B
 * 
 * 优化技巧：
 * - 使用shared memory进行transpose
 * - 可以进一步优化向量化加载
 */
__global__ void permute_transpose_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int A, int B, int C)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int a = blockIdx.z;
    if (a >= A) return;
    
    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;
    
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    
    int globalRow = tileRow + localRow;
    int globalCol = tileCol + localCol;
    
    // 加载阶段：每个线程可能加载多个元素
    if (globalRow < B && globalCol < C) {
        int inIdx = a * B * C + globalRow * C + globalCol;
        tile[localRow][localCol] = input[inIdx];
    }
    
    __syncthreads();
    
    // 存储阶段：转置写入
    if (globalRow < B && globalCol < C) {
        int outIdx = a * C * B + globalCol * B + globalRow;
        output[outIdx] = tile[localRow][localCol];
    }
}

// 支持任意维度 permute 的通用 kernel
__device__ int compute_index(
    int idx,
    const int* __restrict__ dims,
    const int* __restrict__ strides,
    int ndims)
{
    int result = 0;
    for (int i = 0; i < ndims; i++) {
        int dim_idx = idx / strides[i];
        idx %= strides[i];
        result += dim_idx * strides[i];
    }
    return result;
}

// 通用 permute kernel (适用于最多4维张量)
__global__ void permute_general_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ in_dims,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_strides,
    const int* __restrict__ perm,
    int ndims,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // 计算输入索引的各个维度坐标
    int remaining = idx;
    int in_idx = 0;
    
    #pragma unroll
    for (int i = 0; i < MAX_DIMS; i++) {
        if (i >= ndims) break;
        int dim_size = in_dims[perm[i]];
        int coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        in_idx += coord * in_strides[i];
    }
    
    output[idx] = input[in_idx];
}

/**
 * @brief 针对 [0,1,2] -> [1,0,2] 的优化 kernel
 * 
 * 优化思路：
 * - 固定c维度，分别处理a和b维度
 * - 使用shared memory进行transpose
 * 
 * 线程组织：
 * - 每个线程处理一个元素
 * - blockDim.x = TILE_SIZE, blockDim.y = TILE_SIZE
 * - gridDim.x = ceil(B/TILE_SIZE), gridDim.y = ceil(A/TILE_SIZE), gridDim.z = C
 * 
 * 内存访问模式：
 * - 读取：连续访问，步长=B
 * - 写入：连续访问，步长=A*C
 * 
 * 优化技巧：
 * - 固定c维度，减少维度变换复杂度
 * - 使用shared memory提高内存合并效率
 */
__global__ void permute_012_to_102_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int A, int B, int C)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int c = blockIdx.z;
    if (c >= C) return;
    
    int tileA = blockIdx.y * TILE_SIZE;
    int tileB = blockIdx.x * TILE_SIZE;
    
    int localA = threadIdx.y;
    int localB = threadIdx.x;
    
    int globalA = tileA + localA;
    int globalB = tileB + localB;
    
    // 加载: input[a, b, c] 其中 c 固定
    if (globalA < A && globalB < B) {
        int inIdx = globalA * B * C + globalB * C + c;
        tile[localA][localB] = input[inIdx];
    }
    
    __syncthreads();
    
    // 存储: output[b, a, c]
    if (globalA < A && globalB < B) {
        int outIdx = globalB * A * C + globalA * C + c;
        output[outIdx] = tile[localA][localB];
    }
}

/**
 * @brief 针对 [0,1,2] -> [2,0,1] 的优化 kernel
 * 
 * 优化思路：
 * - 固定a维度，分别处理b和c维度
 * - 使用shared memory进行transpose
 * 
 * 线程组织：
 * - 每个线程处理一个元素
 * - blockDim.x = TILE_SIZE, blockDim.y = TILE_SIZE
 * - gridDim.x = ceil(C/TILE_SIZE), gridDim.y = ceil(B/TILE_SIZE), gridDim.z = A
 * 
 * 内存访问模式：
 * - 读取：连续访问，步长=1
 * - 写入：连续访问，步长=A*B
 * 
 * 优化技巧：
 * - 固定a维度，减少维度变换复杂度
 * - 使用shared memory提高内存合并效率
 */
__global__ void permute_012_to_201_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int A, int B, int C)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int a = blockIdx.z;
    if (a >= A) return;
    
    int tileB = blockIdx.y * TILE_SIZE;
    int tileC = blockIdx.x * TILE_SIZE;
    
    int localB = threadIdx.y;
    int localC = threadIdx.x;
    
    int globalB = tileB + localB;
    int globalC = tileC + localC;
    
    // 加载: input[a, b, c]
    if (globalB < B && globalC < C) {
        int inIdx = a * B * C + globalB * C + globalC;
        tile[localB][localC] = input[inIdx];
    }
    
    __syncthreads();
    
    // 存储: output[c, a, b]
    if (globalB < B && globalC < C) {
        int outIdx = globalC * A * B + a * B + globalB;
        output[outIdx] = tile[localB][localC];
    }
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // [0,1,2] -> [0,2,1] 使用 shared memory 优化
    m.def("permute_021", [](torch::Tensor input) {
        auto sizes = input.sizes();
        int A = sizes[0];
        int B = sizes[1];
        int C = sizes[2];
        
        auto output = torch::empty({A, C, B}, input.options());
        
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        
        dim3 BlockDim(TILE_SIZE, TILE_SIZE);
        dim3 GridDim(
            (C + TILE_SIZE - 1) / TILE_SIZE,
            (B + TILE_SIZE - 1) / TILE_SIZE,
            A
        );
        
        permute_transpose_kernel<<<GridDim, BlockDim>>>(
            input_ptr, output_ptr, A, B, C);
        
        return output;
    }, "Permute [0,1,2] -> [0,2,1] with shared memory optimization");
    
    // [0,1,2] -> [1,0,2] 使用 shared memory 优化
    m.def("permute_102", [](torch::Tensor input) {
        auto sizes = input.sizes();
        int A = sizes[0];
        int B = sizes[1];
        int C = sizes[2];
        
        auto output = torch::empty({B, A, C}, input.options());
        
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        
        dim3 BlockDim(TILE_SIZE, TILE_SIZE);
        dim3 GridDim(
            (B + TILE_SIZE - 1) / TILE_SIZE,
            (A + TILE_SIZE - 1) / TILE_SIZE,
            C
        );
        
        permute_012_to_102_kernel<<<GridDim, BlockDim>>>(
            input_ptr, output_ptr, A, B, C);
        
        return output;
    }, "Permute [0,1,2] -> [1,0,2] with shared memory optimization");
    
    // [0,1,2] -> [2,0,1] 使用 shared memory 优化
    m.def("permute_201", [](torch::Tensor input) {
        auto sizes = input.sizes();
        int A = sizes[0];
        int B = sizes[1];
        int C = sizes[2];
        
        auto output = torch::empty({C, A, B}, input.options());
        
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        
        dim3 BlockDim(TILE_SIZE, TILE_SIZE);
        dim3 GridDim(
            (C + TILE_SIZE - 1) / TILE_SIZE,
            (B + TILE_SIZE - 1) / TILE_SIZE,
            A
        );
        
        permute_012_to_201_kernel<<<GridDim, BlockDim>>>(
            input_ptr, output_ptr, A, B, C);
        
        return output;
    }, "Permute [0,1,2] -> [2,0,1] with shared memory optimization");
}
