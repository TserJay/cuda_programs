#include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32



__inline__ __device__ float warp_reduce_sum(float sum) {
    // 这个函数只做一件事：
    // 在一个 warp(32 个线程) 内，把每个线程手里的值继续相加，最后得到一个 warp 的和。
    // _3 用到了 warp shuffle，所以这里保留这个小函数会比把逻辑直接塞进 kernel 更好读一点。
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}


// baseline 实现
__global__ void block_all_reduce_sum_1(const float* input, float* output, int n) {
    __shared__ float tile[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程先从全局内存读一个元素到 shared memory。
    // 如果最后一个 block 不满，就补 0，避免越界。
    tile[tid] = input[gtid];
    __syncthreads();

    // _1 是最朴素的 block 内 reduce 写法：
    // 第 1 轮：0 加 1，2 加 3，4 加 5 ...
    // 第 2 轮：0 加 2，4 加 6 ...
    // 第 3 轮：0 加 4 ...
    // 最后 tile[0] 就是这个 block 的和。
    for (int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) {
        // if (tid & (2 * index - 1) == 0) {
            tile[tid] += tile[tid + index];
        }
        __syncthreads();
    }

    // 这里只写出“当前 block 的和”。
    // 所以 _1 kernel 的直接输出是一个 partial sums 数组，不是最终总和。
    if (tid == 0) {
        output[blockIdx.x] = tile[tid];
    }
}


// _1 中的循环，由于存在 warp divergence ，每次分支所有的线程都会执行，导致带宽的浪费
__global__ void block_all_reduce_sum_11(const float* input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];  // 修复:添加float类型
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = input[gtid];  
    __syncthreads();

    for(unsigned int i = 1; i < blockDim.x; i = i << 1){
        int index = 2*i*tid;
        if(index < blockDim.x)
            sdata[index] += sdata[index + i];
        __syncthreads();  // if外部，for循环体内
    }

    if(tid == 0)
        output[blockIdx.x] = sdata[tid];
}


// 消除 bank conflict 
__global__ void block_all_reduce_sum_2(const float* input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = input[gtid];
    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i = i >> 1){
        if (tid < i){
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0)
        output[blockIdx.x] = sdata[tid];
}

// 解决 idle 线程
__global__ void block_all_reduce_sum_3(const float* input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];

    //each thread loads one element from global memory to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockDim.x * 2) + tid;
    sdata[tid] = input[gtid] + input[gtid + blockDim.x];
    __syncthreads();

    // do reduce in shared memory
    for(unsigned int i = blockDim.x / 2; i > 0; i = i >> 1){
        if(tid < i)
            sdata[tid] += sdata[tid + i];
        __syncthreads();    
    }
    // store result to global mem
    if(tid == 0)
        output[blockIdx.x] = sdata[tid]; 
}


// unroll warp 0
__device__ void warpreduce_4(volatile float* cache, int tid){
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}
__global__ void block_all_reduce_sum_4(const float* input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockDim.x * 2) + tid;
    sdata[tid] = input[gtid] + input[gtid + blockDim.x];
    __syncthreads();

    for(unsigned int i = blockDim.x / 2; i > 32; i = i >> 1){
        if(tid < i)
            sdata[tid] += sdata[tid + i];
        __syncthreads();
    }

    // unroll warp 0 
    if(tid < 32)
        warpreduce_4(sdata, tid);
    //store result to global mem
    if(tid == 0)
        output[blockIdx.x] = sdata[tid];
}


// unroll all warp
template <unsigned int blocksize>
__device__ void warpreduce_5(volatile float* cache, int tid){
    if(blocksize >= 64) cache[tid] += cache[tid + 32];
    if(blocksize >= 32) cache[tid] += cache[tid + 16];
    if(blocksize >= 16) cache[tid] += cache[tid + 8];
    if(blocksize >= 8) cache[tid] += cache[tid + 4];
    if(blocksize >= 4) cache[tid] += cache[tid + 2];
    if(blocksize >= 2) cache[tid] += cache[tid + 1];
}
template <unsigned int blocksize>
__global__ void block_all_reduce_sum_5(const float* input, float* output){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = input[gtid] + input[gtid + blockDim.x];
    __syncthreads();

    // do reudce in block 
    if(blocksize >= 512){
        if(tid < 256)
            sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
    if(blocksize >= 256){
        if(tid < 128)
            sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if(blocksize >= 128){
        if(tid < 64)
            sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
    
    // write result to global mem
    if(tid < 32)
        warpreduce_5<blocksize>(sdata, tid);
    if(tid == 0)
        output[blockIdx.x] = sdata[tid];
}


// grid-stride loop
template<unsigned int blocksize>
__device__ void warpreduce_6(volatile float* cache, int tid){
    if(blocksize >= 64) cache[tid] += cache[tid + 32];
    if(blocksize >= 32) cache[tid] += cache[tid + 16];
    if(blocksize >= 16) cache[tid] += cache[tid + 8];
    if(blocksize >= 8) cache[tid] += cache[tid + 4];
    if(blocksize >= 4) cache[tid] += cache[tid + 2];
    if(blocksize >= 2) cache[tid] += cache[tid + 1];
}
template<unsigned int blocksize>
__global__ void block_all_reduce_sum_6(const float* input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockDim.x * 2) + tid;

    unsigned int stride = blocksize * 2 * gridDim.x;
    sdata[tid] = 0;

    while(gtid < n){
        sdata[tid] += input[gtid] + input[gtid + blocksize];
        gtid += stride;
    }
    __syncthreads();

    // do reudce in block 
    if(blocksize >= 512){
        if(tid < 256)
            sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
    if(blocksize >= 256){
        if(tid < 128)
            sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if(blocksize >= 128){
        if(tid < 64)
            sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
    
    // write result to global mem
    if(tid < 32)
        warpreduce_6<blocksize>(sdata, tid);
    if(tid == 0)
        output[blockIdx.x] = sdata[tid];
}



// grid-stride loop
__global__ void block_all_reduce_sum_101(const float* input, float* output, int n) {
    __shared__ float tile[BLOCK_SIZE];

    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // 每个线程先在寄存器里累加多个元素。
    // 这样线程不是只干一次活，而是沿着 stride 一直往后处理。
    float sum = 0.0f;
    for (int i = start; i < n; i += stride) {
        sum += input[i];
    }

    tile[tid] = sum;
    __syncthreads();

    // “从小跨度往上翻倍”，改成更常见的“每轮减半”。
    // 第 1 轮：前 128 个线程各自加后 128 个线程的值
    // 第 2 轮：前 64 个线程各自加后 64 个线程的值
    // ...
    // 最后 tile[0] 还是当前 block 的和。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }

    // _2 和 _1 一样，这里写出的仍然是每个 block 的 partial sum。
    if (tid == 0) {
        output[blockIdx.x] = tile[0];
    }
}






__global__ void block_all_reduce_sum_102(const float* input, float* output, int n) {
    __shared__ float warp_sum[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // 目标是：直接得到“全局总和”，不再先写一整个 partial sums 数组。
    // 这里每个 block 初始会看 2 * BLOCK_SIZE 个元素：
    // 当前线程先读一个，再尝试多读一个相邻位置。
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    int stride = blockDim.x * gridDim.x * 2;

    float sum = 0.0f;
    while (idx < n) {
        sum += input[idx];

        if (idx + blockDim.x < n) {
            sum += input[idx + blockDim.x];
        }

        idx += stride;
    }

    // 第 1 步：先在 warp 内做 reduce。
    // 这样每个 warp 最后只剩一个结果。
    sum = warp_reduce_sum(sum);

    // 每个 warp 的第 0 号线程把 warp 的结果写到 shared memory。
    if (lane == 0) {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    // 第 2 步：只让第 0 个 warp 再把所有 warp 的结果做一次 reduce。
    if (warp_id == 0) {
        if (tid < blockDim.x / WARP_SIZE) {
            sum = warp_sum[tid];
        } else {
            sum = 0.0f;
        }

        sum = warp_reduce_sum(sum);

        // 第 3 步：当前 block 的和通过 atomicAdd 加到全局 output[0] 上。
        // 所以 _3 的 kernel 自己就能直接产出整个输入的总和。
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Python 模块绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_all_reduce_sum_1", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 256 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 想发挥 grid-stride loop 的优势，不能把 gridSize 开得过大。
        // 如果 block 太多，stride 会接近整个输入长度，线程往往只能处理 1 个元素，
        // 这样 _2 就退化了，性能通常会明显变差。
        // int gridSize = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096);
        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_1<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_1");


    m.def("block_all_reduce_sum_11", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 256 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_11<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_11");


    m.def("block_all_reduce_sum_2", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 256 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_2<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_2");


    m.def("block_all_reduce_sum_3", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 512 个元素开一个 block。
        // block per threads not change,but gridsize
        int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_3<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_3");


    m.def("block_all_reduce_sum_4", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 512 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_4<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_4");


    // unroll all warp 
    m.def("block_all_reduce_sum_5", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 512 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_5<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_5");


    m.def("block_all_reduce_sum_6", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 512 个元素开一个 block。
        int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        // int gridSize = std::min(4096, (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));

        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_6<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // 再把每个 block 的 partial sum 求一次和，得到最终结果。
        return output.sum();
    }, "block_all_reduce_sum_6");


    m.def("block_all_reduce_sum_101", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：每 256 个元素开一个 block。
        // int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // _2 想发挥 grid-stride loop 的优势，不能把 gridSize 开得过大。
        // 如果 block 太多，stride 会接近整个输入长度，线程往往只能处理 1 个元素，
        // 这样 _2 就退化了，性能通常会明显变差。
        int gridSize = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096);
        auto output = torch::zeros(gridSize, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_101<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // _2 也是先写 partial sums，再在 PyTorch 侧做一次 sum。
        return output.sum();
    }, "block_all_reduce_sum_101");

    m.def("block_all_reduce_sum_102", [](torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(input.dim() == 1, "this demo only supports 1D tensors");

        auto x = input.contiguous();
        int n = x.size(0);

        if (n == 0) {
            auto output = torch::zeros(1, x.options());
            return output[0];
        }

        // 这是最直观的写法：按每个 block 初始覆盖 2 * BLOCK_SIZE 个元素来开 block。
        // int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        // _3 也用了 grid-stride loop。为了让每个线程在循环里处理多个元素，
        // 这里同样限制一下 block 数量，避免 stride 过大导致线程只做很少的工作。
        int gridSize = std::min((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2), 4096);
        auto output = torch::zeros(1, x.options());

        const float* input_ptr = x.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        block_all_reduce_sum_102<<<gridSize, BLOCK_SIZE>>>(input_ptr, output_ptr, n);

        // _3 kernel 自己就会把所有 block 的结果加到 output[0] 里。
        return output[0];
    }, "block_all_reduce_sum_102");
}
