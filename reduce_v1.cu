﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>

// v1新版本: 用位运算替换除余操作
// latency: 2.825ms
// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L120)


// 4060 laptop
// allcated 100000 blocks, data counts are 25600000
// the ans is right
// reduce_v0 latency = 5.077600 ms


template<int blockSize>
__global__ void reduce_v1(float* d_in, float* d_out) {
    // 泛指当前线程在其block内的id
    int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    int gtid = threadIdx.x + blockIdx.x * blockSize;
    // load: 每个线程加载一个元素到shared mem对应位置
    __shared__ float smem[blockSize];
    smem[tid] = d_in[gtid];
    // 每对shared memory做读写操作都需要加__syncthreads保证一个block内的threads此刻都同步，以防结果错误
    __syncthreads();

    for (int index = 1; index < blockDim.x; index *= 2) {

        // v1的旧写法: 因为消除了v0的除余操作，速度相比v0有提升: 
        /*if (tid % (2 * index) == 0) {
            smem[tid] += smem[tid + index];
        }*/
        
        // 算法思路和v0一致，仅仅是用位运算替代了v0 if语句中的除余操作
        if ((tid & (2 * index - 1)) == 0) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}
bool CheckResult(float* out, float groudtruth, int n) {
    float res = 0;
    for (int i = 0; i < n; i++) {
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main() {
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float* a = (float*)malloc(N * sizeof(float));
    float* d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));

    float* out = (float*)malloc((GridSize) * sizeof(float));
    float* d_out;
    cudaMalloc((void**)&d_out, (GridSize) * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v1<blockSize> << <Grid, Block >> > (d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d\n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if (is_right) {
        printf("the ans is right\n");
    }
    else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
