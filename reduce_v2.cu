
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// V0,2060
// reduce_v0 latency = 16.971935 ms

// v1,2060: 用位运算替换除余操作
// reduce_v0 latency = 11.568448 

// v2,2060
// reduce_v2 latency = 10.619200 ms



// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L120)
template<int blockSize>

__global__ void reduce_v1(float* d_in, float* d_out) {
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = threadIdx.x + blockIdx.x * blockSize;
    // load: 每个线程加载一个元素到shared mem对应位置
    __shared__ float smem[blockSize];
    smem[tid] = d_in[gtid];
    // 每对shared memory做读写操作都需要加__syncthreads保证一个block内的threads此刻都同步，以防结果错误
    __syncthreads();

    // 消除了bank conflict ，并且使用unsigned int 代替 int
    for(unsigned int index = blockDim.x / 2; index > 0; index >>=1) {
        if (tid < index) {
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
    printf("reduce_v2 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
