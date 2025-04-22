
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

typedef float FLOAT;
//output:
//check pass!
//GPU execution time : 57.808384 ms
//CPU execution time : 275.917206 ms


/*cuda kernel function*/
__global__ void vector_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 2d gird*/
    int idx = blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x;
    /* 1d gird*/
    //int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        z[idx] = x[idx] + y[idx];
}

void vector_add_cpu(FLOAT* x, FLOAT* y, FLOAT* z, int N) 
{
    for (size_t i = 0; i < N; i++)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{   
    int N = 100000000;
    int nbytes = N * sizeof(FLOAT); //N个float所需要分配的内存大小

    /* 1d block */
    int bs = 256;

    /* 2d grid */
    int s = ceil(sqrt((N + bs - 1) / bs));

    /* 构造一个 2D 网格，每行 s 个 block，共 s 行。*/
    dim3 grid(s, s);

    FLOAT * dx, * hx;
    FLOAT * dy, * hy;
    FLOAT * dz, * hz;

    /* gpu分配内存空间 */
    cudaMalloc((void**)&dx, nbytes);
    cudaMalloc((void**)&dy, nbytes);
    cudaMalloc((void**)&dz, nbytes);

    /* 计时器，单位ms*/
    float milliseconds = 0;
    float milliseconds_cpu = 0;

    /* cpu分配内存空间 */
    hx = (FLOAT*)malloc(nbytes);
    hy = (FLOAT*)malloc(nbytes);
    hz = (FLOAT*)malloc(nbytes);


    /* init */
    for (int i = 0; i < N; i++)
    {
        hx[i] = 1;
        hy[i] = 1;
    }

    /* cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost*/
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);


    /*  创建事件函数 */
    cudaEvent_t start, stop; // CUDA 事件 类型的变量
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    
    /* 时间记录起点*/
    cudaEventRecord(start);

    /* launchc kernel */
    vector_add << <grid, bs >> > (dx, dy, dz, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 主机（CPU）等待，直到 stop 事件完成

    cudaEventElapsedTime(&milliseconds, start, stop); //计算时间

    /* copy GPU result to CPU*/
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    /* CPU compute*/
    FLOAT* hz_cpu = (FLOAT*)malloc(nbytes);
  /*  vector_add_cpu(hx, hy, hz_cpu, N);*/

    // 记录 CPU 计算时间
    cudaEventRecord(start_cpu);
    vector_add_cpu(hx, hy, hz_cpu, N);  // 执行 CPU 计算
    cudaEventRecord(stop_cpu); // 结束计时
    cudaEventSynchronize(stop_cpu);

    // 计算时间差，单位为毫秒
    cudaEventElapsedTime(&milliseconds_cpu, start_cpu, stop_cpu); //计算时间
   


    /* check GPU result with CPU*/
    for (int i = 0; i < N; i++)
    {
        if (fabs(hz_cpu[i] - hz[i]) > 1e-6) 
        {
            printf("erro %d\n", i);
        }
    }
    printf("check pass!\n");
    printf("GPU execution time: %f ms\nCPU execution time: %f ms\n  ", milliseconds, milliseconds_cpu);
 
   
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu);


    return 0;
}
