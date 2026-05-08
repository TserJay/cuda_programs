#include <torch/extenson.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


template<
    const int BLOCK_SIZE_M; // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K; // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N; // width of block of C that each thread block calculate
    const int THREAD_SIZE_X; // height of block of C that each thread calculate 
    const int THREAD_SIZE_Y; // height of block of C that each thread calculate

>

__global__ void gemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    float* int M,
    float* int N,
    float* int K,
    float alpha,
    float beta
){
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_N / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszx * bszy;

    //
}
