#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>




// cal offset from row col an ld ,in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + col)


template <
    const int BLOCK_SIZE_M, // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K, // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N, // height of block of C that each thread block calculate
    const int THREAD_SIZE_X, // height of block of C that each thread calculate 
    const int THREAD_SIZE_Y // width of block of C that each thread calculate
    >

__global__ void gemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    float alpha,
    float beta
){
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszx * bszy;

    //
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared mem
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for c
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    // row number and col number that need to be loaded blockIdx.y this thread to shared mem 
    // 本质就是计算需要加载的索引
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    // 主循环沿着 K 维度
    for(int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K){
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE){
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW;
            const int col = A_TILE_COL + tile_idx;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1){
                As[i + A_TILE_ROW][A_TILE_COL] = row < M && col < K ? A[OFFSET(row,col,K)] : 0;
            }
            else{
                As[i + A_TILE_ROW][A_TILE_COL] = A[OFFSET(row,col,K)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1){
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(row,col,N)] : 0;
            }
            else{
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(row,col,N)];
            }
        }
        __syncthreads();

        // compoute c
        #pragma unroll
        for ( int k = 0; k < BLOCK_SIZE_K; ++k){
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y){
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x){
                    accum[thread_y][thread_x] += As[thread_y * A_S + threadIdx.y][k] * Bs[k][thread_x * B_S + threadIdx.x];
                }
            }
        }
        __syncthreads();

    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y){
        #pragma unroll
        for(int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x){
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
                }
            } else{
                C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
            }
        }
    }
        
}









