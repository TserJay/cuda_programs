#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;


// cuda kernel
// 1. permute [0,1,2]->[0,2,1]
__global__ void permute_1(const float* input, float* output, int A, int B, int C) {
    int a = blockIdx.z;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || c >= C) return;
    int in_idx = a * B * C + b * C + c;
    int out_idx = a * C * B + c * B + b;

    output[out_idx] = input[in_idx];
}

// 2. permute [0,1,2]->[1,0,2]

__global__ void permute_2(const float* input, float* output, int A, int B, int C){
    int b = blockIdx.z;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (a >= A || c >= C) return;
    int in_idx = a * B * C + b * C + c;
    int out_idx = b * A * C + a * C +c;
    output[out_idx] = input[in_idx];
}



void permute_1_cpu(const float* input, float* output, int A, int B, int C) {
    for (int a = 0; a < A; a++) {
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                int in_idx = a * B * C + b * C + c;
                int out_idx = a * C * B + c * B + b;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

void permute_2_cpu(const float* input, float* output, int A, int B, int C) {
    for (int b = 0; b < B; b++) {
        for (int a = 0; a < A; a++) {
            for (int c = 0; c < C; c++) {
                int in_idx = a * B * C + b * C + c;
                int out_idx = b * A * C + a * C +c;
                output[out_idx] = input[in_idx];
            }
        }
    }
}


bool verify_result(const float* gpu_output, const float* cpu_output, int size) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_output[i] - cpu_output[i]) > epsilon) {
            printf("验证失败, 索引:%d, GPU值:%f, CPU值:%f\n",
                   i, gpu_output[i], cpu_output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    
    // 形状为[A,B,C]
    int A = 2;
    int B = 3;
    int C = 4;
    int n = A * B * C;

    // 分配内存空间
    float* h_input = nullptr;
    float* h_output = nullptr;
    float* h_output_cpu = nullptr;

    cudaMallocHost(&h_input, n * sizeof(float));
    cudaMallocHost(&h_output, n * sizeof(float));
    cudaMallocHost(&h_output_cpu, n*sizeof(float));

    for(int i = 0; i < n; i++){
        h_input[i] = i;
    }

    float* d_input = nullptr; 
    float* d_output_1 = nullptr;
    float* d_output_2 = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output_1, n * sizeof(float));
    cudaMalloc(&d_output_2, n * sizeof(float));


    // cpy Host to Device
    cudaMemcpy(d_input, h_input, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 BlockDim(16,16);
    dim3 GridDim1(
        (C + BlockDim.x -1) / BlockDim.x,
        (B + BlockDim.y -1) / BlockDim.y,
        A
    );
    dim3 GridDim2(
        (C + BlockDim.x -1) / BlockDim.x,
        (A + BlockDim.y -1) / BlockDim.y,
        B
    );
    // gpu
    permute_1<<<GridDim1,BlockDim>>>(d_input, d_output_1, A, B, C);
    permute_2<<<GridDim2,BlockDim>>>(d_input, d_output_2, A, B, C);
    cudaDeviceSynchronize();

    // cpy Device to Host
    cudaMemcpy(h_output, d_output_1, n*sizeof(float), cudaMemcpyDeviceToHost);

    // cpu 
    permute_1_cpu(h_input, h_output_cpu, A, B, C);
    printf("1.permute [0,1,2]->[0,2,1]: ");
    if(verify_result(h_output, h_output_cpu, n)){
        printf("True\n");
    }else{
        printf("false\n");
    }


    cudaMemcpy(h_output, d_output_2, n*sizeof(float), cudaMemcpyDeviceToHost);
    // cpu 
    permute_2_cpu(h_input, h_output_cpu, A, B, C);
    printf("2.permute [0,1,2]->[1,0,2]: ");
    if(verify_result(h_output, h_output_cpu, n)){
        printf("True\n");
    }else{
        printf("false\n");
    }



    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFreeHost(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output_1);
    cudaFree(d_output_2);

    return 0;
}
