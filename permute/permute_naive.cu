#include <cuda_runtime.h>
#include <stdio.h>


__global__ void permute_naive(const float* input, float* output, int D1, int D2){

    int batch = blockIdx.z;
    int d1 = blockIdx.x * blockDim.x + threadIdx.x;
    int d2 = blockIdx.y * blockDim.y + threadIdx.y;

    if(d1 >= D1 || d2 >=D2) return;

    // 遍历
    int in_idx = batch *  D1 * D2 + d1 * D1 + d2;
    int out_idx = batch * D1 * D2 + d2 * D2 + d1;

    output[out_idx] = input[in_idx];
}


int main() {
    // 参数设置
    int D0 = 2, D1 = 3, D2 = 4;
    int size = D0 * D1 * D2;
    
    // 1. 分配主机内存
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    
    // 2. 初始化输入数据
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }
    
    // 3. 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // 4. 拷贝数据到设备
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 5. 配置kernel启动参数
    dim3 block(4, 4);  // 每个block 16个线程
    dim3 grid((D1 + block.x - 1) / block.x, 
              (D2 + block.y - 1) / block.y, 
              D0);
     
    // 6. 启动kernel
    permute_naive<<<grid, block>>>(d_input, d_output, D1, D2);
    
    // 7. 拷贝结果回主机
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 8. 验证结果
    printf("输入: ");
    for (int i = 0; i < size; i++) {
        printf("%.0f ", h_input[i]);
    }
    printf("\n输出: ");
    for (int i = 0; i < size; i++) {
        printf("%.0f ", h_output[i]);
    }
    printf("\n");
    
    // 9. 清理内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
