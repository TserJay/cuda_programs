
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>


#include <stdio.h>
#include <stdlib.h>


__global__ void histgram(int* hist_data, int* bin_data) {
	int gitd = blockIdx.x * blockDim.x + threadIdx.x;

	// 原子加法指令，计算统计结果
	atomicAdd(&bin_data[hist_data[gitd]], 1);

}
bool CheckResult(int *out, int * groudtruth, int N) {
	for (int i = 0; i < N; i++) {
		if (out[i] != groudtruth[i]) {
			return false;
		}
	}
	return true;

}
int main() {
	float milliseconds = 0;
	const int N = 25600000;
	// 首先定义 cpu 上的变量并申请 cpu 上的内存空间
	int* hist = (int*)malloc(N * sizeof(int));
	int* bin = (int*)malloc(256 * sizeof(int));
	// 然后定义 gpu 上的变量并申请 gpu 上的内存空间
	int* hist_data;
	int* bin_data;
	cudaMalloc((void**)&hist_data, N * sizeof(int));
	cudaMalloc((void**)&bin_data, 256 * sizeof(int));

	for (int i = 0; i < N; i++) {
		hist[i] = i % 256;
	}
	int * groudtruth = (int*)malloc(256 * sizeof(int));
	for (int j = 0; j < 256; j++) {
		groudtruth[j] = 100000;
	}

	cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	
	const int blockSize = 256;
	int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

	dim3 Grid(GridSize);
	dim3 block(blockSize);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// 启动kernel 函数，并传入参数hist_data ，bin_data(结果输出参数)
	histgram << <Grid, block >> > (hist_data, bin_data);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	bool is_right = CheckResult(bin, groudtruth, 256);
	if (is_right) {
		printf("the ans is right\n");
	}
	else {
		printf("the ans is wrong\n");
		for (int i = 0; i < 256; i++) {
			printf("%lf ", bin[i]);
		}
		printf("\n");
	}
	printf("histogram latency = %f ms\n", milliseconds);

	cudaFree(bin_data);
	cudaFree(hist_data);
	free(bin);
	free(hist);

	return 0;	
}
