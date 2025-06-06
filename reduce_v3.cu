
//#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>

// V0
// reduce_v0 latency = 16.971935 ms

// v1  用位运算替换除余操作
// reduce_v0 latency = 11.568448 

// v2,2060
// reduce_v2 latency = 10.619200 ms


// v3: 让空闲线程也干活
// reduce_v3 latency = 5.751552 ms


template<int blockSize>

__global__ void reduce_v3(float* d_in,float * d_out) {
	__shared__ float smem[blockSize];

	unsigned int tid = threadIdx.x; // 线程在线程块block中的索引id
	unsigned int gtid = blockIdx.x * blockSize*2 + threadIdx.x; // 泛指线程在线Grid中的索引id，*2是代表每个线程处理两个元素
	// block中分配实际的thread数量是Block = blockSize / 2
	// 但是在算子中，要实现一个线程处理两个元素，所以thread id 在Grid中*2

	smem[tid] = d_in[gtid] + d_in[gtid + blockSize]; // 每个线程处理两个元素
	__syncthreads(); // 同步线程，确保所有线程都完成了对共享内存的写入

	// 进行归约操作
	// 位运算 等价于 index /= 2
	for (unsigned int index = blockSize / 2; index > 0; index >>= 1) {
		if (tid < index) {
			smem[tid] += smem[tid + index]; // 将相邻的两个元素相加
		}
		__syncthreads(); // 确保所有线程都完成了对共享内存的写入
	}

	if (tid == 0) {
		d_out[blockIdx.x] = smem[0]; // 将结果写回到全局内存
	}
	// 注意：这里的 smem[0] 是每个 block 的归约结果


}


bool CheckResult(float* out, float groudtruth, int n ) {
	float res = 0;
	for (int i = 0; i < n; i++){
		res += out[i];
	}
	if (res != groudtruth) {
		return false;
	}
	return true;
}



int main() {

	float millisconds = 0;

	//const int N = 32 * 1024 * 1024;
	const int N = 25600000;

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);  //获取cuda的设备信息

	const int blockSize = 256; //每个block的线程数

	int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
	//int GridSize = 100000;

	float* a = (float*)malloc(N * sizeof(float));
	float* d_a;
	cudaMalloc((void**)&d_a, N * sizeof(float));

	float* out = (float*)malloc(GridSize * sizeof(float));
	float* d_out;
	cudaMalloc((void**)&d_out, GridSize * sizeof(float));


	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f; // 初始化数组a
	}

	cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice); //将数据从主机复制到设备

	dim3 Grid(GridSize); //每个Grid的block数
	dim3 Block(blockSize / 2); //每个Block的线程数

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	reduce_v3<blockSize / 2> << <Grid, Block>> > (d_a, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);// 等待事件完成
	cudaEventElapsedTime(&millisconds, start, stop); //计算时间

	cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);//将结果从设备复制到主机
	printf("allcated %d blocks, data count are %d\n", GridSize, N);
	
	float groudtruth = N * 1.0f;// 计算预期的结果
	bool is_right = CheckResult(out, groudtruth, GridSize);// 检查结果是否正确
	if (is_right) {
		printf("result is right\n");
	} else {
		printf("result is wrong\n");
	}
	printf("reduce_v3 latency = %f ms\n", millisconds);

	cudaFree(d_a);
	cudaFree(d_out);
	free(a);
	free(out);

}
