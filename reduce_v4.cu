

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdlib.h>
#include <stdio.h>


// V0
// reduce_v0 latency = 16.971935 ms

// v1  用位运算替换除余操作
// reduce_v0 latency = 11.568448 

// v2,2060
// reduce_v2 latency = 10.619200 ms


// v3: 让空闲线程也干活
// reduce_v3 latency = 5.751552 ms

// v4: 最后一个warp不用参与__syncthreads
// 这个版本的reduce_v4使用了共享内存来进行归约操作，
// reduce_v4 latency = 5.658784 ms



__device__ void WarpShareMemReduce(volatile float* smem, int tid) {

	// CUDA 不保证所有的shared memory 读操作都能在写操作之前完成，因此存在竞争关系，可能导致结果错误
	// eg smem[tid] += smem[tid+16] => smem[0] +=smem[16],smem[16] +=smem[32]
	// 此时L9中的smem[16]的读写顺序可能会被改变，导致结果错误
	// 所以在Volta架构后最后加入中间寄存器(L11)配合syncwarp和volatile(使得不会看见其他线程更新smem上的结果)保证读写依赖


	float x = smem[tid];
	if (blockDim.x >= 64) {
		x += smem[tid + 32]; __syncwarp();
		smem[tid] = x; __syncwarp();
	}
	x += smem[tid + 16]; __syncwarp();
	smem[tid] = x; __syncwarp();
	x += smem[tid + 8]; __syncwarp();
	smem[tid] = x; __syncwarp();
	x += smem[tid + 4]; __syncwarp();
	smem[tid] = x; __syncwarp();
	x += smem[tid + 2]; __syncwarp();
	smem[tid] = x;	__syncwarp();
	x += smem[tid + 1]; __syncwarp();
	smem[tid] = x; __syncwarp();


}


template<int blockSize>

__global__ void reduce_v4(float* d_in, float* d_out) {
	__shared__ float smem[blockSize];
	unsigned int tid = threadIdx.x; // 线程索引
	unsigned int gtid = blockDim.x * blockIdx.x + threadIdx.x;// 全局线程索引

	smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
	__syncthreads();

	// 基于v3改进：把最后一个warp抽离出来做reduce，避免多做一次sync threads
	// 此时一个 block 对 d_in 这块数据的 reduce sum 结果保存在 id 为 0 的线程上
	for (int s = blockDim.x / 2; s > 32; s >>= 1) //  >> 为位运算，= s = s/2
	{
		if (tid < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();

	}

	// last warp
	if (tid < 32) {
		WarpShareMemReduce(smem, tid);
	}

	if (tid == 0) {
		d_out[blockIdx.x] = smem[0]; // 将结果写回到全局内存
	}
	// 注意：这里的 smem[0] 是每个 block 的归约结果


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
	reduce_v4<blockSize / 2> << <Grid, Block >> > (d_a, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);// 等待事件完成
	cudaEventElapsedTime(&millisconds, start, stop); //计算时间

	cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);//将结果从设备复制到主机
	printf("allcated %d blocks, data count are %d\n", GridSize, N);

	float groudtruth = N * 1.0f;// 计算预期的结果
	bool is_right = CheckResult(out, groudtruth, GridSize);// 检查结果是否正确
	if (is_right) {
		printf("result is right\n");
	}
	else {
		printf("result is wrong\n");
	}
	printf("reduce_v4 latency = %f ms\n", millisconds);

	cudaFree(d_a);
	cudaFree(d_out);
	free(a);
	free(out);

}
