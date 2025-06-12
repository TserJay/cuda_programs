

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdlib.h>
#include <stdio.h>

#define THREAD_PER_BLOCK 256


//v4: 最后一个warp不用参与__syncthreads
// 这个版本的reduce_v4使用了共享内存来进行归约操作，

//allcated 100000 blocks, data count are 25600000
//result is right
//reduce_v4 latency = 5.658784 ms


// no __syncwarp
// reduce_v4 latency = 3.664032 ms
// reduce_v4 latency = 3.427616 ms
// reduce_v4 latency = 3.782432 ms

// v5: 将所有的循环都展开
template<int blockSize>

__device__ void BlockShareMemReduce(float* smem) {
	// 对v4中的所有循环进行修改，将所有的循环都展开

	if (blockSize >= 1024) {
		if (threadIdx.x < 512) {
			smem[threadIdx.x] += smem[threadIdx.x + 512];
		}
		__syncthreads();
	}
	

	if (blockSize >= 512) {
		if (threadIdx.x < 256) {
			smem[threadIdx.x] += smem[threadIdx.x + 256];
		}
		__syncthreads();
	}
	
	if (blockSize >= 256) {
		if (threadIdx.x < 128) {
			smem[threadIdx.x] += smem[threadIdx.x + 128];
		}
		__syncthreads();
	}
	
	if (blockSize >= 128) {
		if (threadIdx.x < 64) {
			smem[threadIdx.x] += smem[threadIdx.x + 64];
		}
		__syncthreads();
	}
	
	if (blockSize < 32) {
		volatile float* vshm = smem;
		if (blockDim.x >= 64) {
			vshm[threadIdx.x] += vshm[threadIdx.x + 32];
		}
		vshm[threadIdx.x] += vshm[threadIdx.x + 16];
		vshm[threadIdx.x] += vshm[threadIdx.x + 8];
		vshm[threadIdx.x] += vshm[threadIdx.x + 4];
		vshm[threadIdx.x] += vshm[threadIdx.x + 2];
		vshm[threadIdx.x] += vshm[threadIdx.x + 1];
	}
}

template<int blockSize>
__global__ void reduce_v5(float* d_in, float* d_out) {
	__shared__ float smem[THREAD_PER_BLOCK];
	unsigned int tid = threadIdx.x; // 线程索引

	// *2代表当前的 block 要处理 2*blockSize个数据
	// eg blcoksize = 2, blockIdx.x = 1, when tid = 0, gtid = 4, gtid + blockSize = 6;when tid = 1, gtid = 5, gtid + blockSize = 7;
	unsigned int gtid = blockIdx.x * (blockDim.x * 2) + threadIdx.x; 
	
	smem[tid] = d_in[gtid] + d_in[gtid + blockDim.x];
	__syncthreads();

	// comppute reduce sum in shared memory
	BlockShareMemReduce<blockSize>(smem);


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
	reduce_v5<blockSize / 2> << <Grid, Block >> > (d_a, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);// 等待事件完成
	cudaEventElapsedTime(&millisconds, start, stop); //计算时间

	cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);//将结果从设备复制到主机
	printf("allcated %d blocks, data count are %d\n", GridSize, N);
	printf("count is : %d\n", out);

	float groudtruth = N * 1.0f;// 计算预期的结果
	bool is_right = CheckResult(out, groudtruth, GridSize);// 检查结果是否正确
	if (is_right) {
		printf("result is right\n");
	}
	else {
		printf("result is wrong\n");
	}
	printf("reduce_v5 latency = %f ms\n", millisconds);

	cudaFree(d_a);
	cudaFree(d_out);
	free(a);
	free(out);

}
