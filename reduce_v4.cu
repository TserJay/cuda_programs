
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdlib.h>
#include <stdio.h>




//v4: 最后一个warp不用参与__syncthreads
// 这个版本的reduce_v4使用了共享内存来进行归约操作，



__global__ void WarpShareMemReduce(volatile float* smem, int tid) {


	float x = smem[tid];
	if





}



template<int blockSize>

__global__ void reduce_v4(float* d_in,float * d_out) {
	__shared__ float smem[blockSize];
	unsigned int tid = threadIdx.x; // 线程索引
	unsigned int gtid = blockDim.x * blockIdx.x + threadIdx.x;// 全局线程索引

	smem[tid] = d_in[gtid] + d_din[gtid + blockSize];
	__syncthreads();

	//


	

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
	reduce_v4<blockSize / 2> << <Grid, Block>> > (d_a, d_out);
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
	printf("reduce_v4 latency = %f ms\n", millisconds);

	cudaFree(d_a);
	cudaFree(d_out);
	free(a);
	free(out);

}
