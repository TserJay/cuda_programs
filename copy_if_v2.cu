
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>


#include <stdio.h>
#include <stdlib.h>

// 注意：
// 1.本节的文字解析放在了CUDA_lesson.pdf，如有不懂，可以先看看文字解析
// 2.这种warp和shared在老的gpu上面会很有成效，但是在turing后的GPU，nvcc编译器优化了很多，所以导致效果不明显
// 3.我记得在某个CUDA版本之前，atomic是可以保证block或thread严格按照ID串行，但是某个CUDA版本之后，就不行了，至少在现有流行版本不行了，所以会发现CUDA copy if执行后，虽然全都是>0的值，但是顺序和输入不一样

// gpu实现 gpu必须是ampere架构
// block level ,use block level atomics based on shared memory
__global__ void filter_shared_k(int* dst, int* src, int* nres, int n) {
	// 计数器声明为 shared memory ，计数每个block范围内大于0的数量
	__shared__ int l_n;
	int gitd = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread_num = blockDim.x * gridDim.x;

	for (int i = gitd; i < n; i += total_thread_num) {
		if(threadIdx.x == 0)
			l_n = 0;
		__syncthreads();

		int d, pos;
		// l_n表示每个block范围内大于0的数量，block内的线程都能访问
		// pos是每个线程的私有寄存器，且作为atomicAdd的返回值，表示当前线程对l_n原子加1之前的l_n，，比如1 2 4号线程都大于0，那么对于4号线程来说l_n = 3, pos = 2
		if (i < n && src[i]>0) 
			pos = atomicAdd(&l_n, 1);// pos为 l_n 原子加1之前的值
		__syncthreads();

		// 每个block 中选出tid = 0 作为leader
		// leader将每个block范围内大于0的数量加到全局的nres，即所有的block做一个reduce sum
		// l_n 依然是nres 在与 l_n 做原子加之前的值
		if (threadIdx.x == 0) 
			l_n = atomicAdd(nres, l_n);
		__syncthreads();

		// write & store
		// 将结果写回dst 中
		if (i < n && d>0) {
			// 1. pos: src[thread]>0的thread在当前block的index
			// 2. l_n: 在当前block的前面几个block的所有src>0的个数
			// 3. pos + l_n：当前thread的全局offset
			pos += l_n;
			dst[pos] = d;
		}
		__syncthreads();
	}
}


bool CheckResult(int* out, int groudtruth) {
	if (*out != groudtruth)
		return false;
	return true;
}


int main() {
	float milliseconds = 0;
	const int N = 25600000;
	// 首先定义 cpu 上的变量并申请 cpu 上的内存空间
	int *src_h  = (int*)malloc(N * sizeof(int)); // 输入数据
	int *dst_h = (int*)malloc(N * sizeof(int)); 
	int* nres_h = (int*)malloc(1 * sizeof(int));

	// 然后定义 gpu 上的变量并申请 gpu 上的内存空间
	int* src, *dst, *nres;
	cudaMalloc((void**)&src, N * sizeof(int));
	cudaMalloc((void**)&dst, N * sizeof(int));
	cudaMalloc((void**)&nres, 1 * sizeof(int));

	for (int i = 0; i < N; i++) {
		src_h[i] = 1;
	}
	int groudtruth = 0;
	// 当 src > 0 时，groudtruth 加 1
	for (int j = 0; j < N; j++) {
		if (src_h[j] > 0) {
			groudtruth += 1;
		}
	}

	cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);
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
	// 使用 向量化的cuda 内核函数，进行load和store 可以加速计算
	filter_shared_k << <Grid, block >> > (dst, src, nres, N);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	bool is_right = CheckResult(nres_h, groudtruth);
	if (is_right) {
		printf("the ans is right\n");
	}
	else {
		printf("the ans is wrong\n");
		printf("%1f", *nres_h);
		printf("\n");
	}
	printf("histogram latency = %f ms\n", milliseconds);


	cudaFree(src);
	cudaFree(dst);
	cudaFree(nres);
	free(src_h);
	free(dst_h);
	free(nres_h);	
	return 0;	
}
