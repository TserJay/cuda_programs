
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




int main() {
	float millisconds = 0;

	//const int N = 32 * 1024 * 1024;
	const int N = 25600000;

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);  //获取cuda的设备信息

	const int blocksize = 256; //每个block的线程数

	int GridSize = std::min((N + 256 - 1)/256, deviceProp.maxGridSize[0]);
	//int GridSize = 100000;
	


	
	printf("%d",deviceProp.maxGridSize[0]);


	return 0;
    
}
