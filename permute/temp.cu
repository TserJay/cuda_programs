__global__ void permute_11_optimized(const float* input, float* output, int A, int B, int C) {
    // 1. 安全地获取第一维索引
    int a = blockIdx.z;
    if (a >= A) return; // 防止越界

    // 2. 计算当前平面 (B*C) 的总大小
    int total_elements_in_plane = B * C;

    // 3. 计算当前线程在二维平面 (x, y) 上的全局 ID
    // 这种方法比手动乘除法更稳健，适应不同的 Grid/Block 配置
    int plane_thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    int plane_grid_width = gridDim.x * blockDim.x; 
    // 注意：这里假设了 gridDim.x * blockDim.x 覆盖了 C 维度，
    // 或者我们简单地将其视为线性化的二维索引计算。
    
    // 更通用的线性化 ID 计算（假设 grid 和 block 是线性映射到 B*C 的）
    int tid_flat = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.y * blockDim.x) + 
                   (threadIdx.y * blockDim.x + threadIdx.x);

    // 4. 设置步长策略
    // 每个线程处理 4-8 个元素通常比 64 个效果更好，且减少寄存器压力
    const int stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    const int items_per_thread = 4; 
    
    // 5. 循环处理
    // 使用 Grid-Stride Loop 模式，这是 CUDA 官方推荐的处理任意大小数据的模式
    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid_flat + i * stride;
        if (idx >= total_elements_in_plane) break;

        // 坐标还原
        int b = idx / C;
        int c = idx % C;

        // 索引计算
        int in_idx  = a * B * C + b * C + c;
        int out_idx = a * C * B + c * B + b;

        output[out_idx] = input[in_idx];
    }
}