---
name: cuda-reduction
description: CUDA Reduction 优化 - 从朴素实现到 Warp Shuffle 高级技巧
commands:
  - name: benchmark
    description: 运行 reduction benchmark 测试
    command: cd reduce && python reduce.py
  - name: build
    description: 编译 CUDA 扩展
    command: cd reduce && python reduce.py
triggers:
  - reduce
  - cuda reduce
  - gpu reduction
  - warp shuffle
  - block reduce
---

# CUDA Reduction 优化技能

本 skill 涵盖 CUDA reduction 操作的优化路径，从最朴素的实现到高性能的 warp shuffle + atomic 方案。

## 优化阶段

### Stage 1: Naive Block Reduction (`reduce_1`)
- 每个 block 独立做 reduction，输出 partial sums
- 存在 **warp divergence**: `tid % (2 * index) == 0` 导致大量线程空闲
- 代码: `reduce.cu:22-50`

### Stage 2: 消除 Warp Divergence (`reduce_11`)
- 重排索引访问模式，避免分支分歧
- `index = 2*i*tid` 替代条件判断
- 代码: `reduce.cu:54-71`

### Stage 3: Grid-Stride Loop (`reduce_2`)
- 每个线程处理多个元素 (`stride = blockDim.x * gridDim.x`)
- 限制 gridSize 避免 stride 过大导致退化
- 代码: `reduce.cu:75-108`

### Stage 4: Warp Shuffle + AtomicAdd (`reduce_3`)
- 使用 `__shfl_down_sync` 在 warp 内高效 reduce
- 每个 warp 独立 reduce 后写入 shared memory
- 最后用 `atomicAdd` 合并到全局 output
- 代码: `reduce.cu:115-165`

## 关键优化点

| 优化点 | 方法 | 收益 |
|--------|------|------|
| 消除 warp divergence | 重排索引计算 | 更高的并行度 |
| Grid-stride loop | 每线程处理多元素 | 更好的 occupancy |
| Warp shuffle | 使用 shuffle 指令 | 无需 shared memory 同步 |
| Atomic add | 多 block 并行合并 | 避免二次 kernel |

## 运行 Benchmark

```bash
python reduce/reduce.py
```

输出示例:
```
input numel=25600000 dtype=torch.float32 device=cuda
reduce_1 naive             out=25600000.00000000 time=0.12345678 ms
reduce_11 warp divergence  out=25600000.00000000 time=0.11234567 ms
reduce_2 grid_stride       out=25600000.00000000 time=0.09876543 ms
reduce_3 warp_atomic       out=25600000.00000000 time=0.08765432 ms
```

## 常见问题

**Q: 为什么 reduce_3 的 gridSize 要限制?**
A: 如果 block 太多，stride 会接近整个输入长度，线程只能处理很少元素，性能退化。

**Q: warp_reduce_sum 为什么用 inline?**
A: 保持代码可读性，同时让编译器能够内联展开。

**Q: atomicAdd 会有性能问题吗?**
A: 对于大数据集，多 block 并行 atomic 的收益远大于竞争带来的开销。