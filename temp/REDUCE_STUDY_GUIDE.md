# CUDA Reduce 学习路线与性能分析指南

这份文档的目标不是直接给你“最强 reduce 模板”，而是帮你建立一条能自己写、自己看懂、自己优化、自己分析的学习路径。

你现在的阶段很适合这样学：
先把 reduce 的语义和正确性吃透，再逐步引入 shared memory、grid-stride loop、warp 级规约、atomicAdd、再到 Nsight Compute 的性能分析。这样每一步你都知道“为什么要加这个优化”，而不是只会背代码。

## 1. 先建立一个最重要的认知

reduce 不是一个单独的“语法点”，而是一类并行模式。
最常见的是：
- sum
- max
- min
- mean
- argmax / argmin
- softmax 里的一部分归约过程
- LayerNorm / RMSNorm 里的统计量计算

你现在写的 `sum reduce`，本质上是在学习 CUDA 里最经典的一种并行规约模式。
只要这个模式吃透，后面很多算子都会顺很多。

## 2. 你现在最应该先学会什么

学习顺序建议按下面来，不要一上来就追求最优版本。

### 阶段 1：先彻底理解“正确性”

目标：知道 kernel 到底输出的是“每个 block 的和”还是“全局总和”。

要搞清楚这几件事：
- 一个线程负责什么
- 一个 block 负责什么
- 一个 grid 负责什么
- `output[blockIdx.x]` 这种写法是什么意思
- `atomicAdd(output, sum)` 这种写法是什么意思
- 为什么很多 reduce 都不是“一次 kernel 就直接结束”

你需要能一眼判断：
- 如果 `tid == 0` 时写 `output[blockIdx.x] = ...`，那通常输出的是每个 block 的 partial sum
- 如果所有 block 最后都往同一个 `output[0]` 做 `atomicAdd`，那通常输出的是全局总和
- 如果 kernel 只做到 partial sums，外面还要再做一次 `sum()` 或再 launch 一次 kernel，才能得到最终总和

### 阶段 2：学会 block 内 reduce

目标：理解 shared memory 规约的基本写法。

先吃透这几个概念：
- `threadIdx.x`
- `blockIdx.x`
- `blockDim.x`
- 全局索引 `gtid`
- `__shared__`
- `__syncthreads()`

你要能自己解释清楚：
- 为什么数据先从 global memory 搬到 shared memory
- 为什么每轮规约后要同步
- 为什么 `tile[0]` 最后会变成这个 block 的和

建议你先手推一个很小的例子：
- `blockDim.x = 8`
- 输入是 `[1,2,3,4,5,6,7,8]`
- 逐轮画出 shared memory 怎么变化

如果这一步没完全懂，后面 warp 优化会很虚。

### 阶段 3：理解 two-stage reduction

目标：接受一个现实：很多 reduce 不是一个 kernel 直接完成的。

最经典的两阶段流程：
1. 第一个 kernel：每个 block 算一个 partial sum
2. 第二个阶段：再把 partial sums 继续 reduce 成一个值

第二阶段可以有几种写法：
- 再 launch 一个 kernel
- 直接 `output.sum()` 让框架继续做
- 每个 block 最后用 `atomicAdd` 汇总到一个标量

这三种方式都值得你理解，因为它们的易写性和性能都不一样。

## 3. 推荐的 reduce 优化路线

下面这条路线最适合学习，不建议跳步。

### 版本 A：naive block reduce

特征：
- 每个线程加载 1 个元素
- block 内用 shared memory 做树形规约
- 最后输出每个 block 的 partial sum

你主要学到：
- reduce 的基本结构
- shared memory 的作用
- block 内同步的必要性

常见缺点：
- 每个线程只处理 1 个元素，线程工作量偏少
- `%` 这种写法通常不够快
- 只完成了 block 内归约，没有直接得到全局和

### 版本 B：grid-stride loop + block reduce

特征：
- 每个线程先在寄存器里累计多个元素
- 再把每个线程的局部和写入 shared memory
- block 内继续规约

你主要学到：
- grid-stride loop 的写法
- 为什么“一个线程多做一点事”反而常常更快
- 如何减少 block 数量、提升吞吐

这是非常重要的一步，因为它让你开始意识到：
CUDA 优化不是“线程越多越好”，而是“线程组织方式是否合理”。

#### 一个很容易踩坑的点：grid-stride loop 不是 block 越多越好

这是 reduce 学习里特别容易忽略的一点。

很多初学者会自然觉得：
- 既然输入很大
- 那就应该把 block 开得越多越好

但对 grid-stride loop 来说，不一定。

先看这个公式：

```cpp
int stride = blockDim.x * gridDim.x;
```

每个线程第一次处理自己的起始位置，后面每次都跳 `stride` 这么远继续累加。

如果 `gridDim.x` 开得非常大，会发生什么？
- `stride` 会变得非常大
- 大到接近整个输入长度
- 结果就是很多线程在循环里通常只会执行 1 次
- 这样每个线程就只处理了 1 个元素

这时 grid-stride loop 的优势几乎就没了。

换句话说：
你虽然“形式上”写了 grid-stride loop，
但实际上它已经退化回“每个线程只做一点点工作”的模式。

这也是为什么在 `_2`、`_3` 这种版本里，常常会看到类似这样的写法：

```cpp
int gridSize = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096);
```

或者：

```cpp
int gridSize = std::min((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2), 4096);
```

它的思路不是“随便拍脑袋限制 block 数量”，而是：
- 不要把 grid 开得过大
- 给每个线程留出多处理几个元素的机会
- 让寄存器累加真正发挥作用
- 让 grid-stride loop 保持它本来该有的收益

你可以把它理解成一个平衡：
- block 太少，GPU 可能喂不饱
- block 太多，stride 太大，线程又只干很少的活

所以 grid-stride loop 的关键不是“block 数量最大化”，而是“让线程工作量和并行度达到更合适的平衡”。

#### 你以后看到 `_2` 变慢时，优先先检查什么

如果你以后发现一个 grid-stride 版本突然变慢，优先先看这三件事：
- `gridDim.x` 是不是开得太大了
- `stride = blockDim.x * gridDim.x` 是不是已经接近输入长度了
- 每个线程在循环里到底平均处理了几个元素

如果答案是“平均只处理 1 个元素左右”，那通常说明：
grid-stride loop 已经没有真正发挥价值。

### 版本 C：优化 block 内规约方式

目标：把低效的 block 内 reduce 写法换掉。

你可以比较两类写法：
- 从 `index = 1` 开始翻倍，靠 `%` 判断谁参与
- 从 `s = blockDim.x / 2` 开始折半，靠 `tid < s` 判断谁参与

第二种通常更好，原因包括：
- 少掉 `%` 这种开销较高的操作
- 活跃线程分布更自然
- 更接近 CUDA reduction 的常见标准写法

### 版本 D：warp-level reduce

特征：
- 当活跃线程数缩小到一个 warp 范围内时
- 不再继续依赖 shared memory + `__syncthreads()`
- 改用 `__shfl_down_sync` 在 warp 内直接交换寄存器数据

你主要学到：
- warp 是 CUDA 调度的基本单位
- warp 内线程天然同步
- warp shuffle 为什么常比 shared memory 更轻量

这是从“会写 reduce”走向“会优化 reduce”的关键一步。

### 版本 E：一次读两个元素

特征：
- 每个线程不是只读一个元素，而是先读两个相邻位置
- 例如 `idx` 和 `idx + blockDim.x`

你主要学到：
- 如何提高全局内存访问利用率
- 如何减少循环次数
- 为什么很多 reduce kernel 会让一个线程处理多个元素

### 版本 F：atomicAdd 汇总全局结果

特征：
- 每个 block 先得到自己的 block sum
- 最后由一个线程把 block sum `atomicAdd` 到全局输出

优点：
- 代码直观
- 容易实现“单 kernel 直接输出总和”

缺点：
- block 数量多时，atomic 竞争会变重
- 不一定是最快写法

这个版本很适合学习，因为它让你看清：
“语义更直接” 和 “性能最高” 并不总是一回事。

### 版本 G：multi-pass reduction

特征：
- 第一轮先把大数组 reduce 成 partial sums
- 第二轮继续 reduce partial sums
- 直到只剩一个值

这是更经典、更工程化的 reduce 写法。
如果后面你要做更高性能版本，建议把这条路线补上。

## 4. 每一种优化你到底在优化什么

学习 reduce 时，不要只记“技巧名词”，要知道每一招在打什么瓶颈。

### shared memory

在优化：
- 减少对 global memory 的反复访问
- 提高 block 内数据重用效率

### grid-stride loop

在优化：
- 让每个线程处理更多元素
- 减少线程启动后“只干一点点活”的浪费
- 让 kernel 对大输入更灵活

### half reduction 写法

在优化：
- 减少 `%` 等不必要开销
- 改善 block 内规约结构

### warp shuffle

在优化：
- 减少 shared memory 访问
- 减少同步开销
- 提高最后几轮规约效率

### 一次读多个元素

在优化：
- 提高全局内存带宽利用率
- 降低循环与索引计算开销

### atomicAdd

在解决：
- 如何把多个 block 的结果汇总成全局单值

同时它也可能引入新瓶颈：
- 全局原子竞争

### multi-pass reduction

在解决：
- 避免大量 block 同时竞争同一个输出地址
- 用更可扩展的方式完成全局归约

## 5. 建议你按什么顺序自己写

推荐你自己动手时按下面顺序来：

1. 写一个最简单的 block 内 reduce，只求每个 block 的和
2. 在 Python 或 host 侧把 partial sums 再求和，确认语义没错
3. 改成 grid-stride loop，让每个线程处理多个元素
4. 把 `%` 版 block reduce 改成折半规约版
5. 把最后一个 warp 改成 `__shfl_down_sync`
6. 尝试做一个 `atomicAdd` 版的全局和
7. 再写一个 multi-pass 版，对比 `atomicAdd` 版的差别
8. 每一步都做 benchmark 和 ncu 分析

这条路线最关键的是：
每次只改一个点。
不要一次同时改 4 个优化，不然你根本不知道到底是哪一步带来了收益。

## 6. 你在 reduce 学习中一定会反复遇到的关键词

这些词建议你边写边查、边查边记：
- occupancy
- memory bandwidth
- shared memory bank conflict
- warp divergence
- coalesced access
- latency hiding
- arithmetic intensity
- atomic contention
- achieved occupancy
- SM utilization

不用一口气全吃掉，但要逐步熟悉。

## 7. 如何用 NCU 看 reduce 的性能

你的本地环境里已经有 `ncu`：
- `Nsight Compute CLI 2026.1.0.0`

对 reduce 来说，NCU 不是为了“记一堆指标”，而是回答下面这些问题：
- 我的 kernel 是算力瓶颈还是内存瓶颈？
- global memory 访问是否高效？
- block 内同步是不是很多？
- warp 有没有大量空转？
- atomicAdd 有没有成为热点？
- shared memory 有没有 bank conflict？

### 第一步：先只抓你关心的 kernel

如果你的 Python 会调用多个 CUDA kernel，先限定名字，否则报告会很乱。

常用命令示例：

```bash
ncu --kernel-name block_all_reduce_sum_1 python reduce.py
```

```bash
ncu --kernel-name block_all_reduce_sum_2 python reduce.py
```

```bash
ncu --kernel-name block_all_reduce_sum_3 python reduce.py
```

如果想导出报告：

```bash
ncu --kernel-name block_all_reduce_sum_3 -o reduce_3_report python reduce.py
```

这样会得到一个 `.ncu-rep` 文件，后面可以用图形界面再看。

### 第二步：先看 summary，不要一上来就看 full set

入门阶段推荐先用：

```bash
ncu --set summary --kernel-name block_all_reduce_sum_3 python reduce.py
```

等你已经知道 summary 在看什么了，再用：

```bash
ncu --set full --kernel-name block_all_reduce_sum_3 python reduce.py
```

原因很简单：
`full` 信息很多，初学时容易被淹没。

### 第三步：reduce 最值得先看的指标

建议你优先看这几类，而不是全都看。

#### 1. Kernel Duration

它告诉你这个 kernel 花了多久。
这是最直观的结果指标。

你做每一步优化后，第一件事就是看它有没有下降。

#### 2. Memory Throughput / DRAM Throughput

它帮助你判断：
- 这个 kernel 是否主要受内存带宽限制
- 你的全局内存读取效率怎么样

reduce 往往偏 memory-bound，所以这组指标很重要。

#### 3. Achieved Occupancy

它帮助你判断：
- 当前 block size 和资源占用是否让 SM 跑得足够满

但要注意：
occupancy 高不一定性能就高。
它只是参考，不是最终结论。

#### 4. Warp State / Stall Reasons

这一组非常关键。
它回答的是：
- warp 为什么在等
- 是等内存
- 还是等同步
- 还是等原子操作

对于 reduce，很值得关注：
- memory dependency
- barrier
- scoreboard
- atomic related stalls

#### 5. Shared Memory 指标

它帮助你判断：
- shared memory 有没有 bank conflict
- 共享内存读写是否成为明显瓶颈

#### 6. Atomic 指标

如果你在 `_3` 或别的版本里用了 `atomicAdd`，要重点看：
- 原子操作次数
- 原子相关 stall
- atomic throughput / serialization 迹象

如果 atomic 指标很差，通常说明：
你的“单 kernel 直接汇总”可能在大规模 block 下不划算。

## 8. 用 NCU 时推荐的观察顺序

每次 profile 一个 kernel 时，按这个顺序看会比较清楚：

1. 先看 Kernel Duration
2. 再看 memory throughput
3. 再看 achieved occupancy
4. 再看 warp stall reasons
5. 如果用了 shared memory，就看 bank conflict
6. 如果用了 atomicAdd，就看 atomic 相关指标

你要形成一个习惯：
不要看到某个指标高或低就马上下结论，而是把几个指标串起来解释。

例如：
- Duration 高
- DRAM Throughput 也很高
- ALU 压力不大
- Stall 大多是 memory dependency

这通常说明：
这个 kernel 更像是内存带宽受限，而不是算力不足。

再比如：
- `_3` 比 `_2` 更慢
- 同时 atomic stall 很高

那通常说明：
atomic 汇总可能抵消了 warp 优化带来的收益。

## 9. 一个很实用的对比实验表

建议你自己做表记录，每次只改一个变量。

你可以记录：
- 版本名
- block size
- grid size
- 输入规模
- 是否 grid-stride
- 是否 warp shuffle
- 是否 atomicAdd
- 平均时间
- Kernel Duration
- DRAM Throughput
- Achieved Occupancy
- 主要 Stall Reason
- 结论

这样你会很快从“会跑代码”变成“会分析性能”。

## 10. 你现在这个项目里可以怎么对应着学

结合你当前 `reduce` 目录，建议这样看：

- [reduce_baseline.cu](/projects/cuda_programs/reduce/reduce_baseline.cu): 先看你当前在维护的 `_1/_2/_3`，理解版本差异
- [reduce.py](/projects/cuda_programs/reduce/reduce.py): 看 benchmark 的组织方式，理解为什么要热身、为什么要同步、为什么要用 CUDA Event
- [reduce_v0.cu](/projects/cuda_programs/reduce/reduce_v0.cu): 当成最早期、最直观的起点看
- `reduce_v1.cu` 到 `reduce_v6.cu`: 可以按版本顺序继续整理每一步到底引入了什么优化

如果你愿意，后面非常值得做的一件事是：
把 `reduce_v0` 到 `reduce_v6` 各自的“核心变化点”也写成一张表。
这样你会形成自己的 reduce 演进图谱。

## 11. 学 reduce 时最容易犯的几个误区

### 误区 1：只看时间，不看正确性

每改一步，先确认结果对不对。
优化错了、越界了、漏加了，跑得再快也没有意义。

### 误区 2：只背代码，不理解语义

一定要能回答：
- 现在这个 kernel 输出的是 partial sums 还是 final sum？
- 为什么？
- 最终总和是在 kernel 内完成，还是在 kernel 外完成？

### 误区 3：把 occupancy 当成唯一目标

occupancy 只是参考值。
真正重要的是总耗时和瓶颈位置。

### 误区 4：一次改太多

你应该一次只改一个优化点，不然很难定位收益来源。

### 误区 5：一看到 warp shuffle 就觉得一定更快

不一定。
要看规模、访存、atomic、block size 和整体结构。
最后还是要靠 benchmark 和 ncu 说话。

## 12. 推荐你的下一步学习动作

最建议你接下来按这个节奏走：

1. 先把你现在 `_1/_2/_3` 的语义彻底讲清楚，自己能口述出来
2. 手推一个 `blockDim=8` 的 block reduce 示例
3. 自己重新写一遍 `_1`
4. 在 `_1` 的基础上自己写 `_2`
5. 再自己尝试补一个 `_3`
6. 用 `reduce.py` 做 benchmark
7. 用 `ncu --set summary --kernel-name ...` 看三个版本的差异
8. 把你的结论写回 README 或学习笔记

## 13. 一组你可以直接开始用的 NCU 命令

进入 `reduce` 目录后，可以先用这些：

```bash
ncu --set summary --kernel-name block_all_reduce_sum_1 python reduce.py
```

```bash
ncu --set summary --kernel-name block_all_reduce_sum_2 python reduce.py
```

```bash
ncu --set summary --kernel-name block_all_reduce_sum_3 python reduce.py
```

如果想保存报告：

```bash
ncu --set summary --kernel-name block_all_reduce_sum_3 -o reduce_3_summary python reduce.py
```

如果后面你已经熟悉 summary，再上 full：

```bash
ncu --set full --kernel-name block_all_reduce_sum_3 -o reduce_3_full python reduce.py
```

## 14. 最后的建议

对 reduce 来说，真正重要的不是你有没有马上写出一个最强版本，而是你是否能稳定回答这四个问题：
- 这个版本的输出语义是什么
- 这个版本比上一个版本多了什么优化
- 这个优化理论上在改善什么瓶颈
- ncu 结果是否真的支持这个判断

只要这四件事你越来越清楚，你的 CUDA 基础会涨得很快。


codex 巨献