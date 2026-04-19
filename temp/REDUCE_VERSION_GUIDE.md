# CUDA Reduce 版本演进对照表

这份文档专门对应你当前目录里的：
- `reduce_v0.cu`
- `reduce_v1.cu`
- `reduce_v2.cu`
- `reduce_v3.cu`
- `reduce_v4.cu`
- `reduce_v5.cu`
- `reduce_v6.cu`

目标不是只告诉你“哪个更快”，而是帮你建立这种感觉：
每个版本到底比上一个版本多做了什么优化，这个优化在解决什么问题，代价又是什么。

建议你阅读时的顺序是：
1. 先看这一份版本对照表
2. 再打开对应的 `.cu` 文件
3. 只抓每一版“新增的那个点”
4. 不要一口气把所有代码细节都背下来

## 1. 先看总图

你这一套版本，大致可以理解成下面这条路线：

- `v0`：最基础的 block 内 shared memory reduce
- `v1`：把 `%` 换成位运算，减少不必要开销
- `v2`：改成更标准的折半规约写法
- `v3`：让每个线程处理两个元素，提高吞吐
- `v4`：最后一个 warp 单独处理，减少同步开销
- `v5`：把规约循环展开，减少循环控制开销
- `v6`：改成 two-pass reduction，真正得到全局总和

这条路线非常典型，基本覆盖了 reduce 学习时最常见的优化思路。

## 2. 每个版本的核心差异

## v0：naive shared memory reduce

参考文件： [reduce_v0.cu](/projects/cuda_programs/reduce/reduce_v0.cu:30)

### 这一版做了什么

- 每个线程从 global memory 读取 1 个元素
- 把数据放进 shared memory
- 用树形规约把一个 block 内的数据加成一个值
- `tid == 0` 把当前 block 的结果写到 `d_out[blockIdx.x]`

### 这一版最值得学什么

这是 reduce 的原型。
如果这一版你能完全手推明白，后面的优化才不会空。

你要特别理解：
- `smem[tid] = d_in[gtid]`
- 为什么需要 `__syncthreads()`
- 为什么最后 `smem[0]` 是当前 block 的和
- 为什么 `d_out[blockIdx.x]` 说明输出的是 partial sums，而不是 final sum

### 这一版的性能问题

主要问题在这里： [reduce_v0.cu](/projects/cuda_programs/reduce/reduce_v0.cu:44)

```cpp
for (int index = 1; index < blockDim.x; index *= 2)
```

以及这里： [reduce_v0.cu](/projects/cuda_programs/reduce/reduce_v0.cu:53)

```cpp
if (tid % (2 * index) == 0)
```

这个 `%` 取模通常比较慢。
而且这种“从小跨度开始翻倍”的写法，活跃线程分布也不够理想。

### 你看这一版时要问自己的问题

- 一个 block 最终算出了什么
- 所有 block 加起来之后，为什么还没有直接得到最终总和
- `%` 为什么可能成为性能问题

## v1：用位运算替换 `%`

参考文件： [reduce_v1.cu](/projects/cuda_programs/reduce/reduce_v1.cu:19)

### 相比 v0 改了什么

核心变化在这里： [reduce_v1.cu](/projects/cuda_programs/reduce/reduce_v1.cu:39)

```cpp
if ((tid & (2 * index - 1)) == 0)
```

它本质上还是在判断“这个线程这轮要不要参与规约”，
只是把 `%` 改成了位运算。

### 这一版优化了什么

主要是在优化指令开销。

相比 `%`：
- 位运算通常更轻量
- 对这种 2 的幂相关判断更自然

### 这一版没有改变什么

这些都没变：
- 一个线程还是只处理一个元素
- 还是 shared memory block 内规约
- 还是输出每个 block 的 partial sum
- 整体规约结构和 v0 基本一样

### 你应该怎么理解 v1

v1 不是“算法换了”，而是“同样的算法，局部写法更高效了”。

这是很重要的一类优化思路：
不是推翻重写，而是先找最明显的低效指令替掉。

## v2：改成折半规约

参考文件： [reduce_v2.cu](/projects/cuda_programs/reduce/reduce_v2.cu:20)

### 相比 v1 改了什么

核心变化在这里： [reduce_v2.cu](/projects/cuda_programs/reduce/reduce_v2.cu:34)

```cpp
for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1)
```

以及这里： [reduce_v2.cu](/projects/cuda_programs/reduce/reduce_v2.cu:35)

```cpp
if (tid < index)
```

也就是说，规约方式从：
- 小跨度开始，不断翻倍

改成了：
- 从一半开始，每轮减半

### 这一版优化了什么

它优化的不是“功能”，而是 block 内规约结构。

好处通常包括：
- 少掉 `%`
- 活跃线程分布更自然
- 更符合 CUDA reduction 的经典写法
- 通常更容易继续往后优化

### 文档里提到的“bank conflict”怎么理解

你代码注释里写了“消除了 bank conflict”。
更严格地说，`v2` 的主要收益更像是：
- 规约模式更标准
- 线程访问模式更合理
- 指令和控制流更友好

它和 bank conflict 的关系不是一句话就能完全概括，但你可以先把它理解成：
“shared memory 的访问方式更顺了”。

### 你应该怎么理解 v2

v2 是一个很关键的版本。
因为从这里开始，你进入了“比较标准的 reduce 实现框架”。

后面的 v3、v4、v5，很多优化其实都是在 v2 这个结构上继续加的。

## v3：每个线程处理两个元素

参考文件： [reduce_v3.cu](/projects/cuda_programs/reduce/reduce_v3.cu:23)

### 相比 v2 改了什么

关键变化在这里： [reduce_v3.cu](/projects/cuda_programs/reduce/reduce_v3.cu:27)

```cpp
unsigned int gtid = blockIdx.x * blockSize * 2 + threadIdx.x;
```

以及这里： [reduce_v3.cu](/projects/cuda_programs/reduce/reduce_v3.cu:31)

```cpp
smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
```

也就是说：
- 一个线程不再只负责一个元素
- 而是先把两个元素加起来，再放到 shared memory

### 这一版优化了什么

这一版的核心是：
让原来“空着”的计算能力也干活。

你可以把它理解成：
- 以前一个线程只干一份活
- 现在一个线程先干两份，再参加 block 内规约

通常这样做的收益是：
- 减少需要 launch 的 block 数量
- 提高每个线程的工作量
- 更好利用 global memory 读带宽
- 降低一部分 shared memory 规约压力

### 为什么速度会明显改善

因为这一版开始不只是“微调指令”了，
而是在重新组织线程和数据的映射关系。

这类优化往往比“把 `%` 改成位运算”更显著。

### 你看这一版时要重点注意

- `Block(blockSize / 2)` 为什么是对的
- `template<int blockSize>` 这里的 `blockSize` 和真正 launch 的线程数是什么关系
- 为什么每个线程读两个元素后，shared memory 里仍然是一份“局部和数组”

这一步你如果想得很清楚，后面 grid-stride loop 会更容易懂。

## v4：最后一个 warp 单独规约

参考文件： [reduce_v4.cu](/projects/cuda_programs/reduce/reduce_v4.cu:30)

### 相比 v3 改了什么

核心变化有两个：

第一处： [reduce_v4.cu](/projects/cuda_programs/reduce/reduce_v4.cu:70)

```cpp
for (int s = blockDim.x / 2; s > 32; s >>= 1)
```

第二处： [reduce_v4.cu](/projects/cuda_programs/reduce/reduce_v4.cu:80)

```cpp
if (tid < 32) {
    WarpShareMemReduce(smem, tid);
}
```

也就是说：
- 前面的大部分规约仍然按 block 方式做
- 当活跃线程只剩一个 warp 时
- 不再继续让整个 block 做 `__syncthreads()`
- 而是把最后一个 warp 单独拿出来处理

### 这一版优化了什么

它主要在优化同步开销。

因为当只剩 32 个线程活跃时：
- 再做整个 block 的同步通常就不值了
- 最后一个 warp 的规约可以用更轻量的方式处理

### 这里为什么代码开始变复杂

因为你这里开始碰到 warp 级优化了。
代码复杂度上升是正常的。

这也是为什么我一直建议：
一定先把 v0~v3 吃透，再学 v4。

### `WarpShareMemReduce` 你该怎么理解

这个函数本质上就是：
- 只针对最后一个 warp
- 手动展开最后几步规约
- 利用 warp 内更轻量的同步方式完成归约

你可以先不纠结所有 `volatile` 和 `__syncwarp()` 的硬件细节，
先记住主线：
“v4 的重点是把最后一个 warp 从 block 级同步里拆出来”。

## v5：把规约过程展开

参考文件： [reduce_v5.cu](/projects/cuda_programs/reduce/reduce_v5.cu:27)

### 相比 v4 改了什么

关键变化在这里： [reduce_v5.cu](/projects/cuda_programs/reduce/reduce_v5.cu:29)

`BlockShareMemReduce<blockSize>(smem)`

里面把原来循环式的规约，改成了手动展开：
- `if (blockSize >= 1024)`
- `if (blockSize >= 512)`
- `if (blockSize >= 256)`
- `if (blockSize >= 128)`
- warp 内再继续展开

### 这一版优化了什么

核心是减少循环控制开销，给编译器更多优化空间。

循环展开的收益可能来自：
- 少掉循环变量更新
- 少掉循环判断
- 更容易让编译器做指令级优化

### 你应该怎么看 v5

v5 是一个“工程味更重”的版本。

它告诉你：
到了一定阶段，优化已经不是改变算法结构，而是在榨取实现细节上的性能。

### 这一版为什么值得学

因为它是很多高性能 reduce 模板的常见写法。
你以后看 CUB、看一些 kernel 优化文章，会经常看到类似的 loop unroll 思路。

## v6：two-pass reduction，得到最终总和

参考文件： [reduce_v6.cu](/projects/cuda_programs/reduce/reduce_v6.cu:89)

### 相比 v5 改了什么

v6 的本质变化不是 block 内规约，而是“全局归约策略”变了。

关键点 1： [reduce_v6.cu](/projects/cuda_programs/reduce/reduce_v6.cu:101)

```cpp
for (int32_t i = gtid; i < nums; i += total_threads_nums) {
    sum += d_in[i];
}
```

这说明它用了 grid-stride loop：
- 每个线程处理多个元素
- 不再显式写死“一线程处理两个元素”

关键点 2： [reduce_v6.cu](/projects/cuda_programs/reduce/reduce_v6.cu:170)

```cpp
reduce_v6<blockSize><<<Grid, Block>>>(d_a, part_out, N);
reduce_v6<blockSize><<<1, Block>>>(part_out, d_out, GridSize);
```

这就是 two-pass reduction：
- 第 1 次 kernel：把大输入 reduce 成 `part_out`
- 第 2 次 kernel：把 `part_out` 再 reduce 成单个结果

### 这一版优化了什么

它优化的是“全局归约”的组织方式。

和 `v0~v5` 不同，`v0~v5` 本质上只完成了：
- 每个 block 的 partial sum

而 `v6` 才真正完成了：
- 从大输入到最终总和的完整 reduce 流程

### 为什么 v6 很重要

因为这版让你理解一个关键事实：
高性能 reduce 不只是 block 内怎么加，
还包括“多个 block 的结果最后怎么继续合并”。

这也是工程里真正有价值的一步。

### 为什么它可能没有你想象中那么快

你注释里的时间显示 `v6` 并不一定比前面快。
这很正常，因为：
- 它多 launch 了一次 kernel
- 它承担的是“完整最终归约”
- 前面的 `v0~v5` 很多只是 partial sums，不是最终答案

所以不要直接拿 `v6` 和 `v5` 的耗时生硬对比。
更公平的比较应该是：
- `v5 + 第二阶段归约`
- 对比
- `v6` 的两阶段总耗时

## 3. 你可以把这些版本分成三大类

### 第一类：先学基本结构

包括：
- `v0`
- `v1`
- `v2`

这三版主要教你：
- shared memory reduce 的基本样子
- 如何优化 block 内规约判断逻辑

### 第二类：开始提升吞吐

包括：
- `v3`
- `v4`
- `v5`

这三版主要教你：
- 每个线程处理更多数据
- 减少同步开销
- 用 warp 优化最后几步
- 用 loop unroll 进一步榨性能

### 第三类：做完整最终归约

包括：
- `v6`

这版主要教你：
- grid-stride loop
- multi-pass reduction
- partial sums 到 final sum 的完整流程

## 4. 每个版本你最应该观察什么

为了避免你看代码时抓不到重点，我建议每一版只盯一个核心问题。

### 看 v0 时

问自己：
- 为什么 `smem[0]` 会变成 block sum？

### 看 v1 时

问自己：
- 为什么位运算能替代 `%`？

### 看 v2 时

问自己：
- 为什么 `tid < index` 的折半写法更标准？

### 看 v3 时

问自己：
- 为什么一个线程处理两个元素会更快？

### 看 v4 时

问自己：
- 为什么最后一个 warp 不值得继续用整个 block 同步？

### 看 v5 时

问自己：
- loop unroll 减少了哪些额外开销？

### 看 v6 时

问自己：
- 为什么完整 reduce 往往需要 two-pass？

## 5. 结合 NCU 时怎么分析这几个版本

下面是非常实用的一条分析思路。

### `v0 -> v1`

重点看：
- 指令效率变化
- 总时长是否因为替换 `%` 而下降

你的结论通常会是：
- 这是“轻量局部优化”
- 提升存在，但不是结构级飞跃

### `v1 -> v2`

重点看：
- warp 执行效率
- shared memory 访问模式
- stall 是否减少

你的结论通常会是：
- 规约结构更合理了

### `v2 -> v3`

重点看：
- global memory throughput
- kernel duration
- 每线程工作量提升后整体吞吐是否变好

这是最值得重点观察的一跳，因为它通常带来比较明显的性能变化。

### `v3 -> v4`

重点看：
- barrier / synchronization 相关 stall
- warp state

你的结论通常会是：
- 最后一个 warp 单独处理后，同步成本下降

### `v4 -> v5`

重点看：
- 指令数量
- kernel duration

你的结论通常会是：
- loop unroll 更多是在抠实现细节开销

### `v5 -> v6`

重点看：
- 这已经不是单纯的 block 内优化对比了
- 要把“完整最终求和”的语义一起纳入比较

也就是说，这一跳更像是：
- 从“partial sums kernel”
- 走向
- “完整 global reduce pipeline”

## 6. 你当前这套版本里，有几个地方要特别留意

这些不是说代码错了，而是你学习时要知道它们的语义。

### 1. `v0~v5` 大多数只是 partial sums

也就是说它们 kernel 输出的是：
- 每个 block 的归约结果

不是最终总和。

这一点在 `v6` 的注释里其实也说到了： [reduce_v6.cu](/projects/cuda_programs/reduce/reduce_v6.cu:33)

### 2. `v6` 才明显在做完整 two-pass

如果你在比较“最终求和能力”，要把这点考虑进去。

### 3. 一些文件里命名和注释有历史痕迹

比如：
- `reduce_v2.cu` 里的 kernel 名字还是 `reduce_v1`
- 有些注释里的 latency 文本还保留旧名字

这对学习问题不大，但你自己整理笔记时，建议把“版本名”和“文件名”统一一下，不然以后容易绕晕。

## 7. 最推荐你的学习顺序

如果你现在是边看边自己写，我建议你按下面顺序复现：

1. 自己手写一个 `v0`
2. 改出 `v1`
3. 改出 `v2`
4. 改出 `v3`
5. 在 `v3` 基础上理解 `v4`
6. 在 `v4` 基础上理解 `v5`
7. 最后再自己写一个 two-pass 版，对照 `v6`

这条顺序的关键是：
先把 block 内 reduce 学透，再去学“全局最终归约”。

## 8. 你现在最值得自己动手做的练习

### 练习 1：画图

把 `v0` 的 blockDim 设成 8，手推每轮 shared memory 的变化。

### 练习 2：只改一处

从 `v0` 改到 `v1` 时，只改 `%` 那一行。
别同时动别的逻辑。

### 练习 3：只重写 block 内循环

从 `v1` 改到 `v2` 时，只改规约循环，不改 load/store 结构。

### 练习 4：把一个线程处理两个元素画出来

从 `v2` 改到 `v3` 时，画清楚 `gtid` 和 `gtid + blockSize` 分别对应哪两个数据。

### 练习 5：比较 partial sums 和 final sum

把 `v5` 的输出和 `v6` 的最终输出放在一起看。
你要能清楚解释：
- 一个是每个 block 的结果数组
- 一个是整个输入的最终总和

## 9. 最后给你的一个学习建议

如果你要真正把 reduce 学扎实，不要只追求“哪个版本时间最低”。
你更应该能稳定说清楚这三件事：
- 这个版本新增了什么优化
- 这个优化理论上在改善什么瓶颈
- 这个版本输出的是 partial sums 还是 final sum

只要这三件事越来越顺，你的 reduce 学习就在正轨上。
