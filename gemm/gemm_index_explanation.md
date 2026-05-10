# GEMM CUDA 核函数索引详解

## 1. 总体设计思路

这是一个**分块（Tiled）矩阵乘法**的 CUDA 实现，核心思想是：

1. 将大矩阵拆成小块，每个 Block 负责计算 C 的一个子块
2. 利用 **Shared Memory** 缓存 A 和 B 的子块，减少全局内存访问
3. 每个线程负责计算 C 子块中的多个元素（Register Tiling）

假设矩阵维度：
- A: `[M, K]`（行主序）
- B: `[K, N]`（行主序）
- C: `[M, N]`（行主序）

---

## 2. 模板参数与分块策略

```cpp
template <
    const int BLOCK_SIZE_M,  // C 子块的行数（M 方向高度）
    const int BLOCK_SIZE_K,  // K 方向分块大小（A 的列 / B 的行）
    const int BLOCK_SIZE_N,  // C 子块的列数（N 方向宽度）
    const int THREAD_SIZE_X, // 每个线程在 X 方向（N 方向）计算的元素数
    const int THREAD_SIZE_Y  // 每个线程在 Y 方向（M 方向）计算的元素数
>
```

### 2.1 Block 内的线程组织

```cpp
const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;  // Block 内 X 方向线程数
const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;  // Block 内 Y 方向线程数
const int THREAD_NUM_PER_BLOCK = bszx * bszy;   // 一个 Block 的总线程数
```

例如：
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 128`
- `THREAD_SIZE_X = 8`, `THREAD_SIZE_Y = 8`
- 则 `bszx = 16`, `bszy = 16`，Block 大小为 `16 × 16 = 256` 线程

每个线程负责计算 C 中 `8 × 8 = 64` 个元素。

### 2.2 线程 ID 的展平

```cpp
const int tid = threadIdx.y * bszx + threadIdx.x;
```

将 2D 线程索引 `(threadIdx.y, threadIdx.x)` 展平为 1D `tid`，用于协同加载数据到 Shared Memory。

---

## 3. Shared Memory 加载阶段的索引计算

### 3.1 每个线程负责加载 A 的哪些元素

A 子块大小：`[BLOCK_SIZE_M, BLOCK_SIZE_K]`

```cpp
const int A_TILE_ROW = tid / BLOCK_SIZE_K;      // 负责的行偏移
const int A_TILE_COL = tid % BLOCK_SIZE_K;      // 负责的列偏移
const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;  // 行步长
```

**原理**：将 `THREAD_NUM_PER_BLOCK` 个线程均匀映射到 A 子块的 `BLOCK_SIZE_M × BLOCK_SIZE_K` 个元素上。

- 按列优先展开：先填满一列，再下一列（或者说按行分组）。
- `A_TILE_ROW_STRIDE`：同一个线程每次加载相差多少行。

**加载循环**：
```cpp
for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW;
    const int col = A_TILE_COL + tile_idx;
    As[i + A_TILE_ROW][A_TILE_COL] = A[OFFSET(row, col, K)];
}
```

| 索引 | 含义 |
|------|------|
| `BLOCK_SIZE_M * blockIdx.y` | 当前 Block 在全局 A 中的起始行 |
| `tile_idx` | 当前沿 K 维度的分块偏移（主循环变量） |
| `i + A_TILE_ROW` | 在 A 子块内的行偏移（考虑一个线程加载多行） |
| `A_TILE_COL` | 在 A 子块内的列偏移 |
| `OFFSET(row, col, K)` | 全局内存一维索引：`row * K + col` |

**为什么 `i += A_TILE_ROW_STRIDE`？**

因为线程数可能少于 `BLOCK_SIZE_M`，所以每个线程需要加载多行。步长 `A_TILE_ROW_STRIDE` 确保不同线程加载不同行，同一线程隔若干行再加载。

**示例**：
- `THREAD_NUM_PER_BLOCK = 256`, `BLOCK_SIZE_K = 32`
- `A_TILE_ROW_STRIDE = 256 / 32 = 8`
- `tid = 0` 负责加载行：0, 8, 16, 24, ...（每隔 8 行）
- `tid = 1` 负责加载行：1, 9, 17, 25, ...

### 3.2 每个线程负责加载 B 的哪些元素

B 子块大小：`[BLOCK_SIZE_K, BLOCK_SIZE_N]`

```cpp
const int B_TILE_ROW = tid / BLOCK_SIZE_N;
const int B_TILE_COL = tid % BLOCK_SIZE_N;
const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;
```

**加载循环**：
```cpp
for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    const int row = tile_idx + i + B_TILE_ROW;
    const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
    Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(row, col, N)];
}
```

| 索引 | 含义 |
|------|------|
| `BLOCK_SIZE_N * blockIdx.x` | 当前 Block 在全局 B 中的起始列 |
| `tile_idx + i + B_TILE_ROW` | 在全局 B 中的行（K 维度） |
| `B_TILE_COL` | 在 B 子块内的列偏移 |

---

## 4. 核心计算阶段的索引计算

这是**最关键**的部分，理解它就能理解整个分块策略。

```cpp
const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;  // Y 方向有多少个线程
const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;  // X 方向有多少个线程
```

### 4.1 A_S 和 B_S 的含义

- `A_S = BLOCK_SIZE_M / THREAD_SIZE_Y`：在 M 方向上，有多少个线程参与分担计算。
  - 因为 Block 的 Y 维度有 `bszy = BLOCK_SIZE_M / THREAD_SIZE_Y = A_S` 个线程。

- `B_S = BLOCK_SIZE_N / THREAD_SIZE_X`：在 N 方向上，有多少个线程参与分担计算。
  - 因为 Block 的 X 维度有 `bszx = BLOCK_SIZE_N / THREAD_SIZE_X = B_S` 个线程。

### 4.2 计算循环中的索引

```cpp
for (int k = 0; k < BLOCK_SIZE_K; ++k) {
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            accum[thread_y][thread_x] += 
                As[thread_y * A_S + threadIdx.y][k] * 
                Bs[k][thread_x * B_S + threadIdx.x];
        }
    }
}
```

#### As 的索引：`thread_y * A_S + threadIdx.y`

这是**将线程 Y 坐标映射到 A 子块的行**。

**逻辑**：
- `threadIdx.y` 范围：`[0, A_S - 1]`
- 当 `thread_y = 0`：访问行 `threadIdx.y`（第 1 组行）
- 当 `thread_y = 1`：访问行 `A_S + threadIdx.y`（第 2 组行）
- 当 `thread_y = 2`：访问行 `2 * A_S + threadIdx.y`（第 3 组行）

**示例**：
- `BLOCK_SIZE_M = 128`, `THREAD_SIZE_Y = 8`
- `A_S = 128 / 8 = 16`
- `threadIdx.y = 0` 的线程负责计算的行：0, 16, 32, 48, 64, 80, 96, 112
- `threadIdx.y = 1` 的线程负责计算的行：1, 17, 33, 49, 65, 81, 97, 113
- ...
- `threadIdx.y = 15` 的线程负责计算的行：15, 31, 47, 63, 79, 95, 111, 127

**本质**：将 `BLOCK_SIZE_M` 行均匀分给 `A_S` 个线程，每个线程计算 `THREAD_SIZE_Y` 行，行号间隔 `A_S`。

#### Bs 的索引：`thread_x * B_S + threadIdx.x`

同理，将线程 X 坐标映射到 B 子块的列。

- `threadIdx.x` 范围：`[0, B_S - 1]`
- 当 `thread_x = 0`：访问列 `threadIdx.x`
- 当 `thread_x = 1`：访问列 `B_S + threadIdx.x`

**本质**：将 `BLOCK_SIZE_N` 列均匀分给 `B_S` 个线程，每个线程计算 `THREAD_SIZE_X` 列，列号间隔 `B_S`。

### 4.3 累加器 accum 的索引

```cpp
float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
```

每个线程有 `THREAD_SIZE_Y × THREAD_SIZE_X` 个私有寄存器，缓存自己负责的所有 C 元素的局部累加结果。

---

## 5. 写回全局内存的索引计算

```cpp
for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
        const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
        C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
    }
}
```

### 5.1 全局行号 `row`

```
row = BLOCK_SIZE_M * blockIdx.y   // Block 起始行
    + thread_y * A_S              // 线程负责的组偏移（第几个 THREAD_SIZE_Y 组）
    + threadIdx.y                 // 线程在 Block 内的 Y 坐标
```

这与计算阶段 `As[thread_y * A_S + threadIdx.y][k]` 的行索引**完全一致**，确保计算和写回对应同一个 C 元素。

### 5.2 全局列号 `col`

```
col = BLOCK_SIZE_N * blockIdx.x   // Block 起始列
    + thread_x * B_S              // 线程负责的组偏移（第几个 THREAD_SIZE_X 组）
    + threadIdx.x                 // 线程在 Block 内的 X 坐标
```

这与计算阶段 `Bs[k][thread_x * B_S + threadIdx.x]` 的列索引**完全一致**。

---

## 6. 一图胜千言：索引映射关系

### 6.1 Block 与线程分工示意图

```
全局矩阵 C [M, N]

Block (blockIdx.y=1, blockIdx.x=2) 负责 C 子块：
┌──────────────────────────────────────────┐
│  行 [128:256) , 列 [256:384)  (假设 BLOCK=128)  │
│                                          │
│  Block 内 16×16 个线程 (bszy=16, bszx=16)   │
│                                          │
│  threadIdx.y=0 ──► 负责行: 0,16,32,...,112    │
│  threadIdx.y=1 ──► 负责行: 1,17,33,...,113    │
│       ...                                │
│  threadIdx.y=15 ──► 负责行: 15,31,...,127     │
│                                          │
│  threadIdx.x=0 ──► 负责列: 0,16,32,...,112    │
│  threadIdx.x=1 ──► 负责列: 1,17,33,...,113    │
│       ...                                │
│  threadIdx.x=15 ──► 负责列: 15,31,...,127     │
│                                          │
│  每个线程计算 8×8 = 64 个元素                  │
└──────────────────────────────────────────┘
```

### 6.2 A_S / B_S 的物理意义

```
A_S = BLOCK_SIZE_M / THREAD_SIZE_Y = 128 / 8 = 16

意味着：把 128 行分成 16 组，每组由同一个 threadIdx.y 的线程负责。
线程 (threadIdx.y=3) 负责的行索引：
  thread_y=0: 3
  thread_y=1: 3 + 16 = 19
  thread_y=2: 3 + 32 = 35
  ...
  thread_y=7: 3 + 112 = 115

B_S = BLOCK_SIZE_N / THREAD_SIZE_X = 128 / 8 = 16

同理，把 128 列分成 16 组。
```

---

## 7. 代码中存在的笔误/错误

文件中有一些明显的语法错误（不影响索引逻辑理解，但会导致编译失败）：

| 行号 | 错误内容 | 应为 |
|------|---------|------|
| 1 | `<torch/extenson.h>` | `<torch/extension.h>` |
| 14-18 | 模板参数间用 `;` 分隔 | 应用 `,` 分隔 |
| 25-27 | `float* int M` | `int M`（参数类型错误） |
| 55,56 | 行末缺少 `;` | 补分号 |
| 63 | `tile_idx < k` | `tile_idx < K`（变量名冲突） |
| 78 | `I += B_TILE_ROW_STRIDE` | `i += B_TILE_ROW_STRIDE`（大小写） |
| 82 | `row < M && col < K` | `row < K && col < N`（B 的维度判断错误） |
| 92 | `++K` | `++k`（大小写） |
| 94 | `++thead_y` | `++thread_y`（拼写） |

---

## 8. 总结

| 索引 | 公式 | 含义 |
|------|------|------|
| `OFFSET(r,c,ld)` | `r * ld + c` | 行主序矩阵一维偏移 |
| `tid` | `threadIdx.y * bszx + threadIdx.x` | Block 内一维线程 ID |
| `A_TILE_ROW` | `tid / BLOCK_SIZE_K` | 线程加载 A 子块的行偏移 |
| `A_TILE_COL` | `tid % BLOCK_SIZE_K` | 线程加载 A 子块的列偏移 |
| `A_TILE_ROW_STRIDE` | `THREAD_NUM / BLOCK_SIZE_K` | A 加载的行步长 |
| `A_S` | `BLOCK_SIZE_M / THREAD_SIZE_Y` | M 方向线程分组大小 |
| `B_S` | `BLOCK_SIZE_N / THREAD_SIZE_X` | N 方向线程分组大小 |
| As 计算行 | `thread_y * A_S + threadIdx.y` | 将线程映射到 A 子块的行 |
| Bs 计算列 | `thread_x * B_S + threadIdx.x` | 将线程映射到 B 子块的列 |
| C 全局行 | `BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y` | 写回时的全局行号 |
| C 全局列 | `BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x` | 写回时的全局列号 |

**核心设计思想**：通过 `A_S` 和 `B_S` 将 Block 的线程网格“均匀撒”在 C 子块上，每个线程隔行/隔列取元素，保证内存访问的合并性（Coalesced Access），同时利用寄存器缓存局部累加结果，最大化计算吞吐。
