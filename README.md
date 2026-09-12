# fast_mul_ntt

基于数论变换（NTT）与三模数中国剩余定理（CRT）的**大整数乘法**快速实现，纯 C11 编写，无第三方依赖。

## 算法概要

- **三模数 NTT + CRT**：选用三个小于 2^62 的 NTT 素数（见 `src/data.h`），各自完成模域内的卷积后由 `crt3` 合并，模数乘积约 2^184，足以覆盖卷积系数的动态范围；
- **蒙哥马利模乘**：以 R = 2^64 为蒙哥马利域，配合 x86-64 `mul` 指令的 128 位乘法，全程无除法取模；
- **混合基 DIF/DIT 蝶形**：主体为基-4 蝶形，末端落到 4/8 点小变换，配合懒规约（lazy reduction）减少条件分支；
- **缓存友好的递归变换**：变换长度不超过 131 072（约 1 MiB）时走迭代法并使用预计算的旋转因子表；超过阈值后按 n/2 + n/4 + n/4 三段递归分解，使各层工作集逐级缩小、驻留缓存；
- **三种卷积核**：`conv_rec`（两个输入都做正变换）、`conv_single`（一侧已完成正变换）、`conv_sqr`（平方，只需一次正变换）。

对外只暴露两个顶层函数（`src/fast_mul.h`）：

| 接口 | 功能 | 输出长度 |
|------|------|----------|
| `abs_mul64(in1, len1, in2, len2, out)` | 大整数乘法 `out = in1 × in2` | `len1 + len2` |
| `abs_sqr64(in, len, out)` | 大整数平方 `out = in²` | `2 × len` |

所有数都用小端（低位在前）64 位 limb 数组表示。

## 目录结构

```
.
├── src/                核心源码（不含任何测试代码）
│   ├── fast_mul.h          对外 API：abs_mul64 / abs_sqr64
│   ├── fast_mul.c          卷积核（conv_rec / conv_single / conv_sqr）与顶层实现
│   ├── core.h              NTT 正/逆变换：蝶形宏、旋转因子表、crt3
│   ├── data.h              三个 NTT 模数及蒙哥马利域常数
│   └── macro.h             基础类型、128/192 位运算与蒙哥马利乘宏
├── bench/              基准测试
│   └── bench.c             长度扫描计时程序（输出 CSV）
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## 构建

环境要求：**CMake ≥ 3.10** 与 **GCC 或 Clang**（x86-64）。

> 代码中使用了 GNU 风格内联汇编和 Windows 的 `_aligned_malloc`，目前只能在 **Windows（MinGW-w64 的 GCC/Clang）** 下编译运行，移植要点见文末[移植提示](#移植提示)。

```bash
cmake -B build
cmake --build build
```

构建产物（位于 `build/`）：

- 核心静态库（`libfast_mul.a` / `fast_mul.lib`，取决于工具链）；
- `bench.exe` —— 基准测试程序。

## 使用示例

```c
#include "fast_mul.h"

// a、b 为小端 limb 数组
u64 a[N], b[M], out[N + M];

abs_mul64(a, N, b, M, out);   // out ← a × b

u64 c[N], sq[2 * N];
abs_sqr64(c, N, sq);          // sq ← c²
```

注意：`abs_mul64` 要求 `in1` 与 `in2` 指向不同的内存块（值可以相同，平方请直接用 `abs_sqr64`）。

把库接入自己的工程只需编译 `src/fast_mul.c` 并把 `src/` 加入头文件搜索路径，例如：

```bash
gcc -O3 -Isrc my_main.c src/fast_mul.c -o my_app
```

## 基准测试

`bench` 对两个等长随机操作数做乘法计时：操作数长度从 10 000 到 10 000 000 个 limb（约 8 KB ~ 80 MB，即约 2.4 万 ~ 2400 万位十进制数），步长 33 300，共 301 个尺寸点；每个尺寸连续测 3 次取平均（单位：微秒），进度打印到终端，结果写入运行目录下的 `cc_ntt-crt_times.csv`。

```bash
./build/bench.exe
```

完整跑完需要较长时间和较大内存（最大尺寸点约需 1 GB 以上）；只想快速验证可以随时 Ctrl+C 中断，已打印的进度数据仍可参考。

## 工作原理（abs_mul64 流程）

1. 由 `len1 + len2 - 1` 的卷积长度向上取整到 2 的幂，得到 NTT 长度；
2. 输入零填充，并分别转换到三个模数的蒙哥马利域；
3. 对每个模数执行「正变换（DIF）→ 点乘 → 逆变换（DIT，含长度归一化）」；
4. `crt3` 把每个系数在三个模域下的残差合并为一个 192 bit 整数；
5. 从低位到高位做进位传播，得到最终的 64 位 limb 输出。

## 移植提示

- `src/macro.h` 的 `mul64x64to128` 使用 x86-64 GNU 内联汇编，等价于 `__uint128_t` 乘法，可按平台改写；
- `src/fast_mul.c` 的 `ALIGNED_MALLOC` / `ALIGNED_FREE` 目前绑定 Windows 的 `_aligned_malloc` / `_aligned_free`，Linux 下可替换为 `aligned_alloc` / `free`。

## 许可证

[MIT](LICENSE) © 2025 Jecricho Knox
