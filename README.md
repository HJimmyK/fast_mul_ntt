# fast_mul_ntt

本仓库包含两套**大整数乘法**实现，接口完全相同（`abs_mul64` / `abs_sqr64`，小端 64 位 limb 数组）：

| 模块 | 来源 | 指令集 | 许可证 |
|------|------|--------|--------|
| `src/` | 3 模数 NTT + CRT | x86-64（GNU 内联汇编） | MIT |
| `fft_small/` | 移植自 [FLINT](https://flintlib.org) 的 fft_small（单线程化） | AVX2 + FMA | LGPL-3.0-or-later |

两套实现的接口约定：

```c
void abs_mul64(const u64* in1, u64 len1, const u64* in2, u64 len2, u64* out); /* out = in1*in2，长 len1+len2 */
void abs_sqr64(const u64* in1, u64 len1, u64* out);                          /* out = in1²，长 2*len1 */
```

> 注意：两个库导出同名符号，链接时二选一。

## fft_small/（FLINT 移植版，AVX2）

从 FLINT 的 `src/fft_small` 提炼移植的通用乘法/平方，**单线程、AVX2、无外部依赖**（不链接 GMP/FLINT 本体）。算法要点：

- **小数域 NTT**：剩余系用 double 表示（约 50 bit 的 NTT 素数），模乘用 FMA 的 double-double 技巧实现（精确同余，边界条件由 `fft_small_mulmod_satisfies_bounds` 校验）；
- **截断 FFT/IFFT**（van der Hoeven / David Harvey 算法）：按块（256 个 double）组织，行列四步分解 + 基 2/4 蝶形，输入输出均可截断以跳过零填充区；
- **多素数 CRT**：8 个 NTT 素数按操作数长度自动挑选 4~8 个及分组位数（84~192 bit/组），CRT 余因子与 2 的幂表预计算；
- **数据规模上限**：较短操作数不超过约 25 亿 limb（约 20 GB），超过时报错退出。

**模块接口**：公开头文件只有 `fft_small.h`（`abs_mul64` / `abs_sqr64` / `fft_small_init` / `fft_small_clear`）。`fsd.h` 及各 `fsd_*.c` 均为内部实现：单文件内使用的函数一律 `static`（如 `crt_data_*`、`sd_fft_ctx_point_*`、`mpn_ctx_fit_buffer`），跨文件的内部符号集中声明在 `fsd.h`，使用者不应依赖。

```bash
cmake -B build && cmake --build build
./build/fsd_test            # 正确性自测（向量原语 / FFT 往返 / 乘法对拍）
./build/bench_fft_small.exe # 基准测试（输出 cc_fft_small_times.csv）
```

`fft_small/` 内的文件移植自 FLINT，保留原始版权声明；`fsd_vec.h`（AVX2 向量原语）与 `fsd_mpn.c`、`fsd_crt.h`（替代 GMP 的小函数）为重写。

## 两套实现的对比

两个库导出同名接口、基准测试的扫描方式一致（同尺寸序列、每点 3 次取平均），`cc_ntt-crt_times.csv` 与 `cc_fft_small_times.csv` 可直接对画。算法层面的差异：

| 维度 | `src/`（3ntt_crt） | `fft_small/`（FLINT 移植） |
|------|----------------|---------------------------|
| 剩余系表示 | 三个 < 2^62 的 NTT 素数，u64 整数运算 | 4~8 个 ~2^50 素数，剩余用 double 表示 |
| 模乘 | `mulx` 128 位乘 + 蒙哥马利约减（R = 2^64） | FMA 的 double-double 同余乘法（依赖严格舍入语义，禁用 `-ffast-math`） |
| 变换类型 | 完整变换：卷积长度向上取整到 2 的幂，零填充 | 截断 FFT/IFFT（van der Hoeven / Harvey）：输入输出均可截断，跳过零填充区 |
| 蝶形组织 | 基-4 DIF/DIT，末端 4/8 点小变换，懒规约 | 256 double 一块：块内基 2/4 basecase，块间四步分解递归（k/2 + k/2） |
| 缓存策略 | 长度 ≤ 131072 走迭代 + 预计算旋转因子表；超过后 n/2+n/4+n/4 三段递归 | 递归各层工作集逐级减半；旋转因子表按深度惰性扩展 |
| 素数选择 | 固定 3 个（乘积约 2^184 覆盖卷积动态范围） | 按较短操作数长度从 8 个中选 4~8 个、每组 84~192 bit，尺寸方案表打分挑选 |
| 指令集 | x86-64 标量 + GNU 内联汇编 | AVX2 + FMA 向量化（vec4d/vec8d） |
| 平方路径 | `conv_sqr`：一次正变换 | `squaring` 分支：一次正变换 + `point_sqr` |
| 线程 | 单线程 | 单线程（FLINT 原版支持多线程，本移植删去） |
| 上限 | 卷积系数动态范围受三模数乘积（~2^184）约束 | 较短操作数 ≤ 约 25 亿 limb（约 20 GB） |
| 许可证 | MIT | LGPL-3.0-or-later |

> 注意：两个库导出同名符号，链接时二选一，不能同时链入。

## 3NTT-CRT（src/）

基于数论变换（NTT）与三模数中国剩余定理（CRT）的**大整数乘法**快速实现，纯 C11 编写。

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
├── fft_small/          FLINT fft_small 移植（单线程 AVX2，LGPL）
│   ├── fft_small.h/.c      唯一公开头文件与全局上下文（abs_mul64 / abs_sqr64）
│   ├── fsd.h               内部公共定义（仅模块内部使用，跨文件内部符号集中于此）
│   ├── fsd_vec.h           AVX2 向量原语（vec1d/vec4d/vec8d）
│   ├── fsd_fft.c           截断正变换（sd_fft_trunc）
│   ├── fsd_ifft.c          截断逆变换（sd_ifft_trunc）
│   ├── fsd_fft_ctx.c       旋转因子表构建与按深度扩展
│   ├── fsd_mpn_mul.c       乘法主流程（多素数方案选择、FFT 卷积、CRT 重构）
│   ├── fsd_mpn.c           mpn/nmod 小函数、素性检测、模乘精度边界
│   ├── fsd_crt.h           CRT 大数乘加/缩减模板
│   └── fsd_test.c          正确性自测
├── bench/              基准测试
│   ├── bench.c             长度扫描计时（cc_ntt-crt_times.csv）
│   └── bench_fft_small.c   fft_small 版长度扫描计时（cc_fft_small_times.csv）
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

- `fast_mul` / `fft_small` —— 两个静态库（接口同名，链接时二选一）；
- `bench`、`bench_fft_small` —— 各自的基准测试程序；
- `fsd_test` —— fft_small 的正确性自测。

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

完整运行需要较长时间和较大内存（最大尺寸点约需 1 GB 以上）；只想快速验证可以随时 Ctrl+C 中断，已打印的进度数据仍可参考。

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
