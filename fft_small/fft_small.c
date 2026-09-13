// fft_small 公开接口实现：全局上下文的惰性初始化与 abs_mul64 / abs_sqr64

#include "fft_small.h"
#include "fsd.h"

/* 与 FLINT default_ctx.c 一致的默认素数（2 进制赋值 40，支持很长的变换） */
#define FSD_DEFAULT_PRIME UWORD(0x0003f00000000001)

static mpn_ctx_struct fsd_global_ctx[1];
static int fsd_global_ctx_initialized = 0;

/* 运行时检查 CPU 是否支持 AVX2 + FMA（含操作系统 XSAVE 使能） */
static int fsd_cpu_ok(void) {
#if defined(_MSC_VER)
    int c1[4], c7[4];
    __cpuid(c1, 1);
    if (!(c1[2] & (1 << 27)) || /* OSXSAVE */
        !(c1[2] & (1 << 28)) || /* AVX */
        !(c1[2] & (1 << 12)))   /* FMA */
        return 0;
    __cpuid(c7, 7);
    if (!(c7[1] & (1 << 5))) /* AVX2 */
        return 0;
    return (_xgetbv(0) & 0x6) == 0x6; /* XMM/YMM 状态已使能 */
#elif defined(__GNUC__) && defined(__x86_64__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 1;
#endif
}

void fft_small_init(void) {
    if (fsd_global_ctx_initialized)
        return;

    if (!fsd_cpu_ok())
        fsd_abort("CPU 不支持 AVX2/FMA 指令集");

    mpn_ctx_init(fsd_global_ctx, FSD_DEFAULT_PRIME);
    fsd_global_ctx_initialized = 1;
}

void fft_small_clear(void) {
    if (fsd_global_ctx_initialized)
    {
        mpn_ctx_clear(fsd_global_ctx);
        fsd_global_ctx_initialized = 0;
    }
}

static mpn_ctx_struct* fsd_get_ctx(void) {
    if (!fsd_global_ctx_initialized)
        fft_small_init();
    return fsd_global_ctx;
}

void abs_mul64(const u64* in1, u64 len1, const u64* in2, u64 len2, u64* out) {
    /* profile 的界限按第二个操作数给出，交换保证 b 是较短的一方 */
    if (len2 > len1)
    {
        const u64* tp = in1;
        in1 = in2;
        in2 = tp;
        u64 tl = len1;
        len1 = len2;
        len2 = tl;
    }
    mpn_ctx_mpn_mul(fsd_get_ctx(), (ulong*)out, (const ulong*)in1, len1, (const ulong*)in2, len2);
}

void abs_sqr64(const u64* in1, u64 len1, u64* out) {
    mpn_ctx_mpn_mul(fsd_get_ctx(), (ulong*)out, (const ulong*)in1, len1, (const ulong*)in1, len1);
}
