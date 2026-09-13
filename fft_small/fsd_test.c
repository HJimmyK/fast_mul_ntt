// fft_small 正确性自测：
//   1. AVX2 向量原语与标量 u64 模运算对拍（mulmod/reduce/转换/转置）
//   2. mpn_ctx 素数列表核对（应与 FLINT 注释中一致）
//   3. 截断 FFT/IFFT 往返校验（L=0..19，随机截断；对应 FLINT t-sd_fft.c）
//      外加小 L 时的多项式求值直接验证（完整变换，位翻转序）
//   4. abs_mul64 / abs_sqr64 与竖式乘法对拍（对应 FLINT t-mul.c）

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft_small.h"
#include "fsd.h"

/* ---------- 简易随机数 ---------- */
static ulong rng_state = 0x20260912;

static ulong rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static ulong rng_below(ulong n) {
    return rng_next() % n;
}

static ulong nmod_add_u(ulong a, ulong b, nmod_t mod) {
    ulong r = a + b;
    return r >= mod.n ? r - mod.n : r;
}

/* ---------- 1. 向量原语 ---------- */

static int test_vec(void) {
    /* 只测合法的 ~50bit NTT 素数（值超过 2^53 的模数不在本模块适用域内） */
    static const ulong primes[] = {1108307720798209ULL, 659706976665601ULL, 1086317488242689ULL,
                                   910395627798529ULL, 699289395265537ULL};
    int fail = 0;

    for (ulong pi = 0; pi < sizeof(primes)/sizeof(primes[0]); pi++)
    {
        ulong p = primes[pi];
        double n = p, ninv = 1.0/p;

        for (int rep = 0; rep < 20000; rep++)
        {
            /* a, b 覆盖 (-3n, 3n) 内的整数值（蝶形中出现的范围），
               且保证 |a*b| < 4n^2（mulmod 的精度边界） */
            slong ia, ib;
            do {
                ia = (slong)rng_below(6*p) - (slong)(3*p);
                ib = (slong)rng_below(6*p) - (slong)(3*p);
            } while ((double)ia * (double)ib > 3.99*n*n || (double)ia * (double)ib < -3.99*n*n);
            double a = (double)ia, b = (double)ib;

            ulong ra = ((ia % (slong)p) + (slong)p) % (slong)p;
            ulong rb = ((ib % (slong)p) + (slong)p) % (slong)p;
            ulong want = nmod_mul(ra, rb, (nmod_t){p});
            double got_d = vec1d_mulmod(a, b, n, ninv);
            ulong got = (ulong)vec1d_reduce_to_0n(got_d, n, ninv);
            if (got != want)
            {
                printf("vec mulmod FAIL: p=%llu a=%lld b=%lld got=%llu want=%llu\n",
                       (unsigned long long)p, ia, ib, (unsigned long long)got, (unsigned long long)want);
                fail = 1;
            }

            double pm = vec1d_reduce_to_pm1n(a, n, ninv);
            ulong pmr = (ulong)vec1d_reduce_to_0n(pm, n, ninv);
            if (pmr != ra)
            {
                printf("vec reduce FAIL: p=%llu a=%lld\n", (unsigned long long)p, ia);
                fail = 1;
            }
        }

        /* vec4d mulmod：4 路混合 */
        for (int rep = 0; rep < 2000; rep++)
        {
            double aa[4], bb[4];
            ulong want[4];
            for (int k = 0; k < 4; k++)
            {
                slong ia = (slong)rng_below(2*p) - (slong)p;
                slong ib = (slong)rng_below(2*p) - (slong)p;
                aa[k] = (double)ia;
                bb[k] = (double)ib;
                ulong ra = ((ia % (slong)p) + (slong)p) % (slong)p;
                ulong rb = ((ib % (slong)p) + (slong)p) % (slong)p;
                want[k] = nmod_mul(ra, rb, (nmod_t){p});
            }
            vec4d A = vec4d_load(aa), B = vec4d_load(bb);
            vec4d R = vec4d_mulmod(A, B, vec4d_set_d(n), vec4d_set_d(ninv));
            double rr[4];
            vec4d_store(rr, R);
            for (int k = 0; k < 4; k++)
            {
                ulong got = (ulong)vec1d_reduce_to_0n(rr[k], n, ninv);
                if (got != want[k])
                {
                    printf("vec4d mulmod FAIL: p=%llu lane=%d\n", (unsigned long long)p, k);
                    fail = 1;
                }
            }
        }
    }

    /* convert_limited: [0, 2^50) 整数 double -> u64 */
    for (int rep = 0; rep < 20000; rep++)
    {
        double in[4];
        ulong want[4];
        for (int k = 0; k < 4; k++)
        {
            want[k] = rng_next() & ((1ULL << 50) - 1);
            in[k] = (double)want[k];
        }
        vec4n Y = vec4d_convert_limited_vec4n(vec4d_load(in));
        ulong got[4] __attribute__((aligned(32)));
        vec4n_store_unaligned(got, Y);
        if (memcmp(got, want, sizeof(want)) != 0)
        {
            printf("convert_limited FAIL\n");
            fail = 1;
        }
    }

    /* VEC4D_TRANSPOSE */
    {
        double in[16] __attribute__((aligned(32))), out[16] __attribute__((aligned(32)));
        for (int k = 0; k < 16; k++) in[k] = k + 0.0;
        vec4d x0 = vec4d_load(in), x1 = vec4d_load(in+4), x2 = vec4d_load(in+8), x3 = vec4d_load(in+12);
        VEC4D_TRANSPOSE(x0, x1, x2, x3, x0, x1, x2, x3);
        vec4d_store(out, x0); vec4d_store(out+4, x1); vec4d_store(out+8, x2); vec4d_store(out+12, x3);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                if (out[4*r + c] != (double)(4*c + r)) { printf("transpose FAIL\n"); fail = 1; }
    }

    /* unpack/permute 模式 */
    {
        double u[4] __attribute__((aligned(32))) = {0, 1, 2, 3};
        double v[4] __attribute__((aligned(32))) = {4, 5, 6, 7};
        double o[4] __attribute__((aligned(32)));
        vec4d U = vec4d_load(u), V = vec4d_load(v);
        vec4d_store(o, vec4d_unpack_lo_permute_0_2_1_3(U, V));
        if (o[0]!=0 || o[1]!=2 || o[2]!=4 || o[3]!=6) { printf("unpack_lo_0213 FAIL\n"); fail = 1; }
        vec4d_store(o, vec4d_unpack_hi_permute_0_2_1_3(U, V));
        if (o[0]!=1 || o[1]!=3 || o[2]!=5 || o[3]!=7) { printf("unpack_hi_0213 FAIL\n"); fail = 1; }
        vec4d_store(o, vec4d_unpacklo_permute_3_1_2_0(U, V));
        if (o[0]!=6 || o[1]!=4 || o[2]!=2 || o[3]!=0) { printf("unpacklo_3120 FAIL\n"); fail = 1; }
        vec4d_store(o, vec4d_unpackhi_permute_3_1_2_0(U, V));
        if (o[0]!=7 || o[1]!=5 || o[2]!=3 || o[3]!=1) { printf("unpackhi_3120 FAIL\n"); fail = 1; }
        vec4d_store(o, vec4d_permute_3_2_1_0(U));
        if (o[0]!=3 || o[1]!=2 || o[2]!=1 || o[3]!=0) { printf("permute_3210 FAIL\n"); fail = 1; }
    }

    printf("test_vec: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

/* ---------- 2. 上下文核对 ---------- */

static int test_ctx(void) {
    /* FLINT mpn_mul.c 注释中列出的 8 个素数 */
    static const ulong want_primes[] = {
        1108307720798209ULL, 659706976665601ULL, 1086317488242689ULL,
        910395627798529ULL, 699289395265537ULL, 1022545813831681ULL,
        1013749720809473ULL, 868614185943041ULL};

    mpn_ctx_t R;
    mpn_ctx_init(R, 0x0003f00000000001ULL);

    int fail = 0;
    for (int i = 0; i < MPN_CTX_NCRTS; i++)
    {
        if (R->ffts[i].p != (double)want_primes[i])
        {
            printf("prime[%d] = %llu, want %llu\n", i,
                   (unsigned long long)R->ffts[i].p, (unsigned long long)want_primes[i]);
            fail = 1;
        }
    }

    mpn_ctx_clear(R);
    printf("test_ctx: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

/* ---------- 3. FFT/IFFT 往返 ---------- */

static int test_fft(void) {
    int fail = 0;
    mpn_ctx_t R;
    mpn_ctx_init(R, 0x0003f00000000001ULL);
    sd_fft_ctx_struct* Q = R->ffts; /* 第一个素数（2 进制赋值 40，深度足够） */

    for (ulong L = 0; L <= 19; L++)
    {
        ulong Xn = n_pow2(L);
        double* X = (double*)malloc(Xn*sizeof(double));
        double* data = (double*)fsd_aligned_alloc(32, n_max(32, Xn*sizeof(double)));
        nmod_t mod = {(ulong)Q->p};

        ulong nreps = 5 + 2*L;
        for (ulong rep = 0; rep < nreps && !fail; rep++)
        {
            ulong trunc = (rep == 0) ? Xn : 1 + rng_below(Xn);

            for (ulong i = 0; i < trunc; i++)
                X[i] = (double)rng_below((ulong)Q->p);

            /* 完整变换 + 逆变换，应得 2^L * X */
            for (ulong i = 0; i < trunc; i++)
                data[i] = X[i];

            sd_fft_trunc(Q, data, L, trunc, trunc);
            sd_ifft_trunc(Q, data, L, trunc);

            ulong scale = nmod_pow_ui(2, L, mod);
            for (ulong chk = 0; chk < n_min(trunc, 8); chk++)
            {
                ulong i = rng_below(trunc);
                ulong want = nmod_mul((ulong)X[i], scale, mod);
                ulong got = (ulong)vec1d_reduce_to_0n(data[i], Q->p, Q->pinv);
                if (got != want)
                {
                    printf("fft roundtrip FAIL: L=%lu trunc=%lu i=%lu got=%llu want=%llu\n",
                           L, trunc, i, (unsigned long long)got, (unsigned long long)want);
                    fail = 1;
                    break;
                }
            }

            /* 小 L 时的多项式求值直接验证正变换（完整变换，输出位翻转序） */
            if (L <= 3 && trunc == Xn)
            {
                for (ulong i = 0; i < Xn; i++)
                    data[i] = X[i];
                sd_fft_trunc(Q, data, L, Xn, Xn);
                /* 变换使用的原始 2^L 次根：prim^(2^(td-L))，
                   其中 td = (p-1) 的 2 进制赋值（与 w2tab 生成方式一致） */
                ulong td = n_trailing_zeros(Q->mod.n - 1);
                ulong g = nmod_pow_ui(Q->primitive_2power_root,
                                      UWORD(1) << (td - L), Q->mod);
                for (ulong i = 0; i < Xn && !fail; i++)
                {
                    ulong gi = nmod_pow_ui(g, i, mod); /* 本次求值的基底 w^i */
                    ulong y = 0;
                    for (ulong k = Xn; k > 0; k--)
                    {
                        ulong c = (ulong)vec1d_reduce_to_0n(X[k-1], Q->p, Q->pinv);
                        y = nmod_mul(y, gi, mod);
                        y = nmod_add_u(y, c, mod);
                    }
                    ulong got = (ulong)vec1d_reduce_to_0n(data[n_revbin(i, L)], Q->p, Q->pinv);
                    if (got != y)
                    {
                        printf("fft eval FAIL: L=%lu i=%lu got=%llu want=%llu\n",
                               L, i, (unsigned long long)got, (unsigned long long)y);
                        fail = 1;
                    }
                }
            }
        }

        fsd_aligned_free(data);
        free(X);
        if (fail)
            break;
    }

    mpn_ctx_clear(R);
    printf("test_fft: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

/* ---------- 4. 乘法对拍 ---------- */

static void schoolbook_mul(const u64* a, ulong na, const u64* b, ulong nb, u64* out) {
    memset(out, 0, (na + nb)*sizeof(u64));
    for (ulong i = 0; i < na; i++)
    {
        u64 carry = 0;
        for (ulong j = 0; j < nb; j++)
        {
            unsigned __int128 t = (unsigned __int128)a[i]*b[j] + out[i+j] + carry;
            out[i+j] = (u64)t;
            carry = (u64)(t >> 64);
        }
        ulong k = i + nb;
        while (carry)
        {
            u64 s = out[k] + carry;
            carry = s < carry;
            out[k] = s;
            k++;
        }
    }
}

static int test_mul(void) {
    int fail = 0;
    ulong maxn = 3000;
    u64* a = (u64*)malloc(maxn*sizeof(u64));
    u64* b = (u64*)malloc(maxn*sizeof(u64));
    u64* c = (u64*)malloc(2*maxn*sizeof(u64));
    u64* d = (u64*)malloc(2*maxn*sizeof(u64));

    /* 尺寸覆盖：极小、非 2 幂、不等长、跨越各 profile 的 bits 切换点 */
    ulong sizes[] = {1, 2, 3, 5, 17, 33, 64, 100, 255, 256, 257, 300, 511, 512, 513,
                     700, 1024, 1100, 1500, 2048, 2500, 2900};

    for (ulong si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++)
    {
        for (ulong sj = si; sj < sizeof(sizes)/sizeof(sizes[0]); sj++)
        {
            ulong an = sizes[si], bn = sizes[sj];
            for (ulong i = 0; i < an; i++) a[i] = rng_next();
            for (ulong i = 0; i < bn; i++) b[i] = rng_next();

            schoolbook_mul(a, an, b, bn, c);
            abs_mul64(a, an, b, bn, d);
            if (memcmp(c, d, (an+bn)*sizeof(u64)) != 0)
            {
                printf("mul FAIL: %llu x %llu\n", (unsigned long long)an, (unsigned long long)bn);
                fail = 1;
            }

            /* 交换操作数顺序 */
            abs_mul64(b, bn, a, an, d);
            if (memcmp(c, d, (an+bn)*sizeof(u64)) != 0)
            {
                printf("mul(swapped) FAIL: %llu x %llu\n", (unsigned long long)an, (unsigned long long)bn);
                fail = 1;
            }

            /* 平方 */
            schoolbook_mul(a, an, a, an, c);
            abs_sqr64(a, an, d);
            if (memcmp(c, d, 2*an*sizeof(u64)) != 0)
            {
                printf("sqr FAIL: %llu\n", (unsigned long long)an);
                fail = 1;
            }
        }
    }

    /* 跨越 bits 切换的精细扫描 */
    for (ulong an = 400; an <= 3000 && !fail; an += 37)
    {
        ulong bn = an/2 + 1;
        for (ulong i = 0; i < an; i++) a[i] = rng_next();
        for (ulong i = 0; i < bn; i++) b[i] = rng_next();
        schoolbook_mul(a, an, b, bn, c);
        abs_mul64(a, an, b, bn, d);
        if (memcmp(c, d, (an+bn)*sizeof(u64)) != 0)
        {
            printf("mul FAIL: %llu x %llu\n", (unsigned long long)an, (unsigned long long)bn);
            fail = 1;
        }
    }

    free(a); free(b); free(c); free(d);
    printf("test_mul: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= test_vec();
    fail |= test_ctx();
    fail |= test_fft();
    fail |= test_mul();
    fft_small_clear();
    printf(fail ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fail;
}
