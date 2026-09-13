// 大整数乘法主流程：mpn_ctx（8 个 NTT 素数 + CRT）与 mpn_ctx_mpn_mul
//
//
// 输入以 64 位 limb 数组表示，按 bits 位一组拆成多素数多项式，
// 每个素数上做截断 FFT 卷积，再按 CRT 重构出卷积系数并进位。
//
// 预置的尺寸方案（np 个素数、每组 bits 位）与对应的 bn 上限：
//     np = 4, bits =  84, bn <= 2536637511
//     np = 4, bits =  88, bn <= 10380583
//     np = 4, bits =  92, bn <= 42390
//     np = 5, bits = 112, bn <= 32822700
//     np = 5, bits = 116, bn <= 132790
//     np = 5, bits = 120, bn <= 534
//     np = 6, bits = 136, bn <= 144789875
//     np = 6, bits = 140, bn <= 582218
//     np = 6, bits = 144, bn <= 2337
//     np = 7, bits = 160, bn <= 613493872
//     np = 7, bits = 164, bn <= 2456369
//     np = 7, bits = 168, bn <= 9826
//     np = 8, bits = 184, bn <= 2177184315
//     np = 8, bits = 188, bn <= 8689506
//     np = 8, bits = 192, bn <= 34662

#include <stdio.h>
#include "fsd.h"

/* ---------------- CRT 数据的建立与释放（仅本文件使用） ---------------- */

static void crt_data_init(crt_data_t C, ulong prime, ulong coeff_len, ulong nprimes) {
    C->prime = prime;
    C->coeff_len = coeff_len;
    C->nprimes = nprimes;
    C->data = (ulong*)malloc((nprimes*coeff_len + coeff_len + nprimes)*sizeof(ulong));
    if (C->data == NULL)
        fsd_abort("内存分配失败");
}

static void crt_data_clear(crt_data_t C) {
    free(C->data);
}

/*
    计算 profile 的 bn 上限：需要 ceil(64*bn/bits) <= prod_primes/2^(2*bits)。
    也就是让 (prod - 2^(2*bits)) >> 6 仍在 coeff_len 个 limb 内可表示，
    返回该上限值（溢出时返回 ULONG_MAX）。
*/
static ulong crt_data_find_bn_bound(const crt_data_t C, ulong bits) {
    ulong bound = 0;
    ulong q = (2*bits)/64;
    ulong r = (2*bits)%64;
    ulong n = C->coeff_len;
    ulong i;
    ulong* x;

    x = (ulong*)malloc((n+1)*sizeof(ulong));

    x[n] = fsd_mpn_mul_1(x, crt_data_prod_primes(C), n, bits);

    if (q < n+1)
    {
        if (r > 0)
            fsd_mpn_rshift(x + q, x + q, n + 1 - q, r);

        if (!fsd_mpn_sub_1(x + q, x + q, n + 1 - q, bits - 1))
        {
            fsd_mpn_rshift(x + q, x + q, n + 1 - q, 6);
            bound = (x + q)[0];
            for (i = q + 1; i < n + 1; i++)
                if (x[i] != 0)
                    bound = (ulong)-1;
        }
    }

    free(x);
    return bound;
}


#define aindex(i) (((i) < an) ? a[i] : (uint32_t)(0))

#define N_CDIV(a, b) (((a) + (b) - 1) / (b))

/* 困难区段：按输出下标 i 逐个处理（读 a 可能越界，用零填充） */
#define DEFINE_IT(NP) \
static void CAT(mpn_to_ffts_hard, NP)( \
    sd_fft_ctx_struct* Rffts, double* d, ulong dstride, \
    const uint32_t* a, ulong an, ulong atrunc, \
    const vec4d* two_pow, \
    ulong start_hard, ulong stop_hard, \
    ulong bits) \
{ \
    ulong np = NP; \
    ulong nvs = N_CDIV(NP, VEC_SZ); \
    vec4d X[N_CDIV(NP, VEC_SZ)]; \
    vec4d P[N_CDIV(NP, VEC_SZ)]; \
    vec4d PINV[N_CDIV(NP, VEC_SZ)]; \
 \
    for (ulong l = 0; l < nvs; l++) \
    { \
        P[l]    = vec4d_set_d4(Rffts[4*l+0].p, Rffts[4*l+1].p, Rffts[4*l+2].p, Rffts[4*l+3].p); \
        PINV[l] = vec4d_set_d4(Rffts[4*l+0].pinv, Rffts[4*l+1].pinv, Rffts[4*l+2].pinv, Rffts[4*l+3].pinv); \
    } \
 \
    for (ulong i = start_hard; i < stop_hard; i++) \
    { \
        ulong k = (i*bits)/32; \
        ulong j = (i*bits)%32; \
 \
        vec4d ak = vec4d_set_d((double)(aindex(k) >> j)); \
        for (ulong l = 0; l < nvs; l++) \
            X[l] = ak; \
        k++; \
        j = 32 - j; \
        while (j + 32 <= bits) \
        { \
            ak = vec4d_set_d((double)(aindex(k))); \
            for (ulong l = 0; l < nvs; l++) \
                X[l] = vec4d_add(X[l], vec4d_mulmod(ak, two_pow[j*nvs+l], P[l], PINV[l])); \
            k++; \
            j += 32; \
        } \
 \
        if ((bits-j) != 0) \
        { \
            ak = vec4d_set_d((double)(aindex(k) << (32-(bits-j)))); \
            for (ulong l = 0; l < nvs; l++) \
                X[l] = vec4d_add(X[l], vec4d_mulmod(ak, two_pow[(bits-32)*nvs+l], P[l], PINV[l])); \
        } \
 \
        for (ulong l = 0; l < nvs; l++) \
            X[l] = vec4d_reduce_to_pm1n(X[l], P[l], PINV[l]); \
 \
        for (ulong l = 0; l < np; l++) \
            (d + l*dstride)[i] = vec4d_get_index(X[l/VEC_SZ], l%VEC_SZ); \
    } \
 \
    for (ulong l = 0; l < np; l++) \
        for (ulong i = stop_hard; i < atrunc; i++) \
            (d + l*dstride)[i] = 0.0; \
}

DEFINE_IT(4)
DEFINE_IT(5)
DEFINE_IT(6)
DEFINE_IT(7)
DEFINE_IT(8)
#undef DEFINE_IT
#undef aindex
#undef N_CDIV

#define aindex(i) (((i) < an) ? a[i] : (uint32_t)(0))

#define CODE(ir) \
{ \
    ulong k = ((i+ir)*bits)/32; \
    ulong j = ((  ir)*bits)%32; \
 \
    vec4d ak = vec4d_set_d((double)(a[k] >> j)); \
    for (ulong l = 0; l < nvs; l++) \
        X[l] = ak; \
    k++; \
    j = 32 - j; \
    while (j + 32 <= bits) \
    { \
        ak = vec4d_set_d((double)(a[k])); \
        for (ulong l = 0; l < nvs; l++) \
            X[l] = vec4d_add(X[l], vec4d_mulmod(ak, two_pow[j*nvs+l], P[l], PINV[l])); \
        k++; \
        j += 32; \
    } \
 \
    if ((bits-j) != 0) \
    { \
        ak = vec4d_set_d((double)(a[k] << (32-(bits-j)))); \
        for (ulong l = 0; l < nvs; l++) \
            X[l] = vec4d_add(X[l], vec4d_mulmod(ak, two_pow[(bits-32)*nvs+l], P[l], PINV[l])); \
    } \
 \
    for (ulong l = 0; l < nvs; l++) \
        X[l] = vec4d_reduce_to_pm1n(X[l], P[l], PINV[l]); \
 \
    for (ulong l = 0; l < np; l++) \
        (d + l*dstride)[i+ir] = vec4d_get_index(X[l/VEC_SZ], l%VEC_SZ); \
}

#define N_CDIV(a, b) (((a) + (b) - 1) / (b))

/* 把输入 limb 数组按 bits 位一组切成系数，转成 np 个素数上的 double 剩余。
   easy 区段：start/stop 已保证读 a 不越界，可无检查展开 */
#define DEFINE_IT(NP, BITS) \
static void CAT3(mpn_to_ffts, NP, BITS)( \
    sd_fft_ctx_struct* Rffts, double* d, ulong dstride, \
    const ulong* a_, ulong an_, ulong atrunc, \
    const vec4d* two_pow, \
    ulong start_easy, ulong stop_easy, \
    ulong start_hard, ulong stop_hard) \
{ \
    ulong np = NP; \
    ulong bits = BITS; \
    ulong nvs = N_CDIV(NP, VEC_SZ); \
 \
    FSD_ASSERT(bits >= 64); \
    FSD_ASSERT(bits - 32 < MPN_CTX_TWO_POWER_TAB_SIZE); \
 \
    const uint32_t* a = (const uint32_t*)(a_); \
    ulong an = 2*an_; \
 \
    vec4d X[N_CDIV(NP, VEC_SZ)]; \
    vec4d P[N_CDIV(NP, VEC_SZ)]; \
    vec4d PINV[N_CDIV(NP, VEC_SZ)]; \
 \
    for (ulong l = 0; l < nvs; l++) \
    { \
        P[l]    = vec4d_set_d4(Rffts[4*l+0].p, Rffts[4*l+1].p, Rffts[4*l+2].p, Rffts[4*l+3].p); \
        PINV[l] = vec4d_set_d4(Rffts[4*l+0].pinv, Rffts[4*l+1].pinv, Rffts[4*l+2].pinv, Rffts[4*l+3].pinv); \
    } \
 \
    if ((bits % 8) == 0) \
    { \
        FSD_ASSERT(start_easy % 4 == 0); \
        FSD_ASSERT(stop_easy % 4 == 0); \
        for (ulong i = start_easy ; i < stop_easy; i += 4) \
        { \
            CODE(0);CODE(1);CODE(2);CODE(3); \
        } \
    } \
    else if ((bits % 4) == 0) \
    { \
        FSD_ASSERT(start_easy % 8 == 0); \
        FSD_ASSERT(stop_easy % 8 == 0); \
        for (ulong i = start_easy ; i < stop_easy; i += 8) \
        { \
            CODE(0);CODE(1);CODE(2);CODE(3); \
            CODE(4);CODE(5);CODE(6);CODE(7); \
        } \
    } \
    else if ((bits % 2) == 0) \
    { \
        FSD_ASSERT(start_easy % 16 == 0); \
        FSD_ASSERT(stop_easy % 16 == 0); \
        for (ulong i = start_easy ; i < stop_easy; i += 16) \
        { \
            CODE(0);CODE(1);CODE(2);CODE(3); \
            CODE(4);CODE(5);CODE(6);CODE(7); \
            CODE(8);CODE(9);CODE(10);CODE(11); \
            CODE(12);CODE(13);CODE(14);CODE(15); \
        } \
    } \
    else \
    { \
        FSD_ASSERT(0); \
    } \
 \
    CAT(mpn_to_ffts_hard, NP)(Rffts, d, dstride, a, an, atrunc, two_pow, \
                              start_hard, stop_hard, bits); \
}

DEFINE_IT(4, 84)
DEFINE_IT(4, 88)
DEFINE_IT(4, 92)
DEFINE_IT(5,112)
DEFINE_IT(5,116)
DEFINE_IT(5,120)
DEFINE_IT(6,136)
DEFINE_IT(6,140)
DEFINE_IT(6,144)
DEFINE_IT(7,160)
DEFINE_IT(7,164)
DEFINE_IT(7,168)
DEFINE_IT(8,184)
DEFINE_IT(8,188)
DEFINE_IT(8,192)
#undef DEFINE_IT
#undef CODE
#undef aindex
#undef N_CDIV


/* CRT 重构时把 FFT 结果累加进输出 z：
   easy 路径（toff+n <= zn，完整 multi_add 即可）与
   hard 路径（写尾部时可能只剩不足 n 个 limb，退化为 fsd_mpn_add_n） */
#define DEFINE_IT(n, n_plus_1) \
static inline void CAT(_add_to_answer_easy, n)(ulong z[], ulong r[], ulong FSD_UNUSED(zn), ulong toff, ulong tshift) \
{ \
    FSD_ASSERT(zn > toff); \
    if (tshift == 0) \
    { \
        CAT(multi_add, n)(z + toff, r); \
    } \
    else \
    { \
        r[n] = r[n-1] >> (64-tshift); \
        for (ulong k = n; k >= 2; k--) \
            r[k-1] = (r[k-1] << (tshift)) | (r[k-2] >> (64-tshift)); \
        r[0] =  r[0] << (tshift); \
        CAT(multi_add, n_plus_1)(z + toff, r); \
    } \
} \
static inline void CAT(_add_to_answer_hard, n)(ulong z[], ulong r[], ulong zn, ulong toff, ulong tshift) \
{ \
    FSD_ASSERT(zn > toff); \
    if (tshift == 0) \
    { \
        if (zn - toff >= n) \
        { \
            CAT(multi_add, n)(z + toff, r); \
            return; \
        } \
    } \
    else \
    { \
        r[n] = r[n-1] >> (64-tshift); \
        for (ulong k = n; k >= 2; k--) \
            r[k-1] = (r[k-1] << (tshift)) | (r[k-2] >> (64-tshift)); \
        r[0] =  r[0] << (tshift); \
        if (zn - toff > n) \
        { \
            CAT(multi_add, n_plus_1)(z + toff, r); \
            return; \
        } \
    } \
    FSD_ASSERT(zn - toff <= n); \
    (void)fsd_mpn_add_n(z + toff, z + toff, r, zn - toff); \
}

DEFINE_IT(4, 5)
DEFINE_IT(5, 6)
DEFINE_IT(6, 7)
DEFINE_IT(7, 8)
#undef DEFINE_IT

/* 把第 l 个素数在 d + l*dstride 处第 I 块的 FFT 值规约到 [0, p)，
   转成整数并写入 Xs + l*BLK_SZ（供 CRT 重构使用） */
static void _convert_block(
    ulong* Xs,
    sd_fft_ctx_struct* Rffts, double* d, ulong dstride,
    ulong np,
    ulong I)
{
    for (ulong l = 0; l < np; l++)
    {
        vec4d p = vec4d_set_d(Rffts[l].p);
        vec4d pinv = vec4d_set_d(Rffts[l].pinv);
        double* x = sd_fft_ctx_blk_index(d + l*dstride, I);
        ulong j = 0; do {
            vec4d x0, x1, x2, x3;
            vec4n y0, y1, y2, y3;
            x0 = vec4d_load(x + j + 0*VEC_SZ);
            x1 = vec4d_load(x + j + 1*VEC_SZ);
            x2 = vec4d_load(x + j + 2*VEC_SZ);
            x3 = vec4d_load(x + j + 3*VEC_SZ);
            x0 = vec4d_reduce_to_0n(x0, p, pinv);
            x1 = vec4d_reduce_to_0n(x1, p, pinv);
            x2 = vec4d_reduce_to_0n(x2, p, pinv);
            x3 = vec4d_reduce_to_0n(x3, p, pinv);
            y0 = vec4d_convert_limited_vec4n(x0);
            y1 = vec4d_convert_limited_vec4n(x1);
            y2 = vec4d_convert_limited_vec4n(x2);
            y3 = vec4d_convert_limited_vec4n(x3);
            vec4n_store_unaligned(Xs + l*BLK_SZ + j + 0*VEC_SZ, y0);
            vec4n_store_unaligned(Xs + l*BLK_SZ + j + 1*VEC_SZ, y1);
            vec4n_store_unaligned(Xs + l*BLK_SZ + j + 2*VEC_SZ, y2);
            vec4n_store_unaligned(Xs + l*BLK_SZ + j + 3*VEC_SZ, y3);
        } while (j += 4*VEC_SZ, j < BLK_SZ);
        FSD_ASSERT(j == BLK_SZ);
    }
}

typedef void (*from_ffts_func)(
    ulong* z, ulong zn, ulong zlen,
    sd_fft_ctx_struct* Rffts, double* d, ulong dstride,
    crt_data_struct* Rcrts,
    ulong bits,
    ulong start_easy, ulong stop_easy,
    ulong* overhang);

/*
    第 l 个素数的 FFT 数据在 d + l*dstride。
    单线程版：overhang 恒为 NULL，处理 [start_easy, zlen) 全部系数。
*/
#define DEFINE_IT(NP, N, M) \
static void CAT(_mpn_from_ffts, NP)( \
    ulong* z, ulong zn, ulong zlen, \
    sd_fft_ctx_struct* Rffts, double* d, ulong dstride, \
    crt_data_struct* Rcrts, \
    ulong bits, \
    ulong start_easy, ulong stop_easy, \
    ulong* FSD_UNUSED(overhang)) \
{ \
    ulong np = NP; \
    ulong n = N; \
    ulong zn_start = start_easy*bits/64; \
    ulong zn_stop  = zn; \
 \
    FSD_ASSERT(n == Rcrts[np-1].coeff_len); \
    FSD_ASSERT(start_easy <= stop_easy); \
 \
    if (n == M + 1) \
    { \
        for (ulong l = 0; l < np; l++) { \
            FSD_ASSERT(crt_data_co_prime(Rcrts + np - 1, l)[M] == 0); \
        } \
    } \
    else \
    { \
        FSD_ASSERT(n == M); \
    } \
 \
    memset(z + zn_start, 0, (zn_stop - zn_start)*sizeof(ulong)); \
 \
    ulong Xs[BLK_SZ*NP]; \
 \
    for (ulong i = start_easy; i < stop_easy; i += BLK_SZ) \
    { \
        _convert_block(Xs, Rffts, d, dstride, np, i/BLK_SZ); \
 \
        for (ulong j = 0; j < BLK_SZ; j += 1) \
        { \
            ulong r[N + 1]; \
            ulong t[N + 1]; \
            ulong l = 0; \
 \
            CAT3(_big_mul, N, M)(r, t, _crt_data_co_prime(Rcrts + np - 1, l, n), Xs[l*BLK_SZ + j]); \
            for (l++; l < np; l++) \
                CAT3(_big_addmul, N, M)(r, t, _crt_data_co_prime(Rcrts + np - 1, l, n), Xs[l*BLK_SZ + j]); \
 \
            CAT(_reduce_big_sum, N)(r, t, crt_data_prod_primes(Rcrts + np - 1)); \
 \
            ulong toff = ((i+j)*bits)/64; \
            ulong tshift = ((i+j)*bits)%64; \
 \
            FSD_ASSERT(zn_stop > n + toff); \
 \
            CAT(_add_to_answer_easy, N)(z, r, zn_stop, toff, tshift); \
        } \
    } \
 \
    for (ulong i = stop_easy; i < zlen; i++) \
    { \
        ulong r[N + 1]; \
        ulong t[N + 1]; \
        ulong l = 0; \
        double xx = (d + l*dstride)[i]; \
        ulong x = (ulong)vec1d_reduce_to_0n(xx, Rffts[l].p, Rffts[l].pinv); \
 \
        CAT3(_big_mul, N, M)(r, t, crt_data_co_prime(Rcrts + np - 1, l), x); \
        for (l++; l < np; l++) \
        { \
            xx = (d + l*dstride)[i]; \
            x = (ulong)vec1d_reduce_to_0n(xx, Rffts[l].p, Rffts[l].pinv); \
            CAT3(_big_addmul, N, M)(r, t, crt_data_co_prime(Rcrts + np - 1, l), x); \
        } \
 \
        CAT(_reduce_big_sum, N)(r, t, crt_data_prod_primes(Rcrts + np - 1)); \
 \
        ulong toff = (i*bits)/64; \
        ulong tshift = (i*bits)%64; \
 \
        if (toff >= zn) \
            break; \
 \
        CAT(_add_to_answer_hard, N)(z, r, zn, toff, tshift); \
    } \
}

DEFINE_IT(4, 4, 3)
DEFINE_IT(5, 4, 4)
DEFINE_IT(6, 5, 4)
DEFINE_IT(7, 6, 5)
DEFINE_IT(8, 7, 6)
#undef DEFINE_IT

/*
    仅供 mpn_ctx_init 使用的辅助函数：给定奇数 p（p-1 的 2 进制赋值很高），
    返回下一个候选 FFT 数 q（不必是素数），且 q-1 也有较高的 2 进制赋值。
*/
static ulong next_fft_number(ulong p) {
    ulong bits, l, q;
    bits = n_nbits(p);
    l = n_trailing_zeros(p - 1);
    q = p - (UWORD(2) << l);
    if (bits < 15)
        fsd_abort("next_fft_number: 输入过小");
    if (n_nbits(q) == bits)
        /* 最好：q-1 与 p-1 位数相同、2 进制赋值相同 */
        return q;
    if (l < 5)
        return n_pow2(bits - 2) + 1;  /* 最差：位数掉 1（ Fermat 型数） */
    /* 次优：保持位数，2 进制赋值降 1（唯一可能 q > p 的分支） */
    return n_pow2(bits) - n_pow2(l - 1) + 1;
}

/*
    填充 2 的幂表：x[i*nvs + l] 的第 k 路为 2^i mod Rffts[4*l+k].p
    （0 <= l < nvs，0 <= i < len）。输入切分的移位乘法用它代替取模。
*/
static void fill_vec_two_pow_tab(
    vec4d* x,
    sd_fft_ctx_struct* Rffts,
    ulong len,
    ulong nvs)
{
    ulong i, l;
    vec4d* ps;

    ps = (vec4d*)fsd_aligned_alloc(32, 2*nvs*sizeof(vec4d));
    for (l = 0; l < nvs; l++)
    {
        /* just p */
        ps[2*l+0] = vec4d_set_d4(Rffts[4*l+0].p,
                                 Rffts[4*l+1].p,
                                 Rffts[4*l+2].p,
                                 Rffts[4*l+3].p);
        /* 2/p */
        ps[2*l+1] = vec4d_set_d4(Rffts[4*l+0].pinv,
                                 Rffts[4*l+1].pinv,
                                 Rffts[4*l+2].pinv,
                                 Rffts[4*l+3].pinv);
        ps[2*l+1] = vec4d_add(ps[2*l+1], ps[2*l+1]);
    }

    for (l = 0; l < nvs; l++)
        x[0*nvs + l] = vec4d_one();

    for (i = 1; i < len; i++)
    for (l = 0; l < nvs; l++)
    {
        vec4d t = x[(i-1)*nvs+l];
        vec4d p = ps[2*l+0];
        vec4d two_over_p = ps[2*l+1];
        vec4d q = vec4d_round(vec4d_mul(t, two_over_p));
        x[i*nvs+l] = vec4d_fnmadd(q, p, vec4d_add(t, t));
    }

    fsd_aligned_free(ps);
}

void mpn_ctx_init(mpn_ctx_t R, ulong p)
{
    ulong i;

    R->buffer = NULL;
    R->buffer_alloc = 0;

    for (i = 0; i < MPN_CTX_NCRTS; i++)
    {
        if (i > 0)
            p = next_fft_number(p);

        while (!n_is_prime(p))
            p = next_fft_number(p);

        /* ffts */
        sd_fft_ctx_init_prime(R->ffts + i, p);

        /* crts */
        if (i == 0)
        {
            crt_data_init(R->crts + 0, p, 1, 1);
            *crt_data_co_prime_red(R->crts + 0, 0) = 1;
            crt_data_co_prime(R->crts + 0, 0)[0] = 1;
            crt_data_prod_primes(R->crts + 0)[0] = p;
        }
        else
        {
            ulong pi;
            ulong len = R->crts[i - 1].coeff_len;
            ulong* t, * tt;

            t = (ulong*)malloc(2*(len + 2)*sizeof(ulong));
            if (t == NULL)
                fsd_abort("内存分配失败");
            tt = t + (len + 2);

            t[len + 1] = 0;
            t[len] = fsd_mpn_mul_1(t, crt_data_prod_primes(R->crts + i - 1), len, p);

            /* leave enough room for (product of primes)*(number of primes) */
            len += 2;
            (void)fsd_mpn_mul_1(tt, t, len, i + 1);
            while (tt[len - 1] == 0)
                len--;

            crt_data_init(R->crts + i, p, len, i + 1);

            /* set product of primes */
            flint_mpn_copyi(crt_data_prod_primes(R->crts + i), t, len);

            /* set cofactors */
            for (pi = 0; pi < i + 1; pi++)
            {
                ulong* cofac = crt_data_co_prime(R->crts + i, pi);
                fsd_mpn_divexact_1(cofac, t, len, R->crts[pi].prime);
                *crt_data_co_prime_red(R->crts + i, pi) =
                                      fsd_mpn_mod_1(cofac, len, R->crts[pi].prime);
            }

            free(t);
        }
    }

    /* powers of two for fast mod */
    {
        ulong len = MPN_CTX_TWO_POWER_TAB_SIZE;
        ulong max_nvs = n_cdiv(MPN_CTX_NCRTS, VEC_SZ);
        vec4d* x = (vec4d*)fsd_aligned_alloc(32,
                                    max_nvs*(max_nvs + 1)/2*len*sizeof(vec4d));
        R->vec_two_pow_buffer = x;
        for (ulong nvs = 1; nvs <= max_nvs; nvs++)
        {
            R->vec_two_pow_tab[nvs - 1] = x;
            fill_vec_two_pow_tab(x, R->ffts, len, nvs);
            x += nvs*len;
        }
    }

    R->profiles_size = 0;

#define PUSH_PROFILE(np_, bits_, n, m) \
    i = R->profiles_size; \
    R->profiles[i].np        = np_; \
    R->profiles[i].bits      = bits_; \
    R->profiles[i].bn_bound  = crt_data_find_bn_bound(R->crts + np_ - 1, bits_); \
    R->profiles[i].to_ffts   = CAT3(mpn_to_ffts, np_, bits_); \
    R->profiles_size = i + 1;

    PUSH_PROFILE(4, 84, 4,3);
    PUSH_PROFILE(4, 88, 4,3);
    PUSH_PROFILE(4, 92, 4,3);
    PUSH_PROFILE(5,112, 4,4);
    PUSH_PROFILE(5,116, 4,4);
    PUSH_PROFILE(5,120, 4,4);
    PUSH_PROFILE(6,136, 5,4);
    PUSH_PROFILE(6,140, 5,4);
    PUSH_PROFILE(6,144, 5,4);
    PUSH_PROFILE(7,160, 6,5);
    PUSH_PROFILE(7,164, 6,5);
    PUSH_PROFILE(7,168, 6,5);
    PUSH_PROFILE(8,184, 7,6);
    PUSH_PROFILE(8,188, 7,6);
    PUSH_PROFILE(8,192, 7,6);

#undef PUSH_PROFILE

    FSD_ASSERT(R->profiles_size <= MAX_NPROFILES);
}

void mpn_ctx_clear(mpn_ctx_t R)
{
    slong i;

    for (i = 0; i < MPN_CTX_NCRTS; i++)
    {
        sd_fft_ctx_clear(R->ffts + i);
        crt_data_clear(R->crts + i);
    }

    fsd_aligned_free(R->vec_two_pow_buffer);

    fsd_aligned_free(R->buffer);
}

/*
    为 an x bn 的乘法挑选尺寸方案，返回选中的 profile 下标。

    profiles 按 np 分组（4..8），组内 bits 递增、bn_bound 递减。
    每组中取「bits 最大且 bn_bound 仍能容纳 bn」的一项作为候选，
    以 score = np * depth * ztrunc * (1 - 0.25*零填充占比) 估计总计算量，
    取 score 最小的候选。（冷路径，每次乘法只调用一次）
*/
static ulong mpn_ctx_best_profile(const mpn_ctx_t R, ulong an, ulong bn)
{
    ulong best_i = 0;
    double best_score = 1e300;

    /* profiles[0] 的 bn_bound 全表最大；连它都装不下说明操作数超限 */
    if (bn > R->profiles[0].bn_bound)
        fsd_abort("操作数过长：超过 fft_small 支持的最大 limb 数（约 2.5e9）");

    for (ulong i = 0; i < R->profiles_size; )
    {
        if (bn > R->profiles[i].bn_bound)
        {
            i++; /* 组内 bound 只减不增，装不下就跳过这一项 */
            continue;
        }

        /* 组内推进到 bits 最大且仍能容纳 bn 的 profile */
        ulong j = i;
        while (j + 1 < R->profiles_size &&
               R->profiles[j + 1].np == R->profiles[i].np &&
               bn <= R->profiles[j + 1].bn_bound)
            j++;

        ulong np = R->profiles[j].np;
        ulong bits = R->profiles[j].bits;
        ulong alen = n_cdiv(64*an, bits);
        ulong blen = n_cdiv(64*bn, bits);
        ulong zlen = alen + blen - 1;
        ulong ztrunc = n_round_up(zlen, BLK_SZ);
        ulong depth = n_max(LG_BLK_SZ, n_clog2(ztrunc));

        double ratio = (double)(ztrunc)/(double)(n_pow2(depth));
        double score = (1-0.25*ratio)*(1.0/1000000);
        score *= np*depth;
        score *= ztrunc;
        if (score < best_score)
        {
            best_i = j;
            best_score = score;
        }

        i = j + 1; /* 跳到下一个 np 组 */
    }

    return best_i;
}

/* 保证 R->buffer 至少 n 字节；按 17/16 比例增长以均摊 realloc（仅本文件使用） */
static void* mpn_ctx_fit_buffer(mpn_ctx_t R, ulong n) {
    if (n > R->buffer_alloc)
    {
        fsd_aligned_free(R->buffer);
        n = n_round_up(n_max(n, R->buffer_alloc*17/16), 4096);
        R->buffer = fsd_aligned_alloc(4096, n);
        if (R->buffer == NULL)
            fsd_abort("内存分配失败");
        R->buffer_alloc = n;
    }
    return R->buffer;
}

/* 点乘：a <- a * b * m，按块（BLK_SZ）用 vec8d 处理（仅本文件使用）。
   m 已吸收 CRT 修正因子与 2^-depth 的逆，使 IFFT 输出直接是卷积系数 */
static void sd_fft_ctx_point_mul(
    const sd_fft_ctx_t Q,
    double* a,
    const double* b,
    ulong m_,
    ulong depth)
{
    vec8d m = vec8d_set_d(vec1d_reduce_0n_to_pmhn((slong)m_, Q->p));
    vec8d n    = vec8d_set_d(Q->p);
    vec8d ninv = vec8d_set_d(Q->pinv);
    FSD_ASSERT(depth >= LG_BLK_SZ);
    for (ulong I = 0; I < n_pow2(depth - LG_BLK_SZ); I++)
    {
        double* ax = a + sd_fft_ctx_blk_offset(I);
        const double* bx = b + sd_fft_ctx_blk_offset(I);
        ulong j = 0; do {
            vec8d x0, x1, b0, b1;
            x0 = vec8d_load(ax+j+0);
            x1 = vec8d_load(ax+j+8);
            b0 = vec8d_load(bx+j+0);
            b1 = vec8d_load(bx+j+8);
            x0 = vec8d_mulmod(x0, m, n, ninv);
            x1 = vec8d_mulmod(x1, m, n, ninv);
            x0 = vec8d_mulmod(x0, b0, n, ninv);
            x1 = vec8d_mulmod(x1, b1, n, ninv);
            vec8d_store(ax+j+0, x0);
            vec8d_store(ax+j+8, x1);
        } while (j += 16, j < BLK_SZ);
    }
}

/* 平方专用的点乘：a <- a^2 * m（省掉一次正变换，仅本文件使用） */
static void sd_fft_ctx_point_sqr(
    const sd_fft_ctx_t Q,
    double* a,
    ulong m_,
    ulong depth)
{
    vec8d m = vec8d_set_d(vec1d_reduce_0n_to_pmhn((slong)m_, Q->p));
    vec8d n    = vec8d_set_d(Q->p);
    vec8d ninv = vec8d_set_d(Q->pinv);
    FSD_ASSERT(depth >= LG_BLK_SZ);

    for (ulong I = 0; I < n_pow2(depth - LG_BLK_SZ); I++)
    {
        double* ax = a + sd_fft_ctx_blk_offset(I);
        ulong j = 0; do {
            vec8d x0, x1;
            x0 = vec8d_load(ax+j+0);
            x1 = vec8d_load(ax+j+8);
            x0 = vec8d_mulmod(x0, x0, n, ninv);
            x1 = vec8d_mulmod(x1, x1, n, ninv);
            x0 = vec8d_mulmod(x0, m, n, ninv);
            x1 = vec8d_mulmod(x1, m, n, ninv);
            vec8d_store(ax+j+0, x0);
            vec8d_store(ax+j+8, x1);
        } while (j += 16, j < BLK_SZ);
    }
}


void mpn_ctx_mpn_mul(mpn_ctx_t R, ulong* z, const ulong* a, ulong an, const ulong* b, ulong bn)
{
    ulong zn, alen, blen, zlen, atrunc, btrunc, ztrunc, depth, stride;
    double* abuf;
    profile_struct P;
    int squaring;

    FSD_ASSERT(an > 0);
    FSD_ASSERT(bn > 0);

    P = R->profiles[mpn_ctx_best_profile(R, an, bn)];

    squaring = (a == b) && (an == bn);
    zn = an + bn;
    alen = n_cdiv(64*an, P.bits);
    blen = n_cdiv(64*bn, P.bits);
    zlen = alen + blen - 1;
    atrunc = n_round_up(alen, BLK_SZ);
    btrunc = n_round_up(blen, BLK_SZ);
    ztrunc = n_round_up(zlen, BLK_SZ);
    depth = n_max(LG_BLK_SZ, n_clog2(ztrunc));
    stride = n_round_up(sd_fft_ctx_data_size(depth), 128);

    FSD_ASSERT(0 <= flint_mpn_cmp_ui_2exp(
                                crt_data_prod_primes(R->crts + P.np - 1),
                                R->crts[P.np - 1].coeff_len, blen, 2*P.bits));

    {
        ulong bits = P.bits;
        /* if i*bits + 32 < 64*an, then the index into a is always in bounds */
        ulong a_stop_easy = n_min(atrunc, (64*an - 33)/bits);
        /* if i*bits >= 64*an, then the index into a is always out of bounds */
        ulong a_stop_hard = n_min(atrunc, (64*an + bits - 1)/bits);
        /* ditto */
        ulong b_stop_easy = n_min(btrunc, (64*bn - 33)/bits);
        ulong b_stop_hard = n_min(btrunc, (64*bn + bits - 1)/bits);
        ulong rounding = (bits%8 == 0) ? 4 : (bits%4 == 0) ? 8 : 16;
        double* bbuf;

        abuf = (double*)mpn_ctx_fit_buffer(R, 2*P.np*stride*sizeof(double));
        bbuf = abuf + P.np*stride;

        /* some fixups for loop unrollings: round down the easy stops */
        FSD_ASSERT(bits%2 == 0);
        a_stop_easy &= -rounding;
        b_stop_easy &= -rounding;

        /* 1. 输入转成各素数上的 double 剩余 */
        P.to_ffts(R->ffts, abuf, stride, a, an, atrunc,
                  R->vec_two_pow_tab[n_cdiv(P.np, VEC_SZ) - 1],
                  0, a_stop_easy, a_stop_easy, a_stop_hard);

        if (!squaring)
        {
            P.to_ffts(R->ffts, bbuf, stride, b, bn, btrunc,
                      R->vec_two_pow_tab[n_cdiv(P.np, VEC_SZ) - 1],
                      0, b_stop_easy, b_stop_easy, b_stop_hard);
        }

        /* 2. 每个素数：正变换 -> 点乘（含 CRT 修正与 2^-depth）-> 逆变换 */
        for (ulong l = 0; l < P.np; l++)
        {
            sd_fft_ctx_struct* Q = R->ffts + l;
            ulong m;

            if (!squaring)
                sd_fft_trunc(Q, bbuf + l*stride, depth, btrunc, ztrunc);

            sd_fft_trunc(Q, abuf + l*stride, depth, atrunc, ztrunc);
            /* m = (red * 2^depth)^-1 mod p：把 CRT 余因子的缩放与
               IFFT 固有的 2^L 因子一并折进点乘 */
            {
                ulong red = *crt_data_co_prime_red(R->crts + P.np - 1, l);
                m = nmod_inv(nmod_red2(red >> (64 - depth), red << depth, Q->mod), Q->mod);
            }

            if (squaring)
                sd_fft_ctx_point_sqr(Q, abuf + l*stride, m, depth);
            else
                sd_fft_ctx_point_mul(Q, abuf + l*stride, bbuf + l*stride, m, depth);

            sd_ifft_trunc(Q, abuf + l*stride, depth, ztrunc);
        }

        /* 3. CRT 重构 + 进位输出 */
        {
            ulong n = R->crts[P.np-1].coeff_len;
            ulong end_easy = (zn >= n+1 ? zn - (n+1) : 0)*64/P.bits;

            FSD_ASSERT(4 <= P.np && P.np <= 8);
            static from_ffts_func tab[8-4+1] = {_mpn_from_ffts_4,
                                                _mpn_from_ffts_5,
                                                _mpn_from_ffts_6,
                                                _mpn_from_ffts_7,
                                                _mpn_from_ffts_8};

            end_easy &= -BLK_SZ;

            tab[P.np - 4](z, zn, zlen, R->ffts, abuf, stride, R->crts, bits, 0, end_easy, NULL);
        }
    }
}
