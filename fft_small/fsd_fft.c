// 截断正变换 sd_fft_trunc（DIF，输出按位翻转序）
//
// 移植自 FLINT src/fft_small/sd_fft.c（LGPL-3.0-or-later）。
// 数据按 256 个 double 一块（BLK_SZ）组织：块内走基 2/4 的小变换
// （basecase），块间走四步分解的递归层（先列后行），配合扭转参数 j。
// 截断（itrunc/otrunc）用于跳过零填充区与不需要的输出，见文末
// sd_fft_trunc 的接口说明。

#include "fsd.h"

#define N 8
#define VECND vec8d
#define VECNOP(op) CAT(VECND, op)

/***************************** 正向蝶形 **************************************
    b0 = a0 + w*a1
    b1 = a0 - w*a1
*/

#define RADIX_2_FORWARD_PARAM_J_IS_Z(V, Q) \
    V n    = CAT(V, set_d)(Q->p); \
    V ninv = CAT(V, set_d)(Q->pinv);

#define RADIX_2_FORWARD_MOTH_J_IS_Z(V, X0, X1) \
{ \
    V x0, x1; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x1 = CAT(V, reduce_to_pm1n)(x1, n, ninv); \
    CAT(V, store)(X0, CAT(V, add)(x0, x1)); \
    CAT(V, store)(X1, CAT(V, sub)(x0, x1)); \
}

#define RADIX_2_FORWARD_PARAM_J_IS_NZ(V, Q, j_r, j_bits) \
    V w = CAT(V, set_d)(Q->w2tab[j_bits][j_r]); \
    V n    = CAT(V, set_d)(Q->p); \
    V ninv = CAT(V, set_d)(Q->pinv);

#define RADIX_2_FORWARD_MOTH_J_IS_NZ(V, X0, X1) \
{ \
    V x0, x1; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x1 = CAT(V, mulmod)(x1, w, n, ninv); \
    CAT(V, store)(X0, CAT(V, add)(x0, x1)); \
    CAT(V, store)(X1, CAT(V, sub)(x0, x1)); \
}

/**************** 带截断的正向蝶形：只写回 X0（otrunc < itrunc 的场合）******/

#define RADIX_2_FORWARD_MOTH_TRUNC_2_1_J_IS_Z(V, X0, X1) \
{ \
    V x0, x1; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x1 = CAT(V, reduce_to_pm1n)(x1, n, ninv); \
    CAT(V, store)(X0, CAT(V, add)(x0, x1)); \
}

#define RADIX_2_FORWARD_MOTH_TRUNC_2_1_J_IS_NZ(V, X0, X1) \
{ \
    V x0, x1; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x1 = CAT(V, mulmod)(x1, w, n, ninv); \
    CAT(V, store)(X0, CAT(V, add)(x0, x1)); \
}

/***************************** 正向基 4 蝶形 **********************************
    b0 = a0 + w^2*a2 +   w*(a1 + w^2*a3)
    b1 = a0 + w^2*a2 -   w*(a1 + w^2*a3)
    b2 = a0 - w^2*a2 + i*w*(a1 - w^2*a3)
    b3 = a0 - w^2*a2 - i*w*(a1 - w^2*a3)

    即两层基 2 蝶形的复合。
*/

#define RADIX_4_FORWARD_PARAM_J_IS_Z(V, Q) \
    V iw = CAT(V, set_d)(Q->w2tab[1][0]); \
    V n    = CAT(V, set_d)(Q->p); \
    V ninv = CAT(V, set_d)(Q->pinv);

#define RADIX_4_FORWARD_MOTH_J_IS_Z(V, X0, X1, X2, X3) \
{ \
    V x0, x1, x2, x3, y0, y1, y2, y3; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x2 = CAT(V, load)(X2); \
    x3 = CAT(V, load)(X3); \
    x2 = CAT(V, reduce_to_pm1n)(x2, n, ninv); \
    x3 = CAT(V, reduce_to_pm1n)(x3, n, ninv); \
    y0 = CAT(V, add)(x0, x2); \
    y1 = CAT(V, add)(x1, x3); \
    y2 = CAT(V, sub)(x0, x2); \
    y3 = CAT(V, sub)(x1, x3); \
    y1 = CAT(V, reduce_to_pm1n)(y1, n, ninv); \
    y3 = CAT(V, mulmod)(y3, iw, n, ninv); \
    x0 = CAT(V, add)(y0, y1); \
    x1 = CAT(V, sub)(y0, y1); \
    x2 = CAT(V, add)(y2, y3); \
    x3 = CAT(V, sub)(y2, y3); \
    CAT(V, store)(X0, x0); \
    CAT(V, store)(X1, x1); \
    CAT(V, store)(X2, x2); \
    CAT(V, store)(X3, x3); \
}

#define RADIX_4_FORWARD_PARAM_J_IS_NZ(V, Q, j_r, j_bits) \
    FSD_ASSERT(j_bits > 0); \
    V w  = CAT(V, set_d)(Q->w2tab[1+j_bits][2*j_r]); \
    V w2 = CAT(V, set_d)(Q->w2tab[0+j_bits][j_r]); \
    V iw = CAT(V, set_d)(Q->w2tab[1+j_bits][2*j_r+1]); \
    V n    = CAT(V, set_d)(Q->p); \
    V ninv = CAT(V, set_d)(Q->pinv);

#define RADIX_4_FORWARD_MOTH_J_IS_NZ(V, X0, X1, X2, X3) \
{ \
    V x0, x1, x2, x3, y0, y1, y2, y3; \
    x0 = CAT(V, load)(X0); \
    x0 = CAT(V, reduce_to_pm1n)(x0, n, ninv); \
    x1 = CAT(V, load)(X1); \
    x2 = CAT(V, load)(X2); \
    x3 = CAT(V, load)(X3); \
    x2 = CAT(V, mulmod)(x2, w2, n, ninv); \
    x3 = CAT(V, mulmod)(x3, w2, n, ninv); \
    y0 = CAT(V, add)(x0, x2); \
    y1 = CAT(V, add)(x1, x3); \
    y2 = CAT(V, sub)(x0, x2); \
    y3 = CAT(V, sub)(x1, x3); \
    y1 = CAT(V, mulmod)(y1, w, n, ninv); \
    y3 = CAT(V, mulmod)(y3, iw, n, ninv); \
    x0 = CAT(V, add)(y0, y1); \
    x1 = CAT(V, sub)(y0, y1); \
    x2 = CAT(V, add)(y2, y3); \
    x3 = CAT(V, sub)(y2, y3); \
    CAT(V, store)(X0, x0); \
    CAT(V, store)(X1, x1); \
    CAT(V, store)(X2, x2); \
    CAT(V, store)(X3, x3); \
}

/* 寄存器内的 4 点小变换（j 任意）：w2 = w^2，w、iw = ±w。
   输入 x0 未规约时先做 reduce_to_pm1n */
#define LENGTH4_ANY_J(T, x0, x1, x2, x3, n, ninv, w2, w, iw) \
{ \
    T X0 = x0, X1 = x1, X2 = x2, X3 = x3, Y0, Y1, Y2, Y3; \
    X0 = T##_##reduce_to_pm1n(X0, n, ninv); \
    X2 = T##_##mulmod(X2, w2, n, ninv); \
    X3 = T##_##mulmod(X3, w2, n, ninv); \
    Y0 = T##_##add(X0, X2); \
    Y1 = T##_##add(X1, X3); \
    Y2 = T##_##sub(X0, X2); \
    Y3 = T##_##sub(X1, X3); \
    Y1 = T##_##mulmod(Y1, w, n, ninv); \
    Y3 = T##_##mulmod(Y3, iw, n, ninv); \
    x0 = T##_##add(Y0, Y1); \
    x1 = T##_##sub(Y0, Y1); \
    x2 = T##_##add(Y2, Y3); \
    x3 = T##_##sub(Y2, Y3); \
}

/* 寄存器内的 4 点小变换（j = 0）：扭转全部是单位根的特殊值，
   省掉两次 mulmod（用 reduce + 一次乘 i 代替） */
#define LENGTH4_ZERO_J(T, x0, x1, x2, x3, n, ninv, e14) \
{ \
    T X0 = x0, X1 = x1, X2 = x2, X3 = x3, Y0, Y1, Y2, Y3; \
    X0 = T##_##reduce_to_pm1n(X0, n, ninv); \
    X2 = T##_##reduce_to_pm1n(X2, n, ninv); \
    X3 = T##_##reduce_to_pm1n(X3, n, ninv); \
    Y0 = T##_##add(X0, X2); \
    Y1 = T##_##add(X1, X3); \
    Y2 = T##_##sub(X0, X2); \
    Y3 = T##_##sub(X1, X3); \
    Y1 = T##_##reduce_to_pm1n(Y1, n, ninv); \
    Y3 = T##_##mulmod(Y3, e14, n, ninv); \
    x0 = T##_##add(Y0, Y1); \
    x1 = T##_##sub(Y0, Y1); \
    x2 = T##_##add(Y2, Y3); \
    x3 = T##_##sub(Y2, Y3); \
}

/* 寄存器内的 8 点小变换（j 任意）：三层结构，先按 w2 分四对，
   再按 w/iw 组合，最后每路乘自己的扭转 ww0..ww3 */
#define LENGTH8_ANY_J(T, x0, x1, x2, x3, x4, x5, x6, x7, n, ninv, w2, w, iw, ww0, ww1, ww2, ww3) \
{ \
    T X0 = x0, X1 = x1, X2 = x2, X3 = x3, X4 = x4, X5 = x5, X6 = x6, X7 = x7; \
    T Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7; \
    X0 = T##_##reduce_to_pm1n(X0, n, ninv); \
    X1 = T##_##reduce_to_pm1n(X1, n, ninv); \
    X4 = T##_##mulmod(X4, w2, n, ninv); \
    X5 = T##_##mulmod(X5, w2, n, ninv); \
    X6 = T##_##mulmod(X6, w2, n, ninv); \
    X7 = T##_##mulmod(X7, w2, n, ninv); \
    Y0 = T##_##add(X0, X4); \
    Y1 = T##_##add(X1, X5); \
    Y2 = T##_##add(X2, X6); \
    Y3 = T##_##add(X3, X7); \
    Y4 = T##_##sub(X0, X4); \
    Y5 = T##_##sub(X1, X5); \
    Y6 = T##_##sub(X2, X6); \
    Y7 = T##_##sub(X3, X7); \
    Y2 = T##_##mulmod(Y2, w, n, ninv); \
    Y3 = T##_##mulmod(Y3, w, n, ninv); \
    Y6 = T##_##mulmod(Y6, iw, n, ninv); \
    Y7 = T##_##mulmod(Y7, iw, n, ninv); \
    Z0 = T##_##add(Y0, Y2); \
    Z1 = T##_##add(Y1, Y3); \
    Z2 = T##_##sub(Y0, Y2); \
    Z3 = T##_##sub(Y1, Y3); \
    Z4 = T##_##add(Y4, Y6); \
    Z5 = T##_##add(Y5, Y7); \
    Z6 = T##_##sub(Y4, Y6); \
    Z7 = T##_##sub(Y5, Y7); \
    Z0 = T##_##reduce_to_pm1n(Z0, n, ninv); \
    Z1 = T##_##mulmod(Z1, ww0, n, ninv); \
    Z2 = T##_##reduce_to_pm1n(Z2, n, ninv); \
    Z3 = T##_##mulmod(Z3, ww1, n, ninv); \
    Z4 = T##_##reduce_to_pm1n(Z4, n, ninv); \
    Z5 = T##_##mulmod(Z5, ww2, n, ninv); \
    Z6 = T##_##reduce_to_pm1n(Z6, n, ninv); \
    Z7 = T##_##mulmod(Z7, ww3, n, ninv); \
    x0 = T##_##add(Z0, Z1); \
    x1 = T##_##sub(Z0, Z1); \
    x2 = T##_##add(Z2, Z3); \
    x3 = T##_##sub(Z2, Z3); \
    x4 = T##_##add(Z4, Z5); \
    x5 = T##_##sub(Z4, Z5); \
    x6 = T##_##add(Z6, Z7); \
    x7 = T##_##sub(Z6, Z7); \
}

/* 寄存器内的 8 点小变换（j = 0）：第一层只有 ±，用 reduce 代替 mulmod */
#define LENGTH8_ZERO_J(T, x0, x1, x2, x3, x4, x5, x6, x7, n, ninv, e14, e18, e38) \
{ \
    T X0 = x0, X1 = x1, X2 = x2, X3 = x3, X4 = x4, X5 = x5, X6 = x6, X7 = x7; \
    T Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7; \
    Y0 = T##_##reduce_to_pm1n(T##_##add(X0, X4), n, ninv); \
    Y1 = T##_##reduce_to_pm1n(T##_##add(X1, X5), n, ninv); \
    Y2 = T##_##reduce_to_pm1n(T##_##add(X2, X6), n, ninv); \
    Y3 = T##_##reduce_to_pm1n(T##_##add(X3, X7), n, ninv); \
    Y4 = T##_##reduce_to_pm1n(T##_##sub(X0, X4), n, ninv); \
    Y5 = T##_##reduce_to_pm1n(T##_##sub(X1, X5), n, ninv); \
    Y6 = T##_##reduce_to_pm1n(T##_##sub(X2, X6), n, ninv); \
    Y7 = T##_##reduce_to_pm1n(T##_##sub(X3, X7), n, ninv); \
    Z0 = T##_##add(Y0, Y2); \
    Z1 = T##_##add(Y1, Y3); \
    Z2 = T##_##sub(Y0, Y2); \
    Z3 = T##_##sub(Y1, Y3); \
    Y6 = T##_##mulmod(e14, Y6, n, ninv); \
    Y7 = T##_##mulmod(e14, Y7, n, ninv); \
    Z4 = T##_##add(Y4, Y6); \
    Z5 = T##_##add(Y5, Y7); \
    Z6 = T##_##sub(Y4, Y6); \
    Z7 = T##_##sub(Y5, Y7); \
    x0 = T##_##add(Z0, Z1); \
    x1 = T##_##sub(Z0, Z1); \
    Z3 = T##_##mulmod(e14, Z3, n, ninv); \
    Z5 = T##_##mulmod(e18, Z5, n, ninv); \
    Z7 = T##_##mulmod(e38, Z7, n, ninv); \
    x2 = T##_##add(Z2, Z3); \
    x3 = T##_##sub(Z2, Z3); \
    x4 = T##_##add(Z4, Z5); \
    x5 = T##_##sub(Z4, Z5); \
    x6 = T##_##add(Z6, Z7); \
    x7 = T##_##sub(Z6, Z7); \
}


/**************** 长度 2^m 的 basecase 变换（块内小变换） ******************/
/* 模板<层数 m, j 是否为 0> sd_fft_basecase(Q, X, j_r, j_bits)

   记号同上：sd_fft_basecase_{m} 就地完成连续 m 层变换，
   各层长度依次为 (2^m, 2^(m-1), ..., 2^1)。j_r/j_bits 描述块的整体扭转 j。
   m <= 3 用标量（vec1d），m = 4、5 用 vec4d（4x4 转置配合），
   m >= 6 由 EXTEND_BASECASE 递归生成。 */

/* 长度 1：空操作 */
static void sd_fft_basecase_0_1(const sd_fft_ctx_t FSD_UNUSED(Q), double* FSD_UNUSED(X)) {
}

static void sd_fft_basecase_1_1(const sd_fft_ctx_t Q, double* X) {
    /* 长度 2：单个基 2 蝶形 */
    double n    = Q->p;
    double ninv = Q->pinv;
    double x0 = vec1d_reduce_to_pm1n(X[0], n, ninv);
    double x1 = vec1d_reduce_to_pm1n(X[1], n, ninv);
    X[0] = vec1d_add(x0, x1);
    X[1] = vec1d_sub(x0, x1);
}


static void sd_fft_basecase_2_1(const sd_fft_ctx_t Q, double* X) {
    LENGTH4_ZERO_J(vec1d, X[0], X[1], X[2], X[3], Q->p, Q->pinv, Q->w2tab[1][0])
}


static void sd_fft_basecase_3_1(const sd_fft_ctx_t Q, double* X) {
    LENGTH8_ZERO_J(vec1d, X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], Q->p, Q->pinv,
                   Q->w2tab[1][0], Q->w2tab[2][0], Q->w2tab[2][1])
}


/* 长度 16（m=4, j=0）：两次 4x4 小变换夹一次转置。
   最后的转置被省略，因此长度 >= 16 的输出序比位翻转序更乱——
   上层调用方都按同样的约定使用，见 m=5 处的说明 */
static void sd_fft_basecase_4_1(const sd_fft_ctx_t Q, double* X) {
    vec4d n    = vec4d_set_d(Q->p);
    vec4d ninv = vec4d_set_d(Q->pinv);
    vec4d w, w2, iw, u, v;

    vec4d x0 = vec4d_load(X+4*0);
    vec4d x1 = vec4d_load(X+4*1);
    vec4d x2 = vec4d_load(X+4*2);
    vec4d x3 = vec4d_load(X+4*3);

    FSD_ASSERT(SD_FFT_CTX_W2TAB_INIT >= 4); /* Q.w2tab[0] points to consecutive entries */
    iw = vec4d_set_d(Q->w2tab[0][1]);
    LENGTH4_ZERO_J(vec4d, x0,x1,x2,x3, n,ninv, iw);

    u = vec4d_load_aligned(&Q->w2tab[0][0]);
    v = vec4d_load_aligned(&Q->w2tab[0][4]);
    w2 = u;
    w  = vec4d_unpack_lo_permute_0_2_1_3(u, v);
    iw = vec4d_unpack_hi_permute_0_2_1_3(u, v);
    VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3);
    LENGTH4_ANY_J(vec4d, x0,x1,x2,x3, n,ninv, w2, w,iw);

    /* VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3) */ /* skipped */
    vec4d_store(X+4*0, x0);
    vec4d_store(X+4*1, x1);
    vec4d_store(X+4*2, x2);
    vec4d_store(X+4*3, x3);
}

static void sd_fft_basecase_4_0(const sd_fft_ctx_t Q, double* X, ulong j_r, ulong j_bits) {
    vec4d n    = vec4d_set_d(Q->p);
    vec4d ninv = vec4d_set_d(Q->pinv);
    vec4d w, w2, iw, u, v;

    vec4d x0 = vec4d_load(X+4*0);
    vec4d x1 = vec4d_load(X+4*1);
    vec4d x2 = vec4d_load(X+4*2);
    vec4d x3 = vec4d_load(X+4*3);

    w2 = vec4d_set_d(Q->w2tab[0+j_bits][1*j_r+0]);
    w  = vec4d_set_d(Q->w2tab[1+j_bits][2*j_r+0]);
    iw = vec4d_set_d(Q->w2tab[1+j_bits][2*j_r+1]);
    LENGTH4_ANY_J(vec4d, x0,x1,x2,x3, n,ninv, w2, w,iw);

    u  = vec4d_load_aligned(&Q->w2tab[3+j_bits][8*j_r+0]);
    v  = vec4d_load_aligned(&Q->w2tab[3+j_bits][8*j_r+4]);
    w2 = vec4d_load_aligned(&Q->w2tab[2+j_bits][4*j_r+0]);
    w  = vec4d_unpack_lo_permute_0_2_1_3(u, v);
    iw = vec4d_unpack_hi_permute_0_2_1_3(u, v);
    VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3);
    LENGTH4_ANY_J(vec4d, x0,x1,x2,x3, n,ninv, w2, w,iw);

    /* VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3) */ /* skipped */
    vec4d_store(X+4*0, x0);
    vec4d_store(X+4*1, x1);
    vec4d_store(X+4*2, x2);
    vec4d_store(X+4*3, x3);
}

/*
长度 32 的变换有两种拆法：
   (a) 列方向 8 个长度 4 的变换 + 行方向 4 个长度 8 的变换，或
   (b) 列方向 4 个长度 8 的变换 + 行方向 8 个长度 4 的变换
长度 16 的 basecase 少做了最后一次 4x4 转置，输出序比位翻转序更乱。
若长度 32 使用的序与 16 不同，上层就难以统一追踪各块到底用了哪种
basecase。因此 16 与 32 必须产生相同的序，而 (b) 更容易做到这一点。
（实现顺序：先 8 点（LENGTH8）再转置再 4 点（LENGTH4），与 (b) 等价。）
*/
static void sd_fft_basecase_5_1(const sd_fft_ctx_t Q, double* X) {
    vec4d n    = vec4d_set_d(Q->p);
    vec4d ninv = vec4d_set_d(Q->pinv);
    vec4d u, v, w0, ww0, ww1, www2, www3;

    vec4d x0 = vec4d_load(X+4*0);
    vec4d x1 = vec4d_load(X+4*1);
    vec4d x2 = vec4d_load(X+4*2);
    vec4d x3 = vec4d_load(X+4*3);
    vec4d x4 = vec4d_load(X+4*4);
    vec4d x5 = vec4d_load(X+4*5);
    vec4d x6 = vec4d_load(X+4*6);
    vec4d x7 = vec4d_load(X+4*7);

    ww1  = vec4d_set_d(Q->w2tab[1][0]);
    www2 = vec4d_set_d(Q->w2tab[2][0]);
    www3 = vec4d_set_d(Q->w2tab[2][1]);
    LENGTH8_ZERO_J(vec4d, x0,x1,x2,x3,x4,x5,x6,x7, n,ninv, ww1, www2,www3);

    VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3);
    VEC4D_TRANSPOSE(x4,x5,x6,x7, x4,x5,x6,x7);

    /* j = 0, 1, 2, 3 */
    w0  = vec4d_set_d4(Q->w2tab[0][0], Q->w2tab[0+(1)][1*(0)+0], Q->w2tab[0+(2)][1*(0)+0], Q->w2tab[0+(2)][1*(1)+0]);
    ww0 = vec4d_set_d4(Q->w2tab[0][0], Q->w2tab[1+(1)][2*(0)+0], Q->w2tab[1+(2)][2*(0)+0], Q->w2tab[1+(2)][2*(1)+0]);
    ww1 = vec4d_set_d4(Q->w2tab[1][0], Q->w2tab[1+(1)][2*(0)+1], Q->w2tab[1+(2)][2*(0)+1], Q->w2tab[1+(2)][2*(1)+1]);
    LENGTH4_ANY_J(vec4d, x0,x1,x2,x3, n,ninv, w0,ww0,ww1);

    /* j = 4, 5, 6, 7 */
    w0 = vec4d_set_d4(Q->w2tab[0+(3)][1*(0)+0], Q->w2tab[0+(3)][1*(1)+0], Q->w2tab[0+(3)][1*(2)+0], Q->w2tab[0+(3)][1*(3)+0]);
    u  = vec4d_load_aligned(&Q->w2tab[1+(3)][2*(0)+0]);
    v  = vec4d_load_aligned(&Q->w2tab[1+(3)][2*(0)+4]);
    ww0 = vec4d_unpack_lo_permute_0_2_1_3(u, v);
    ww1 = vec4d_unpack_hi_permute_0_2_1_3(u, v);
    LENGTH4_ANY_J(vec4d, x4,x5,x6,x7, n,ninv, w0,ww0,ww1);

    /* VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3); */ /* skipped */
    /* VEC4D_TRANSPOSE(x4,x5,x6,x7, x4,x5,x6,x7); */
    vec4d_store(X+4*0, x0);
    vec4d_store(X+4*1, x1);
    vec4d_store(X+4*2, x2);
    vec4d_store(X+4*3, x3);
    vec4d_store(X+4*4, x4);
    vec4d_store(X+4*5, x5);
    vec4d_store(X+4*6, x6);
    vec4d_store(X+4*7, x7);
}

static void sd_fft_basecase_5_0(const sd_fft_ctx_t Q, double* X, ulong j_r, ulong j_bits) {
    vec4d n    = vec4d_set_d(Q->p);
    vec4d ninv = vec4d_set_d(Q->pinv);
    vec4d u, v, w0, ww0, ww1, www0, www1, www2, www3;

    vec4d x0 = vec4d_load(X+4*0);
    vec4d x1 = vec4d_load(X+4*1);
    vec4d x2 = vec4d_load(X+4*2);
    vec4d x3 = vec4d_load(X+4*3);
    vec4d x4 = vec4d_load(X+4*4);
    vec4d x5 = vec4d_load(X+4*5);
    vec4d x6 = vec4d_load(X+4*6);
    vec4d x7 = vec4d_load(X+4*7);

    w0   = vec4d_set_d(Q->w2tab[0+j_bits][1*j_r+0]);
    ww0  = vec4d_set_d(Q->w2tab[1+j_bits][2*j_r+0]);
    ww1  = vec4d_set_d(Q->w2tab[1+j_bits][2*j_r+1]);
    www0 = vec4d_set_d(Q->w2tab[2+j_bits][4*j_r+0]);
    www1 = vec4d_set_d(Q->w2tab[2+j_bits][4*j_r+1]);
    www2 = vec4d_set_d(Q->w2tab[2+j_bits][4*j_r+2]);
    www3 = vec4d_set_d(Q->w2tab[2+j_bits][4*j_r+3]);
    LENGTH8_ANY_J(vec4d, x0,x1,x2,x3,x4,x5,x6,x7, n,ninv, w0, ww0,ww1, www0,www1,www2,www3);

    VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3);
    VEC4D_TRANSPOSE(x4,x5,x6,x7, x4,x5,x6,x7);

    /* j = 8*j+0, 8*j+1, 8*j+2, 8*j+3 */
    w0 = vec4d_load_aligned(&Q->w2tab[0+(3+j_bits)][1*(8*j_r+0)+0]);
    u  = vec4d_load_aligned(&Q->w2tab[1+(3+j_bits)][2*(8*j_r+0)+0]);
    v  = vec4d_load_aligned(&Q->w2tab[1+(3+j_bits)][2*(8*j_r+0)+4]);
    ww0 = vec4d_unpack_lo_permute_0_2_1_3(u, v);
    ww1 = vec4d_unpack_hi_permute_0_2_1_3(u, v);
    LENGTH4_ANY_J(vec4d, x0,x1,x2,x3,n,ninv, w0, ww0, ww1);

    /* j = 8*j+4, 8*j+5, 8*j+6, 8*j+7 */
    w0 = vec4d_load_aligned(&Q->w2tab[0+(3+j_bits)][1*(8*j_r+4)+0]);
    u  = vec4d_load_aligned(&Q->w2tab[1+(3+j_bits)][2*(8*j_r+4)+0]);
    v  = vec4d_load_aligned(&Q->w2tab[1+(3+j_bits)][2*(8*j_r+4)+4]);
    ww0 = vec4d_unpack_lo_permute_0_2_1_3(u, v);
    ww1 = vec4d_unpack_hi_permute_0_2_1_3(u, v);
    LENGTH4_ANY_J(vec4d, x4,x5,x6,x7,n,ninv, w0, ww0, ww1);

    /* VEC4D_TRANSPOSE(x0,x1,x2,x3, x0,x1,x2,x3); */ /* skipped */
    /* VEC4D_TRANSPOSE(x4,x5,x6,x7, x4,x5,x6,x7); */
    vec4d_store(X+4*0, x0);
    vec4d_store(X+4*1, x1);
    vec4d_store(X+4*2, x2);
    vec4d_store(X+4*3, x3);
    vec4d_store(X+4*4, x4);
    vec4d_store(X+4*5, x5);
    vec4d_store(X+4*6, x6);
    vec4d_store(X+4*7, x7);
}


/* 由 n = m-2 层的 basecase 递推 m 层（m >= 6）：
   先做一层基 4 蝶形（把 2^m 分成 4 段 2^(m-2)），
   再对四段分别递归，扭转 j 变为 4j+{0,1,2,3} */
#define EXTEND_BASECASE(n, m) \
static void CAT3(sd_fft_basecase, m, 1)(const sd_fft_ctx_t Q, double* X) \
{ \
    ulong l = n_pow2(m - 2); \
    RADIX_4_FORWARD_PARAM_J_IS_Z(VECND, Q) \
    ulong i = 0; do { \
        RADIX_4_FORWARD_MOTH_J_IS_Z(VECND, X+0*l+i, X+1*l+i, X+2*l+i, X+3*l+i) \
    } while (i += N, i < l); \
    CAT3(sd_fft_basecase, n, 1)(Q, X+0*l); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+1*l, 0, 1); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+2*l, 0, 2); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+3*l, 1, 2); \
} \
static void CAT3(sd_fft_basecase, m, 0)(const sd_fft_ctx_t Q, double* X, ulong j_r, ulong j_bits) \
{ \
    ulong l = n_pow2(m - 2); \
    RADIX_4_FORWARD_PARAM_J_IS_NZ(VECND, Q, j_r, j_bits) \
    ulong i = 0; do { \
        RADIX_4_FORWARD_MOTH_J_IS_NZ(VECND, X+0*l+i, X+1*l+i, X+2*l+i, X+3*l+i) \
    } while (i += N, i < l); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+0*l, 4*j_r+0, j_bits+2); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+1*l, 4*j_r+1, j_bits+2); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+2*l, 4*j_r+2, j_bits+2); \
    CAT3(sd_fft_basecase, n, 0)(Q, X+3*l, 4*j_r+3, j_bits+2); \
}
EXTEND_BASECASE(4, 6)
EXTEND_BASECASE(5, 7)
EXTEND_BASECASE(6, 8)
EXTEND_BASECASE(7, 9)
#undef EXTEND_BASECASE

/* `sd_fft_base_{m}_*` 系列直接接收扭转 j；
   而 `sd_fft_basecase_{m}_*` 接收 j_r 和 j_bits（j 为 0 时无参数） */

/* 后缀 1：允许 j == 0 */
static void sd_fft_base_8_1(const sd_fft_ctx_t Q, double* x, ulong j) {
    ulong j_bits, j_r;

    FSD_ASSERT(8 == LG_BLK_SZ);

    SET_J_BITS_AND_J_R(j_bits, j_r, j);

    if (j == 0)
        sd_fft_basecase_8_1(Q, x);
    else
        sd_fft_basecase_8_0(Q, x, j_r, j_bits);
}

/* 后缀 0：j 必须非零 */
static void sd_fft_base_8_0(const sd_fft_ctx_t Q, double* x, ulong j) {
    ulong j_bits, j_r;

    FSD_ASSERT(j != 0);
    FSD_ASSERT(8 == LG_BLK_SZ);

    SET_J_BITS_AND_J_R(j_bits, j_r, j);

    sd_fft_basecase_8_0(Q, x, j_r, j_bits);
}

static void sd_fft_base_9_1(const sd_fft_ctx_t Q, double* x, ulong j) {
    ulong j_bits, j_r;

    FSD_ASSERT(8 == LG_BLK_SZ);

    SET_J_BITS_AND_J_R(j_bits, j_r, j);

    if (j == 0)
        sd_fft_basecase_9_1(Q, x);
    else
        sd_fft_basecase_9_0(Q, x, j_r, j_bits);
}


/**************** 带截断的基 4 蝶形（块级） ************************************/

/*
    设 `D = X1-X0`（以 double 计），且 D = X2-X1 = X3-X2 >= BLK_SZ。
    `sd_fft_moth_trunc_block_{itrunc}_{otrunc}_{j_is_zero}` 计算一个
    可截断的两层变换，层长 (4*D, 2*D)，掩码 (0..BLK_SZ)。
    只读前 `itrunc` 个输入块（2 <= itrunc <= 4），缺失的块按零处理；
    只写前 `otrunc` 个输出块（1 <= otrunc <= 4）。
*/

/* 第三个参数（模板实参）表示 j 是否为 0：j=0 的版本省掉查表乘法 */
#define DEFINE_IT(itrunc, otrunc) \
static void CAT4(sd_fft_moth_trunc_block, itrunc, otrunc, 1)( \
    const sd_fft_ctx_t Q, \
    ulong FSD_UNUSED(j_r), ulong FSD_UNUSED(j_bits), \
    double* X0, double* X1, double* X2, double* X3) \
{ \
    RADIX_4_FORWARD_PARAM_J_IS_Z(VECND, Q); \
    ulong i = 0; do { \
        VECND x0, x1, x2, x3, y0, y1, y2, y3; \
        x0 = x1 = x2 = x3 = VECNOP(zero)(); \
        if (0 < itrunc) x0 = VECNOP(load)(X0+i); \
        if (0 < itrunc) x0 = VECNOP(reduce_to_pm1n)(x0, n, ninv); \
        if (1 < itrunc) x1 = VECNOP(load)(X1+i); \
        if (2 < itrunc) x2 = VECNOP(load)(X2+i); \
        if (2 < itrunc) x2 = VECNOP(reduce_to_pm1n)(x2, n, ninv); \
        if (3 < itrunc) x3 = VECNOP(load)(X3+i); \
        if (3 < itrunc) x3 = VECNOP(reduce_to_pm1n)(x3, n, ninv); \
        y0 = (2 < itrunc) ? VECNOP(add)(x0, x2) : x0; \
        y1 = (3 < itrunc) ? VECNOP(add)(x1, x3) : x1; \
        y2 = (2 < itrunc) ? VECNOP(sub)(x0, x2) : x0; \
        y3 = (3 < itrunc) ? VECNOP(sub)(x1, x3) : x1; \
        y1 = VECNOP(reduce_to_pm1n)(y1, n, ninv); \
        y3 = VECNOP(mulmod)(y3, iw, n, ninv); \
        x0 = VECNOP(add)(y0, y1); \
        x1 = VECNOP(sub)(y0, y1); \
        x2 = VECNOP(add)(y2, y3); \
        x3 = VECNOP(sub)(y2, y3); \
        if (0 < otrunc) VECNOP(store)(X0+i, x0); \
        if (1 < otrunc) VECNOP(store)(X1+i, x1); \
        if (2 < otrunc) VECNOP(store)(X2+i, x2); \
        if (3 < otrunc) VECNOP(store)(X3+i, x3); \
    } while (i += N, i < BLK_SZ); \
    FSD_ASSERT(i == BLK_SZ); \
} \
static void CAT4(sd_fft_moth_trunc_block, itrunc, otrunc, 0)( \
    const sd_fft_ctx_t Q, \
    ulong j_r, ulong j_bits, \
    double* X0, double* X1, double* X2, double* X3) \
{ \
    RADIX_4_FORWARD_PARAM_J_IS_NZ(VECND, Q, j_r, j_bits); \
    ulong i = 0; do { \
        VECND x0, x1, x2, x3, y0, y1, y2, y3; \
        x0 = x1 = x2 = x3 = VECNOP(zero)(); \
        if (0 < itrunc) x0 = VECNOP(load)(X0+i); \
        if (0 < itrunc) x0 = VECNOP(reduce_to_pm1n)(x0, n, ninv); \
        if (1 < itrunc) x1 = VECNOP(load)(X1+i); \
        if (2 < itrunc) x2 = VECNOP(load)(X2+i); \
        if (2 < itrunc) x2 = VECNOP(mulmod)(x2, w2, n, ninv); \
        if (3 < itrunc) x3 = VECNOP(load)(X3+i); \
        if (3 < itrunc) x3 = VECNOP(mulmod)(x3, w2, n, ninv); \
        y0 = (2 < itrunc) ? VECNOP(add)(x0, x2) : x0; \
        y1 = (3 < itrunc) ? VECNOP(add)(x1, x3) : x1; \
        y2 = (2 < itrunc) ? VECNOP(sub)(x0, x2) : x0; \
        y3 = (3 < itrunc) ? VECNOP(sub)(x1, x3) : x1; \
        y1 = VECNOP(mulmod)(y1, w, n, ninv); \
        y3 = VECNOP(mulmod)(y3, iw, n, ninv); \
        x0 = VECNOP(add)(y0, y1); \
        x1 = VECNOP(sub)(y0, y1); \
        x2 = VECNOP(add)(y2, y3); \
        x3 = VECNOP(sub)(y2, y3); \
        if (0 < otrunc) VECNOP(store)(X0+i, x0); \
        if (1 < otrunc) VECNOP(store)(X1+i, x1); \
        if (2 < otrunc) VECNOP(store)(X2+i, x2); \
        if (3 < otrunc) VECNOP(store)(X3+i, x3); \
    } while (i += N, i < BLK_SZ); \
    FSD_ASSERT(i == BLK_SZ); \
}

DEFINE_IT(2, 1)
DEFINE_IT(2, 2)
DEFINE_IT(2, 3)
DEFINE_IT(2, 4)
DEFINE_IT(3, 1)
DEFINE_IT(3, 2)
DEFINE_IT(3, 3)
DEFINE_IT(3, 4)
DEFINE_IT(4, 1)
DEFINE_IT(4, 2)
DEFINE_IT(4, 3)
DEFINE_IT(4, 4)
#undef DEFINE_IT

/************************ 递归部分 ******************************************/

/*
    计算不截断的 k 层变换，各层长度为
    (BLK_SZ*S*2^k, BLK_SZ*S*2^(k-1), ..., BLK_SZ*S*2)，掩码 (0..BLK_SZ)。
    S 是块间跨距；k > 4 时按 k1 = k/2、k2 = k - k1 四步分解（先列后行），
    落到 k <= 4 后直接展开蝶形循环。
*/
static void sd_fft_no_trunc_block(
    const sd_fft_ctx_t Q,
    double* x,
    ulong S, /* stride */
    ulong k, /* BLK_SZ transforms each of length 2^k */
    ulong j)
{
    ulong j_bits, j_r;

    if (k > 4)
    {
        ulong k1 = k/2;
        ulong k2 = k - k1;

        ulong l2 = n_pow2(k2);
        ulong a = 0; do {
            sd_fft_no_trunc_block(Q, x + BLK_SZ*(a*S), S<<k2, k1, j);
        } while (a++, a < l2);

        /* 行变换 */
        ulong l1 = n_pow2(k1);
        ulong b = 0; do {
            sd_fft_no_trunc_block(Q, x + BLK_SZ*((b<<k2)*S), S, k2, (j<<k1) + b);
        } while (b++, b < l1);

        return;
    }

    SET_J_BITS_AND_J_R(j_bits, j_r, j);

    if (k >= 2)
    {
        ulong k1 = 2;
        ulong k2 = k - k1;
        ulong l2 = n_pow2(k2);

        /* 列变换 */
        if (j_bits == 0)
        {
            RADIX_4_FORWARD_PARAM_J_IS_Z(VECND, Q)
            ulong a = 0; do {
                double* X0 = x + BLK_SZ*(a*S + (S<<k2)*0);
                double* X1 = x + BLK_SZ*(a*S + (S<<k2)*1);
                double* X2 = x + BLK_SZ*(a*S + (S<<k2)*2);
                double* X3 = x + BLK_SZ*(a*S + (S<<k2)*3);
                ulong i = 0; do {
                    RADIX_4_FORWARD_MOTH_J_IS_Z(VECND, X0+i, X1+i, X2+i, X3+i);
                } while (i += N, i < BLK_SZ);
            } while (a++, a < l2);
        }
        else
        {
            RADIX_4_FORWARD_PARAM_J_IS_NZ(VECND, Q, j_r, j_bits)
            ulong a = 0; do {
                double* X0 = x + BLK_SZ*(a*S + (S<<k2)*0);
                double* X1 = x + BLK_SZ*(a*S + (S<<k2)*1);
                double* X2 = x + BLK_SZ*(a*S + (S<<k2)*2);
                double* X3 = x + BLK_SZ*(a*S + (S<<k2)*3);
                ulong i = 0; do {
                    RADIX_4_FORWARD_MOTH_J_IS_NZ(VECND, X0+i, X1+i, X2+i, X3+i);
                } while (i += N, i < BLK_SZ);
            } while (a++, a < l2);
        }

        if (l2 == 1)
            return;

        /* 行变换 */
        ulong l1 = n_pow2(k1);
        ulong b = 0; do {
            sd_fft_no_trunc_block(Q, x + BLK_SZ*((b<<k2)*S), S, k2, (j<<k1) + b);
        } while (b++, b < l1);
    }
    else if (k == 1)
    {
        double* X0 = x + BLK_SZ*(S*0);
        double* X1 = x + BLK_SZ*(S*1);
        if (j_bits == 0)
        {
            RADIX_2_FORWARD_PARAM_J_IS_Z(VECND, Q)
            ulong i = 0; do {
                RADIX_2_FORWARD_MOTH_J_IS_Z(VECND, X0+i, X1+i);
            } while (i += N, i < BLK_SZ);
        }
        else
        {
            RADIX_2_FORWARD_PARAM_J_IS_NZ(VECND, Q, j_r, j_bits)
            ulong i = 0; do {
                RADIX_2_FORWARD_MOTH_J_IS_NZ(VECND, X0+i, X1+i);
            } while (i += N, i < BLK_SZ);
        }
    }
}

/*
    计算不截断的 (LG_BLK_SZ + k) 层连续变换，各层长度为
    (BLK_SZ*2^k, BLK_SZ*2^(k-1), ..., BLK_SZ, BLK_SZ/2, ..., 2)：
    块间层走 sd_fft_no_trunc_block，块内层落到 sd_fft_base_{8,9}。
*/
static void sd_fft_no_trunc_internal(
    const sd_fft_ctx_t Q,
    double* x,
    ulong k,    /* 1 transform of length BLK_SZ*2^k */
    ulong j)    /* twist param */
{
    if (k > 2)
    {
        ulong k1 = k/2;
        ulong k2 = k - k1;

        /* 列变换 */
        ulong l2 = n_pow2(k2);
        ulong a = 0; do {
            sd_fft_no_trunc_block(Q, x + BLK_SZ*a, n_pow2(k2), k1, j);
        } while (a++, a < l2);

        /* 行变换 */
        ulong l1 = n_pow2(k1);
        ulong b = 0; do {
            sd_fft_no_trunc_internal(Q, x + BLK_SZ*(b<<k2), k2, (j<<k1) + b);
        } while (b++, b < l1);

        return;
    }

    if (k == 2)
    {
        /* k1 = 2; k2 = 0：只剩块内层 */
        sd_fft_no_trunc_block(Q, x, 1, 2, j);
        sd_fft_base_8_1(Q, x + BLK_SZ*0, 4*j + 0);
        sd_fft_base_8_0(Q, x + BLK_SZ*1, 4*j + 1);
        sd_fft_base_8_0(Q, x + BLK_SZ*2, 4*j + 2);
        sd_fft_base_8_0(Q, x + BLK_SZ*3, 4*j + 3);
    }
    else if (k == 1)
    {
        sd_fft_base_9_1(Q, x, j);
    }
    else
    {
        /* currently unreachable because all ffts are called with k > 0 */
        sd_fft_base_8_1(Q, x, j);
    }
}


/* sd_fft_no_trunc_block 的截断版：只处理前 itrunc 个输入块
   （其余视为零）、只产出前 otrunc 个输出块。
   k > 2 时同样按 k1/k2 分解，子问题的截断量由位段拆出 */
static void sd_fft_trunc_block(
    const sd_fft_ctx_t Q,
    double* x,
    ulong S,
    ulong k, /* transform length 2^k */
    ulong j,
    ulong itrunc,
    ulong otrunc)
{
    ulong j_bits, j_r;

    FSD_ASSERT(itrunc <= n_pow2(k));
    FSD_ASSERT(otrunc <= n_pow2(k));

    if (otrunc < 1)
        return;

    if (itrunc <= 1)
    {
        if (itrunc < 1)
        {
            for (ulong a = 0; a < otrunc; a++)
            {
                double* X0 = x + BLK_SZ*(S*a);
                VECND z = VECNOP(zero)();
                ulong i = 0; do {
                    VECNOP(store)(X0+i, z);
                } while (i += N, i < BLK_SZ);
            }
        }
        else
        {
            double* X0 = x + BLK_SZ*(S*0);
            for (ulong a = 1; a < otrunc; a++)
            {
                double* X1 = x + BLK_SZ*(S*a);
                ulong i = 0; do {
                    VECND u = VECNOP(load)(X0+i);
                    VECNOP(store)(X1+i, u);
                } while (i += N, i < BLK_SZ);
            }
        }

        return;
    }

    if (itrunc == otrunc && otrunc == n_pow2(k))
    {
        sd_fft_no_trunc_block(Q, x, S, k, j);
        return;
    }

    if (k > 2)
    {
        ulong k1 = k/2;
        ulong k2 = k - k1;

        ulong l2 = n_pow2(k2);
        ulong n1 = otrunc >> k2;
        ulong n2 = otrunc & (l2 - 1);
        ulong z1 = itrunc >> k2;
        ulong z2 = itrunc & (l2 - 1);
        ulong n1p = n1 + (n2 != 0);
        ulong z2p = n_min(l2, itrunc);

        /* 列变换（半截断） */
        for (ulong a = 0; a < z2p; a++)
            sd_fft_trunc_block(Q, x + BLK_SZ*(a*S), S << k2, k1, j, z1 + (a < z2), n1p);

        /* 完整行 */
        for (ulong b = 0; b < n1; b++)
            sd_fft_trunc_block(Q, x + BLK_SZ*(b*(S << k2)), S, k2, (j << k1) + b, z2p, l2);

        /* 最后一段不完整的行 */
        if (n2 > 0)
            sd_fft_trunc_block(Q, x + BLK_SZ*(n1*(S << k2)), S, k2, (j << k1) + n1, z2p, n2);

        return;
    }

    SET_J_BITS_AND_J_R(j_bits, j_r, j);

    if (k == 2)
    {
#define IT(ii, oo) CAT4(sd_fft_moth_trunc_block, ii, oo, 0), \
                   CAT4(sd_fft_moth_trunc_block, ii, oo, 1)
#define LOOKUP_IT(ii, oo, j_is_zero) tab[(j_is_zero) + 2*((oo)-1 + 4*((ii)-2))]
        static void (*tab[3*4*2])(const sd_fft_ctx_t, ulong, ulong, double*, double*, double*, double*) =
                        {IT(2,1), IT(2,2), IT(2,3), IT(2,4),
                         IT(3,1), IT(3,2), IT(3,3), IT(3,4),
                         IT(4,1), IT(4,2), IT(4,3), IT(4,4)};

        double* X0 = x + BLK_SZ*(S*0);
        double* X1 = x + BLK_SZ*(S*1);
        double* X2 = x + BLK_SZ*(S*2);
        double* X3 = x + BLK_SZ*(S*3);
        LOOKUP_IT(itrunc, otrunc, j == 0)(Q, j_r, j_bits, X0, X1, X2, X3);
#undef LOOKUP_IT
#undef IT
    }
    else if (k == 1)
    {
        double* X0 = x + BLK_SZ*(S*0);
        double* X1 = x + BLK_SZ*(S*1);
        FSD_ASSERT(itrunc == 2);
        FSD_ASSERT(otrunc == 1);
        if (j_bits == 0)
        {
            RADIX_2_FORWARD_PARAM_J_IS_Z(VECND, Q)
            ulong i = 0; do {
                RADIX_2_FORWARD_MOTH_TRUNC_2_1_J_IS_Z(VECND, X0 + i, X1 + i);
            } while (i += N, i < BLK_SZ);
        }
        else
        {
            RADIX_2_FORWARD_PARAM_J_IS_NZ(VECND, Q, j_r, j_bits)
            ulong i = 0; do {
                RADIX_2_FORWARD_MOTH_TRUNC_2_1_J_IS_NZ(VECND, X0 + i, X1 + i);
            } while (i += N, i < BLK_SZ);
        }
    }
}


/* sd_fft_no_trunc_internal 的截断版（截断单位为块） */
static void sd_fft_trunc_internal(
    const sd_fft_ctx_t Q,
    double* x,      /* x = data + BLK_SZ*I  where I = starting index */
    ulong k,        /* transform length BLK_SZ*2^k */
    ulong j,
    ulong itrunc,   /* actual trunc is BLK_SZ*itrunc */
    ulong otrunc)   /* actual trunc is BLK_SZ*otrunc */
{
    if (otrunc < 1)
        return;

    if (itrunc < 1)
    {
        for (ulong a = 0; a < otrunc; a++)
        {
            double* X0 = x + BLK_SZ*a;
            VECND z = VECNOP(zero)();
            ulong i = 0; do {
                VECNOP(store)(X0 + i, z);
            } while (i += N, i < BLK_SZ);
        }

        return;
    }

    if (itrunc == otrunc && otrunc == n_pow2(k))
    {
        sd_fft_no_trunc_internal(Q, x, k, j);
        return;
    }

    if (k > 2)
    {
        ulong k1 = k/2;
        ulong k2 = k - k1;

        ulong l2 = n_pow2(k2);
        ulong n1 = otrunc >> k2;
        ulong n2 = otrunc & (l2 - 1);
        ulong z1 = itrunc >> k2;
        ulong z2 = itrunc & (l2 - 1);
        ulong n1p = n1 + (n2 != 0);
        ulong z2p = n_min(l2, itrunc);

        /* 列变换（半截断） */
        for (ulong a = 0; a < z2p; a++)
            sd_fft_trunc_block(Q, x + BLK_SZ*a, n_pow2(k2), k1, j, z1 + (a < z2), n1p);

        /* 完整行 */
        for (ulong b = 0; b < n1; b++)
            sd_fft_trunc_internal(Q, x + BLK_SZ*(b << k2), k2, (j << k1) + b, z2p, l2);

        /* 最后一段不完整的行 */
        if (n2 > 0)
            sd_fft_trunc_internal(Q, x + BLK_SZ*(n1 << k2), k2, (j << k1) + n1, z2p, n2);

        return;
    }

    if (k == 2)
    {
        sd_fft_trunc_block(Q, x, 1, 2, j, itrunc, otrunc);
                        sd_fft_base_8_1(Q, x + BLK_SZ*0, 4*j + 0);
        if (otrunc > 1) sd_fft_base_8_0(Q, x + BLK_SZ*1, 4*j + 1);
        if (otrunc > 2) sd_fft_base_8_0(Q, x + BLK_SZ*2, 4*j + 2);
        if (otrunc > 3) sd_fft_base_8_0(Q, x + BLK_SZ*3, 4*j + 3);
    }
    else if (k == 1)
    {
        sd_fft_trunc_block(Q, x, 1, 1, j, itrunc, otrunc);
                        sd_fft_base_8_1(Q, x + BLK_SZ*0, 2*j + 0);
        if (otrunc > 1) sd_fft_base_8_0(Q, x + BLK_SZ*1, 2*j + 1);
    }
    else
    {
        /* currently unreachable for the same reason */
        sd_fft_base_8_1(Q, x, j);
    }
}

/********************* 对外接口 *********************************************/

/*
    就地计算 d 的截断 FFT，假设第 itrunc 项之后的输入全为零。

    输出满足（输出按位翻转序排布）
        eval_poly(in_data, sd_fft_ctx_w(Q, i)) = out_data[revbin(i, L)]
    对所有 0 <= i < otrunc 成立（该不变式由 fsd_test.c 校验）。
    通常只在 otrunc >= itrunc 且 max(itrunc, otrunc) >= 2^(L-1) 时有意义。
    d 的长度必须至少为 2^L。
*/
void sd_fft_trunc(
    sd_fft_ctx_t Q,
    double* d,
    ulong L,    /* 卷积长度 2^L */
    ulong itrunc, ulong otrunc)
{
    FSD_ASSERT(itrunc <= n_pow2(L));
    FSD_ASSERT(otrunc <= n_pow2(L));

    sd_fft_ctx_fit_depth(Q, L);

    if (L > LG_BLK_SZ)
    {
        ulong new_itrunc, new_otrunc;

        /* 截断量换算成块数（向上取整），块间走递归层 */
        new_itrunc = n_cdiv(itrunc, BLK_SZ);
        new_otrunc = n_cdiv(otrunc, BLK_SZ);
        /* 把 itrunc 到块边界之间的尾巴清零 */
        for (int i = 0; i < (int)((-(ulong)itrunc)&(BLK_SZ-1)); i++)
            d[itrunc+i] = 0.0;

        sd_fft_trunc_internal(Q, d, L - LG_BLK_SZ, 0, new_itrunc, new_otrunc);
        return;
    }

    /* L <= LG_BLK_SZ：块内直接清零 + basecase */
    for (ulong i = itrunc; i < (1<<L); i++)
        d[i] = 0;

    /* L=8 会读到 w2tab[7]，初始化已保证连续建到第 9 张表 */
    FSD_ASSERT(LG_BLK_SZ <= SD_FFT_CTX_W2TAB_INIT);

    switch (L) {
        case 0: sd_fft_basecase_0_1(Q, d); break;
        case 1: sd_fft_basecase_1_1(Q, d); break;
        case 2: sd_fft_basecase_2_1(Q, d); break;
        case 3: sd_fft_basecase_3_1(Q, d); break;
        case 4: sd_fft_basecase_4_1(Q, d); break;
        case 5: sd_fft_basecase_5_1(Q, d); break;
        case 6: sd_fft_basecase_6_1(Q, d); break;
        case 7: sd_fft_basecase_7_1(Q, d); break;
        case 8: sd_fft_basecase_8_1(Q, d); break;
        default: FSD_ASSERT(0);
    }
}
