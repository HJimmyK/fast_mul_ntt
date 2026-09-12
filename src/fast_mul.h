// MIT License
//
// Copyright (c) 2025 Jecricho Knox
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// fast_mul.h —— 对外接口：基于三模数 NTT + CRT 的大整数乘法 / 平方

#ifndef FAST_MUL_H
#define FAST_MUL_H

#include "macro.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 大整数乘法：out = in1 * in2
 * in1、in2、out 均为小端 64 位 limb 数组，out 长度必须为 len1 + len2
 * 注意：in1 与 in2 必须指向不同的内存块（值可以相同）
 */
void abs_mul64(const u64* in1, u64 len1, const u64* in2, u64 len2, u64* out);

/*
 * 大整数平方：out = in1 * in1
 * out 长度必须为 2 * len1
 */
void abs_sqr64(const u64* in1, u64 len1, u64* out);

#ifdef __cplusplus
}
#endif

#endif // FAST_MUL_H
