// src/kernels/cuda/dsa_topk.cuh - the DSA indexer's choice of the top pools in O(n_vis) (one block, 1024 threads).
//
// The order it writes is the one the O(n_vis^2) rank loop wrote (score descending, ties by the lower pool index), so
// the selected cells - and every result built on them - are unchanged; only the cost stops growing with the square of
// the context (at 30k tokens the old loop was ~56M comparisons per DSA layer per token, in one block).
//   1. a radix select (4 passes of 8 bits) over the scores' order-preserving uint32 keys finds the cut-off key - the
//      top-th largest - and how many of the keys equal to it are taken;
//   2. one pass in pool order (each thread a contiguous run, exclusive scans for the offsets) collects the winners:
//      every key above the cut-off and the first `need` equal to it - the lower indices, as the rank loop chose;
//   3. a bitonic sort of the <= 1024 winners, packed as (key << 32) | ~index, descending.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels::dsa {

// a float's order-preserving key: larger float, larger key (-0 is +0; the scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis) in rank order; 1 <= top <= 1024,
// top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that includes
// this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[NT], s_eq[NT];
    __shared__ unsigned long long s_win[NT];
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
    }
    __syncthreads();
    // 1. the cut-off key, a byte at a time from the top
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_vis; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need;
            for (int b = 255; b >= 0; --b) {
                const int c = (int) s_hist[b];
                if (c >= need) {
                    s_prefix = prefix | ((unsigned) b << shift);
                    s_mask = mask | (255u << shift);
                    s_need = need;
                    break;
                }
                need -= c;
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    // 2. the winners, in pool order
    const int chunk = (n_vis + NT - 1) / NT;
    const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    s_gt[tid] = gt;
    s_eq[tid] = eq;
    __syncthreads();
    for (int off = 1; off < NT; off <<= 1) {   // inclusive scans of both counts
        int a = 0, b = 0;
        if (tid >= off) {
            a = s_gt[tid - off];
            b = s_eq[tid - off];
        }
        __syncthreads();
        s_gt[tid] += a;
        s_eq[tid] += b;
        __syncthreads();
    }
    const int n_gt = s_gt[NT - 1];   // == top - need_eq
    int og = s_gt[tid] - gt, oe = s_eq[tid] - eq;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) {
            s_win[og++] = w;
        } else if (k == cut) {
            if (oe < need_eq) s_win[n_gt + oe] = w;
            ++oe;
        }
    }
    // 3. descending bitonic sort, padded with the smallest value
    int n2 = 1;
    while (n2 < top) n2 <<= 1;
    for (int i = top + tid; i < n2; i += NT) s_win[i] = 0ull;
    __syncthreads();
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n2; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const unsigned long long a = s_win[i], b = s_win[j];
                    if ((i & size) == 0 ? a < b : a > b) {
                        s_win[i] = b;
                        s_win[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int r = tid; r < top; r += NT) {
        const int p = (int) ~(unsigned) (s_win[r] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
}

// select_top_pools2 (10-26): select_top_pools' result, bit for bit, with less serial work in its one block:
// the histograms warp-aggregated (most scores share their top byte: one bin took every atomic of the first pass),
// the cut-off bin found by one warp's suffix scan (was thread 0's loop over 256 bins, 4 times), and the winners'
// offsets by warp scans (was 10 block-wide steps of 2 syncs).  The bitonic sort is select_top_pools'.  s_keys (shared
// memory of n_vis words, or nullptr): the keys are read from memory once, in one unrolled burst, and every pass reads
// them there - the passes' dependent global loads (7 a thread at 6.7K pools, 5 passes) were most of the time.
static __device__ void select_top_pools2(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells,
                                         unsigned* s_keys) {
#if defined(STRATA_USE_HIP)
    select_top_pools(sc, n_vis, top, kpool, cells);
#else
    constexpr int NT = 1024, NW = NT / 32;
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_wg[NW], s_we[NW];
    __shared__ unsigned long long s_win[NT];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
    }
    if (s_keys != nullptr) {
#pragma unroll 8
        for (int i = tid; i < n_vis; i += NT) s_keys[i] = score_key(sc[i]);
    }
    __syncthreads();
    const auto key_at = [&](int i) { return s_keys != nullptr ? s_keys[i] : score_key(sc[i]); };
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i0 = 0; i0 < n_vis; i0 += NT) {
            const int i = i0 + tid;
            const unsigned k = i < n_vis ? key_at(i) : 0u;
            const bool in = i < n_vis && (k & mask) == prefix;
            const unsigned act = __ballot_sync(0xffffffffu, in);
            if (in) {
                const unsigned b = (k >> shift) & 255u;
                const unsigned peers = __match_any_sync(act, b);
                if (lane == __ffs(peers) - 1) atomicAdd(&s_hist[b], (unsigned) __popc(peers));
            }
        }
        __syncthreads();
        if (warp == 0) {
            // lane l: bins [8 l, 8 l + 8); bins above lane l's: the suffix sum over the lanes past it
            unsigned c[8], sl = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sl += (c[j] = s_hist[8 * lane + j]);
            unsigned suf = sl;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned t = __shfl_down_sync(0xffffffffu, suf, o);
                if (lane + o < 32) suf += t;
            }
            const int need = s_need;
            int cum = (int) (suf - sl);
#pragma unroll
            for (int j = 7; j >= 0; --j) {
                if (cum < need && cum + (int) c[j] >= need) {
                    s_prefix = prefix | ((unsigned) (8 * lane + j) << shift);
                    s_mask = mask | (255u << shift);
                    s_need = need - cum;
                }
                cum += (int) c[j];
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    const int chunk = (n_vis + NT - 1) / NT;
    const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = key_at(i);
        gt += k > cut;
        eq += k == cut;
    }
    // exclusive offsets in thread order: a warp scan, then the warps' totals scanned by warp 0
    int ig = gt, ie = eq;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, ig, o), b = __shfl_up_sync(0xffffffffu, ie, o);
        if (lane >= o) ig += a, ie += b;
    }
    if (lane == 31) {
        s_wg[warp] = ig;
        s_we[warp] = ie;
    }
    __syncthreads();
    if (warp == 0) {
        int a = s_wg[lane], b = s_we[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, a, o), y = __shfl_up_sync(0xffffffffu, b, o);
            if (lane >= o) a += x, b += y;
        }
        s_wg[lane] = a;   // inclusive over the warps
        s_we[lane] = b;
    }
    __syncthreads();
    const int n_gt = s_wg[NW - 1];
    int og = (warp > 0 ? s_wg[warp - 1] : 0) + ig - gt, oe = (warp > 0 ? s_we[warp - 1] : 0) + ie - eq;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = key_at(i);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) {
            s_win[og++] = w;
        } else if (k == cut) {
            if (oe < need_eq) s_win[n_gt + oe] = w;
            ++oe;
        }
    }
    int n2 = 1;
    while (n2 < top) n2 <<= 1;
    for (int i = top + tid; i < n2; i += NT) s_win[i] = 0ull;
    __syncthreads();
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n2; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const unsigned long long a = s_win[i], b = s_win[j];
                    if ((i & size) == 0 ? a < b : a > b) {
                        s_win[i] = b;
                        s_win[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int r = tid; r < top; r += NT) {
        const int p = (int) ~(unsigned) (s_win[r] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
#endif
}

}  // namespace strata::kernels::dsa
