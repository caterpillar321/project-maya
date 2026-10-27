// tests/glm_mla_decode_test.cu - the decode MLA attention (glm_fast mla_attend / mla_attend_batch): the all-heads
// kernel (mla_set_path 2) and the former head-group kernels (1) against an FP64 reference over the same FP16 latent
// values, a batch against single tokens, and both kernels' times with L2 flushed between calls (needs a GPU).
// glm_mla_decode_test [iters]
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

constexpr int H = 64, KV = 512, NOPE = 256, REC = KV + KV / 16, NCELL = 40000;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "glm_mla_decode_test: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

float h2f(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}
uint16_t f2h(float f) {
    const __half h = __float2half(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

struct Err {
    double abs = 0, rel = 0;
};
// worst |a - ref| and worst per-head |a - ref| / max |ref|
Err compare(const std::vector<float>& a, const std::vector<double>& ref) {
    Err e;
    for (int h = 0; h < H; ++h) {
        double mx = 0, ea = 0;
        for (int c = 0; c < KV; ++c) {
            mx = std::max(mx, std::fabs(ref[(size_t) h * KV + c]));
            ea = std::max(ea, std::fabs(a[(size_t) h * KV + c] - ref[(size_t) h * KV + c]));
        }
        e.abs = std::max(e.abs, ea);
        e.rel = std::max(e.rel, mx > 0 ? ea / mx : ea);
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the latent rows: INT8 records (lat8_store_warp's quantization) and the FP16 values they read as
    std::vector<uint8_t> rec((size_t) REC * NCELL);
    std::vector<uint16_t> f16((size_t) KV * NCELL);
    for (int r = 0; r < NCELL; ++r) {
        float y[KV];
        for (int c = 0; c < KV; ++c) y[c] = nd(rng) * (c % 7 == 0 ? 3.0f : 1.0f);
        uint8_t* p = rec.data() + (size_t) REC * r;
        for (int g = 0; g < KV / 32; ++g) {
            float amax = 0;
            for (int c = 32 * g; c < 32 * g + 32; ++c) amax = std::max(amax, std::fabs(y[c]));
            const uint16_t d16 = f2h(amax / 127.0f);
            const float d = h2f(d16);
            std::memcpy(p + KV + 2 * g, &d16, 2);
            for (int c = 32 * g; c < 32 * g + 32; ++c) {
                const int q = d > 0 ? std::max(-127, std::min(127, (int) std::nearbyint(y[c] / d))) : 0;
                p[c] = (uint8_t) (int8_t) q;
                f16[(size_t) KV * r + c] = f2h((float) q * d);   // the same FP16 the INT8 read gives
            }
        }
    }
    uint16_t *d_rec = nullptr, *d_f16 = nullptr;
    ok(cudaMalloc(&d_rec, rec.size()), "malloc");
    ok(cudaMalloc(&d_f16, f16.size() * 2), "malloc");
    ok(cudaMemcpy(d_rec, rec.data(), rec.size(), cudaMemcpyHostToDevice), "upload");
    ok(cudaMemcpy(d_f16, f16.data(), f16.size() * 2, cudaMemcpyHostToDevice), "upload");
    const int T = 4;
    float *d_q = nullptr, *d_ctx = nullptr;
    int* d_cells = nullptr;
    const int max_sel = 4096;
    ok(cudaMalloc(&d_q, (size_t) T * H * KV * 4), "malloc");
    ok(cudaMalloc(&d_ctx, (size_t) T * H * KV * 4), "malloc");
    ok(cudaMalloc(&d_cells, (size_t) T * max_sel * 4), "malloc");
    void* d_flush = nullptr;
    const size_t flush_bytes = (size_t) 128 << 20;
    ok(cudaMalloc(&d_flush, flush_bytes), "malloc");
    cudaStream_t s;
    ok(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    bool all_ok = true;
    const int sizes[] = {183, 640, 2048, 2175, 4096};
    const float qsig[] = {1.0f, 3.0f};   // score spread: moderate, peaky
    for (const float qs : qsig) {
        for (const int n_sel : sizes) {
            std::vector<float> q((size_t) T * H * KV);
            for (auto& v : q) v = nd(rng) * qs;
            std::vector<int> cells((size_t) T * max_sel, -1);
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < n_sel; ++i) cells[(size_t) t * max_sel + i] = (int) (rng() % NCELL);
                // a tail of unselected cells, as dsa_select leaves past the context
                for (int i = std::max(0, n_sel - 37); i < n_sel; i += 3) cells[(size_t) t * max_sel + i] = -1;
            }
            ok(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "upload");
            ok(cudaMemcpy(d_cells, cells.data(), cells.size() * 4, cudaMemcpyHostToDevice), "upload");
            // the FP64 reference of token 0
            std::vector<double> ref((size_t) H * KV, 0.0);
            const double scale = 1.0 / std::sqrt((double) NOPE);
            for (int h = 0; h < H; ++h) {
                std::vector<double> sc(n_sel, -INFINITY);
                double mx = -INFINITY;
                for (int i = 0; i < n_sel; ++i) {
                    const int cell = cells[i];
                    if (cell < 0) continue;
                    double a = 0;
                    for (int c = 0; c < KV; ++c) a += (double) q[(size_t) h * KV + c] * h2f(f16[(size_t) KV * cell + c]);
                    sc[i] = a * scale;
                    mx = std::max(mx, sc[i]);
                }
                double den = 0;
                for (int i = 0; i < n_sel; ++i)
                    if (cells[i] >= 0) den += std::exp(sc[i] - mx);
                for (int i = 0; i < n_sel; ++i) {
                    const int cell = cells[i];
                    if (cell < 0) continue;
                    const double w = std::exp(sc[i] - mx) / den;
                    for (int c = 0; c < KV; ++c) ref[(size_t) h * KV + c] += w * h2f(f16[(size_t) KV * cell + c]);
                }
            }
            for (int q8 = 1; q8 >= 0; --q8) {
                const uint16_t* lat = q8 ? d_rec : d_f16;
                std::vector<float> out[3];
                double ms[3] = {0, 0, 0};
                for (int path = 1; path <= 2; ++path) {
                    gf::mla_set_path(path);
                    ok(cudaMemsetAsync(d_ctx, 0, (size_t) H * KV * 4, s), "memset");
                    if (!gf::mla_attend(d_q, lat, d_cells, n_sel, H, NOPE, KV, d_ctx, s, q8 != 0)) {
                        std::fprintf(stderr, "mla_attend unavailable\n");
                        return 1;
                    }
                    ok(cudaStreamSynchronize(s), "run");
                    out[path].resize((size_t) H * KV);
                    ok(cudaMemcpy(out[path].data(), d_ctx, out[path].size() * 4, cudaMemcpyDeviceToHost), "read");
                    // time, L2 flushed before each call
                    float tot = 0;
                    for (int it = 0; it < iters; ++it) {
                        if (!getenv("MLA_NOFLUSH")) cudaMemsetAsync(d_flush, it & 255, flush_bytes, s);
                        cudaEventRecord(e0, s);
                        gf::mla_attend(d_q, lat, d_cells, n_sel, H, NOPE, KV, d_ctx, s, q8 != 0);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float m = 0;
                        cudaEventElapsedTime(&m, e0, e1);
                        tot += m;
                    }
                    ms[path] = tot / iters;
                }
                // the batch of T tokens against single calls, the all-heads kernel (bit for bit)
                gf::mla_set_path(2);
                const int ns[T] = {n_sel, n_sel, n_sel, n_sel};
                ok(cudaMemsetAsync(d_ctx, 0, (size_t) T * H * KV * 4, s), "memset");
                gf::mla_attend_batch(d_q, H * KV, lat, d_cells, max_sel, ns, T, H, NOPE, KV, d_ctx, H * KV, s, q8 != 0);
                std::vector<float> bat((size_t) T * H * KV), one((size_t) H * KV);
                ok(cudaMemcpy(bat.data(), d_ctx, bat.size() * 4, cudaMemcpyDeviceToHost), "read");
                size_t bdiff = 0;
                for (int t = 0; t < T; ++t) {
                    gf::mla_attend(d_q + (size_t) t * H * KV, lat, d_cells + (size_t) t * max_sel, n_sel, H, NOPE, KV,
                                   d_ctx, s, q8 != 0);
                    ok(cudaMemcpy(one.data(), d_ctx, one.size() * 4, cudaMemcpyDeviceToHost), "read");
                    bdiff += (size_t) std::memcmp(bat.data() + (size_t) t * H * KV, one.data(), one.size() * 4) != 0;
                }
                const Err eo = compare(out[1], ref), en = compare(out[2], ref);
                double dn = 0;
                for (size_t i = 0; i < out[1].size(); ++i) dn = std::max(dn, (double) std::fabs(out[1][i] - out[2][i]));
                const bool good = en.rel <= std::max(2.0 * eo.rel, 1e-5) && bdiff == 0;
                all_ok = all_ok && good;
                std::printf("q %.0f n_sel %4d %s | vs fp64: former abs %.2e rel %.2e, new abs %.2e rel %.2e | new-former "
                            "%.2e | batch==single %s | former %7.1f us, new %7.1f us (%.2fx)%s\n",
                            qs, n_sel, q8 ? "int8" : "fp16", eo.abs, eo.rel, en.abs, en.rel, dn,
                            bdiff == 0 ? "yes" : "NO", 1000 * ms[1], 1000 * ms[2], ms[1] / ms[2], good ? "" : "  FAIL");
            }
        }
    }
    gf::mla_set_path(0);
    std::printf("glm_mla_decode_test: %s\n", all_ok ? "OK" : "FAIL");
    return all_ok ? 0 : 1;
}
