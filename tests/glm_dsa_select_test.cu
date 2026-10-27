// tests/glm_dsa_select_test.cu - the decode indexer (glm_fast dsa_score / dsa_select): the faster kernels
// (dsa_set_path 2) against the former (1), scores and selected cells bit for bit, over random and tie-heavy scores,
// and both kernels' times (needs a GPU).  glm_dsa_select_test [iters]
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

static void ok(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_dsa_select_test: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int K = 128, NH = 32, KPOOL = 4, TOP = 512, NSEL = TOP * KPOOL + KPOOL - 1, MAXP = 65536;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> iq(NH * K), iw(NH), pooled((size_t) MAXP * K);
    for (auto& v : iq) v = nd(rng);
    for (auto& v : iw) v = nd(rng) * 0.1f;
    for (auto& v : pooled) v = nd(rng);
    float *d_iq, *d_iw, *d_pool, *d_s1, *d_s2;
    int *d_c1, *d_c2;
    ok(cudaMalloc(&d_iq, iq.size() * 4), "malloc");
    ok(cudaMalloc(&d_iw, iw.size() * 4), "malloc");
    ok(cudaMalloc(&d_pool, pooled.size() * 4), "malloc");
    ok(cudaMalloc(&d_s1, MAXP * 4), "malloc");
    ok(cudaMalloc(&d_s2, MAXP * 4), "malloc");
    ok(cudaMalloc(&d_c1, NSEL * 4), "malloc");
    ok(cudaMalloc(&d_c2, NSEL * 4), "malloc");
    ok(cudaMemcpy(d_iq, iq.data(), iq.size() * 4, cudaMemcpyHostToDevice), "up");
    ok(cudaMemcpy(d_iw, iw.data(), iw.size() * 4, cudaMemcpyHostToDevice), "up");
    ok(cudaMemcpy(d_pool, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "up");
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    bool all = true;
    auto timed = [&](auto f) {
        f();
        cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return 1000.0 * ms / iters;
    };
    for (const int np : {183 / 4, 600, 2048, 6730, 16384, 65536}) {
        const int nvis = np, pos = nvis * KPOOL + 1;
        // scores: old vs new, bits
        double t_s[3], t_c[3];
        for (int path = 1; path <= 2; ++path) {
            gf::dsa_set_path(path);
            float* dst = path == 1 ? d_s1 : d_s2;
            t_s[path] = timed([&] { gf::dsa_score(d_iq, d_pool, d_iw, K, NH, np, dst, s); });
        }
        std::vector<float> s1(np), s2(np);
        ok(cudaMemcpy(s1.data(), d_s1, np * 4, cudaMemcpyDeviceToHost), "down");
        ok(cudaMemcpy(s2.data(), d_s2, np * 4, cudaMemcpyDeviceToHost), "down");
        const bool score_same = std::memcmp(s1.data(), s2.data(), np * 4) == 0;
        // selections over: the real scores, scores rounded to few values (ties), zeros/negatives mixed
        bool sel_same = true;
        for (int variant = 0; variant < 3; ++variant) {
            std::vector<float> sc = s1;
            if (variant == 1)
                for (auto& v : sc) v = std::round(v * 4.0f) / 4.0f;
            if (variant == 2)
                for (size_t i = 0; i < sc.size(); ++i) sc[i] = (i % 3 == 0) ? 0.0f : (i % 3 == 1 ? -0.0f : -std::fabs(sc[i]));
            ok(cudaMemcpy(d_s1, sc.data(), np * 4, cudaMemcpyHostToDevice), "up");
            for (int path = 1; path <= 2; ++path) {
                gf::dsa_set_path(path);
                int* dst = path == 1 ? d_c1 : d_c2;
                const double t = timed([&] { gf::dsa_select(d_s1, nvis, KPOOL, TOP, NSEL, pos, dst, s); });
                if (variant == 0) t_c[path] = t;
            }
            std::vector<int> c1(NSEL), c2(NSEL);
            ok(cudaMemcpy(c1.data(), d_c1, NSEL * 4, cudaMemcpyDeviceToHost), "down");
            ok(cudaMemcpy(c2.data(), d_c2, NSEL * 4, cudaMemcpyDeviceToHost), "down");
            sel_same = sel_same && std::memcmp(c1.data(), c2.data(), NSEL * 4) == 0;
        }
        all = all && score_same && sel_same;
        std::printf("pools %6d | score former %6.1f us, new %6.1f us, bits %s | select former %6.1f us, new %6.1f us, "
                    "cells (real/ties/zeros) %s\n", np, t_s[1], t_s[2], score_same ? "same" : "DIFFER", t_c[1], t_c[2],
                    sel_same ? "same" : "DIFFER");
    }
    gf::dsa_set_path(0);
    std::printf("glm_dsa_select_test: %s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
