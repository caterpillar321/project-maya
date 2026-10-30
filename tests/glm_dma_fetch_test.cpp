// Explicit GPU integration check for the actual host submission lane, not a replacement copy implementation.
#include "../src/core/glm_fast_state.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace gf = strata::kernels::glmf;
static void require(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "DMA TEST FAIL: %s\n", msg); std::exit(1); }
}
static void check(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    check(cudaSetDevice(0));
    constexpr size_t blob = 1 << 20, total = 8 * blob;
    unsigned char *host = nullptr, *mapped = nullptr, *dst = nullptr;
    check(cudaHostAlloc((void**) &host, total, cudaHostAllocMapped));
    check(cudaHostGetDevicePointer((void**) &mapped, host, 0));
    check(cudaMalloc((void**) &dst, total));
    for (size_t i = 0; i < total; ++i) host[i] = (unsigned char) ((i * 13 + i / blob * 29) % 251);
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    gf::MoeDev dev{};
    check(cudaMalloc((void**) &dev.fetch_src, 8 * sizeof(unsigned long long)));
    check(cudaMalloc((void**) &dev.plan_ptr, 8 * sizeof(unsigned long long)));
    int *error_h = nullptr, *error_d = nullptr;
    check(cudaHostAlloc((void**) &error_h, sizeof(int), cudaHostAllocMapped));
    check(cudaHostGetDevicePointer((void**) &error_d, error_h, 0));
    *error_h = 0; dev.route_error = error_d;
    {
        strata::core::glmfast::WindowDmaFetch dma;
        std::string err;
        const bool ready = dma.init(0, err);
        require(ready, err.c_str());
        std::vector<unsigned char> result(total);
        int cases = 0;
        for (size_t n : {size_t(16), blob - 16, blob}) {
            for (unsigned mask : {0u, 1u, 128u, 85u, 255u, 3u, 0u}) {
                unsigned long long srcs[8], dsts[8];
                for (int i = 0; i < 8; ++i) {
                    srcs[i] = (mask >> i & 1) ? (unsigned long long) (mapped + i * blob) : 0ull;
                    dsts[i] = (unsigned long long) (dst + i * blob);
                }
                check(cudaMemsetAsync(dst, 0xfd, total, s));
                check(cudaMemcpyAsync(dev.fetch_src, srcs, sizeof(srcs), cudaMemcpyHostToDevice, s));
                check(cudaMemcpyAsync(dev.plan_ptr, dsts, sizeof(dsts), cudaMemcpyHostToDevice, s));
                if (cases == 10) {
                    const unsigned v[2] = {std::numeric_limits<unsigned>::max(), std::numeric_limits<unsigned>::max()};
                    check(cudaMemcpyAsync(dma.state, v, sizeof(v), cudaMemcpyHostToDevice, s));
                    check(cudaStreamSynchronize(s));
                }
                gf::moe_dma_submit(dev, 8, n, dma.device, dma.state, s);
                gf::moe_dma_wait(dev, dma.device, dma.state, s);
                check(cudaStreamSynchronize(s));
                require(*error_h == 0, "copy completion error");
                check(cudaMemcpy(result.data(), dst, total, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < total; ++i) {
                    const bool copied = (mask >> (i / blob) & 1) && i % blob < n;
                    require(result[i] == (copied ? host[i] : 0xfd), "copy contents / untouched bytes");
                }
                ++cases;
            }
        }
        const bool drained = dma.drain(err);
        require(drained, err.c_str());
        // A host-side failure must reject the forward, not silently consume stale expert weights or hang.
        dma.host->error = 1;
        gf::moe_dma_submit(dev, 8, blob, dma.device, dma.state, s);
        gf::moe_dma_wait(dev, dma.device, dma.state, s);
        check(cudaStreamSynchronize(s));
        require(*error_h == -101, "host error propagation");
        require(!dma.drain(err), "failed lane cannot be reused as healthy");
        std::printf("glm_dma_fetch_test: PASS (%d copies, empty/mixed/all masks, sequence wrap, untouched bytes, host failure)\n", cases);
    }
    check(cudaStreamDestroy(s));
    check(cudaFree(dev.fetch_src)); check(cudaFree(dev.plan_ptr)); check(cudaFree(dst));
    check(cudaFreeHost(error_h)); check(cudaFreeHost(host));
}
