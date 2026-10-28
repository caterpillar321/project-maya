// Private T2/T3 same-layer weight reuse: real route/wait/fetch kernels, admission parity, and grouped arithmetic.
// No model is needed. Runs only on the GPU selected by CUDA_VISIBLE_DEVICES; no benchmark or sustained load.
#include "strata/kernels/glm_fast.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace {
constexpr int NE = 32, E = 512, FF = 256, K = 8, NL = 2;
const int ids[3][K] = {{0,1,2,3,4,5,6,7}, {4,5,6,7,8,9,10,11}, {0,4,8,12,13,14,15,16}};
int failures = 0;
#define CK(expr) do { auto status = (expr); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(status)); \
    std::exit(2); } } while (0)
void expect(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
template<class V> std::vector<V> download(const V* p, size_t n) {
    std::vector<V> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(V), cudaMemcpyDeviceToHost)); return v;
}
template<class V> void upload(V* p, const std::vector<V>& v) {
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice));
}
struct Settings { bool reuse = false, ram = true, helper = false; int spares = 0, promote_min = 0; unsigned long long cpu = 0; };
struct Result { std::vector<gf::MoeRequest> requests; std::vector<uint8_t> bytes; std::vector<float> weights; int groups = 0; };

struct Fixture {
    const int T;
    size_t blob;
    gf::MoeDev base;
    std::vector<void*> allocations;
    uint8_t* host = nullptr;
    uint8_t* host_d = nullptr;
    gf::MoeRequest* ring = nullptr;
    gf::MoeResponse* response = nullptr;
    uint8_t* scratch;
    uint8_t* spare;
    unsigned long long* slots;
    unsigned long long* plans;
    int* plan_ids;
    float* plan_weights;
    unsigned long long* fetch;
    float* x;
    float* logits;
    uint8_t* owned;
    int* skip;
    int layer = 0;
    Settings setting;
    template<class V> V* alloc(size_t n) {
        V* p = nullptr; CK(cudaMalloc((void**) &p, n * sizeof(V))); allocations.push_back(p);
        CK(cudaMemset(p, 0, n * sizeof(V))); return p;
    }
    explicit Fixture(size_t bytes, int rows) : T(rows), blob(bytes) {
        CK(cudaHostAlloc((void**) &host, NL * NE * blob, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &host_d, host, 0));
        for (size_t i = 0; i < NL * NE * blob; ++i) host[i] = (uint8_t) ((i / blob * 17 + i % blob) % 251);
        CK(cudaHostAlloc((void**) &ring, sizeof(*ring) * gf::kRingSize, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&base.ring, ring, 0));
        CK(cudaHostAlloc((void**) &response, sizeof(*response), cudaHostAllocMapped));
        void* response_d = nullptr; CK(cudaHostGetDevicePointer(&response_d, response, 0)); base.resp = response_d;
        base.n_keys = NL * NE;
        base.tab = alloc<unsigned long long>(2 * NL * NE + NL * gf::kSpares);
        scratch = alloc<uint8_t>(T * K * blob); spare = alloc<uint8_t>(gf::kSpares * blob);
        slots = alloc<unsigned long long>(T * K);
        std::vector<unsigned long long> sp(T * K);
        for (int i = 0; i < T * K; ++i) sp[i] = (unsigned long long) (scratch + i * blob);
        upload(slots, sp);
        plans = alloc<unsigned long long>(T * K); plan_ids = alloc<int>(T * K);
        plan_weights = alloc<float>(T * K); fetch = alloc<unsigned long long>(T * K);
        base.cpu_part = alloc<float>(T * E); base.cpu_flag = alloc<int>(T);
        base.cpu_seq = alloc<unsigned int>(T * 16); base.seq = alloc<unsigned int>(1);
        base.wait_seq = alloc<unsigned int>(1); base.pf_n = alloc<int>(1);
        base.pf_src = alloc<unsigned long long>(K); base.pf_dst = alloc<unsigned long long>(K);
        base.dcnt = alloc<unsigned int>(NL * NE);
        owned = alloc<uint8_t>(NL * NE); base.helper_owned = owned;
        logits = alloc<float>(NE); x = alloc<float>(T * E); skip = alloc<int>(1);
        std::mt19937 rng(20261028); std::normal_distribution<float> nd(0.0f, 0.4f);
        std::vector<float> input(T * E); for (auto& v : input) v = nd(rng); upload(x, input);
    }
    ~Fixture() {
        gf::set_skip(nullptr);
        for (void* p : allocations) cudaFree(p);
        cudaFreeHost(response); cudaFreeHost(ring); cudaFreeHost(host);
    }
    void reset(Settings s, int il = 0, bool clear_plans = true) {
        setting = s; layer = il; gf::set_skip(nullptr);
        CK(cudaMemset(base.seq, 0, sizeof(unsigned int))); CK(cudaMemset(base.dcnt, 0, NL * NE * sizeof(unsigned int)));
        CK(cudaMemset(base.cpu_seq, 0, T * 16 * sizeof(unsigned int)));
        CK(cudaMemset(scratch, 0xa5, T * K * blob)); CK(cudaMemset(owned, s.helper ? 1 : 0, NL * NE));
        std::memset(ring, 0, sizeof(*ring) * gf::kRingSize); std::memset(response, 0, sizeof(*response));
        if (clear_plans) { CK(cudaMemset(plans, 0, T * K * sizeof(unsigned long long))); CK(cudaMemset(plan_ids, 0xff, T * K * sizeof(int))); }
        std::vector<unsigned long long> table(2 * NL * NE + NL * gf::kSpares, 0);
        if (s.ram) for (int key = 0; key < NL * NE; ++key) table[NL * NE + key] = (unsigned long long) (host_d + key * blob);
        for (int i = 0; i < s.spares; ++i) table[2 * NL * NE + il * gf::kSpares + i] = (unsigned long long) (spare + i * blob);
        upload(base.tab, table);
    }
    gf::MoeDev row(int t) {
        gf::MoeDev d = base;
        d.scratch = slots + t * K; d.plan_ptr = plans + t * K; d.plan_id = plan_ids + t * K;
        d.plan_w = plan_weights + t * K; d.fetch_src = fetch + t * K;
        d.cpu_flag += t; d.cpu_part += t * E; d.cpu_seq += t * 16;
        if (setting.reuse) { d.win_reuse_ptr = plans; d.win_reuse_id = plan_ids; d.win_reuse_scratch = slots; }
        return d;
    }
    gf::MoeRequest route(int t, bool complete = true, int win_n = -1) {
        if (win_n < 0) win_n = T;
        std::vector<float> l(NE, -8.0f);
        for (int i = 0; i < K; ++i) l[ids[t][i]] = 4.0f - 0.2f * i;
        upload(logits, l);
        const auto d = row(t);
        gf::moe_route(logits, nullptr, NE, K, 1.0f, true, layer, x + t * E, E, d,
                      nullptr, nullptr, 1.0f, 0, nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0,
                      K, setting.cpu, setting.promote_min, 0, 0, t, nullptr, win_n);
        CK(cudaStreamSynchronize(0));
        const unsigned int seq = download(base.seq, 1)[0];
        gf::MoeRequest request = ring[seq % gf::kRingSize];
        if (complete) {
            if (request.miss_mask) {
                std::memset(response, 0, sizeof(*response)); response->ptr_mask = request.miss_mask;
                for (int i = 0; i < K; ++i) response->ptr[i] = (unsigned long long) (host_d + (layer * NE + request.ids[i]) * blob);
                response->seq = seq;
            }
            gf::moe_wait(d, E, 0); gf::moe_fetch(d, K, blob, 0); CK(cudaStreamSynchronize(0));
        }
        return request;
    }
    Result run(Settings s, int il = 0, bool clear_plans = true) {
        reset(s, il, clear_plans); Result result; result.requests.resize(T);
        for (int t = 0; t < T; ++t) result.requests[t] = route(t);
        const auto pp = download(plans, T * K);
        result.bytes.resize(T * K * blob, 0);
        for (int i = 0; i < T * K; ++i) if (pp[i]) CK(cudaMemcpy(result.bytes.data() + i * blob, (void*) pp[i], blob, cudaMemcpyDeviceToHost));
        result.weights = download(plan_weights, T * K);
        gf::MoeWin w; w.plan_ptr = plans; w.T = T; w.k = K;
        w.grp_ptr = alloc<unsigned long long>(T * K); w.grp_start = alloc<int>(T * K + 1);
        w.grp_ent = alloc<int>(T * K); w.n_groups = alloc<int>(1); gf::moe_win_group(w, 0);
        result.groups = download(w.n_groups, 1)[0];
        return result;
    }
};
int count(const Result& r, bool reuse) {
    int n = 0; for (const auto& q : r.requests) n += __builtin_popcount(reuse ? q.reuse_mask : q.fetch_mask); return n;
}
void parity(const Result& off, const Result& on) {
    expect(off.bytes == on.bytes, "every plan reads the same weight bytes");
    expect(std::memcmp(off.weights.data(), on.weights.data(), off.weights.size() * sizeof(float)) == 0, "route weights unchanged");
    expect(off.requests.size() == on.requests.size(), "same number of route rows");
    for (size_t t = 0; t < off.requests.size(); ++t) {
        const auto& a = off.requests[t]; const auto& b = on.requests[t];
        expect(a.cpu_mask == b.cpu_mask && a.promo_mask == b.promo_mask && a.miss_mask == b.miss_mask &&
               a.skip_mask == b.skip_mask && std::memcmp(a.ids, b.ids, sizeof(a.ids)) == 0,
               "helper/CPU/promotion/disk/selection policy unchanged");
        expect(a.fetch_mask == (b.fetch_mask | b.reuse_mask) && !(b.fetch_mask & b.reuse_mask), "actual fetch/reuse masks partition old fetches");
    }
}
void test_routes(int T) {
    Fixture f(4096, T);
    const int reused = T == 2 ? 4 : 7;
    for (int scenario = 0; scenario < 6; ++scenario) {
        Settings s;
        if (scenario == 1) s.spares = 3;
        if (scenario == 2) s.cpu = 8ull << 32;
        if (scenario == 3) s.helper = true;
        if (scenario == 4) s.ram = false;
        if (scenario == 5) { s.spares = 3; s.promote_min = 100; }
        const auto off = f.run(s); s.reuse = true; const auto on = f.run(s); parity(off, on);
        if (scenario == 0) {
            expect(count(off, false) == T * K && count(on, false) == T * K - reused && count(on, true) == reused,
                   "each repeated scratch expert avoids one copy");
            expect(off.groups == T * K && on.groups == T * K - reused, "aliases merge exactly the repeated expert groups");
            const auto pp = download(f.plans, T * K), fs = download(f.fetch, T * K);
            expect(pp[8] == pp[4] && fs[8] == 0, "row1 rank0 aliases row0 rank4 with no fetch source");
            const auto unused = download(f.scratch + 8 * f.blob, f.blob);
            expect(std::all_of(unused.begin(), unused.end(), [](uint8_t c) { return c == 0xa5; }), "aliased destination was not copied");
        }
        if (scenario >= 2 && scenario <= 4) expect(count(on, true) == 0, "CPU/helper/disk paths never use RAM scratch reuse");
        if (scenario == 5) expect(count(on, true) == reused, "reuse does not bypass the promotion threshold");
    }
    Settings s; s.reuse = true; f.reset(s); f.route(0);
    // Even matching expert IDs must not borrow this/later row's scratch (or any unverified global pointer).
    auto pp = download(f.plans, T * K);
    for (int i = 0; i < K; ++i) pp[i] = (unsigned long long) (f.scratch + ((T - 1) * K + i) * f.blob);
    upload(f.plans, pp); expect(f.route(1).reuse_mask == 0, "current/future-row pointers are rejected");
    f.reset(s); f.route(0); upload(f.plan_ids, std::vector<int>(T * K, NE - 1));
    expect(f.route(1).reuse_mask == 0, "private pointer with a different expert ID is rejected");
    for (int unsupported : {0, 1, 4}) {
        f.reset(s); f.route(0);
        expect(f.route(1, true, unsupported).reuse_mask == 0, "unsupported window size never aliases scratch");
    }
    if (T == 3) {
        f.reset(s); f.route(0); f.route(1);
        expect(f.route(2, true, 2).reuse_mask == 0, "row outside T2 never aliases scratch");
    }
    f.reset(s); f.route(0); CK(cudaMemset(f.skip, 1, sizeof(int))); gf::set_skip(f.skip);
    const auto skipped = f.route(1); gf::set_skip(nullptr);
    expect(skipped.fetch_mask == 0 && skipped.reuse_mask == 0 && skipped.miss_mask == 0, "aborted/nonresident route does not fetch or alias");
    const auto a = f.run(s, 0); (void) a;
    const auto next = f.run(s, 1, false);
    expect(next.requests[0].reuse_mask == 0 && next.requests[0].fetch_mask == 255, "new layer/window row0 never reuses stale plans");
    for (int t = 0; t < T; ++t) for (int i = 0; i < K; ++i)
        expect(std::memcmp(next.bytes.data() + (t * K + i) * f.blob, f.host + (NE + ids[t][i]) * f.blob, f.blob) == 0,
               "new layer/window uses its own immutable weights");
    std::printf("T%d route/fetch checks complete (%d -> %d copies; bytes, guards, admission and group counts)\n",
                T, T * K, T * K - reused);
}

std::vector<uint8_t> quantized(int type, int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 0.05f); std::vector<float> x((size_t) rows * cols);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> q((size_t) rows * gf::row_bytes(type, cols)); std::vector<float> importance(cols, 1.0f);
    const size_t n = ggml_quantize_chunk((ggml_type) type, x.data(), q.data(), 0, rows, cols,
                                      ggml_quantize_requires_imatrix((ggml_type) type) ? importance.data() : nullptr);
    expect(n == q.size(), "quantized fixture size"); return q;
}
std::vector<float> grouped(Fixture& f, int gu_type, size_t down_off) {
    const int T = f.T;
    gf::MoeWin w; w.plan_ptr = f.plans; w.plan_w = f.plan_weights; w.cpu_part = f.base.cpu_part;
    w.cpu_flag = f.base.cpu_flag; w.T = T; w.k = K; w.n_embd = E; w.n_ff = FF; w.limit = 7.0f; w.down_off = down_off;
    const size_t q8 = 36; w.xq_stride = E / 32 * q8;
    auto* xq = f.alloc<uint8_t>(T * w.xq_stride); w.xq = xq;
    for (int t = 0; t < T; ++t) gf::quantize_q8_1(f.x + t * E, xq + t * w.xq_stride, E, 0);
    w.hq = f.alloc<uint8_t>(T * K * (FF / 32) * q8); w.part = f.alloc<float>(T * K * E);
    w.out = f.alloc<float>(T * E); w.out_stride = E;
    w.grp_ptr = f.alloc<unsigned long long>(T * K); w.grp_start = f.alloc<int>(T * K + 1);
    w.grp_ent = f.alloc<int>(T * K); w.n_groups = f.alloc<int>(1);
    expect(gf::moe_win_experts(gu_type, 23, w, 0), "grouped expert types supported");
    CK(cudaStreamSynchronize(0)); return download(w.out, T * E);
}
void test_arithmetic(int T) {
    for (int up : {23, 12, 13}) {
        const size_t gate_bytes = FF * gf::row_bytes(23, E), up_bytes = FF * gf::row_bytes(up, E);
        const size_t down_off = gate_bytes + up_bytes, bytes = down_off + E * gf::row_bytes(23, FF);
        const int gu_type = up == 23 ? 23 : 23 | (up << 8);
        Fixture f(bytes, T); std::mt19937 rng(42 + up);
        for (int e = 0; e < NE; ++e) {
            const auto gate = quantized(23, FF, E, rng), u = quantized(up, FF, E, rng), down = quantized(23, E, FF, rng);
            std::memcpy(f.host + e * bytes, gate.data(), gate_bytes);
            std::memcpy(f.host + e * bytes + gate_bytes, u.data(), up_bytes);
            std::memcpy(f.host + e * bytes + down_off, down.data(), down.size());
        }
        Settings s; const auto off_routes = f.run(s); const auto off = grouped(f, gu_type, down_off);
        s.reuse = true; const auto on_routes = f.run(s); const auto on = grouped(f, gu_type, down_off);
        parity(off_routes, on_routes);
        expect(off_routes.groups == T * K && on_routes.groups == T * K - (T == 2 ? 4 : 7), "arithmetic test changes pointer grouping");
        expect(std::all_of(off.begin(), off.end(), [](float v) { return std::isfinite(v); }) &&
               std::any_of(off.begin(), off.end(), [](float v) { return v != 0.0f; }), "finite nontrivial expert output");
        expect(std::memcmp(off.data(), on.data(), off.size() * sizeof(float)) == 0,
               "independent scratch vs aliased grouped expert output is bitwise identical");
        std::printf("T%d grouped arithmetic: gu_type=%d down=23, %zu float outputs compared bitwise\n", T, gu_type, off.size());
    }
}
} // namespace
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: CUDA GPU required"); return 77; }
    CK(cudaSetDevice(0));
    for (int rows : {2, 3}) { test_routes(rows); test_arithmetic(rows); }
    expect(gf::launch_errors() == 0, "no kernel launch failures");
    std::printf("glm_window_fetch_reuse_test: %s (%d failures)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
