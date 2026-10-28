// maya-helperd: one GPU of another machine as a GLM expert helper (STRATA_GLM_NET_HELPER=host:port on the trunk).
// It wraps the local GlmExpertHelper: the trunk sends the experts to hold (or where they lie in this machine's copy
// of the pack), then one compute request a route.  One trunk at a time; the GPU is the first one CUDA sees, so pick
// it with CUDA_VISIBLE_DEVICES.
//
//   maya-helperd --port 18800 [--model-dir ~/models/maya/Maya-L] [--bind 0.0.0.0]
//   maya-helperd --dial TRUNK:PORT ...   (the trunk listens: STRATA_GLM_NET_HELPER=listen:PORT; redials after each)
//
// --ram-gib N: a RAM tier behind the GPU.  The trunk is offered VRAM + N GiB; experts past the VRAM go to this
// machine's RAM and this machine's CPU computes them (ggml-cpu's dots, as the trunk's CPU lane), while the GPU does
// its share of the same route.  Pilot: static, no RAM -> VRAM promotion.  --cpu-threads T (default: cores / 2).

#include "strata/core/net_helper.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "helper_queue_probe.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace np = strata::core::netproto;
namespace kc = strata::kernels::cpu;

namespace {
size_t g_ram_bytes = 0;
int g_cpu_threads = 0;
int g_selftest = 0;
int g_rows_check = 0;
bool g_ram_cold = false;   // --ram-cold
bool g_rows_serial = false;
bool g_queue_probe = false;   // HELPERD_QUEUE_PROBE: diagnostics only; does not consume/batch requests

uint64_t cpu_ns(clockid_t clock) {
    timespec t{};
    ::clock_gettime(clock, &t);
    return (uint64_t) t.tv_sec * 1000000000 + (uint64_t) t.tv_nsec;
}

// A spinning pool: a route's CPU share is a fraction of a millisecond, too short to wake sleeping threads for.
// The workers drop to short sleeps after 20 ms without work (between requests), so an idle helper is quiet.
class SpinPool {
public:
    explicit SpinPool(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~SpinPool() {
        stop_.store(true);
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }
    uint64_t work_ns() const { return work_ns_.load(std::memory_order_relaxed); }
    void profile_work(bool enabled) { profile_work_.store(enabled, std::memory_order_relaxed); }
    void start(int jobs, std::function<void(int)> fn) {
        fn_ = std::move(fn);
        jobs_ = jobs;
        next_.store(0, std::memory_order_relaxed);
        left_.store(jobs, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
    }
    void wait() {   // the caller works too
        take();
        while (left_.load(std::memory_order_acquire) > 0) __builtin_ia32_pause();
    }
    void run(int jobs, std::function<void(int)> fn) { start(jobs, std::move(fn)); wait(); }

private:
    void take() {
        for (;;) {
            const int j = next_.fetch_add(1, std::memory_order_acq_rel);
            if (j >= jobs_) return;
            if (g_queue_probe && profile_work_.load(std::memory_order_relaxed)) {
                const uint64_t start = cpu_ns(CLOCK_THREAD_CPUTIME_ID);
                fn_(j);
                work_ns_.fetch_add(cpu_ns(CLOCK_THREAD_CPUTIME_ID) - start, std::memory_order_relaxed);
            } else {
                fn_(j);
            }
            left_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void loop() {
        uint64_t seen = gen_.load(std::memory_order_acquire);
        auto idle = std::chrono::steady_clock::now();
        while (!stop_.load(std::memory_order_relaxed)) {
            const uint64_t g = gen_.load(std::memory_order_acquire);
            if (g != seen) {
                seen = g;
                take();
                idle = std::chrono::steady_clock::now();
                continue;
            }
            if (std::chrono::steady_clock::now() - idle > std::chrono::milliseconds(20))
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            else
                __builtin_ia32_pause();
        }
    }
    std::vector<std::thread> th_;
    std::function<void(int)> fn_;
    int jobs_ = 0;
    std::atomic<int> next_{0}, left_{0};
    std::atomic<uint64_t> gen_{0};
    std::atomic<uint64_t> work_ns_{0};
    std::atomic<bool> profile_work_{false};
    std::atomic<bool> stop_{false};
};

bool recv_all(int fd, void* p, size_t n) {
    uint8_t* at = (uint8_t*) p;
    while (n > 0) {
        const ssize_t k = ::recv(fd, at, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        at += k;
        n -= (size_t) k;
    }
    return true;
}
bool send_all(int fd, const void* p, size_t n) {
    const uint8_t* at = (const uint8_t*) p;
    while (n > 0) {
        const ssize_t k = ::send(fd, at, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        at += k;
        n -= (size_t) k;
    }
    return true;
}
bool answer(int fd, bool ok, uint64_t value, const void* data, size_t len, uint32_t pad = 0) {
    const np::Reply r{ok ? 1 : 0, pad, value, (uint64_t) len};
    return send_all(fd, &r, sizeof(r)) && (len == 0 || send_all(fd, data, len));
}
bool fail(int fd, const std::string& text) {
    std::fprintf(stderr, "helperd: %s\n", text.c_str());
    return answer(fd, false, 0, text.data(), text.size());
}

struct Session {
    strata::core::GlmExpertHelper helper;
    std::string model_dir;
    std::vector<int> shard_fd;
    std::vector<uint8_t> payload;
    uint8_t* stage = nullptr;   // pinned: a blob read from the pack on its way to VRAM
    size_t stage_bytes = 0;
    std::vector<float> out;
    int n_embd = 0, n_ff = 0;
    uint64_t computed = 0, routes = 0, inserted = 0;
    double ms_compute = 0;
    // the RAM tier: experts past the VRAM, computed by this machine's CPU
    uint8_t* ram = nullptr;
    size_t ram_cap = 0, ram_used = 0;
    std::vector<int64_t> ram_off;      // key -> offset in `ram` (-1: not here)
    std::vector<uint32_t> ram_up;      // key -> the up slice's offset in its blob
    uint64_t ram_held = 0, cpu_experts = 0, cpu_routes = 0;
    uint64_t c2_routes = 0, c2_gpu_checks = 0, c2_gpu_bad = 0, c2_act_checks = 0, c2_act_bad = 0, c2_r16_checks = 0;
    double c2_r16_max = 0.0, c2_r16_mean = 0.0;
    double ms_cpu = 0;
    std::map<std::pair<int, int>, kc::NativeFmt> fmts;
    std::unique_ptr<SpinPool> pool;
    std::vector<uint8_t> act, hq;
    std::vector<float> ff, dn, cpu_out;
    std::vector<uint16_t> rx16, ans16;   // a prompt layer's rows (kRows)
    std::vector<np::RowRef> rrefs;
    std::vector<np::TokEntry> tents;
    uint64_t pf_rows = 0, pf_layers = 0;
    double ms_rows = 0, ms_rows_recv = 0, ms_rows_send = 0, ms_tok_gpu = 0;   // (ms_tok_gpu: kTokRows' GPU share)
    std::vector<float> rxf, rowsf, gout;
    int rows_checks = 0;
    std::vector<int> check_keys;
    std::vector<std::vector<uint8_t>> check_blob;
    std::vector<uint32_t> check_up;
    std::vector<uint8_t> queue_scratch;
    uint64_t probe_groups[4]{}, probe_routes = 0, probe_covered = 0, probe_cpu_occurrences = 0;
    uint64_t probe_cpu_repeats = 0, probe_cpu_bytes = 0, probe_cpu_repeat_bytes = 0, probe_fmt_errors = 0;
    uint64_t probe_work_ns = 0;
    uint64_t probe_sampled_routes = 0;
    double probe_ms = 0, probe_service_ms = 0, probe_gpu_ms = 0, probe_sampled_service_ms = 0;
    int probe_skip = 0;

    void queue_probe(int fd, const np::ComputeReq& q) {
        ++probe_routes;
        // Disjoint hypothetical groups, not overlapping snapshots whose repeated experts would be overcounted.
        if (probe_skip > 0) { --probe_skip; return; }
        const auto start = std::chrono::steady_clock::now();
        np::ComputeReq next[2]{};
        const int n = strata::helper_probe::peek_ready(fd, n_embd, q, next, queue_scratch);
        ++probe_groups[n + 1];
        probe_covered += (uint64_t) n + 1;
        probe_skip = n;
        const np::ComputeReq* rows[3] = {&q, &next[0], &next[1]};
        int seen[24]{}, used = 0;
        std::string err;
        const kc::NativeFmt* nf = nullptr;
        for (int t = 0; t <= n; ++t) {
            for (int e = 0; e < rows[t]->ne && e < 8; ++e) {
                const int key = rows[t]->keys[e];
                if (key < 0 || (size_t) key >= ram_off.size() || ram_off[(size_t) key] < 0) continue;
                if (!nf && !(nf = fmt(q.gu_type, q.d_type, err))) { ++probe_fmt_errors; continue; }
                const uint64_t bytes = (uint64_t) n_ff * (nf->gu_row + (nf->up_row ? nf->up_row : nf->gu_row)) +
                                       (uint64_t) n_embd * nf->d_row;
                ++probe_cpu_occurrences;
                probe_cpu_bytes += bytes;
                bool repeated = false;
                for (int i = 0; i < used; ++i) repeated |= seen[i] == key;
                if (repeated) { ++probe_cpu_repeats; probe_cpu_repeat_bytes += bytes; }
                else seen[used++] = key;
            }
        }
        probe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    void print_queue_probe(const char* stage) const {
        if (!g_queue_probe) return;
        std::fprintf(stderr, "helperd queue probe %s: routes=%llu covered=%llu groups1=%llu groups2=%llu groups3=%llu "
                             "cpu_occurrences=%llu cpu_repeats=%llu cpu_weight_bytes=%llu cpu_repeat_bytes=%llu "
                             "fmt_errors=%llu probe_ms=%.3f service_ms=%.3f gpu_ms=%.3f "
                             "sampled_routes=%llu sampled_service_ms=%.3f sampled_pool_job_cpu_ms=%.3f pending_skip=%d\n",
                     stage, (unsigned long long) probe_routes, (unsigned long long) probe_covered,
                     (unsigned long long) probe_groups[1], (unsigned long long) probe_groups[2],
                     (unsigned long long) probe_groups[3], (unsigned long long) probe_cpu_occurrences,
                     (unsigned long long) probe_cpu_repeats, (unsigned long long) probe_cpu_bytes,
                     (unsigned long long) probe_cpu_repeat_bytes, (unsigned long long) probe_fmt_errors,
                     probe_ms, probe_service_ms, probe_gpu_ms, (unsigned long long) probe_sampled_routes,
                     probe_sampled_service_ms, probe_work_ns / 1e6, probe_skip);
    }

    ~Session() {
        for (int f : shard_fd) if (f >= 0) ::close(f);
        if (stage) cudaFreeHost(stage);
        if (ram) ::munmap(ram, ram_cap);
        helper.close();
    }
    bool ram_insert(int key, size_t stride, uint32_t up_off, uint8_t*& dst) {
        if (ram == nullptr || ram_used + stride > ram_cap || key < 0 || (size_t) key >= ram_off.size()) return false;
        dst = ram + ram_used;
        ram_off[(size_t) key] = (int64_t) ram_used;
        ram_up[(size_t) key] = up_off;
        ram_used += stride;
        ++ram_held;
        return true;
    }
    const kc::NativeFmt* fmt(int gu_type, int d_type, std::string& err) {
        auto it = fmts.find({gu_type, d_type});
        if (it != fmts.end()) return &it->second;
        kc::NativeFmt f;
        if (!kc::native_fmt(gu_type, d_type, n_embd, n_ff, f, err)) return nullptr;
        return &fmts.emplace(std::make_pair(gu_type, d_type), f).first->second;
    }
    // the CPU's share of a route into cpu_out: sum_i w[i] * expert_i(x), the trunk CPU lane's arithmetic
    void cpu_start(const kc::NativeFmt& nf, size_t down_off, float limit, int ne, const uint8_t* const* blob,
                   const uint32_t* up, const float* x) {
        kc::native_quant_act(nf, x, act.data());
        (void) down_off;
        cpu_start_act(nf, limit, ne, blob, up);
    }
    // ... with the activation already in act (kCompute2's kXk)
    void cpu_start_act(const kc::NativeFmt& nf, float limit, int ne, const uint8_t* const* blob, const uint32_t* up) {
        const int T = pool->size() * 4, gu_tot = ne * n_ff;
        pool->start(T, [=, this, &nf](int job) {
            const void* a[1] = {act.data()};
            const int q1 = (int) ((int64_t) gu_tot * (job + 1) / T);
            for (int q = (int) ((int64_t) gu_tot * job / T); q < q1;) {
                const int i = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
                float* o[1] = {ff.data() + (size_t) i * n_ff};
                kc::native_gu_rows_split(nf, blob[i], blob[i] + up[i], a, 1, o, r0, r1, limit);
                q += r1 - r0;
            }
        });
    }
    void cpu_finish(const kc::NativeFmt& nf, size_t down_off, int ne, const uint8_t* const* blob, const float* w) {
        pool->wait();
        for (int i = 0; i < ne; ++i)
            kc::native_quant_h(nf, ff.data() + (size_t) i * n_ff, hq.data() + (size_t) i * kc::kNativeHBytes);
        const int T = pool->size() * 4, dn_tot = ne * n_embd;
        pool->run(T, [=, this, &nf](int job) {
            const int q1 = (int) ((int64_t) dn_tot * (job + 1) / T);
            for (int q = (int) ((int64_t) dn_tot * job / T); q < q1;) {
                const int i = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
                const void* h[1] = {hq.data() + (size_t) i * kc::kNativeHBytes};
                float* o[1] = {dn.data() + (size_t) i * n_embd};
                kc::native_down_rows_split(nf, blob[i] + down_off, h, 1, o, r0, r1);
                q += r1 - r0;
            }
        });
        for (int r = 0; r < n_embd; ++r) {
            float m = 0.0f;
            for (int i = 0; i < ne; ++i) m += w[i] * dn[(size_t) i * n_embd + r];
            cpu_out[(size_t) r] = m;
        }
    }
    uint8_t* staging(size_t bytes) {
        if (bytes > stage_bytes) {
            if (stage) cudaFreeHost(stage);
            stage = nullptr;
            stage_bytes = 0;
            if (cudaHostAlloc((void**) &stage, bytes, cudaHostAllocDefault) != cudaSuccess) return nullptr;
            stage_bytes = bytes;
        }
        return stage;
    }

    // false: the connection is done (closed, or a reply could not be sent)
    bool serve(int fd, const np::Hdr& h) {
        std::string err;
        switch (h.op) {
        case np::kOpen: {
            np::OpenReq q{};
            if (h.len < sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            std::string names(h.len - sizeof(q), '\0');
            if (!names.empty() && !recv_all(fd, names.data(), names.size())) return false;
            for (int f : shard_fd) if (f >= 0) ::close(f);
            shard_fd.clear();
            size_t at = 0;
            int missing = 0;
            while (at < names.size()) {
                const size_t nl = names.find('\n', at);
                const std::string name = names.substr(at, nl - at);
                at = nl == std::string::npos ? names.size() : nl + 1;
                const int f = model_dir.empty() ? -1 : ::open((model_dir + "/" + name).c_str(), O_RDONLY | O_CLOEXEC);
                missing += f < 0;
                shard_fd.push_back(f);
            }
            if (!helper.open(0, q.n_embd, q.n_ff, {}, err)) return fail(fd, err);
            n_embd = q.n_embd;
            n_ff = q.n_ff;
            out.assign((size_t) n_embd, 0.0f);
            cpu_out.assign((size_t) n_embd, 0.0f);
            act.assign(kc::kNativeActBytes, 0);
            hq.assign((size_t) 8 * kc::kNativeHBytes, 0);
            ff.assign((size_t) 8 * n_ff, 0.0f);
            dn.assign((size_t) 8 * n_embd, 0.0f);
            std::fprintf(stderr, "helperd: trunk opened (n_embd %d, n_ff %d, %zu shards, %d not in %s)\n", q.n_embd,
                         q.n_ff, shard_fd.size(), missing, model_dir.empty() ? "(no --model-dir)" : model_dir.c_str());
            return answer(fd, true, (uint64_t) missing, nullptr, 0, np::kProto);   // (pad: the protocol, kCompute2)
        }
        case np::kReserve: {
            np::ReserveReq q{};
            if (h.len != sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            if (!helper.reserve(q.keys, (size_t) q.bytes, err)) return fail(fd, err);
            const size_t vram = helper.available();
            if (ram) ::munmap(ram, ram_cap);
            ram = nullptr;
            ram_cap = ram_used = 0;
            ram_held = 0;
            ram_off.assign((size_t) q.keys, -1);
            ram_up.assign((size_t) q.keys, 0);
            // --ram-cold: the RAM tier is offered apart (Reply.pad, MiB) for the trunk's coldest experts - the ones its
            // own tiers leave on the disk - and the reply's value stays the VRAM's
            if (g_ram_bytes > 0 && (g_ram_cold || (size_t) q.bytes > vram)) {
                // not q.bytes - vram: the VRAM's tail (less than a stride) stays empty, so the RAM takes up to all
                ram_cap = g_ram_cold ? g_ram_bytes : std::min(g_ram_bytes, (size_t) q.bytes);
                void* p = ::mmap(nullptr, ram_cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) return fail(fd, std::string("RAM tier: ") + std::strerror(errno));
                ::madvise(p, ram_cap, MADV_HUGEPAGE);
                ram = (uint8_t*) p;
                if (!pool) pool = std::make_unique<SpinPool>(std::max(0, g_cpu_threads - 1));
            }
            std::fprintf(stderr, "helperd: %.2f GiB for experts in VRAM + %.2f GiB in RAM (%d CPU threads)\n",
                         (double) vram / 1073741824.0, (double) ram_cap / 1073741824.0, pool ? pool->size() : 0);
            if (g_ram_cold) {
                const np::Reply r{1, (uint32_t) (ram_cap >> 20), (uint64_t) vram, 0};
                return send_all(fd, &r, sizeof(r));
            }
            return answer(fd, true, (uint64_t) (vram + ram_cap), nullptr, 0);
        }
        case np::kInsert: {
            np::InsertReq q{};
            if (h.len < sizeof(q) || !recv_all(fd, &q, sizeof(q)) || h.len - sizeof(q) != q.bytes) return false;
            uint8_t* buf = staging((size_t) q.bytes);
            if (buf == nullptr) {
                payload.resize((size_t) q.bytes);
                buf = payload.data();
            }
            if (!recv_all(fd, buf, (size_t) q.bytes)) return false;
            if (q.stride > helper.available()) {   // past the VRAM: the RAM tier (the up slice where native_fmt puts it)
                uint8_t* dst = nullptr;
                if (!ram_insert(q.key, (size_t) q.stride, 0xffffffffu, dst)) return fail(fd, "the RAM tier is full");
                std::memcpy(dst, buf, (size_t) q.bytes);
            } else if (!helper.insert(q.key, buf, (size_t) q.bytes, (size_t) q.stride, err)) return fail(fd, err);
            ++inserted;
            return answer(fd, true, 0, nullptr, 0);
        }
        case np::kInsertFile: {
            np::InsertFileReq q{};
            if (h.len != sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            const bool to_ram = q.stride > helper.available();
            uint8_t* buf = nullptr;
            if (to_ram) {   // read straight into the RAM tier
                if (!ram_insert(q.key, (size_t) q.stride, (uint32_t) q.dst[1], buf)) return fail(fd, "the RAM tier is full");
            } else {
                buf = staging((size_t) q.bytes);
                if (buf == nullptr) return fail(fd, "no pinned staging buffer");
            }
            for (int j = 0; j < 3; ++j) {
                if (q.shard[j] >= shard_fd.size() || shard_fd[q.shard[j]] < 0)
                    return fail(fd, "shard " + std::to_string(q.shard[j]) + " is not in " + model_dir);
                if (q.dst[j] + q.len[j] > q.bytes) return fail(fd, "blob part outside the blob");
                size_t done = 0;
                while (done < q.len[j]) {
                    const ssize_t k = ::pread(shard_fd[q.shard[j]], buf + q.dst[j] + done, (size_t) q.len[j] - done,
                                              (off_t) (q.off[j] + done));
                    if (k < 0 && errno == EINTR) continue;
                    if (k <= 0) return fail(fd, std::string("pack read: ") + (k == 0 ? "short file" : std::strerror(errno)));
                    done += (size_t) k;
                }
            }
            if (!to_ram && !helper.insert(q.key, buf, (size_t) q.bytes, (size_t) q.stride, err)) return fail(fd, err);
            if (!to_ram && check_keys.size() < (size_t) g_selftest) {   // a CPU copy of a GPU expert, to compare
                check_keys.push_back(q.key);
                check_blob.emplace_back(buf, buf + q.bytes);
                check_up.push_back((uint32_t) q.dst[1]);
            }
            ++inserted;
            return answer(fd, true, 0, nullptr, 0);
        }
        case np::kHash: {   // the trunk checks its copies here (STRATA_GLM_NET_COPY): a hash of each RAM-tier blob
            if (h.len % sizeof(np::HashRef) != 0 || h.len == 0 || h.len > ((uint64_t) 1 << 24)) return false;
            std::vector<np::HashRef> refs(h.len / sizeof(np::HashRef));
            if (!recv_all(fd, refs.data(), (size_t) h.len)) return false;
            std::vector<uint64_t> hs(refs.size(), 0);
            std::atomic<size_t> next{0};
            std::vector<std::thread> th;
            const int nt = std::max(1, std::min(12, (int) std::thread::hardware_concurrency()));
            for (int t = 0; t < nt; ++t)
                th.emplace_back([&] {
                    for (size_t i; (i = next.fetch_add(1)) < refs.size();) {
                        const np::HashRef& r = refs[i];
                        const int64_t ro = r.key >= 0 && (size_t) r.key < ram_off.size() ? ram_off[(size_t) r.key] : -1;
                        if (ro < 0 || r.dst[2] + r.len[2] > ram_cap - (size_t) ro) continue;
                        hs[i] = np::blob_hash(ram + ro, r.dst, r.len);
                    }
                });
            for (auto& t : th) t.join();
            size_t held = 0;
            for (uint64_t v : hs) held += v != 0;
            std::fprintf(stderr, "helperd: hashed %zu RAM-tier experts for the trunk's check (%zu not held)\n", refs.size(),
                         refs.size() - held);
            return answer(fd, true, (uint64_t) refs.size(), hs.data(), hs.size() * sizeof(uint64_t));
        }
        case np::kCompute: {
            np::ComputeReq q{};
            if (h.len != sizeof(q) + (size_t) n_embd * sizeof(float) || !recv_all(fd, &q, sizeof(q))) return false;
            payload.resize((size_t) n_embd * sizeof(float));
            if (!recv_all(fd, payload.data(), payload.size())) return false;
            if (g_queue_probe) queue_probe(fd, q);
            const auto t0 = std::chrono::steady_clock::now();
            // Job clocks can perturb sub-ms work. Sample one in 64 routes; the queue snapshot remains on each
            // disjoint group. These CPU counters describe only sampled service intervals, never socket-idle time.
            const bool sample_cpu = g_queue_probe && probe_routes % 64 == 0;
            const uint64_t work0 = sample_cpu && pool ? pool->work_ns() : 0;
            if (g_queue_probe && pool) pool->profile_work(sample_cpu);
            const float* x = (const float*) payload.data();
            int gn = 0, cn = 0, gk[8];
            float gw[8], cw[8];
            const uint8_t* cb[8];
            uint32_t cu[8];
            const kc::NativeFmt* nf = nullptr;
            for (int i = 0; i < q.ne && i < 8; ++i) {
                const int key = q.keys[i];
                const int64_t ro = key >= 0 && (size_t) key < ram_off.size() ? ram_off[(size_t) key] : -1;
                if (ro < 0) { gk[gn] = key; gw[gn++] = q.w[i]; continue; }
                if (nf == nullptr && (nf = fmt(q.gu_type, q.d_type, err)) == nullptr) return fail(fd, err);
                cb[cn] = ram + ro;
                cu[cn] = ram_up[(size_t) key] == 0xffffffffu ? (uint32_t) nf->up_off : ram_up[(size_t) key];
                cw[cn++] = q.w[i];
            }
            // the CPU starts its gate/up rows, the GPU does its share meanwhile, then the CPU's down rows
            if (cn > 0) cpu_start(*nf, (size_t) q.down_off, q.limit, cn, cb, cu, x);
            if (gn > 0) {
                const auto gpu0 = g_queue_probe ? std::chrono::steady_clock::now() : t0;
                if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, gn, gk, gw, x, out.data(), err)) {
                    if (cn > 0) pool->wait();
                    return fail(fd, err);
                }
                if (g_queue_probe)
                    probe_gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu0).count();
            } else {
                std::fill(out.begin(), out.end(), 0.0f);
            }
            if (cn > 0) {
                const auto c0 = std::chrono::steady_clock::now();
                cpu_finish(*nf, (size_t) q.down_off, cn, cb, cw);
                for (int r = 0; r < n_embd; ++r) out[(size_t) r] += cpu_out[(size_t) r];
                ms_cpu += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                cpu_experts += (uint64_t) cn;
                ++cpu_routes;
            }
            ms_compute += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (g_queue_probe) {
                const double service_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                probe_service_ms += service_ms;
                if (sample_cpu) {
                    ++probe_sampled_routes;
                    probe_sampled_service_ms += service_ms;
                    if (pool) probe_work_ns += pool->work_ns() - work0;
                }
                if (pool) pool->profile_work(false);
                if (probe_routes % 10000 == 0) print_queue_probe("progress");
            }
            // HELPERD_SELFTEST=N: the first N GPU experts are also kept on the CPU; their first route compares the two
            for (int i = 0; i < gn && !check_keys.empty(); ++i) {
                size_t c = 0;
                while (c < check_keys.size() && check_keys[c] != gk[i]) ++c;
                if (c == check_keys.size()) continue;
                const kc::NativeFmt* cf = fmt(q.gu_type, q.d_type, err);
                if (cf == nullptr) return fail(fd, err);
                std::vector<float> g1((size_t) n_embd);
                const float one = 1.0f;
                if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, 1, &gk[i], &one, x, g1.data(), err))
                    return fail(fd, err);
                const uint8_t* b1[1] = {check_blob[c].data()};
                const uint32_t u1[1] = {check_up[c]};
                if (!pool) pool = std::make_unique<SpinPool>(std::max(0, g_cpu_threads - 1));
                cpu_start(*cf, (size_t) q.down_off, q.limit, 1, b1, u1, x);
                cpu_finish(*cf, (size_t) q.down_off, 1, b1, &one);
                double num = 0, den = 0, mx = 0;
                for (int r = 0; r < n_embd; ++r) {
                    const double d = (double) g1[(size_t) r] - cpu_out[(size_t) r];
                    num += d * d;
                    den += (double) g1[(size_t) r] * g1[(size_t) r];
                    mx = std::max(mx, std::fabs(d));
                }
                std::fprintf(stderr, "helperd: selftest key %d (types %d/%d): CPU vs GPU rel err %.2e, max abs %.2e\n",
                             gk[i], q.gu_type, q.d_type, std::sqrt(num / std::max(den, 1e-30)), mx);
                check_keys.erase(check_keys.begin() + (long) c);
                check_blob.erase(check_blob.begin() + (long) c);
                check_up.erase(check_up.begin() + (long) c);
            }
            computed += (uint64_t) q.ne;
            ++routes;
            return answer(fd, true, 0, out.data(), out.size() * sizeof(float));
        }
        case np::kCompute2: {   // kProto 2: x as the trunk's q8_1 / the CPU activation / FP32; FP16 answers on request
            np::Compute2Req q{};
            if (h.len < sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            const size_t nxf = (q.flags & np::kXf) ? (size_t) n_embd * sizeof(float) : 0;
            if (h.len != sizeof(q) + q.nxq + q.nxk + nxf || q.nxq > 65536 || q.nxk > kc::kNativeActBytes) return false;
            payload.resize((size_t) q.nxq + q.nxk + nxf);
            if (!payload.empty() && !recv_all(fd, payload.data(), payload.size())) return false;
            const auto t0 = std::chrono::steady_clock::now();
            const uint8_t* pxq = (q.flags & np::kXq) ? payload.data() : nullptr;
            const uint8_t* pxk = (q.flags & np::kXk) ? payload.data() + q.nxq : nullptr;
            const float* x = nxf ? (const float*) (payload.data() + q.nxq + q.nxk) : nullptr;
            if (pxq != nullptr && q.nxq != (uint32_t) (n_embd / 32 * 36)) return fail(fd, "kCompute2: q8_1 input of the wrong size");
            int gn = 0, cn = 0, gk[8];
            float gw[8], cw[8];
            const uint8_t* cb[8];
            uint32_t cu[8];
            const kc::NativeFmt* nf = nullptr;
            for (int i = 0; i < q.ne && i < 8; ++i) {
                const int key = q.keys[i];
                const int64_t ro = key >= 0 && (size_t) key < ram_off.size() ? ram_off[(size_t) key] : -1;
                if (ro < 0) { gk[gn] = key; gw[gn++] = q.w[i]; continue; }
                if (nf == nullptr && (nf = fmt(q.gu_type, q.d_type, err)) == nullptr) return fail(fd, err);
                cb[cn] = ram + ro;
                cu[cn] = ram_up[(size_t) key] == 0xffffffffu ? (uint32_t) nf->up_off : ram_up[(size_t) key];
                cw[cn++] = q.w[i];
            }
            if (gn > 0 && pxq == nullptr && x == nullptr) return fail(fd, "kCompute2: GPU experts without x");
            if (cn > 0 && pxk == nullptr && x == nullptr) return fail(fd, "kCompute2: CPU experts without x");
            if (cn > 0 && pxk != nullptr && q.nxk != (uint32_t) nf->act_bytes) return fail(fd, "kCompute2: CPU activation of the wrong size");
            const bool check = (q.flags & np::kCheck) != 0 && x != nullptr;
            // the CPU starts its gate/up rows (from the trunk's activation bytes when sent), the GPU does its share
            if (cn > 0) {
                if (pxk != nullptr) {
                    if (check) {
                        kc::native_quant_act(*nf, x, act.data());
                        ++c2_act_checks;
                        c2_act_bad += std::memcmp(act.data(), pxk, q.nxk) != 0;
                    }
                    std::memcpy(act.data(), pxk, q.nxk);
                    cpu_start_act(*nf, q.limit, cn, cb, cu);
                } else {
                    cpu_start(*nf, (size_t) q.down_off, q.limit, cn, cb, cu, x);
                }
            }
            if (gn > 0) {
                const bool okg = pxq != nullptr
                                     ? helper.compute_q8(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, gn, gk, gw, pxq,
                                                         out.data(), err)
                                     : helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, gn, gk, gw, x,
                                                      out.data(), err);
                if (!okg) {
                    if (cn > 0) pool->wait();
                    return fail(fd, err);
                }
                if (check && pxq != nullptr) {   // the same experts from the FP32 x: the bytes must agree
                    gout.resize((size_t) n_embd);
                    if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, gn, gk, gw, x, gout.data(), err)) {
                        if (cn > 0) pool->wait();
                        return fail(fd, err);
                    }
                    ++c2_gpu_checks;
                    c2_gpu_bad += std::memcmp(gout.data(), out.data(), (size_t) n_embd * sizeof(float)) != 0;
                }
            } else {
                std::fill(out.begin(), out.end(), 0.0f);
            }
            if (cn > 0) {
                const auto c0 = std::chrono::steady_clock::now();
                cpu_finish(*nf, (size_t) q.down_off, cn, cb, cw);
                for (int r = 0; r < n_embd; ++r) out[(size_t) r] += cpu_out[(size_t) r];
                ms_cpu += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                cpu_experts += (uint64_t) cn;
                ++cpu_routes;
            }
            ms_compute += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            computed += (uint64_t) q.ne;
            ++routes;
            ++c2_routes;
            // the FP16 answer: a power-of-two scale puts the largest |sum| near 2^14 (FP16's precision is relative,
            // the scale keeps small sums out of its subnormals); checked: the error it would have against the FP32 one
            float amax = 0.0f;
            if ((q.flags & np::kR16) || check)
                for (int r = 0; r < n_embd; ++r) amax = std::max(amax, std::fabs(out[(size_t) r]));
            const int ex = amax > 0.0f ? 14 - std::ilogb(amax) : 0;
            const float sc = std::ldexp(1.0f, ex), inv = std::ldexp(1.0f, -ex);
            if (check) {
                double emax = 0.0, esum = 0.0;
                for (int r = 0; r < n_embd; ++r) {
                    const float v = out[(size_t) r];
                    const double e = std::fabs((double) np::h2f(np::f2h(v * sc)) * inv - v);
                    emax = std::max(emax, e);
                    esum += e;
                }
                ++c2_r16_checks;
                c2_r16_max = std::max(c2_r16_max, amax > 0.0f ? emax / amax : 0.0);
                c2_r16_mean += amax > 0.0f ? esum / n_embd / amax : 0.0;
                if (c2_r16_checks % 500 == 0)
                    std::fprintf(stderr, "helperd: kCompute2 checks: GPU from q8_1 vs FP32 x %llu (%llu differ), CPU activation "
                                         "%llu (%llu differ); FP16 answer error / max|sum|: max %.2e, mean %.2e (%llu)\n",
                                 (unsigned long long) c2_gpu_checks, (unsigned long long) c2_gpu_bad,
                                 (unsigned long long) c2_act_checks, (unsigned long long) c2_act_bad, c2_r16_max,
                                 c2_r16_mean / (double) c2_r16_checks, (unsigned long long) c2_r16_checks);
            }
            if (q.flags & np::kR16) {
                ans16.resize((size_t) n_embd + 2);
                std::memcpy(ans16.data(), &inv, sizeof(inv));
                for (int r = 0; r < n_embd; ++r) ans16[(size_t) r + 2] = np::f2h(out[(size_t) r] * sc);
                return answer(fd, true, 0, ans16.data(), sizeof(float) + (size_t) n_embd * sizeof(uint16_t));
            }
            return answer(fd, true, 0, out.data(), out.size() * sizeof(float));
        }
        case np::kRows: {   // a prompt layer's rows: each (token, expert) unweighted, FP16 both ways
            np::RowsReq q{};
            const auto tr0 = std::chrono::steady_clock::now();
            if (h.len < sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            const size_t nx = (size_t) q.n_tok * (size_t) n_embd * sizeof(uint16_t);
            const size_t nr = (size_t) q.n_rows * sizeof(np::RowRef);
            if (q.n_tok < 1 || q.n_rows < 1 || h.len != sizeof(q) + nx + nr) return false;
            rx16.resize((size_t) q.n_tok * n_embd);
            rrefs.resize((size_t) q.n_rows);
            if (!recv_all(fd, rx16.data(), nx) || !recv_all(fd, rrefs.data(), nr)) return false;
            const auto t0 = std::chrono::steady_clock::now();
            ms_rows_recv += std::chrono::duration<double, std::milli>(t0 - tr0).count();
            const size_t E = (size_t) n_embd;
            rxf.resize((size_t) q.n_tok * E);
            {
                const _Float16* src = (const _Float16*) rx16.data();
                for (size_t i = 0; i < rxf.size(); ++i) rxf[i] = (float) src[i];
            }
            // GPU rows in one batch (compute_rows); RAM-tier rows grouped by expert on the CPU (multi-token kernels)
            std::vector<int> gk, gt, gidx;
            std::map<int, std::vector<int>> cpu_rows;   // key -> its rows
            for (int r = 0; r < q.n_rows; ++r) {
                const np::RowRef rr = rrefs[(size_t) r];
                if (rr.tok < 0 || rr.tok >= q.n_tok) return fail(fd, "prompt row outside its tokens");
                const int64_t ro = rr.key >= 0 && (size_t) rr.key < ram_off.size() ? ram_off[(size_t) rr.key] : -1;
                if (ro >= 0) cpu_rows[rr.key].push_back(r);
                else { gk.push_back(rr.key); gt.push_back(rr.tok); gidx.push_back(r); }
            }
            rowsf.resize((size_t) q.n_rows * E);
            if (!gk.empty()) {
                gout.resize(gk.size() * E);
                if (g_rows_serial) {   // HELPERD_ROWS_SERIAL=1: the old way, one compute() call a row (for comparison)
                    const float one = 1.0f;
                    for (size_t i = 0; i < gk.size(); ++i)
                        if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, 1, &gk[i], &one,
                                            rxf.data() + (size_t) gt[i] * E, gout.data() + i * E, err)) return fail(fd, err);
                } else if (!helper.compute_rows(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, q.n_tok, rxf.data(),
                                                (int) gk.size(), gk.data(), gt.data(), gout.data(), err)) return fail(fd, err);
                for (size_t i = 0; i < gidx.size(); ++i)
                    std::memcpy(rowsf.data() + (size_t) gidx[i] * E, gout.data() + i * E, E * sizeof(float));
                // HELPERD_ROWS_CHECK=N: the first N batches also go row by row through compute(), compared
                if (rows_checks < g_rows_check) {
                    ++rows_checks;
                    double num = 0, den = 0;
                    std::vector<float> one_out(E);
                    const float one = 1.0f;
                    for (size_t i = 0; i < gk.size(); ++i) {
                        if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, 1, &gk[i], &one,
                                            rxf.data() + (size_t) gt[i] * E, one_out.data(), err)) return fail(fd, err);
                        for (size_t j = 0; j < E; ++j) {
                            const double d = (double) one_out[j] - gout[i * E + j];
                            num += d * d;
                            den += (double) one_out[j] * one_out[j];
                        }
                    }
                    std::fprintf(stderr, "helperd: rows check %d: %zu GPU rows batched vs one by one: rel err %.2e\n",
                                 rows_checks, gk.size(), std::sqrt(num / std::max(den, 1e-30)));
                }
            }
            if (!cpu_rows.empty()) {
                const kc::NativeFmt* nf = fmt(q.gu_type, q.d_type, err);
                if (nf == nullptr) return fail(fd, err);
                if (!pool) pool = std::make_unique<SpinPool>(std::max(0, g_cpu_threads - 1));
                std::vector<uint8_t> acts((size_t) q.n_tok * kc::kNativeActBytes);
                std::vector<char> quant((size_t) q.n_tok, 0);
                for (auto& [key, rl] : cpu_rows) {
                    const int nt = (int) rl.size();
                    const uint8_t* blob = ram + ram_off[(size_t) key];
                    const uint32_t up = ram_up[(size_t) key] == 0xffffffffu ? (uint32_t) nf->up_off : ram_up[(size_t) key];
                    std::vector<const void*> a((size_t) nt);
                    std::vector<float> ffb((size_t) nt * n_ff);
                    std::vector<float*> ffp((size_t) nt), op((size_t) nt);
                    std::vector<uint8_t> hqb((size_t) nt * kc::kNativeHBytes);
                    std::vector<const void*> hp((size_t) nt);
                    for (int t = 0; t < nt; ++t) {
                        const int tk = rrefs[(size_t) rl[(size_t) t]].tok;
                        uint8_t* ab = acts.data() + (size_t) tk * kc::kNativeActBytes;
                        if (!quant[(size_t) tk]) { kc::native_quant_act(*nf, rxf.data() + (size_t) tk * E, ab); quant[(size_t) tk] = 1; }
                        a[(size_t) t] = ab;
                        ffp[(size_t) t] = ffb.data() + (size_t) t * n_ff;
                        op[(size_t) t] = rowsf.data() + (size_t) rl[(size_t) t] * E;
                        hp[(size_t) t] = hqb.data() + (size_t) t * kc::kNativeHBytes;
                    }
                    const int T1 = pool->size() * 2;
                    pool->run(T1, [&](int job) {
                        const int r0 = (int) ((int64_t) n_ff * job / T1), r1 = (int) ((int64_t) n_ff * (job + 1) / T1);
                        if (r1 > r0) kc::native_gu_rows_split(*nf, blob, blob + up, a.data(), nt, ffp.data(), r0, r1, q.limit);
                    });
                    for (int t = 0; t < nt; ++t) kc::native_quant_h(*nf, ffp[(size_t) t], hqb.data() + (size_t) t * kc::kNativeHBytes);
                    pool->run(T1, [&](int job) {
                        const int r0 = (int) ((int64_t) n_embd * job / T1), r1 = (int) ((int64_t) n_embd * (job + 1) / T1);
                        if (r1 > r0) kc::native_down_rows_split(*nf, blob + q.down_off, hp.data(), nt, op.data(), r0, r1);
                    });
                    cpu_experts += 1;
                }
            }
            ans16.resize((size_t) q.n_rows * E);
            {
                _Float16* dst = (_Float16*) ans16.data();
                for (size_t i = 0; i < rowsf.size(); ++i) dst[i] = (_Float16) rowsf[i];
            }
            const auto t1 = std::chrono::steady_clock::now();
            ms_rows += std::chrono::duration<double, std::milli>(t1 - t0).count();
            pf_rows += (uint64_t) q.n_rows;
            ++pf_layers;
            const bool ok = answer(fd, true, 0, ans16.data(), ans16.size() * sizeof(uint16_t));
            ms_rows_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            return ok;
        }
        case np::kTokRows: {   // a prompt layer's rows answered as each token's weighted sum (FP16 both ways)
            np::RowsReq q{};
            const auto tr0 = std::chrono::steady_clock::now();
            if (h.len < sizeof(q) || !recv_all(fd, &q, sizeof(q))) return false;
            const size_t nx = (size_t) q.n_tok * (size_t) n_embd * sizeof(uint16_t);
            const size_t ne = (size_t) q.n_rows * sizeof(np::TokEntry);
            if (q.n_tok < 1 || q.n_rows < 1 || h.len != sizeof(q) + nx + ne) return false;
            rx16.resize((size_t) q.n_tok * n_embd);
            tents.resize((size_t) q.n_rows);
            if (!recv_all(fd, rx16.data(), nx) || !recv_all(fd, tents.data(), ne)) return false;
            const auto t0 = std::chrono::steady_clock::now();
            ms_rows_recv += std::chrono::duration<double, std::milli>(t0 - tr0).count();
            const size_t E = (size_t) n_embd;
            rxf.resize((size_t) q.n_tok * E);
            {
                const _Float16* src = (const _Float16*) rx16.data();
                for (size_t i = 0; i < rxf.size(); ++i) rxf[i] = (float) src[i];
            }
            rowsf.assign((size_t) q.n_tok * E, 0.0f);   // the tokens' sums
            std::vector<int> gtok, gne, gkeys;
            std::vector<float> gw;
            std::map<int, std::vector<std::pair<int, float>>> cpu_rows;   // RAM-tier key -> (token, weight)
            for (size_t i = 0; i < tents.size();) {
                const int tk = tents[i].tok;
                if (tk < 0 || tk >= q.n_tok) return fail(fd, "prompt row outside its tokens");
                int n = 0;
                const size_t g0 = gkeys.size();
                gkeys.resize(g0 + 8, 0);
                gw.resize(g0 + 8, 0.0f);
                for (; i < tents.size() && tents[i].tok == tk; ++i) {
                    const int key = tents[i].key;
                    const int64_t ro = key >= 0 && (size_t) key < ram_off.size() ? ram_off[(size_t) key] : -1;
                    if (ro >= 0) { cpu_rows[key].push_back({tk, tents[i].w}); continue; }
                    if (n == 8) return fail(fd, "more than 8 experts a token");
                    gkeys[g0 + (size_t) n] = key;
                    gw[g0 + (size_t) n] = tents[i].w;
                    ++n;
                }
                if (n == 0) { gkeys.resize(g0); gw.resize(g0); continue; }
                gtok.push_back(tk);
                gne.push_back(n);
            }
            if (!gtok.empty()) {
                gout.resize(gtok.size() * E);
                // HELPERD_TOK_MMQ=0: the per-token kernels (an expert read once per row); default (10-27): MMQ, each
                // expert read once for all its rows - a 1.4K-token prompt's layer had ~0.8 rows a token here
                static const bool tok_mmq = [] { const char* v = std::getenv("HELPERD_TOK_MMQ"); return !v || std::atoi(v) != 0; }();
                const auto tc0 = std::chrono::steady_clock::now();
                if (!(tok_mmq ? helper.compute_tok_mmq(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, q.n_tok, rxf.data(),
                                                       (int) gtok.size(), gtok.data(), gne.data(), gkeys.data(), gw.data(),
                                                       gout.data(), err)
                              : helper.compute_tok(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, q.n_tok, rxf.data(),
                                                   (int) gtok.size(), gtok.data(), gne.data(), gkeys.data(), gw.data(),
                                                   gout.data(), err))) return fail(fd, err);
                ms_tok_gpu += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
                for (size_t gi = 0; gi < gtok.size(); ++gi)
                    std::memcpy(rowsf.data() + (size_t) gtok[gi] * E, gout.data() + gi * E, E * sizeof(float));
                if (rows_checks < g_rows_check) {   // HELPERD_ROWS_CHECK: the same sums through compute(), compared
                    ++rows_checks;
                    double num = 0, den = 0;
                    std::vector<float> o(E);
                    for (size_t gi = 0; gi < gtok.size(); ++gi) {
                        if (!helper.compute(q.gu_type, q.d_type, (size_t) q.down_off, q.limit, gne[gi], gkeys.data() + gi * 8,
                                            gw.data() + gi * 8, rxf.data() + (size_t) gtok[gi] * E, o.data(), err))
                            return fail(fd, err);
                        for (size_t j = 0; j < E; ++j) {
                            const double d = (double) o[j] - gout[gi * E + j];
                            num += d * d;
                            den += (double) o[j] * o[j];
                        }
                    }
                    std::fprintf(stderr, "helperd: tok check %d: %zu token sums batched vs compute(): rel err %.2e\n",
                                 rows_checks, gtok.size(), std::sqrt(num / std::max(den, 1e-30)));
                }
            }
            if (!cpu_rows.empty()) {   // the RAM tier: per expert over its tokens (multi-token kernels), weighted in
                const kc::NativeFmt* nf = fmt(q.gu_type, q.d_type, err);
                if (nf == nullptr) return fail(fd, err);
                if (!pool) pool = std::make_unique<SpinPool>(std::max(0, g_cpu_threads - 1));
                std::vector<uint8_t> acts((size_t) q.n_tok * kc::kNativeActBytes);
                std::vector<char> quant((size_t) q.n_tok, 0);
                for (auto& [key, tl] : cpu_rows) {
                    const int nt = (int) tl.size();
                    const uint8_t* blob = ram + ram_off[(size_t) key];
                    const uint32_t up = ram_up[(size_t) key] == 0xffffffffu ? (uint32_t) nf->up_off : ram_up[(size_t) key];
                    std::vector<const void*> a((size_t) nt), hp((size_t) nt);
                    std::vector<float> ffb((size_t) nt * n_ff), ob((size_t) nt * E);
                    std::vector<float*> ffp((size_t) nt), op((size_t) nt);
                    std::vector<uint8_t> hqb((size_t) nt * kc::kNativeHBytes);
                    for (int t = 0; t < nt; ++t) {
                        const int tk = tl[(size_t) t].first;
                        uint8_t* ab = acts.data() + (size_t) tk * kc::kNativeActBytes;
                        if (!quant[(size_t) tk]) { kc::native_quant_act(*nf, rxf.data() + (size_t) tk * E, ab); quant[(size_t) tk] = 1; }
                        a[(size_t) t] = ab;
                        ffp[(size_t) t] = ffb.data() + (size_t) t * n_ff;
                        op[(size_t) t] = ob.data() + (size_t) t * E;
                        hp[(size_t) t] = hqb.data() + (size_t) t * kc::kNativeHBytes;
                    }
                    const int T1 = pool->size() * 2;
                    pool->run(T1, [&](int job) {
                        const int r0 = (int) ((int64_t) n_ff * job / T1), r1 = (int) ((int64_t) n_ff * (job + 1) / T1);
                        if (r1 > r0) kc::native_gu_rows_split(*nf, blob, blob + up, a.data(), nt, ffp.data(), r0, r1, q.limit);
                    });
                    for (int t = 0; t < nt; ++t) kc::native_quant_h(*nf, ffp[(size_t) t], hqb.data() + (size_t) t * kc::kNativeHBytes);
                    pool->run(T1, [&](int job) {
                        const int r0 = (int) ((int64_t) n_embd * job / T1), r1 = (int) ((int64_t) n_embd * (job + 1) / T1);
                        if (r1 > r0) kc::native_down_rows_split(*nf, blob + q.down_off, hp.data(), nt, op.data(), r0, r1);
                    });
                    for (int t = 0; t < nt; ++t) {
                        float* dst = rowsf.data() + (size_t) tl[(size_t) t].first * E;
                        const float w = tl[(size_t) t].second;
                        for (size_t j = 0; j < E; ++j) dst[j] += w * ob[(size_t) t * E + j];
                    }
                    cpu_experts += 1;
                }
            }
            ans16.resize((size_t) q.n_tok * E);
            {
                _Float16* dst = (_Float16*) ans16.data();
                for (size_t i = 0; i < rowsf.size(); ++i) dst[i] = (_Float16) rowsf[i];
            }
            const auto t1 = std::chrono::steady_clock::now();
            ms_rows += std::chrono::duration<double, std::milli>(t1 - t0).count();
            pf_rows += (uint64_t) q.n_rows;
            ++pf_layers;
            const bool ok = answer(fd, true, 0, ans16.data(), ans16.size() * sizeof(uint16_t));
            ms_rows_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            return ok;
        }
        case np::kClose:
            return false;
        default:
            return fail(fd, "unknown request " + std::to_string(h.op));
        }
    }
};
}   // namespace

int main(int argc, char** argv) {
    int port = 18800;
    std::string bind_addr = "0.0.0.0", model_dir, dial;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--bind" && i + 1 < argc) bind_addr = argv[++i];
        else if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--dial" && i + 1 < argc) dial = argv[++i];
        else if (a == "--ram-gib" && i + 1 < argc) g_ram_bytes = (size_t) (std::atof(argv[++i]) * 1073741824.0);
        else if (a == "--cpu-threads" && i + 1 < argc) g_cpu_threads = std::atoi(argv[++i]);
        else if (a == "--ram-cold") g_ram_cold = true;
        else {
            std::fprintf(stderr, "usage: maya-helperd (--port N [--bind ADDR] | --dial HOST:PORT) [--model-dir DIR] "
                                 "[--ram-gib N [--cpu-threads T]]\n");
            return 2;
        }
    }
    if (g_cpu_threads <= 0) g_cpu_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    if (const char* st = getenv("HELPERD_SELFTEST")) g_selftest = std::atoi(st);
    if (const char* rc = getenv("HELPERD_ROWS_CHECK")) g_rows_check = std::atoi(rc);
    if (const char* rs = getenv("HELPERD_ROWS_SERIAL")) g_rows_serial = std::atoi(rs) != 0;
    if (const char* qp = getenv("HELPERD_QUEUE_PROBE")) g_queue_probe = std::atoi(qp) != 0;
    if (g_queue_probe) std::fprintf(stderr, "helperd: queue probe enabled (no batching, bounded nonblocking peek, CPU job clocks)\n");
    if (g_ram_bytes > 0 && !kc::native_experts_available()) {
        std::fprintf(stderr, "helperd: --ram-gib needs the ggml-cpu path (STRATA_NATIVE_EXPERTS)\n");
        return 2;
    }
    if (!model_dir.empty() && model_dir[0] == '~')
        if (const char* home = getenv("HOME")) model_dir = std::string(home) + model_dir.substr(1);
    // as RemoteExperts::preflight (which refuses device 0, the trunk's): spin while waiting for the GPU - a route's
    // answer is on the trunk's critical path - and map host memory, before the context exists
    if (const char* spin = getenv("STRATA_REMOTE_SPIN"); !(spin && spin[0] == '0'))
        cudaInitDevice(0, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSetDevice(0) != cudaSuccess || cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::fprintf(stderr, "helperd: no usable CUDA device (pick one with CUDA_VISIBLE_DEVICES)\n");
        return 1;
    }
    const double free_gib = (double) free_bytes / 1073741824.0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const int one = 1;
    const auto session = [&](int fd, const char* who) {
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::fprintf(stderr, "helperd: trunk %s connected\n", who);
        {
            Session s;
            s.model_dir = model_dir;
            np::Hdr h{};
            // HELPERD_SPIN_US=<us> (default 0: off): after each answer, poll the socket without blocking for up to
            // that long before the blocking read - a decode's next request (~0.5-1.5 ms later) is then seen as it
            // lands instead of after a wake-up of this thread
            static const long spin_us = [] {
                const char* v = std::getenv("HELPERD_SPIN_US");
                return v ? std::max(0L, std::atol(v)) : 0L;
            }();
            const auto spin = [&] {
                if (spin_us <= 0) return;
                const auto t0 = std::chrono::steady_clock::now();
                char c;
                for (;;) {
                    const ssize_t k = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
                    if (k >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) return;
                    if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us)) return;
                    __builtin_ia32_pause();
                }
            };
            while ((spin(), recv_all(fd, &h, sizeof(h))) && h.magic == np::kMagic && s.serve(fd, h)) {}
            s.print_queue_probe("final");
            std::fprintf(stderr, "helperd: trunk %s gone: %llu experts held (%llu in RAM), %llu experts in %llu routes, "
                                 "%.1f ms computing; CPU %llu experts in %llu routes, %.1f ms past the GPU's share; prompt rows %llu in %llu layers: recv %.1f ms, compute %.1f ms (GPU %.1f), send %.1f ms\n",
                         who, (unsigned long long) s.inserted, (unsigned long long) s.ram_held,
                         (unsigned long long) s.computed, (unsigned long long) s.routes, s.ms_compute,
                         (unsigned long long) s.cpu_experts, (unsigned long long) s.cpu_routes, s.ms_cpu,
                         (unsigned long long) s.pf_rows, (unsigned long long) s.pf_layers, s.ms_rows_recv, s.ms_rows,
                         s.ms_tok_gpu, s.ms_rows_send);
        }   // the session frees this GPU's cache before the next trunk
        ::close(fd);
    };
    if (!dial.empty()) {
        const size_t colon = dial.rfind(':');
        if (colon == std::string::npos) { std::fprintf(stderr, "helperd: --dial takes HOST:PORT\n"); return 2; }
        const std::string host = dial.substr(0, colon), port_s = dial.substr(colon + 1);
        std::fprintf(stderr, "helperd: %s (%.1f GiB free) dialing %s\n", prop.name, free_gib, dial.c_str());
        for (;;) {   // the trunk listens only while it starts: try every second
            addrinfo hints{}, *res = nullptr;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            int fd = -1;
            if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) == 0) {
                fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
                if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) != 0) { ::close(fd); fd = -1; }
                freeaddrinfo(res);
            }
            if (fd < 0) { ::sleep(1); continue; }
            session(fd, dial.c_str());
        }
    }
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr) != 1 ||
        ::bind(ls, (sockaddr*) &addr, sizeof(addr)) != 0 || ::listen(ls, 1) != 0) {
        std::fprintf(stderr, "helperd: cannot listen on %s:%d: %s\n", bind_addr.c_str(), port, std::strerror(errno));
        return 1;
    }
    std::fprintf(stderr, "helperd: %s (%.1f GiB free) listening on %s:%d\n", prop.name, free_gib, bind_addr.c_str(), port);
    for (;;) {
        sockaddr_in peer{};
        socklen_t pl = sizeof(peer);
        const int fd = ::accept(ls, (sockaddr*) &peer, &pl);
        if (fd < 0) continue;
        char who[64] = {};
        inet_ntop(AF_INET, &peer.sin_addr, who, sizeof(who));
        session(fd, who);
    }
}
