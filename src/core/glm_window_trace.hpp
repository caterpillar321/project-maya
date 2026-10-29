// Bounded, opt-in stream timeline for single-trunk T1 pipeline / batched T2/T3 windows.
// Events bracket consecutive work on one stream. These are elapsed intervals,
// including host submission gaps and waits, NOT isolated kernel/PCIe service times.
// The caller collects only after its existing window synchronization, and prints
// at request end so logging cannot delay the next layer's helper requests.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>

namespace strata::core::glmfast {

class WindowTrace {
    struct Event {
        cudaEvent_t event = nullptr;
        const char* stage = nullptr;
        int layer = -1, row = -1;
    };
    struct Segment {
        const char* stage;
        int layer, row;
        float ms;
    };
    struct Sample {
        uint64_t window = 0;
        int64_t p0 = 0;
        int T = 0, attempt = 0, layer_begin = 0, layer_end = 0;
        const char* outcome = "aborted";
        bool complete = false;
        float span_ms = 0;
        double segments_ms = 0;
        std::vector<Segment> segments;
    };
    static constexpr size_t kMaxEvents = 4096;
    int device_;
    uint64_t session_, request_ = 1, serial_ = 0;
    int every_, limit_;
    uint64_t seen_ = 0, sampled_ = 0, unsupported_ = 0, errors_ = 0;
    bool selected_ = false, active_ = false, failed_ = false;
    size_t used_ = 0;
    std::vector<Event> events_;
    std::vector<Sample> samples_;
    Sample current_;

    static int bounded_env(const char* name, int fallback, int maximum) {
        const char* value = std::getenv(name);
        return value == nullptr ? fallback : std::max(1, std::min(maximum, std::atoi(value)));
    }
    void abort_active() {
        if (!active_) return;
        current_.outcome = "aborted";
        current_.complete = false;
        samples_.push_back(std::move(current_));
        active_ = false;
        ++errors_;
    }
    void identity(char (&prefix)[512], const char* kind, const Sample* sample = nullptr) const {
        const int n = std::snprintf(prefix, sizeof(prefix),
                                   "glm window trace: {\"schema\":1,\"kind\":\"%s\",\"session\":\"%llu\","
                                   "\"device\":%d,\"request\":%llu", kind,
                                   (unsigned long long) session_, device_, (unsigned long long) request_);
        if (sample != nullptr && n > 0 && (size_t) n < sizeof(prefix))
            std::snprintf(prefix + n, sizeof(prefix) - (size_t) n,
                          ",\"window\":%llu,\"p0\":%lld,\"T\":%d,\"attempt\":%d",
                          (unsigned long long) sample->window, (long long) sample->p0, sample->T, sample->attempt);
    }

public:
    static bool enabled() {
        static const bool value = [] {
            const char* v = std::getenv("STRATA_GLM_WIN_TRACE");
            return v != nullptr && std::atoi(v) != 0;
        }();
        return value;
    }
    explicit WindowTrace(int device) : device_(device),
        session_((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count()),
        every_(bounded_env("STRATA_GLM_WIN_TRACE_EVERY", 16, 4096)),
        limit_(bounded_env("STRATA_GLM_WIN_TRACE_LIMIT", 8, 64)) {}
    WindowTrace(const WindowTrace&) = delete;
    WindowTrace& operator=(const WindowTrace&) = delete;
    ~WindowTrace() {
        if (events_.empty()) return;
        int previous = device_;
        cudaGetDevice(&previous);
        cudaSetDevice(device_);
        for (const Event& e : events_) if (e.event) cudaEventDestroy(e.event);
        cudaSetDevice(previous);
    }

    void skip_window() { abort_active(); selected_ = false; }

    // Called once per ordinary window(), after its existing boundary work. The
    // eligibility gate excludes intra-window multi-GPU pipelining, stagger and retry execution.
    void begin_window(int64_t p0, int T, int layer_begin, int layer_end, bool supported) {
        abort_active();
        ++serial_;
        ++seen_;
        selected_ = false;
        if (!supported) {
            if (++unsupported_ == 1)
                std::fprintf(stderr, "glm window trace unavailable: requires one trunk, private T1/T2/T3 scratch, "
                                     "batched experts and WIN_STAGGER=0; unsupported windows are counted, "
                                     "not attributed\n");
            return;
        }
        if ((seen_ - 1) % (uint64_t) every_ != 0 || sampled_ >= (uint64_t) limit_) return;
        ++sampled_;
        selected_ = true;
        active_ = true;   // A failure while preparing the embedding is an aborted sample too.
        current_ = Sample{};
        current_.window = serial_;
        current_.p0 = p0;
        current_.T = T;
        current_.layer_begin = layer_begin;
        current_.layer_end = layer_end;
        // Create outside the timed attempt, once, then reuse. Forty marks per
        // layer accommodates both attention variants plus all T3 route phases.
        const size_t capacity = std::min(kMaxEvents, (size_t) std::max(0, layer_end - layer_begin) * 40 + 32);
        failed_ = false;
        while (events_.size() < capacity) {
            cudaEvent_t event = nullptr;
            if (cudaEventCreate(&event) != cudaSuccess) { failed_ = true; break; }
            events_.push_back(Event{event, nullptr, -1, -1});
        }
    }

    void begin_attempt(int attempt, bool per_token, cudaStream_t stream) {
        if (!selected_) return;
        current_.attempt = attempt;
        used_ = 0;
        if (per_token) {
            current_.outcome = "unsupported_retry";
            current_.complete = false;
            current_.segments.clear();
            current_.span_ms = 0;
            current_.segments_ms = 0;
            samples_.push_back(current_);
            ++unsupported_;
            selected_ = false;
            return;
        }
        active_ = true;
        mark(-1, "start", -1, stream);
    }

    void mark(int layer, const char* stage, int row, cudaStream_t stream) {
        if (!active_ || failed_) return;
        if (used_ >= events_.size()) { failed_ = true; return; }
        Event& e = events_[used_];
        e.layer = layer;
        e.stage = stage;
        e.row = row;
        if (cudaEventRecord(e.event, stream) != cudaSuccess) { failed_ = true; return; }
        ++used_;
    }

    // Caller must have completed its pre-existing stream synchronization. This
    // function never synchronizes; not-ready/error samples fail closed.
    void collect(bool scratch_retry, bool execution_ok) {
        if (!active_) return;
        failed_ = failed_ || !execution_ok;
        current_.outcome = scratch_retry ? "scratch_retry" : "completed";
        current_.segments.clear();
        current_.segments_ms = 0;
        current_.span_ms = 0;
        if (!failed_ && used_ > 1 &&
            cudaEventElapsedTime(&current_.span_ms, events_[0].event, events_[used_ - 1].event) == cudaSuccess) {
            current_.segments.reserve(used_ - 1);
            for (size_t i = 1; i < used_; ++i) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, events_[i - 1].event, events_[i].event) != cudaSuccess) {
                    failed_ = true;
                    break;
                }
                current_.segments.push_back(Segment{events_[i].stage, events_[i].layer, events_[i].row, ms});
                current_.segments_ms += ms;
            }
        } else failed_ = true;
        current_.complete = !failed_;
        if (failed_) { current_.outcome = "trace_error"; ++errors_; }
        samples_.push_back(current_);
        active_ = false;
    }

    // Called from the normal end-of-request report. Its counter is not a global
    // API request ID: it identifies this model's successive report periods.
    void report() {
        abort_active();
        char prefix[512];
        for (const Sample& sample : samples_) {
            for (size_t i = 0; i < sample.segments.size(); ++i) {
                const Segment& segment = sample.segments[i];
                identity(prefix, "segment", &sample);
                std::fprintf(stderr, "%s,\"index\":%zu,\"layer\":%d,\"stage\":\"%s\",\"row\":%d,\"ms\":%.6f}\n",
                             prefix, i, segment.layer, segment.stage, segment.row, segment.ms);
            }
            identity(prefix, "window", &sample);
            std::fprintf(stderr, "%s,\"layer_begin\":%d,\"layer_end\":%d,\"segments\":%zu,\"span_ms\":%.6f,"
                                 "\"segments_ms\":%.6f,\"complete\":%s,\"outcome\":\"%s\"}\n",
                         prefix, sample.layer_begin, sample.layer_end, sample.segments.size(), sample.span_ms,
                         sample.segments_ms, sample.complete ? "true" : "false", sample.outcome);
        }
        identity(prefix, "report");
        std::fprintf(stderr, "%s,\"sampled\":%llu,\"every\":%d,\"limit\":%d,\"unsupported\":%llu,"
                             "\"errors\":%llu,\"windows_seen\":%llu}\n",
                     prefix, (unsigned long long) sampled_, every_, limit_, (unsigned long long) unsupported_,
                     (unsigned long long) errors_, (unsigned long long) seen_);
        samples_.clear();
        seen_ = sampled_ = unsupported_ = errors_ = 0;
        selected_ = false;
        ++request_;
    }
};

}  // namespace strata::core::glmfast
