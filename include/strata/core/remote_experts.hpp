#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// GLM helper-owned experts. UMA aliases pinned RAM; discrete helpers cache in
/// their own VRAM. Neither backend changes the legacy Qwen RemoteExperts path.
class GlmExpertHelper {
public:
    GlmExpertHelper() = default;
    ~GlmExpertHelper();
    GlmExpertHelper(const GlmExpertHelper&) = delete;
    GlmExpertHelper& operator=(const GlmExpertHelper&) = delete;
    bool open(int device, int n_embd, int n_ff,
              const std::vector<std::pair<void*, size_t>>& regions, std::string& err);
    bool reserve(int keys, size_t requested_bytes, std::string& err);
    bool insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err);
    bool replace(int out, int in, const uint8_t* host, std::string& err);
    struct Swap { uint32_t gain; int in, out; };
    std::vector<Swap> swaps(const std::vector<uint32_t>& usage, const std::vector<int>& primary,
                            int experts_per_layer, int max_swaps) const;
    const std::vector<uint8_t>& owners() const { return owned_; }
    bool owns(int key) const { return !owned_.empty() && owned_[(size_t) key] != 0; }
    bool integrated() const { return integrated_; }
    size_t available() const { return capacity_ - used_; }
    size_t used() const { return used_; }
    bool compute(int gu_type, int d_type, size_t down_off, float limit, int ne,
                 const int* keys, const float* weights, const float* x,
                 float* out, std::string& err);
    /// compute() in two halves: the launches (out is written by the stream), then the wait - the caller works between
    bool compute_begin(int gu_type, int d_type, size_t down_off, float limit, int ne,
                       const int* keys, const float* weights, const float* x,
                       float* out, std::string& err);
    bool compute_finish(std::string& err);
    /// compute() from x's q8_1 blocks (n_embd / 32 of them, as quantize_q8_1 makes them) instead of x
    bool compute_q8(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys,
                    const float* weights, const void* xq, float* out, std::string& err);
    /// A prompt's rows (maya-helperd kRows): row r is expert keys[r] on token tok[r] of x (n_tok x n_embd floats),
    /// unweighted, into out (n_rows x n_embd floats).  One upload, the rows' kernels back to back, one download - the
    /// same kernels and arithmetic as compute() with one expert of weight 1.
    bool compute_rows(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const float* x, int n_rows,
                      const int* keys, const int* tok, float* out, std::string& err);
    /// Tokens' weighted sums: group g is token tok[g] of x with ne[g] (<= 8) experts keys[8 g ..] at weights w[8 g ..],
    /// its sum into out row g (n_groups x n_embd floats) - compute()'s kernels, one upload/download for all of them.
    bool compute_tok(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const float* x, int n_groups,
                     const int* tok, const int* ne, const int* keys, const float* w, float* out, std::string& err);
    /// compute_tok's sums through the prompt path's MMQ kernels (10-27): the groups' rows sorted by expert, each expert
    /// read once for all its rows (compute_tok reads it once per row), then each token's weighted sum.  q8_1 activations
    /// as in the trunk's own prompt path; the same answer up to that rounding.
    bool compute_tok_mmq(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const float* x, int n_groups,
                         const int* tok, const int* ne, const int* keys, const float* w, float* out, std::string& err);
    void close();
    int device() const { return device_; }

private:
    struct Region { const uint8_t* host; uint8_t* mapped; size_t bytes; };
    std::vector<Region> regions_;
    struct Slot { uint8_t* host = nullptr; uint8_t* device = nullptr; size_t bytes = 0; };
    std::vector<Slot> slots_;
    std::vector<uint8_t> owned_;
    bool integrated_ = false;
    uint8_t* pool_ = nullptr;
    size_t capacity_ = 0, used_ = 0;
    int device_ = -1, n_embd_ = 0, n_ff_ = 0;
    cudaStream_t stream_ = nullptr;
    uint8_t* arena_ = nullptr;
    float *x_ = nullptr, *out_ = nullptr;
    void *xq_ = nullptr, *hq_ = nullptr;
    strata::kernels::glmf::MoeDev plan_;
    float *rx_ = nullptr, *rout_ = nullptr, *rone_ = nullptr;   // compute_rows' device buffers (grown on demand)
    unsigned long long* rptr_ = nullptr;
    float* rw_ = nullptr;
    size_t rx_n_ = 0, rrows_n_ = 0, rents_n_ = 0;
    bool rows_room(size_t rows, size_t ents, std::string& err);
    // compute_tok_mmq's device buffers (grown on demand) and its MMQ launch context
    void* mq_ = nullptr;
    int *mrt_ = nullptr, *mbnd_ = nullptr, *miota_ = nullptr, *mgofs_ = nullptr, *mgrow_ = nullptr;
    float *mgw_ = nullptr, *mgu_ = nullptr, *mout_ = nullptr;
    void *mxq_ = nullptr, *mhq_ = nullptr;
    size_t m_rows_n_ = 0, m_groups_n_ = 0, m_bnd_n_ = 0, m_xq_n_ = 0, m_hq_n_ = 0;
    bool compute_begin_x(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys,
                         const float* weights, const float* x, const void* xq_host, float* out, std::string& err);
};

/// A static, profile-filled expert tier on another CUDA device. CUDA0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err);
    void close();

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

private:
    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr;
    float* h_x_ = nullptr;
    float* h_out_ = nullptr;
    void* h_meta_ = nullptr;
    float* d_x_ = nullptr;
    float* z_x_ = nullptr;     ///< h_x_ as the helper GPU sees it (zero-copy: no input copy per layer)
    float* z_out_ = nullptr;   ///< h_out_ as the helper GPU sees it (zero-copy: no result copy)
    bool zero_copy_ = false;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    void* d_meta_ = nullptr;  ///< one contiguous upload of grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;
};

} // namespace strata::core
