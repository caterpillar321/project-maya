#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The wire format between a trunk host and maya-helperd (another machine's GPU holding a static share of the GLM
/// experts).  Both ends are the same build on x86-64 Linux: little-endian, structs as laid out here.
namespace netproto {
constexpr uint32_t kMagic = 0x3148'4e4du;   // "MNH1"
enum Op : uint32_t { kOpen = 1, kReserve = 2, kInsert = 3, kCompute = 4, kClose = 5, kInsertFile = 6, kRows = 7, kTokRows = 8, kHash = 9,
                    kCompute2 = 10 };
/// the protocol a helperd speaks: its kOpen reply's pad (0 from the older ones: kCompute only)
constexpr uint32_t kProto = 2;
struct Hdr { uint32_t magic, op; uint64_t len; };   // len payload bytes follow
struct OpenReq { int32_t n_embd, n_ff; };   // + the pack's shard file names, '\n'-separated (for kInsertFile)
struct ReserveReq { int32_t keys; uint32_t pad; uint64_t bytes; };
struct InsertReq { int32_t key; uint32_t pad; uint64_t bytes, stride; };   // + bytes of the expert's blob
/// the helper reads the blob from its own copy of the pack: part j is len[j] bytes at off[j] of shard[j], put at dst[j]
struct InsertFileReq { int32_t key; uint32_t shard[3]; uint64_t off[3], len[3], dst[3], bytes, stride; };
struct ComputeReq {
    int32_t gu_type, d_type;
    uint64_t down_off;
    float limit;
    int32_t ne;
    int32_t keys[8];
    float w[8];
};   // + n_embd floats of x
/// kCompute2 (kProto 2, 10-26): the route's input as the trunk already has it, smaller than kCompute's FP32 x:
/// the q8_1 blocks its GPU experts read (kXq; the helper's own quantize of that x gives the same bytes), the CPU
/// lane's activation (kXk: the layer's vec_dot_type rows, the bytes native_quant_act makes), and/or the FP32 x (kXf);
/// kR16: the answer as FP16 of (sum x scale) after a float scale (a power of two); kCheck (with kXf): the helper also
/// computes from the FP32 x and counts the answers / activations that differ, and the FP16 answer's error
struct Compute2Req {
    int32_t gu_type, d_type;
    uint64_t down_off;
    float limit;
    int32_t ne;
    int32_t keys[8];
    float w[8];
    uint32_t flags, nxq, nxk, pad;
};   // + nxq bytes of q8_1, nxk bytes of the CPU activation, (kXf) n_embd floats
enum C2 : uint32_t { kXq = 1, kXk = 2, kXf = 4, kR16 = 8, kCheck = 16 };
inline float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do { ++ee; mm <<= 1; } while ((mm & 0x400u) == 0);
            u = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
inline uint16_t f2h(float f) {   // round to nearest even; |f| past 65504 saturates (the scale keeps it below)
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0u));
    if (e >= 31) return (uint16_t) (s | 0x7bffu);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t hm = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1u))) ++hm;
        return (uint16_t) (s | hm);
    }
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1fffu;
    uint32_t r = (uint32_t) (s | ((uint32_t) e << 10) | hm);
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) ++r;   // (a carry into the exponent is the right rounding)
    if ((r & 0x7c00u) == 0x7c00u) r = s | 0x7bffu;
    return (uint16_t) r;
}
/// a prompt layer's rows for this helper's experts (short prompts, STRATA_GLM_NET_PREFILL_MAX): each row is one
/// (token, expert) pair, answered unweighted - the trunk's combine weighs it with the other rows
struct RowsReq {
    int32_t gu_type, d_type;
    uint64_t down_off;
    float limit;
    int32_t n_tok, n_rows;
    uint32_t pad;
};   // + n_tok x n_embd FP16 (the tokens' inputs) + n_rows x RowRef; the answer: n_rows x n_embd FP16
struct RowRef { int32_t tok, key; };
/// kTokRows: the same rows answered as each token's weighted sum over its rows here (one n_embd answer a token instead
/// of one a row): RowsReq with n_rows = entries, + n_tok x n_embd FP16 + n_rows x TokEntry grouped by token (ascending);
/// the answer: n_tok x n_embd FP16 (a token without entries: zeros)
struct TokEntry { int32_t tok, key; float w; };
struct Reply { int32_t ok; uint32_t pad; uint64_t value, len; };   // + len bytes: the error text, or n_embd floats
/// kHash: n x HashRef (the request's len / sizeof) - each a RAM-tier expert's three parts, len[j] bytes at dst[j] of its
/// blob; the answer: n uint64_t blob_hash values (0: the key is not in the RAM tier).  The trunk checks its copies with it
struct HashRef { int32_t key; uint32_t pad; uint64_t dst[3], len[3]; };
/// the copies' check hash: four multiply-xor lanes over 8-byte words (a tail byte-wise), then the parts chained
inline uint64_t blob_hash(const uint8_t* p, const uint64_t dst[3], const uint64_t len[3]) {
    constexpr uint64_t kM = 0x9e3779b97f4a7c15ull;
    uint64_t h = 0x243f6a8885a308d3ull;
    for (int j = 0; j < 3; ++j) {
        const uint8_t* a = p + dst[j];
        const uint64_t n = len[j], nw = n / 32;
        uint64_t l[4] = {1, 2, 3, 4};
        for (uint64_t i = 0; i < nw; ++i)
            for (int k = 0; k < 4; ++k) {
                uint64_t w;
                __builtin_memcpy(&w, a + i * 32 + k * 8, 8);
                l[k] = (l[k] ^ w) * kM;
                l[k] ^= l[k] >> 29;
            }
        for (uint64_t i = nw * 32; i < n; ++i) l[i & 3] = (l[i & 3] ^ a[i]) * kM;
        for (int k = 0; k < 4; ++k) h = (h ^ l[k] ^ (n + (uint64_t) k)) * kM, h ^= h >> 31;
    }
    return h == 0 ? 1 : h;
}
}   // namespace netproto

/// The trunk side of one maya-helperd: owns experts like GlmExpertHelper, but over TCP.  A route is begin() (the
/// request goes out at once) then finish() (waits for the answer and adds it), so the local helper and the CPU lane
/// work while it is on the wire.  Static: no swaps after the start.
class GlmNetHelper {
public:
    GlmNetHelper() = default;
    ~GlmNetHelper() { close(); }
    GlmNetHelper(const GlmNetHelper&) = delete;
    GlmNetHelper& operator=(const GlmNetHelper&) = delete;

    bool connect(const std::string& host_port, std::string& err);
    bool open(int n_embd, int n_ff, const std::vector<std::string>& shards, std::string& err);
    bool reserve(int keys, size_t requested_bytes, std::string& err);
    /// pipelined: the replies are read by flush()
    bool insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err);
    /// pipelined too: the helper reads the blob from its own copy of the model files
    bool insert_file(int key, const uint32_t shard[3], const uint64_t off[3], const uint64_t len[3],
                     const uint64_t dst[3], size_t bytes, size_t stride, std::string& err);
    /// into the helper's RAM tier (maya-helperd --ram-cold: its CPU computes it) - read from its own copy of the pack
    bool insert_file_ram(int key, const uint32_t shard[3], const uint64_t off[3], const uint64_t len[3],
                         const uint64_t dst[3], size_t bytes, size_t stride, std::string& err);
    bool flush(std::string& err);
    /// kHash: the helper's hashes of n of its RAM-tier experts (netproto::blob_hash; 0 for a key it does not hold)
    bool hash_ram(int n, const netproto::HashRef* refs, uint64_t* out, std::string& err);
    bool begin(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys, const float* weights,
               const float* x, std::string& err);
    /// kCompute2 (proto() >= 2): flags netproto::C2, xq / xk / x as the flags say
    bool begin2(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys, const float* weights,
                uint32_t flags, const void* xq, size_t nxq, const void* xk, size_t nxk, const float* x, std::string& err);
    /// the helperd's protocol (its kOpen reply), and whether it holds key in its RAM tier (its CPU computes it)
    uint32_t proto() const { return proto_; }
    bool owns_ram(int key) const { return !owned_ram_.empty() && owned_ram_[(size_t) key] != 0; }
    /// adds the answer into out (n_embd floats)
    bool finish(float* out, std::string& err);
    /// a prompt layer's rows (x: n_tok x n_embd FP16, rows: n_rows refs into it); rows_finish reads the n_rows x n_embd
    /// FP16 answer.  Not while a route is pending
    bool rows_begin(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const uint16_t* x, int n_rows,
                    const netproto::RowRef* rows, std::string& err);
    bool rows_finish(uint16_t* out, int n_rows, std::string& err);
    /// kTokRows: the token-summed variant (entries grouped by token); rows_finish(out, n_tok) reads n_tok answers
    bool tok_begin(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const uint16_t* x, int n_ent,
                   const netproto::TokEntry* ent, std::string& err);
    uint64_t pf_rows = 0, pf_layers = 0;
    double pf_ms_send = 0, pf_ms_wait = 0;   ///< rows_begin's sending, rows_finish's waiting for the answer
    void close();

    bool owns(int key) const { return !owned_.empty() && owned_[(size_t) key] != 0; }
    size_t available() const { return capacity_ - used_; }
    size_t ram_available() const { return ram_capacity_ - ram_used_; }
    size_t ram_used() const { return ram_used_; }
    int resident_ram() const { return resident_ram_; }
    size_t used() const { return used_; }
    int resident() const { return resident_; }
    int missing() const { return missing_; }   ///< pack shards the helper has no copy of (0: it reads them itself)
    const std::string& name() const { return name_; }

    uint64_t experts = 0, routes = 0;
    double ms_wait = 0;   ///< time finish() spent waiting for the answer

private:
    bool accept_on(int port, std::string& err);
    bool send_all(const void* p, size_t n, std::string& err);
    bool recv_all(void* p, size_t n, std::string& err);
    bool request(uint32_t op, const void* a, size_t na, const void* b, size_t nb, std::string& err);
    bool reply(netproto::Reply& r, std::string& err);
    int fd_ = -1;
    std::string name_;
    std::vector<uint8_t> owned_;
    size_t capacity_ = 0, used_ = 0, ram_capacity_ = 0, ram_used_ = 0;
    int resident_ram_ = 0;
    int n_embd_ = 0, resident_ = 0, unread_ = 0, missing_ = -1;
    bool pending_ = false;            // a prompt layer's rows on the wire
    int routes_pending_ = 0;           // decode routes on the wire (begin without finish yet, in order)
    static constexpr int kMaxRoutes = 16;
    std::vector<float> rx_;
    std::vector<uint16_t> rx16_;
    uint32_t proto_ = 0;
    std::vector<uint8_t> owned_ram_;
    bool r16_[kMaxRoutes] = {};   // the routes on the wire, in order: FP16 answers
    int r16_head_ = 0;
};

}   // namespace strata::core
