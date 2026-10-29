#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core::stageproto {
// Experimental homogeneous x86-64 protocol. FP32 residuals are transmitted without quantization.
constexpr uint32_t magic = 0x3153474d, version = 5, reply = 1, failure = 2, cache_hit = 4, more = 8;
constexpr uint32_t max_prefill_rows = 32768;
constexpr uint32_t max_payload = max_prefill_rows * 4u * 4096u * sizeof(float);
enum Op : uint32_t { Hello = 1, Verify = 2, Commit = 3, Reset = 4, Save = 5, Restore = 6, Close = 7,
                     StreamStep = 8, StreamEnd = 9, Prefill = 10, CheckSave = 11, CheckSaveSnapshot = 12, CheckRestore = 13,
                     SlotRead = 14, SlotWrite = 15, PrefillStream = 16 };
struct Header {
    uint32_t magic_value = magic, version_value = version, op = 0, flags = 0;
    uint64_t session = 0, sequence = 0, window = 0;
    int64_t position = 0;
    uint32_t rows = 0, bytes = 0;
};
static_assert(sizeof(Header) == 56);
struct Info {
    uint64_t manifest = 0;
    int64_t context = 0;
    uint32_t cut = 0, layers = 0, embd = 0, hc = 0, vocab = 0, experts = 0, top_k = 0, nextn = 0;
    uint32_t kv_bits = 0, kv_resident = 0;
};
static_assert(sizeof(Info) == 56);
bool same_info(const Info& a, const Info& b);
bool valid_header(const Header& h, std::string& err);
bool send_packet(int fd, Header h, const void* payload, size_t bytes, std::string& err);
bool receive_packet(int fd, Header& h, std::vector<uint8_t>& payload, std::string& err);
bool configure_socket(int fd, std::string& err);
int accept_one(const std::string& endpoint, std::string& err); // listen:port, finite startup wait
int dial(const std::string& endpoint, std::string& err);       // IPv4:port, finite retry
uint64_t manifest_hash(const std::string& pack, std::string& err); // metadata identity, NOT a full weights digest
} // namespace strata::core::stageproto

namespace strata::core {
class GlmRemoteStage {
public:
    ~GlmRemoteStage();
    bool open(const std::string& endpoint, const stageproto::Info& expected, std::string& err);
    // Own an already connected descriptor; also used by CPU socketpair protocol tests.
    bool open_connected(int fd, const stageproto::Info& expected, std::string& err);
    bool verify(int64_t p0, const int32_t* tokens, int rows, std::vector<float>& residual, std::string& err);
    bool prefill(int64_t position, const std::vector<int32_t>& tokens, int32_t next_token,
                 std::vector<float>& residual, std::string& err);
    bool prefill_stream(int64_t position, const std::vector<int32_t>& tokens, int32_t next_token, int chunk,
        const std::function<bool(size_t,const std::vector<float>&,std::string&)>& consume, std::string& err);
    bool commit(int keep, std::string& err);
    bool reset(std::string& err);
    bool save(int64_t position, std::string& err);
    bool restore(int64_t position, std::string& err);
    bool stream_step(int64_t position, int32_t token, int32_t next_guess, std::vector<float>& residual,
                     bool& hit, std::string& err);
    bool stream_end(std::string& err);
    bool checkpoint(uint32_t op, int slot, int64_t position, std::string& err);
    bool slot_read(int64_t snapshot_position, std::vector<uint8_t>& bytes, std::string& err);
    bool slot_write(int64_t position, const std::vector<uint8_t>& bytes, std::string& err);
    bool streaming() const { return streaming_; }
    bool ready() const { return fd_ >= 0; }
    bool pending() const { return pending_rows_ > 0; }
    void close();
private:
    int fd_ = -1, pending_rows_ = 0;
    bool streaming_ = false;
    int32_t stream_guess_ = -1;
    int64_t position_ = 0, pending_position_ = 0, snapshot_position_ = -1;
    uint64_t session_ = 0, sequence_ = 0, window_ = 0;
    stageproto::Info info_{};
    bool exchange(stageproto::Header h, const void* p, size_t n,
                  stageproto::Header& answer, std::vector<uint8_t>& data, std::string& err);
    bool control(uint32_t op, int64_t position, uint32_t rows, std::string& err);
};
} // namespace strata::core
