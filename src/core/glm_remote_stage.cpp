#include "strata/core/glm_remote_stage.hpp"
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <thread>

namespace strata::core::stageproto {
namespace {
bool transfer(int fd, void* p, size_t n, bool sending, std::string& err) {
    auto* at = static_cast<uint8_t*>(p);
    while (n) {
        const ssize_t k = sending ? ::send(fd, at, n, MSG_NOSIGNAL) : ::recv(fd, at, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { err = k == 0 ? "stage peer closed the connection" : std::string("stage I/O: ") + std::strerror(errno); return false; }
        at += k; n -= (size_t) k;
    }
    return true;
}
bool port_of(const std::string& s, uint16_t& port) {
    if (s.empty() || s.size() > 5 || s.find_first_not_of("0123456789") != std::string::npos) return false;
    const unsigned long n = std::stoul(s);
    if (n == 0 || n > 65535) return false;
    port = (uint16_t) n; return true;
}
} // namespace
bool same_info(const Info& a, const Info& b) {
    return a.manifest == b.manifest && a.context == b.context && a.cut == b.cut && a.layers == b.layers &&
           a.embd == b.embd && a.hc == b.hc && a.vocab == b.vocab && a.experts == b.experts &&
           a.top_k == b.top_k && a.nextn == b.nextn && a.kv_bits == b.kv_bits && a.kv_resident == b.kv_resident;
}
bool valid_header(const Header& h, std::string& err) {
    if (h.magic_value != magic || h.version_value != version || h.op < Hello || h.op > BlockForecast ||
        (h.flags & ~(reply | failure | cache_hit | more)) ||
        ((h.flags & (failure | cache_hit | more)) && !(h.flags & reply)) ||
        ((h.flags & cache_hit) && ((h.op != StreamStep && h.op != BlockStep) || (h.flags & failure))) ||
        ((h.flags & more) && (h.op != PrefillStream || (h.flags & failure))) ||
        h.bytes > ((h.op == Prefill || h.op == PrefillStream || h.op == SlotRead || h.op == SlotWrite) ? max_payload : 16u * 4u * 4096u * sizeof(float)) ||
        h.rows > ((h.op == Prefill || h.op == PrefillStream) ? max_prefill_rows : 16u) || h.position < 0) {
        err = "stage protocol: invalid magic/version/op/flags/shape"; return false;
    }
    return true;
}
bool configure_socket(int fd, std::string& err) {
    const int one = 1;
    // UNIX socketpairs in tests have no TCP_NODELAY, but the finite receive/send timeouts still apply.
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0 && errno != EOPNOTSUPP && errno != ENOTSUP) {
        err = "stage TCP_NODELAY: " + std::string(std::strerror(errno)); return false;
    }
    const timeval timeout{120, 0};
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        err = "stage socket timeout setup failed"; return false;
    }
    return true;
}
bool send_packet(int fd, Header h, const void* payload, size_t bytes, std::string& err) {
    if (bytes > max_payload || (bytes && payload == nullptr)) { err = "stage payload too large or missing"; return false; }
    h.bytes = (uint32_t) bytes;
    return valid_header(h, err) && transfer(fd, &h, sizeof(h), true, err) &&
           transfer(fd, const_cast<void*>(payload), bytes, true, err);
}
bool receive_packet(int fd, Header& h, std::vector<uint8_t>& payload, std::string& err) {
    if (!transfer(fd, &h, sizeof(h), false, err) || !valid_header(h, err)) return false;
    payload.resize(h.bytes);
    return transfer(fd, payload.data(), payload.size(), false, err);
}
int accept_one(const std::string& endpoint, std::string& err) {
    uint16_t port;
    if (endpoint.rfind("listen:", 0) != 0 || !port_of(endpoint.substr(7), port)) {
        err = "remote front needs listen:<port>"; return -1;
    }
    int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { err = "stage socket failed"; return -1; }
    const int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port); addr.sin_addr.s_addr = INADDR_ANY;
    if (::bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(ls, 1) < 0) {
        err = "stage listen: " + std::string(std::strerror(errno)); ::close(ls); return -1;
    }
    pollfd poller{ls, POLLIN, 0};
    const int n = ::poll(&poller, 1, 120000);
    if (n <= 0) { err = "stage worker did not connect within 120 seconds"; ::close(ls); return -1; }
    const int fd = ::accept(ls, nullptr, nullptr); ::close(ls);
    if (fd < 0) { err = "stage accept failed"; return -1; }
    if (!configure_socket(fd, err)) { ::close(fd); return -1; }
    return fd;
}
int dial(const std::string& endpoint, std::string& err) {
    const size_t at = endpoint.rfind(':'); uint16_t port;
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    if (at == std::string::npos || !port_of(endpoint.substr(at + 1), port) ||
        ::inet_pton(AF_INET, endpoint.substr(0, at).c_str(), &addr.sin_addr) != 1) {
        err = "stage dial needs IPv4:port"; return -1;
    }
    addr.sin_port = htons(port);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < until) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) break;
        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) { ::close(fd); break; }
        int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc < 0 && errno == EINPROGRESS) {
            pollfd p{fd, POLLOUT, 0};
            if (::poll(&p, 1, 1000) > 0) {
                int socket_error = 0; socklen_t len = sizeof(socket_error);
                if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &len) == 0 && socket_error == 0) rc = 0;
            }
        }
        if (rc == 0 && ::fcntl(fd, F_SETFL, flags) == 0) {
            if (configure_socket(fd, err)) return fd;
            ::close(fd); return -1;
        }
        ::close(fd); std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    err = "stage worker could not connect to coordinator"; return -1;
}
uint64_t manifest_hash(const std::string& pack, std::string& err) {
    uint64_t h = 14695981039346656037ull;
    for (const char* name : {"index.txt", "native_experts.txt"}) {
        std::ifstream f(pack + "/" + name, std::ios::binary);
        if (!f) { err = std::string("stage manifest missing ") + name; return 0; }
        char b;
        while (f.get(b)) h = (h ^ (uint8_t) b) * 1099511628211ull;
        if (!f.eof()) { err = "stage manifest read failed"; return 0; }
        h = (h ^ 255u) * 1099511628211ull;
    }
    return h;
}
} // namespace strata::core::stageproto

namespace strata::core {
namespace st = stageproto;
GlmRemoteStage::~GlmRemoteStage() { close(); }
void GlmRemoteStage::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; pending_rows_ = 0; streaming_ = false; stream_guess_ = -1;
    block_streaming_ = false; block_rows_ = block_keep_ = block_guesses_ = 0;
    block_flight_ = false;
}
bool GlmRemoteStage::open(const std::string& endpoint, const st::Info& expected, std::string& err) {
    const int fd = st::accept_one(endpoint, err);
    return fd >= 0 && open_connected(fd, expected, err);
}
bool GlmRemoteStage::open_connected(int fd, const st::Info& expected, std::string& err) {
    close(); fd_ = fd;
    st::Header h; std::vector<uint8_t> data;
    if (!st::configure_socket(fd_, err) || !st::receive_packet(fd_, h, data, err)) { close(); return false; }
    st::Info got{};
    if (data.size() == sizeof(got)) std::memcpy(&got, data.data(), sizeof(got));
    if (h.op != st::Hello || h.flags || h.session || h.sequence || h.window || h.rows || h.position ||
        data.size() != sizeof(got) || !st::same_info(got, expected)) {
        err = "stage HELLO mismatch (manifest, geometry, context or boundary)"; close(); return false;
    }
    info_ = expected;
    session_ = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() ^ (uint64_t) ::getpid();
    if (!session_) session_ = 1;
    h.flags = st::reply; h.session = session_;
    if (!st::send_packet(fd_, h, &info_, sizeof(info_), err)) { close(); return false; }
    sequence_ = window_ = 0; position_ = 0; snapshot_position_ = -1;
    return true;
}
bool GlmRemoteStage::exchange(st::Header h, const void* p, size_t n, st::Header& answer,
                              std::vector<uint8_t>& data, std::string& err) {
    if (fd_ < 0) { err = "remote stage is disconnected; reload the model"; return false; }
    h.session = session_; h.sequence = ++sequence_;
    if (!st::send_packet(fd_, h, p, n, err) || !st::receive_packet(fd_, answer, data, err)) { close(); return false; }
    if (answer.session != session_ || answer.sequence != h.sequence || answer.op != h.op ||
        answer.window != h.window || !(answer.flags & st::reply)) {
        err = "stage response identity mismatch"; close(); return false;
    }
    if (answer.flags & st::failure) {
        err = "stage worker: " + std::string(data.begin(), data.end()); close(); return false;
    }
    return true;
}
bool GlmRemoteStage::verify(int64_t p0, const int32_t* tokens, int rows, std::vector<float>& residual, std::string& err) {
    if (pending() || streaming() || rows < 1 || rows > 16 || tokens == nullptr || p0 != position_ || p0 > info_.context - rows) {
        err = "stage verify position/window mismatch"; return false;
    }
    for (int i = 0; i < rows; ++i) if (tokens[i] < 0 || (uint32_t) tokens[i] >= info_.vocab) {
        err = "stage verify invalid token"; return false;
    }
    st::Header h; h.op = st::Verify; h.position = p0; h.rows = rows; h.window = ++window_;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, tokens, (size_t) rows * sizeof(int32_t), a, data, err)) return false;
    const size_t floats = (size_t) rows * info_.hc * info_.embd;
    if (a.position != p0 + rows || a.rows != (uint32_t) rows || data.size() != floats * sizeof(float)) {
        err = "stage residual shape/position mismatch"; close(); return false;
    }
    residual.resize(floats); std::memcpy(residual.data(), data.data(), data.size());
    if (const char* prefix = std::getenv("STRATA_GLM_STAGE_DIAG"); prefix && *prefix && p0 < 4) {
        const std::string path = std::string(prefix) + "-s" + std::to_string(sequence_) + "-p" +
                                 std::to_string(p0) + "-T" + std::to_string(rows) + ".f32";
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(residual.data()), (std::streamsize) data.size());
        if (!f) { err = "stage diagnostic residual write failed"; close(); return false; }
    }
    pending_rows_ = rows; pending_position_ = p0;
    return true;
}
bool GlmRemoteStage::prefill(int64_t position, const std::vector<int32_t>& tokens, int32_t next_token,
                              std::vector<float>& residual, std::string& err) {
    if (pending() || streaming() || position != position_ || tokens.empty() ||
        tokens.size() > st::max_prefill_rows || position > info_.context - (int64_t) tokens.size() ||
        next_token < -1 || (next_token >= 0 && (uint32_t) next_token >= info_.vocab)) {
        err = "invalid remote prefill range/lifecycle"; return false;
    }
    for (int32_t t : tokens) if (t < 0 || (uint32_t) t >= info_.vocab) {
        err = "invalid remote prefill token"; return false;
    }
    std::vector<int32_t> payload(tokens); payload.push_back(next_token);
    st::Header h; h.op = st::Prefill; h.position = position; h.rows = tokens.size(); h.window = ++window_;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, payload.data(), payload.size()*sizeof(int32_t), a, data, err)) return false;
    const size_t floats = tokens.size() * info_.hc * info_.embd;
    if (a.position != position + (int64_t) tokens.size() || a.rows != tokens.size() ||
        data.size() != floats * sizeof(float)) {
        err = "remote prefill residual shape/position mismatch"; close(); return false;
    }
    residual.resize(floats); std::memcpy(residual.data(), data.data(), data.size());
    if (const char* prefix = std::getenv("STRATA_GLM_STAGE_DIAG"); prefix && *prefix && position == 0) {
        std::ofstream f(std::string(prefix) + "-prefill-" + std::to_string(tokens.size()) + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(residual.data()), residual.size()*sizeof(float));
    }
    position_ = a.position;
    return true;
}
bool GlmRemoteStage::prefill_stream(int64_t position, const std::vector<int32_t>& tokens, int32_t next_token, int chunk,
    const std::function<bool(size_t,const std::vector<float>&,std::string&)>& consume, std::string& err) {
    if (!ready() || pending() || streaming() || position!=position_ || tokens.empty() ||
        tokens.size()>st::max_prefill_rows || position>info_.context-(int64_t)tokens.size() ||
        chunk<1 || chunk>(int)st::max_prefill_rows || next_token < -1 ||
        (next_token>=0 && (uint32_t)next_token>=info_.vocab)) {err="invalid streamed prefill request";return false;}
    for(int t:tokens)if(t<0 || (uint32_t)t>=info_.vocab){err="invalid streamed prefill token";return false;}
    std::vector<int32_t> ids;ids.reserve(tokens.size()+2);ids.push_back(chunk);
    ids.insert(ids.end(),tokens.begin(),tokens.end());ids.push_back(next_token);
    st::Header h;h.op=st::PrefillStream;h.position=position;h.rows=tokens.size();
    h.session=session_;h.sequence=++sequence_;h.window=++window_;
    if(!st::send_packet(fd_,h,ids.data(),ids.size()*sizeof(int32_t),err)){close();return false;}
    struct Packet {st::Header h;std::vector<uint8_t> data;std::string error;bool ok=false;};
    const auto read=[this](){Packet p;p.ok=st::receive_packet(fd_,p.h,p.data,p.error);return p;};
    auto pending=std::async(std::launch::async,read);
    bool consumed=true;std::string consume_error;
    for(size_t at=0;at<tokens.size();) {
        const size_t count=std::min<size_t>(chunk,tokens.size()-at);
        Packet p=pending.get();const auto& a=p.h;
        if(!p.ok){err=p.error;close();return false;}
        if(a.session!=h.session || a.sequence!=h.sequence || a.window!=h.window || a.op!=h.op || !(a.flags&st::reply)) {
            err="streamed prefill response identity mismatch";close();return false;
        }
        if(a.flags&st::failure){err="stage worker: "+std::string(p.data.begin(),p.data.end());close();return false;}
        const bool another=at+count<tokens.size();
        if(a.rows!=count || a.position!=position+(int64_t)(at+count) || bool(a.flags&st::more)!=another ||
           p.data.size()!=count*(size_t)info_.hc*info_.embd*sizeof(float)) {
            err="streamed prefill response shape/order mismatch";close();return false;
        }
        // One reader overlaps the next FP32 transfer with this chunk's local GPU work. Cancellation
        // drains the bounded outstanding prompt before reset, keeping the connection reusable.
        if(another)pending=std::async(std::launch::async,read);
        if(consumed) {
            std::vector<float> residual(p.data.size()/sizeof(float));
            std::memcpy(residual.data(),p.data.data(),p.data.size());
            p.data.clear();p.data.shrink_to_fit();
            consumed=consume(at,residual,consume_error);
        }
        at+=count;position_=position+at;
    }
    if(!consumed){err=consume_error.empty()?"streamed prefill consumer failed":consume_error;return false;}
    return true;
}
bool GlmRemoteStage::control(uint32_t op, int64_t position, uint32_t rows, std::string& err) {
    st::Header h; h.op = op; h.position = position; h.rows = rows;
    h.window = op == st::Commit ? window_ : 0;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, nullptr, 0, a, data, err)) return false;
    const int64_t result_pos = op == st::Commit ? position + rows : position;
    if (a.position != result_pos || a.rows != rows || !data.empty()) {
        err = "stage control acknowledgment mismatch"; close(); return false;
    }
    return true;
}
bool GlmRemoteStage::commit(int keep, std::string& err) {
    if (streaming_ || !pending() || keep < 1 || keep > pending_rows_) { err = "invalid stage commit"; return false; }
    if (!control(st::Commit, pending_position_, keep, err)) return false;
    position_ = pending_position_ + keep; pending_rows_ = 0; return true;
}
bool GlmRemoteStage::reset(std::string& err) {
    if (block_flight_) { err = "finish the block response before reset"; return false; }
    if (!control(st::Reset, 0, 0, err)) return false;
    position_ = 0; pending_rows_ = 0; snapshot_position_ = -1; streaming_ = false; stream_guess_ = -1;
    block_streaming_ = false; block_rows_ = block_keep_ = block_guesses_ = 0; return true;
}
bool GlmRemoteStage::save(int64_t position, std::string& err) {
    if (pending() || streaming() || position != position_) { err = "invalid stage snapshot position"; return false; }
    if (!control(st::Save, position, 0, err)) return false;
    snapshot_position_ = position; return true;
}
bool GlmRemoteStage::restore(int64_t position, std::string& err) {
    if (pending() || streaming() || position < 0 || position != snapshot_position_) { err = "stage snapshot unavailable"; return false; }
    if (!control(st::Restore, position, 0, err)) return false;
    position_ = position; return true;
}
bool GlmRemoteStage::checkpoint(uint32_t op, int slot, int64_t position, std::string& err) {
    if (pending() || streaming() || slot < 0 || slot >= 256 || position < 0 || position > info_.context ||
        (op != st::CheckSave && op != st::CheckSaveSnapshot && op != st::CheckRestore) ||
        (op == st::CheckSave && position != position_) ||
        (op == st::CheckSaveSnapshot && position != snapshot_position_)) {
        err = "invalid remote checkpoint lifecycle/identity"; return false;
    }
    st::Header h; h.op = op; h.window = slot + 1; h.position = position;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, nullptr, 0, a, data, err)) return false;
    const int64_t expected = op == st::CheckRestore ? position : position_;
    if (a.position != expected || a.rows || !data.empty()) {
        err = "remote checkpoint acknowledgment mismatch"; close(); return false;
    }
    if (op == st::CheckRestore) position_ = position;
    return true;
}
bool GlmRemoteStage::slot_read(int64_t position, std::vector<uint8_t>& bytes, std::string& err) {
    if (pending() || streaming() || position < 1 || position != snapshot_position_) {
        err = "invalid remote slot snapshot"; return false;
    }
    st::Header h; h.op = st::SlotRead; h.position = position;
    st::Header a;
    if (!exchange(h, nullptr, 0, a, bytes, err)) return false;
    if (a.position != position_ || a.rows || bytes.empty()) {
        err = "remote slot export acknowledgment mismatch"; close(); return false;
    }
    return true;
}
bool GlmRemoteStage::slot_write(int64_t position, const std::vector<uint8_t>& bytes, std::string& err) {
    if (pending() || streaming() || position < 1 || position > info_.context || bytes.empty() || bytes.size() > st::max_payload) {
        err = "invalid remote slot import"; return false;
    }
    st::Header h; h.op = st::SlotWrite; h.position = position;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, bytes.data(), bytes.size(), a, data, err)) return false;
    if (a.position != position || a.rows || !data.empty()) {
        err = "remote slot import acknowledgment mismatch"; close(); return false;
    }
    position_ = snapshot_position_ = position;
    return true;
}
bool GlmRemoteStage::stream_step(int64_t position, int32_t token, int32_t next_guess,
                                 std::vector<float>& residual, bool& hit, std::string& err) {
    hit = false;
    if (pending() || block_streaming_ || position != position_ || position >= info_.context || token < 0 ||
        (uint32_t) token >= info_.vocab || next_guess < -1 ||
        (next_guess >= 0 && ((uint32_t) next_guess >= info_.vocab || position + 1 >= info_.context))) {
        err = "stage stream token/position mismatch"; return false;
    }
    const bool expected_hit = streaming_ && stream_guess_ == token;
    const int32_t input[2] = {token, next_guess};
    st::Header h; h.op = st::StreamStep; h.position = position; h.rows = 1; h.window = ++window_;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, input, sizeof(input), a, data, err)) return false;
    const size_t floats = (size_t) info_.hc * info_.embd;
    if (a.position != position + 1 || a.rows != 1 || data.size() != floats * sizeof(float) ||
        bool(a.flags & st::cache_hit) != expected_hit) {
        err = "stage stream response shape/position/cache identity mismatch"; close(); return false;
    }
    residual.resize(floats); std::memcpy(residual.data(), data.data(), data.size());
    hit = expected_hit; position_ = position + 1; streaming_ = true; stream_guess_ = next_guess;
    return true;
}
bool GlmRemoteStage::block_step(int64_t position, const int32_t* tokens, int rows, const int32_t* next, int guesses,
                              std::vector<float>& residual, int& reused, std::string& err) {
    return block_begin(position, tokens, rows, next, guesses, err) && block_finish(residual, reused, err);
}
int GlmRemoteStage::block_reusable_prefix(const int32_t* tokens, int rows) const {
    if (!tokens || rows < 1 || rows > 2 || !block_streaming_ || block_flight_ || block_keep_ != block_rows_ || block_keep_ < 1)
        return 0;
    int matched = 0;
    while (matched < rows && matched < block_guesses_ && tokens[matched] == block_next_[matched]) ++matched;
    return matched;
}
bool GlmRemoteStage::block_begin(int64_t position, const int32_t* tokens, int rows, const int32_t* next, int guesses,
                               std::string& err) {
    if (pending() || streaming_ || position != position_ || rows < 1 || rows > 2 || !tokens ||
        guesses < 0 || guesses > 2 || (guesses && !next) || position > info_.context - rows - guesses ||
        (block_streaming_ && (block_keep_ < 1 || block_keep_ > block_rows_))) {
        err = "invalid block stream range/lifecycle"; return false;
    }
    st::BlockInput input;
    input.keep = block_streaming_ ? block_keep_ : 0; input.guesses = guesses;
    for (int i = 0; i < rows; ++i) {
        if (tokens[i] < 0 || (uint32_t) tokens[i] >= info_.vocab) { err = "invalid block token"; return false; }
        input.tokens[i] = tokens[i];
    }
    for (int i = 0; i < guesses; ++i) {
        if (next[i] < 0 || (uint32_t) next[i] >= info_.vocab) { err = "invalid block guess"; return false; }
        input.next[i] = next[i];
    }
    const int expected = block_reusable_prefix(tokens, rows);
    st::Header h; h.op = st::BlockStep; h.position = position; h.rows = rows; h.window = ++window_;
    h.session = session_; h.sequence = ++sequence_;
    if (!st::send_packet(fd_, h, &input, sizeof(input), err)) { close(); return false; }
    block_request_ = h; block_input_ = input; block_expected_ = expected; block_flight_ = true;
    return true;
}
bool GlmRemoteStage::block_finish(std::vector<float>& residual, int& reused, std::string& err) {
    reused = 0;
    if (!block_flight_) { err = "no block response pending"; return false; }
    const auto h = block_request_;
    const auto input = block_input_;
    const int64_t position = h.position; const int rows = h.rows, guesses = input.guesses, expected = block_expected_;
    st::Header a; std::vector<uint8_t> data;
    if (!st::receive_packet(fd_, a, data, err)) { close(); return false; }
    if (a.session != session_ || a.sequence != h.sequence || a.op != h.op || a.window != h.window || !(a.flags & st::reply)) {
        err = "block response identity mismatch"; close(); return false;
    }
    if (a.flags & st::failure) { err = "stage worker: " + std::string(data.begin(), data.end()); close(); return false; }
    const size_t floats = (size_t) rows * info_.hc * info_.embd;
    st::BlockReply meta;
    if (data.size() >= sizeof(meta)) std::memcpy(&meta, data.data(), sizeof(meta));
    if (a.position != position + rows || a.rows != (uint32_t) rows ||
        data.size() != sizeof(meta) + floats * sizeof(float) || meta.reserved || meta.reused != (uint32_t) expected ||
        bool(a.flags & st::cache_hit) != (expected == rows)) {
        err = "block response shape/position/reuse mismatch"; close(); return false;
    }
    residual.resize(floats); std::memcpy(residual.data(), data.data() + sizeof(meta), floats * sizeof(float));
    reused = expected; block_streaming_ = true; block_position_ = position; block_rows_ = rows; block_keep_ = 0;
    block_guesses_ = guesses;
    for (int i = 0; i < 2; ++i) block_next_[i] = input.next[i];
    position_ = position + rows;
    block_flight_ = false;
    return true;
}
bool GlmRemoteStage::block_forecast(const int32_t* next, int guesses, std::string& err) {
    if (!block_streaming_ || block_flight_ || block_keep_ != 0 || block_guesses_ != 0 || guesses < 1 || guesses > 2 ||
        !next || position_ > info_.context - guesses) { err = "invalid block forecast lifecycle"; return false; }
    for (int i = 0; i < guesses; ++i)
        if (next[i] < 0 || (uint32_t) next[i] >= info_.vocab) { err = "invalid block forecast token"; return false; }
    st::Header h; h.op = st::BlockForecast; h.position = position_; h.rows = guesses; h.window = window_;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, next, guesses*sizeof(int32_t), a, data, err)) return false;
    if (a.position != position_ || a.rows != h.rows || !data.empty()) {
        err = "block forecast acknowledgment mismatch"; close(); return false;
    }
    block_guesses_ = guesses;
    for (int i = 0; i < guesses; ++i) block_next_[i] = next[i];
    return true;
}
bool GlmRemoteStage::block_commit(int keep, std::string& err) {
    if (!block_streaming_ || block_keep_ != 0 || keep < 1 || keep > block_rows_) {
        err = "invalid block stream commit"; return false;
    }
    block_keep_ = keep; position_ = block_position_ + keep; return true;
}
bool GlmRemoteStage::block_end(std::string& err) {
    if (block_flight_) { err = "finish the block response before end"; return false; }
    if (!block_streaming_) return ready();
    if (block_keep_ < 1 || block_keep_ > block_rows_) { err = "uncommitted block stream end"; return false; }
    st::Header h; h.op = st::BlockEnd; h.position = position_; h.rows = block_keep_;
    st::Header a; std::vector<uint8_t> data;
    if (!exchange(h, nullptr, 0, a, data, err)) return false;
    if (a.position != position_ || a.rows != h.rows || !data.empty()) {
        err = "block end acknowledgment mismatch"; close(); return false;
    }
    block_streaming_ = false; block_rows_ = block_keep_ = block_guesses_ = 0;
    return true;
}
bool GlmRemoteStage::stream_end(std::string& err) {
    if (block_streaming_) { err = "block stream needs block_end"; return false; }
    if (!streaming_) return ready();
    if (!control(st::StreamEnd, position_, 0, err)) return false;
    streaming_ = false; stream_guess_ = -1; return true;
}
} // namespace strata::core
