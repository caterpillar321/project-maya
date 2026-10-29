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
    if (h.magic_value != magic || h.version_value != version || h.op < Hello || h.op > Close ||
        h.flags > (reply | failure) || (h.flags == failure) || h.bytes > max_payload || h.rows > 16 || h.position < 0) {
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
void GlmRemoteStage::close() { if (fd_ >= 0) ::close(fd_); fd_ = -1; pending_rows_ = 0; }
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
    if (pending() || rows < 1 || rows > 16 || tokens == nullptr || p0 != position_ || p0 > info_.context - rows) {
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
    if (!pending() || keep < 1 || keep > pending_rows_) { err = "invalid stage commit"; return false; }
    if (!control(st::Commit, pending_position_, keep, err)) return false;
    position_ = pending_position_ + keep; pending_rows_ = 0; return true;
}
bool GlmRemoteStage::reset(std::string& err) {
    if (!control(st::Reset, 0, 0, err)) return false;
    position_ = 0; pending_rows_ = 0; snapshot_position_ = -1; return true;
}
bool GlmRemoteStage::save(int64_t position, std::string& err) {
    if (pending() || position != position_) { err = "invalid stage snapshot position"; return false; }
    if (!control(st::Save, position, 0, err)) return false;
    snapshot_position_ = position; return true;
}
bool GlmRemoteStage::restore(int64_t position, std::string& err) {
    if (pending() || position < 0 || position != snapshot_position_) { err = "stage snapshot unavailable"; return false; }
    if (!control(st::Restore, position, 0, err)) return false;
    position_ = position; return true;
}
} // namespace strata::core
