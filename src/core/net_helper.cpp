#include "strata/core/net_helper.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace np = netproto;

bool GlmNetHelper::send_all(const void* p, size_t n, std::string& err) {
    const uint8_t* at = (const uint8_t*) p;
    while (n > 0) {
        const ssize_t k = ::send(fd_, at, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { err = "net helper " + name_ + ": send: " + std::strerror(errno); return false; }
        at += k;
        n -= (size_t) k;
    }
    return true;
}
bool GlmNetHelper::recv_all(void* p, size_t n, std::string& err) {
    uint8_t* at = (uint8_t*) p;
    while (n > 0) {
        const ssize_t k = ::recv(fd_, at, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { err = "net helper " + name_ + ": " + (k == 0 ? "connection closed" : std::strerror(errno)); return false; }
        at += k;
        n -= (size_t) k;
    }
    return true;
}
bool GlmNetHelper::request(uint32_t op, const void* a, size_t na, const void* b, size_t nb, std::string& err) {
    const np::Hdr h{np::kMagic, op, (uint64_t) (na + nb)};
    return send_all(&h, sizeof(h), err) && (na == 0 || send_all(a, na, err)) && (nb == 0 || send_all(b, nb, err));
}
// a reply; an error's text becomes err (an answer's floats are left for the caller to read)
bool GlmNetHelper::reply(np::Reply& r, std::string& err) {
    if (!recv_all(&r, sizeof(r), err)) return false;
    if (r.ok) return true;
    std::string text(r.len, '\0');
    if (r.len && !recv_all(text.data(), r.len, err)) return false;
    err = "net helper " + name_ + ": " + text;
    return false;
}

// "listen:PORT": the helper dials in (maya-helperd --dial), for when only the trunk's firewall can be opened
bool GlmNetHelper::accept_on(int port, std::string& err) {
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (ls < 0 || ::bind(ls, (sockaddr*) &addr, sizeof(addr)) != 0 || ::listen(ls, 1) != 0) {
        err = "net helper " + name_ + ": listen: " + std::strerror(errno);
        if (ls >= 0) ::close(ls);
        return false;
    }
    std::fprintf(stderr, "glm net helper: waiting on port %d for maya-helperd --dial (up to 120 s)\n", port);
    pollfd p{ls, POLLIN, 0};
    const int ready = ::poll(&p, 1, 120000);
    fd_ = ready == 1 ? ::accept(ls, nullptr, nullptr) : -1;
    ::close(ls);
    if (fd_ < 0) {
        err = "net helper " + name_ + ": no helper dialed in";
        return false;
    }
    return true;
}

bool GlmNetHelper::connect(const std::string& host_port, std::string& err) {
    close();
    name_ = host_port;
    if (host_port.rfind("listen:", 0) == 0) {
        if (!accept_on(std::atoi(host_port.c_str() + 7), err)) return false;
        const int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        return true;
    }
    const size_t colon = host_port.rfind(':');
    if (colon == std::string::npos) { err = "net helper: give host:port, not '" + host_port + "'"; return false; }
    const std::string host = host_port.substr(0, colon), port = host_port.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "net helper " + name_ + ": " + gai_strerror(rc);
        return false;
    }
    for (addrinfo* a = res; a != nullptr && fd_ < 0; a = a->ai_next) {
        fd_ = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd_ >= 0 && ::connect(fd_, a->ai_addr, a->ai_addrlen) != 0) {
            err = "net helper " + name_ + ": connect: " + std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
        }
    }
    freeaddrinfo(res);
    if (fd_ < 0) return false;
    const int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));   // a route's few KB must not wait for Nagle
    return true;
}

bool GlmNetHelper::open(int n_embd, int n_ff, const std::vector<std::string>& shards, std::string& err) {
    std::string names;
    for (const auto& s : shards) names += s.substr(s.find_last_of('/') + 1) + "\n";
    const np::OpenReq q{n_embd, n_ff};
    np::Reply r{};
    if (!request(np::kOpen, &q, sizeof(q), names.data(), names.size(), err) || !reply(r, err)) return false;
    n_embd_ = n_embd;
    missing_ = (int) r.value;
    proto_ = r.pad;
    rx_.assign((size_t) n_embd, 0.0f);
    return true;
}

bool GlmNetHelper::reserve(int keys, size_t requested_bytes, std::string& err) {
    const np::ReserveReq q{keys, 0, (uint64_t) requested_bytes};
    np::Reply r{};
    if (!request(np::kReserve, &q, sizeof(q), nullptr, 0, err) || !reply(r, err)) return false;
    capacity_ = (size_t) r.value;   // what the helper's VRAM holds (its own 512 MiB headroom taken off)
    ram_capacity_ = (size_t) r.pad << 20;   // maya-helperd --ram-cold: its RAM tier, for the trunk's coldest experts
    ram_used_ = 0;
    used_ = 0;
    resident_ = 0;
    owned_.assign((size_t) keys, 0);
    owned_ram_.assign((size_t) keys, 0);
    return true;
}

bool GlmNetHelper::insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err) {
    if (owns(key) || stride > available()) { err = "net helper " + name_ + ": invalid cache insertion"; return false; }
    const np::InsertReq q{key, 0, (uint64_t) bytes, (uint64_t) stride};
    if (!request(np::kInsert, &q, sizeof(q), host, bytes, err)) return false;
    owned_[(size_t) key] = 1;
    used_ += stride;
    ++resident_;
    ++unread_;
    return true;
}

bool GlmNetHelper::insert_file(int key, const uint32_t shard[3], const uint64_t off[3], const uint64_t len[3],
                               const uint64_t dst[3], size_t bytes, size_t stride, std::string& err) {
    if (owns(key) || stride > available()) { err = "net helper " + name_ + ": invalid cache insertion"; return false; }
    np::InsertFileReq q{};
    q.key = key;
    for (int j = 0; j < 3; ++j) q.shard[j] = shard[j], q.off[j] = off[j], q.len[j] = len[j], q.dst[j] = dst[j];
    q.bytes = bytes;
    q.stride = stride;
    if (!request(np::kInsertFile, &q, sizeof(q), nullptr, 0, err)) return false;
    owned_[(size_t) key] = 1;
    used_ += stride;
    ++resident_;
    ++unread_;
    return true;
}

bool GlmNetHelper::insert_file_ram(int key, const uint32_t shard[3], const uint64_t off[3], const uint64_t len[3],
                                   const uint64_t dst[3], size_t bytes, size_t stride, std::string& err) {
    if (owns(key) || stride > ram_available()) { err = "net helper " + name_ + ": invalid RAM-tier insertion"; return false; }
    np::InsertFileReq q{};
    q.key = key;
    for (int j = 0; j < 3; ++j) q.shard[j] = shard[j], q.off[j] = off[j], q.len[j] = len[j], q.dst[j] = dst[j];
    q.bytes = bytes;
    q.stride = stride;
    if (!request(np::kInsertFile, &q, sizeof(q), nullptr, 0, err)) return false;
    owned_[(size_t) key] = 1;
    if (!owned_ram_.empty()) owned_ram_[(size_t) key] = 1;
    ram_used_ += stride;
    ++resident_ram_;
    ++unread_;
    return true;
}

bool GlmNetHelper::flush(std::string& err) {
    for (; unread_ > 0; --unread_) {
        np::Reply r{};
        if (!reply(r, err)) return false;
    }
    return true;
}

bool GlmNetHelper::hash_ram(int n, const np::HashRef* refs, uint64_t* out, std::string& err) {
    if (n < 1) return true;
    if (pending_ || routes_pending_ > 0 || unread_ > 0) { err = "net helper " + name_ + ": hash request while busy"; return false; }
    if (!request(np::kHash, refs, (size_t) n * sizeof(np::HashRef), nullptr, 0, err)) return false;
    np::Reply r{};
    if (!reply(r, err)) return false;
    if (r.len != (uint64_t) n * sizeof(uint64_t)) { err = "net helper " + name_ + ": bad hash answer size"; return false; }
    return recv_all(out, (size_t) r.len, err);
}

bool GlmNetHelper::begin(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys,
                         const float* weights, const float* x, std::string& err) {
    if (pending_ || routes_pending_ >= kMaxRoutes || ne < 1 || ne > 8) { err = "net helper " + name_ + ": bad route"; return false; }
    np::ComputeReq q{};
    q.gu_type = gu_type;
    q.d_type = d_type;
    q.down_off = down_off;
    q.limit = limit;
    q.ne = ne;
    for (int i = 0; i < ne; ++i) q.keys[i] = keys[i], q.w[i] = weights[i];
    if (!request(np::kCompute, &q, sizeof(q), x, (size_t) n_embd_ * sizeof(float), err)) return false;
    r16_[(r16_head_ + routes_pending_) % kMaxRoutes] = false;
    ++routes_pending_;   // (several may be on the wire - a verify window's tokens - answered in order)
    experts += (uint64_t) ne;
    ++routes;
    return true;
}

bool GlmNetHelper::begin2(int gu_type, int d_type, size_t down_off, float limit, int ne, const int* keys,
                          const float* weights, uint32_t flags, const void* xq, size_t nxq, const void* xk, size_t nxk,
                          const float* x, std::string& err) {
    if (proto_ < 2) { err = "net helper " + name_ + ": kCompute2 to a helperd of protocol " + std::to_string(proto_); return false; }
    if (pending_ || routes_pending_ >= kMaxRoutes || ne < 1 || ne > 8) { err = "net helper " + name_ + ": bad route"; return false; }
    np::Compute2Req q{};
    q.gu_type = gu_type;
    q.d_type = d_type;
    q.down_off = down_off;
    q.limit = limit;
    q.ne = ne;
    for (int i = 0; i < ne; ++i) q.keys[i] = keys[i], q.w[i] = weights[i];
    q.flags = flags;
    q.nxq = (flags & np::kXq) ? (uint32_t) nxq : 0u;
    q.nxk = (flags & np::kXk) ? (uint32_t) nxk : 0u;
    const size_t nxf = (flags & np::kXf) ? (size_t) n_embd_ * sizeof(float) : 0;
    const np::Hdr h{np::kMagic, np::kCompute2, (uint64_t) (sizeof(q) + q.nxq + q.nxk + nxf)};
    // one buffer, one send: the header, the request and its inputs (a few KB) leave in as few segments as they fit
    static thread_local std::vector<uint8_t> buf;
    buf.resize(sizeof(h) + sizeof(q) + q.nxq + q.nxk + nxf);
    uint8_t* at = buf.data();
    std::memcpy(at, &h, sizeof(h)); at += sizeof(h);
    std::memcpy(at, &q, sizeof(q)); at += sizeof(q);
    if (q.nxq) { std::memcpy(at, xq, q.nxq); at += q.nxq; }
    if (q.nxk) { std::memcpy(at, xk, q.nxk); at += q.nxk; }
    if (nxf) std::memcpy(at, x, nxf);
    if (!send_all(buf.data(), buf.size(), err)) return false;
    r16_[(r16_head_ + routes_pending_) % kMaxRoutes] = (flags & np::kR16) != 0;
    ++routes_pending_;
    experts += (uint64_t) ne;
    ++routes;
    return true;
}

// STRATA_GLM_NET_SPIN_US=<us> (default 0: off): before an answer's blocking read, poll the socket without blocking
// for up to that long - this thread then sees the answer as it lands instead of after a wake-up (the LAN's round
// trip measured 280-320 us beyond the helpers' compute, 10-26; ping 0.12 min / 0.38 avg ms)
static void spin_readable(int fd) {
    static const long spin_us = [] {
        const char* v = std::getenv("STRATA_GLM_NET_SPIN_US");
        return v ? std::max(0L, std::atol(v)) : 0L;
    }();
    if (spin_us <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    char c;
    for (;;) {
        const ssize_t k = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
        if (k > 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK) || k == 0) return;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us)) return;
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
}

bool GlmNetHelper::finish(float* out, std::string& err) {
    if (routes_pending_ == 0) return true;
    --routes_pending_;
    spin_readable(fd_);
    const bool r16 = r16_[r16_head_];
    r16_head_ = (r16_head_ + 1) % kMaxRoutes;
    const auto t0 = std::chrono::steady_clock::now();
    np::Reply r{};
    if (!reply(r, err)) return false;
    if (r16) {
        if (r.len != sizeof(float) + (uint64_t) n_embd_ * sizeof(uint16_t)) { err = "net helper " + name_ + ": bad FP16 answer size"; return false; }
        float inv = 0.0f;
        rx16_.resize((size_t) n_embd_);
        if (!recv_all(&inv, sizeof(inv), err) || !recv_all(rx16_.data(), (size_t) n_embd_ * sizeof(uint16_t), err)) return false;
        ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (int j = 0; j < n_embd_; ++j) out[j] += np::h2f(rx16_[(size_t) j]) * inv;
        return true;
    }
    if (r.len != (uint64_t) n_embd_ * sizeof(float)) { err = "net helper " + name_ + ": bad answer size"; return false; }
    if (!recv_all(rx_.data(), (size_t) r.len, err)) return false;
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int j = 0; j < n_embd_; ++j) out[j] += rx_[(size_t) j];
    return true;
}

bool GlmNetHelper::rows_begin(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const uint16_t* x,
                              int n_rows, const np::RowRef* rows, std::string& err) {
    if (pending_ || routes_pending_ > 0 || n_rows < 1 || n_tok < 1) { err = "net helper " + name_ + ": bad prompt rows"; return false; }
    np::RowsReq q{};
    q.gu_type = gu_type;
    q.d_type = d_type;
    q.down_off = down_off;
    q.limit = limit;
    q.n_tok = n_tok;
    q.n_rows = n_rows;
    const size_t nx = (size_t) n_tok * (size_t) n_embd_ * sizeof(uint16_t), nr = (size_t) n_rows * sizeof(np::RowRef);
    const np::Hdr h{np::kMagic, np::kRows, (uint64_t) (sizeof(q) + nx + nr)};
    const auto t0 = std::chrono::steady_clock::now();
    if (!send_all(&h, sizeof(h), err) || !send_all(&q, sizeof(q), err) || !send_all(x, nx, err) ||
        !send_all(rows, nr, err)) return false;
    pf_ms_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pending_ = true;
    pf_rows += (uint64_t) n_rows;
    ++pf_layers;
    return true;
}

bool GlmNetHelper::tok_begin(int gu_type, int d_type, size_t down_off, float limit, int n_tok, const uint16_t* x,
                             int n_ent, const np::TokEntry* ent, std::string& err) {
    if (pending_ || routes_pending_ > 0 || n_ent < 1 || n_tok < 1) { err = "net helper " + name_ + ": bad prompt rows"; return false; }
    np::RowsReq q{};
    q.gu_type = gu_type;
    q.d_type = d_type;
    q.down_off = down_off;
    q.limit = limit;
    q.n_tok = n_tok;
    q.n_rows = n_ent;
    const size_t nx = (size_t) n_tok * (size_t) n_embd_ * sizeof(uint16_t), ne = (size_t) n_ent * sizeof(np::TokEntry);
    const np::Hdr h{np::kMagic, np::kTokRows, (uint64_t) (sizeof(q) + nx + ne)};
    const auto t0 = std::chrono::steady_clock::now();
    if (!send_all(&h, sizeof(h), err) || !send_all(&q, sizeof(q), err) || !send_all(x, nx, err) ||
        !send_all(ent, ne, err)) return false;
    pf_ms_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pending_ = true;
    pf_rows += (uint64_t) n_ent;
    ++pf_layers;
    return true;
}

bool GlmNetHelper::rows_finish(uint16_t* out, int n_rows, std::string& err) {
    if (!pending_) return true;
    pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    np::Reply r{};
    if (!reply(r, err)) return false;
    if (r.len != (uint64_t) n_rows * (uint64_t) n_embd_ * sizeof(uint16_t)) {
        err = "net helper " + name_ + ": bad prompt answer size";
        return false;
    }
    if (!recv_all(out, (size_t) r.len, err)) return false;
    pf_ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void GlmNetHelper::close() {
    if (fd_ >= 0) {
        std::string ignored;
        request(np::kClose, nullptr, 0, nullptr, 0, ignored);
        ::close(fd_);
    }
    fd_ = -1;
    owned_.clear();
    capacity_ = used_ = ram_capacity_ = ram_used_ = 0;
    resident_ram_ = 0;
    resident_ = unread_ = 0;
    pending_ = false;
    routes_pending_ = 0;
}

}   // namespace strata::core
