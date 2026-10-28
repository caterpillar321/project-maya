#include "../src/program/helper_queue_probe.hpp"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace hp = strata::helper_probe;
namespace np = strata::core::netproto;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
static std::vector<uint8_t> frame(np::ComputeReq q, uint32_t op = np::kCompute) {
    np::Hdr h{np::kMagic, op, sizeof(q) + 4 * sizeof(float)};
    std::vector<uint8_t> b(sizeof(h) + h.len, 0);
    std::memcpy(b.data(), &h, sizeof(h));
    std::memcpy(b.data() + sizeof(h), &q, sizeof(q));
    return b;
}
int main() {
    np::ComputeReq q{};
    q.ne = 2; q.keys[0] = 7; q.keys[1] = 8; q.limit = 7.0f;
    np::ComputeReq next[2]{};
    const auto one = frame(q);
    auto two = one;
    two.insert(two.end(), one.begin(), one.end());
    for (size_t n = 0; n < one.size(); ++n)
        check(hp::ready_prefix(one.data(), n, 4, q, next) == 0, "partial first frame");
    for (size_t n = one.size(); n < two.size(); ++n)
        check(hp::ready_prefix(two.data(), n, 4, q, next) == 1, "partial second frame");
    check(hp::ready_prefix(two.data(), two.size(), 4, q, next) == 2, "two complete frames");
    for (auto bad : {0, 1, 2, 3, 4, 5, 6, 7}) {
        auto v = q;
        if (bad == 0) v.ne = 9;
        if (bad == 1) v.ne = 0;
        if (bad == 2) v.keys[1] = -1;
        if (bad == 3) v.keys[1] = v.keys[0];
        if (bad == 4) ++v.gu_type;
        if (bad == 5) ++v.d_type;
        if (bad == 6) ++v.down_off;
        if (bad == 7) v.limit = 8.0f;
        const auto b = frame(v);
        check(hp::ready_prefix(b.data(), b.size(), 4, q, next) == 0, "incompatible request");
    }
    auto barrier = frame(q, np::kHash);
    barrier.insert(barrier.end(), one.begin(), one.end());
    check(hp::ready_prefix(barrier.data(), barrier.size(), 4, q, next) == 0, "intervening operation");
    for (bool magic : {false, true}) {
        auto b = one;
        np::Hdr h{};
        std::memcpy(&h, b.data(), sizeof(h));
        if (magic) h.magic = 0; else ++h.len;
        std::memcpy(b.data(), &h, sizeof(h));
        check(hp::ready_prefix(b.data(), b.size(), 4, q, next) == 0, "malformed header");
    }
    int fd[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "socketpair");
    std::vector<uint8_t> scratch;
    check(hp::peek_ready(fd[0], 4, q, next, scratch) == 0, "empty socket never blocks");
    check(::send(fd[1], two.data(), two.size(), 0) == (ssize_t) two.size(), "send");
    check(hp::peek_ready(fd[0], 4, q, next, scratch) == 2, "socket complete frames");
    std::vector<uint8_t> received(two.size());
    check(::recv(fd[0], received.data(), received.size(), MSG_DONTWAIT) == (ssize_t) two.size(), "peek consumes nothing");
    check(received == two, "wire bytes unchanged");
    ::close(fd[0]); ::close(fd[1]);
    std::puts("helper_queue_probe_test: OK");
}
