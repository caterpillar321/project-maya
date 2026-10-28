#pragma once

#include "strata/core/net_helper.hpp"
#include <cstring>
#include <sys/socket.h>
#include <vector>

namespace strata::helper_probe {
namespace np = core::netproto;

inline bool valid(const np::ComputeReq& q) {
    if (q.ne < 1 || q.ne > 8) return false;
    for (int i = 0; i < q.ne; ++i) {
        if (q.keys[i] < 0) return false;
        for (int j = 0; j < i; ++j) if (q.keys[i] == q.keys[j]) return false;
    }
    return true;
}

inline bool compatible(const np::ComputeReq& a, const np::ComputeReq& b) {
    return valid(b) && a.gu_type == b.gu_type && a.d_type == b.d_type && a.down_off == b.down_off &&
           std::memcmp(&a.limit, &b.limit, sizeof(float)) == 0;
}

// A bounded, non-consuming snapshot. Never wait for a partial frame, skip an intervening op, or inspect more
// than two following FP32 requests. A compatible expert key is global, so no layer-number assumption is needed.
inline int ready_prefix(const void* data, size_t bytes, int n_embd, const np::ComputeReq& first,
                        np::ComputeReq (&next)[2]) {
    if (n_embd <= 0 || n_embd > 65536 || !valid(first)) return 0;
    const size_t payload = sizeof(np::ComputeReq) + (size_t) n_embd * sizeof(float);
    const size_t frame = sizeof(np::Hdr) + payload;
    const auto* at = static_cast<const uint8_t*>(data);
    int n = 0;
    while (n < 2 && bytes >= frame) {
        np::Hdr h{};
        np::ComputeReq q{};
        std::memcpy(&h, at, sizeof(h));
        if (h.magic != np::kMagic || h.op != np::kCompute || h.len != payload) break;
        std::memcpy(&q, at + sizeof(h), sizeof(q));
        if (!compatible(first, q)) break;
        next[n++] = q;
        at += frame;
        bytes -= frame;
    }
    return n;
}

inline int peek_ready(int fd, int n_embd, const np::ComputeReq& first, np::ComputeReq (&next)[2],
                      std::vector<uint8_t>& scratch) {
    if (n_embd <= 0 || n_embd > 65536 || !valid(first)) return 0;
    scratch.resize(2 * (sizeof(np::Hdr) + sizeof(np::ComputeReq) + (size_t) n_embd * sizeof(float)));
    const ssize_t bytes = ::recv(fd, scratch.data(), scratch.size(), MSG_PEEK | MSG_DONTWAIT);
    return bytes <= 0 ? 0 : ready_prefix(scratch.data(), (size_t) bytes, n_embd, first, next);
}
}  // namespace strata::helper_probe
