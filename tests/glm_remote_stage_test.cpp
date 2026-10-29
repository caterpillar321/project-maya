// Real socketpair transport with a CPU-only peer. No CUDA initialization, model weights, or GPU work.
#include "strata/core/glm_remote_stage.hpp"
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace st = strata::core::stageproto;
using strata::core::GlmRemoteStage;
namespace {
int checks = 0;
void require(bool value, const char* msg) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}
st::Info info() {
    st::Info i{}; i.manifest = 123; i.context = 64; i.cut = 18; i.layers = 45;
    i.embd = 4096; i.hc = 4; i.vocab = 154880; i.experts = 288; i.top_k = 8; i.nextn = 1; i.kv_bits = 8;
    return i;
}
// The peer exits whenever the owning client closes. Negative modes intentionally corrupt one response.
void peer(int fd, int mode) {
    std::string err;
    st::Info i = info();
    if (mode == 1) ++i.manifest;
    st::Header h; h.op = st::Hello;
    if (!st::send_packet(fd, h, &i, sizeof(i), err)) std::abort();
    std::vector<uint8_t> data;
    if (!st::receive_packet(fd, h, data, err)) { ::close(fd); return; }
    const uint64_t session = h.session;
    uint64_t seq = 0;
    while (st::receive_packet(fd, h, data, err)) {
        if (h.session != session || h.sequence != ++seq) std::abort();
        h.flags = st::reply;
        if (h.op == st::Verify) {
            if (data.size() != h.rows * sizeof(int32_t)) std::abort();
            std::vector<float> residual((size_t) h.rows * i.hc * i.embd, 0.25f);
            h.position += h.rows;
            if (mode == 2) ++h.sequence;
            if (mode == 3) ++h.window;
            if (mode == 4) ++h.session;
            if (mode == 5) --h.position;
            if (mode == 6) {
                h.bytes = st::max_payload + 1;
                ::send(fd, &h, sizeof(h), MSG_NOSIGNAL); break;
            }
            if (mode == 7) {
                h.bytes = (uint32_t) (residual.size() * sizeof(float));
                ::send(fd, &h, sizeof(h), MSG_NOSIGNAL);
                ::send(fd, residual.data(), 16, MSG_NOSIGNAL); break;
            }
            if (!st::send_packet(fd, h, residual.data(), residual.size()*sizeof(float), err)) break;
        } else {
            if (!data.empty()) std::abort();
            if (h.op == st::Commit) h.position += h.rows;
            if (!st::send_packet(fd, h, nullptr, 0, err)) break;
        }
    }
    ::close(fd);
}
}
int main() {
    std::string err;
    st::Header h;
    for (uint32_t op = st::Hello; op <= st::Close; ++op) {
        h.op = op;
        require(st::valid_header(h, err), "valid opcode");
    }
    for (int which = 0; which < 7; ++which) {
        h = {}; h.op = st::Verify;
        if (which == 0) h.magic_value = 0;
        if (which == 1) h.version_value = 99;
        if (which == 2) h.op = 99;
        if (which == 3) h.flags = st::failure;
        if (which == 4) h.bytes = st::max_payload + 1;
        if (which == 5) h.rows = 17;
        if (which == 6) h.position = -1;
        require(!st::valid_header(h, err), "reject malformed framing before payload allocation");
    }
    for (int mode = 0; mode < 8; ++mode) {
        int sockets[2]; require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        std::thread server(peer, sockets[1], mode);
        GlmRemoteStage client;
        const bool opened = client.open_connected(sockets[0], info(), err);
        if (mode == 1) {
            require(!opened && !client.ready(), "HELLO identity mismatch closes connection");
        } else {
            require(opened, err.c_str());
            std::vector<float> residual;
            int32_t ids[] = {1, 2, 3};
            require(!client.commit(1, err), "cannot commit before verify");
            require(!client.verify(1, ids, 3, residual, err), "cannot skip a position");
            require(!client.verify(0, ids, 17, residual, err), "oversized window rejected before reading token array");
            ids[1] = -1;
            require(!client.verify(0, ids, 3, residual, err), "invalid token rejected without advancing sequence");
            ids[1] = 2;
            const bool ok = client.verify(0, ids, 3, residual, err);
            if (mode >= 2) {
                require(!ok && !client.ready(), "bad response or truncation poisons connection");
            } else {
                require(ok && client.pending(), "full residual returned with tentative state");
                require(residual.size() == 3*4*4096 && residual.front() == .25f && residual.back() == .25f,
                        "FP32 payload shape and contents");
                require(!client.verify(0, ids, 2, residual, err), "cannot overwrite a tentative window");
                require(!client.save(0, err), "cannot snapshot a tentative window");
                require(!client.commit(0, err) && !client.commit(4, err), "invalid keep leaves pending state intact");
                require(client.commit(2, err) && !client.pending(), "partial acceptance advances exactly keep");
                require(!client.commit(2, err), "duplicate commit is not applied twice");
                require(client.save(2, err), "save committed position");
                require(client.verify(2, ids, 2, residual, err) && client.commit(1, err), "next window and rejection");
                require(!client.restore(1, err), "cannot restore an unrelated prefix");
                require(client.restore(2, err), "paired snapshot restore");
                require(client.verify(2, ids, 1, residual, err) && client.commit(1, err), "T1 after restore");
                require(client.reset(err), "reset clears snapshot and state");
                require(!client.restore(2, err), "old snapshot unavailable after reset");
                require(client.verify(0, ids, 3, residual, err) && client.commit(3, err), "full acceptance after reset");
            }
        }
        client.close(); server.join();
    }
    std::printf("glm_remote_stage_test: PASS (%d checks, actual CPU socket transport)\n", checks);
}
