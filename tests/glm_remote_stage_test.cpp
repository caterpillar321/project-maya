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
    int32_t guessed = -1;
    while (st::receive_packet(fd, h, data, err)) {
        if (h.session != session || h.sequence != ++seq) std::abort();
        h.flags = st::reply;
        if (h.op == st::StreamStep) {
            if (data.size() != 2 * sizeof(int32_t)) std::abort();
            int32_t input[2]; std::memcpy(input, data.data(), sizeof(input));
            if (guessed == input[0]) h.flags |= st::cache_hit;
            guessed = input[1];
            ++h.position;
            std::vector<float> residual((size_t) i.hc * i.embd, .5f);
            if (mode == 8) --h.position;
            if (mode == 9) h.flags ^= st::cache_hit;
            if (mode == 10) residual.pop_back();
            if (!st::send_packet(fd, h, residual.data(), residual.size()*sizeof(float), err)) break;
        } else if (h.op == st::Verify) {
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
            if (h.op == st::Reset || h.op == st::StreamEnd) guessed = -1;
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
    for (uint32_t op = st::Hello; op <= st::StreamEnd; ++op) {
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
    h = {}; h.op = st::Verify; h.flags = st::reply | st::cache_hit;
    require(!st::valid_header(h, err), "cache-hit flag only belongs to streaming replies");
    h.op = st::StreamStep;
    require(st::valid_header(h, err), "stream cache-hit reply");
    h.flags |= st::failure;
    require(!st::valid_header(h, err), "failed response cannot claim a cached result");
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
                bool hit = true;
                require(!client.stream_step(3, -1, 2, residual, hit, err), "stream rejects invalid actual token");
                require(!client.stream_step(3, 1, -2, residual, hit, err), "stream rejects invalid guess");
                require(client.stream_step(3, 1, 2, residual, hit, err) && !hit && client.streaming(), "first stream step");
                require(residual.size() == 4*4096 && residual.back() == .5f, "stream FP32 residual");
                require(!client.verify(4, ids, 1, residual, err) && !client.save(4, err) && !client.commit(1, err),
                        "ordinary operations cannot interleave a tentative streamed guess");
                require(client.stream_step(4, 2, 3, residual, hit, err) && hit, "matching guess reused");
                require(client.stream_step(5, 4, -1, residual, hit, err) && !hit, "wrong guess replaced by actual token");
                require(client.stream_end(err) && !client.streaming(), "stream end returns to ordinary lifecycle");
                require(client.verify(6, ids, 1, residual, err) && client.commit(1, err), "ordinary window after stream");
                require(client.stream_step(7, 1, 2, residual, hit, err) && client.reset(err) && !client.streaming(),
                        "reset discards a streamed guess");
            }
        }
        client.close(); server.join();
    }
    for (int mode = 8; mode <= 10; ++mode) {
        int sockets[2]; require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "stream negative socketpair");
        std::thread server(peer, sockets[1], mode);
        GlmRemoteStage client; require(client.open_connected(sockets[0], info(), err), "stream negative HELLO");
        std::vector<float> residual; bool hit = false;
        require(!client.stream_step(0, 1, 2, residual, hit, err) && !client.ready(), "corrupt stream result closes session");
        client.close(); server.join();
    }
    std::printf("glm_remote_stage_test: PASS (%d checks, actual CPU socket transport)\n", checks);
}
