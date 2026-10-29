// One local front trunk + its local expert helper. Persistent model; one coordinator/one tentative window.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_remote_stage.hpp"
#include "strata/platform/env_alias.hpp"
#include <unistd.h>
#include <poll.h>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace st = strata::core::stageproto;
int main(int argc, char** argv) {
    strata::platform::env_aliases();
    std::string pack, endpoint, profile, err;
    int cut = 0, device = 0, context = 262144;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::puts("maya-stage-worker --pack DIR --dial IPv4:port --cut LAYER [--context N] [--device N] [--profile FILE]");
            return 0;
        }
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", key.c_str()); return 2; }
        const std::string value = argv[++i];
        if (key == "--pack") pack = value;
        else if (key == "--dial") endpoint = value;
        else if (key == "--profile") profile = value;
        else {
            int n = 0;
            const auto r = std::from_chars(value.data(), value.data() + value.size(), n);
            if (r.ec != std::errc() || r.ptr != value.data() + value.size() || n < 0) return 2;
            if (key == "--cut") cut = n;
            else if (key == "--context") context = n;
            else if (key == "--device") device = n;
            else { std::fprintf(stderr, "unknown option %s\n", key.c_str()); return 2; }
        }
    }
    if (pack.empty() || endpoint.empty() || cut < 1 || cut >= 45 || context < 1) return 2;
    for (const char* key : {"STRATA_GLM_REMOTE_FRONT", "STRATA_GLM_NET_HELPER", "STRATA_GLM_SPLIT_NET", "STRATA_GLM_DFLASH"}) {
        const char* v = std::getenv(key);
        if (v && *v && std::strcmp(v, "0") != 0) { std::fprintf(stderr, "stage worker refuses %s\n", key); return 2; }
    }
    if (const char* v = std::getenv("STRATA_GLM_SEQS"); v && std::strcmp(v, "1") != 0) return 2;
    // Match the home baseline's KV format; explicit environment values remain authoritative and are checked in HELLO.
    if (!std::getenv("STRATA_GLM_KV_INT8")) ::setenv("STRATA_GLM_KV_INT8", "1", 0);
    if (!std::getenv("STRATA_GLM_KV_RESIDENT")) ::setenv("STRATA_GLM_KV_RESIDENT", "32768", 0);
    strata::core::Glm5Model model;
    model.set_expert_profile(profile);
    if (!model.enable_stage_windows(err) || !model.load_pack(pack, context, err, device, 0, cut)) {
        std::fprintf(stderr, "stage worker load: %s\n", err.c_str()); return 1;
    }
    st::Info info;
    if (!model.stage_info(cut, info, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (const auto& r : model.vram_report())
        std::fprintf(stderr, "stage worker GPU%d %s layers [%d,%d) expert pool %.2f GiB\n",
                     r.dev, r.pci.c_str(), r.l0, r.l1, (double) r.pool / 1073741824.0);
    const int fd = st::dial(endpoint, err);
    if (fd < 0) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    struct Fd { int value; ~Fd() { ::close(value); } } closer{fd};
    st::Header hello; hello.op = st::Hello;
    st::Header ack; std::vector<uint8_t> data;
    if (!st::send_packet(fd, hello, &info, sizeof(info), err) || !st::receive_packet(fd, ack, data, err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 1;
    }
    st::Info peer{};
    if (data.size() == sizeof(peer)) std::memcpy(&peer, data.data(), sizeof(peer));
    if (ack.op != st::Hello || ack.flags != st::reply || !ack.session || ack.sequence || ack.window || ack.rows ||
        ack.position || data.size() != sizeof(peer) || !st::same_info(peer, info)) {
        std::fprintf(stderr, "stage HELLO acknowledgment mismatch\n"); return 1;
    }
    const uint64_t session = ack.session;
    uint64_t sequence = 0, last_window = 0, active_window = 0;
    int64_t active_position = 0;
    int64_t saved_position = -1;
    bool streaming = false;
    int64_t stream_position = 0;
    int32_t cached_token = -1;
    strata::core::Glm5Model::StageWindowResult cached;
    std::fprintf(stderr, "stage worker ready: session %llu, layers [0,%d), FP32 boundary\n",
                 (unsigned long long) session, cut);
    for (;;) {
        st::Header h;
        err.clear();
        // An idle model is allowed to wait indefinitely for its next request. Socket timeouts still bound a
        // partially delivered packet after its first byte, and the coordinator's active RPC waits remain finite.
        pollfd ready{fd, POLLIN, 0};
        int readable;
        do { readable = ::poll(&ready, 1, -1); } while (readable < 0 && errno == EINTR);
        if (readable < 0) { std::fprintf(stderr, "stage worker poll failed\n"); return 1; }
        if (!st::receive_packet(fd, h, data, err)) {
            const bool clean = err == "stage peer closed the connection" && !model.stage_window_pending();
            std::fprintf(stderr, "stage worker end: %s\n", err.c_str()); return clean ? 0 : 1;
        }
        bool ok = h.flags == 0 && h.session == session && h.sequence == sequence + 1;
        sequence = h.sequence;
        st::Header response = h; response.flags = st::reply;
        strata::core::Glm5Model::StageWindowResult result;
        int32_t next_prefetch = -1;
        if (ok && h.op == st::Verify) {
            ok = !streaming && h.rows > 0 && h.rows <= 16 && data.size() == h.rows * sizeof(int32_t) &&
                 h.position == model.position() && h.window == last_window + 1 && !model.stage_window_pending();
            if (ok) {
                std::vector<int32_t> tokens(h.rows);
                std::memcpy(tokens.data(), data.data(), data.size());
                ok = model.stage_window(tokens.data(), h.rows, nullptr, 0, result, err);
                if (ok) { active_window = last_window = h.window; active_position = h.position; }
            }
        } else if (ok && h.op == st::Commit) {
            ok = !streaming && data.empty() && h.window == active_window && active_window != 0 && h.position == active_position &&
                 model.stage_commit((int) h.rows, err);
            if (ok) active_window = 0;
        } else if (ok && h.op == st::Reset) {
            ok = data.empty() && !h.rows && !h.window && !h.position;
            if (ok) {
                model.reset(); active_window = 0; saved_position = -1;
                streaming = false; stream_position = 0; cached_token = -1; cached = {};
            }
        } else if (ok && h.op == st::Save) {
            ok = !streaming && data.empty() && !h.rows && !h.window && !model.stage_window_pending() &&
                 h.position == model.position() && model.snapshot_save();
            if (ok) saved_position = h.position;
        } else if (ok && h.op == st::Restore) {
            ok = !streaming && data.empty() && !h.rows && !h.window && !model.stage_window_pending() &&
                 h.position == saved_position && h.position == model.snapshot_pos() && model.snapshot_restore();
        } else if (ok && h.op == st::StreamStep) {
            ok = h.rows == 1 && data.size() == 2 * sizeof(int32_t) && h.window == last_window + 1 &&
                 active_window == 0 && (streaming || !model.stage_window_pending());
            int32_t input[2] = {-1, -1};
            if (ok) std::memcpy(input, data.data(), sizeof(input));
            const int64_t position = streaming ? stream_position : model.position();
            ok = ok && h.position == position && position < info.context && input[0] >= 0 &&
                 (uint32_t) input[0] < info.vocab && input[1] >= -1 &&
                 (input[1] < 0 || ((uint32_t) input[1] < info.vocab && position + 1 < info.context));
            if (ok && cached_token >= 0) {
                ok = model.stage_window_pending() && model.position() == position + 1;
                if (ok && cached_token == input[0]) {
                    result = std::move(cached);
                    response.flags |= st::cache_hit;
                } else if (ok) ok = model.stage_discard(err);
            }
            if (ok && result.residual.empty())
                ok = model.position() == position && model.stage_window(input, 1, nullptr, 0, result, err);
            if (ok) ok = model.stage_commit(1, err);
            if (ok) {
                stream_position = position + 1; streaming = true;
                cached_token = -1; cached = {};
                last_window = h.window; next_prefetch = input[1];
            }
        } else if (ok && h.op == st::StreamEnd) {
            ok = streaming && data.empty() && !h.rows && !h.window && h.position == stream_position;
            if (ok && cached_token >= 0) ok = model.stage_discard(err);
            if (ok) {
                ok = model.position() == stream_position && !model.stage_window_pending();
                streaming = false; cached_token = -1; cached = {};
            }
        } else if (ok && h.op == st::Close && data.empty() && !h.rows && !h.window) {
            return 0;
        } else ok = false;
        if (!ok) {
            if (err.empty()) err = "invalid stage request identity, shape, position or lifecycle";
            response.flags = st::reply | st::failure;
            if (err.size() > 1024) err.resize(1024);
            std::string io_err;
            st::send_packet(fd, response, err.data(), err.size(), io_err);
            std::fprintf(stderr, "stage worker: %s\n", err.c_str()); return 1;
        }
        response.position = model.position();
        const void* payload = result.residual.empty() ? nullptr : result.residual.data();
        if (!st::send_packet(fd, response, payload, result.residual.size() * sizeof(float), err)) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 1;
        }
        // The actual token's response is already on the wire. Compute the next guess while the coordinator
        // runs its current tail; only this one tentative token may stand beyond the committed prefix.
        if (next_prefetch >= 0) {
            if (!model.stage_window(&next_prefetch, 1, nullptr, 0, cached, err)) {
                std::fprintf(stderr, "stage stream prefetch: %s\n", err.c_str()); return 1;
            }
            cached_token = next_prefetch;
        }
    }
}
