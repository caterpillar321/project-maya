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
#include <future>
#include <algorithm>
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
    bool block_active = false;
    int block_rows = 0;
    int64_t block_position = 0;
    std::vector<int32_t> block_cached_tokens;
    strata::core::Glm5Model::StageWindowResult block_cached;
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
        if (block_active && h.op != st::BlockStep && h.op != st::BlockEnd && h.op != st::BlockForecast &&
            h.op != st::Reset && h.op != st::Close)
            ok = false;
        sequence = h.sequence;
        st::Header response = h; response.flags = st::reply;
        strata::core::Glm5Model::StageWindowResult result;
        int32_t next_prefetch = -1;
        std::vector<int32_t> block_next;
        std::vector<uint8_t> slot_bytes;
        if (ok && h.op == st::PrefillStream) {
            ok=!streaming && !model.stage_window_pending() && !active_window && h.position==model.position() &&
               h.rows>0 && h.rows<=st::max_prefill_rows && h.window==last_window+1 && data.size()==((size_t)h.rows+2)*sizeof(int32_t);
            std::vector<int32_t> ids;
            if(ok){ids.resize(h.rows+2);std::memcpy(ids.data(),data.data(),data.size());}
            const int chunk=ok?ids[0]:0;
            ok=ok && chunk>=1 && chunk<=(int)st::max_prefill_rows;
            std::future<std::pair<bool,std::string>> sending;
            for(size_t at=0;ok && at<h.rows;) {
                const size_t count=std::min<size_t>(chunk,h.rows-at);
                std::vector<int32_t> part(ids.begin()+1+at,ids.begin()+1+at+count);
                std::vector<float> residual;
                ok=model.stage_prefill(part,ids[1+at+count],nullptr,0,residual,err);
                if(sending.valid()){auto sent=sending.get();if(!sent.first){ok=false;err=sent.second;}}
                if(!ok)break;
                st::Header frame=response;frame.position=model.position();frame.rows=count;
                if(at+count<h.rows)frame.flags|=st::more;
                sending=std::async(std::launch::async,[fd,frame,r=std::move(residual)](){
                    std::string e;bool done=st::send_packet(fd,frame,r.data(),r.size()*sizeof(float),e);
                    return std::make_pair(done,e);
                });
                at+=count;
            }
            if(sending.valid()){auto sent=sending.get();if(!sent.first){ok=false;err=sent.second;}}
            if(ok){last_window=h.window;continue;}
            if(err.empty())err="invalid streamed prefill request";
            response.flags=st::reply|st::failure;
            std::string io_error;st::send_packet(fd,response,err.data(),std::min<size_t>(1024,err.size()),io_error);
            std::fprintf(stderr,"stage streamed prefill: %s\n",err.c_str());return 1;
        }
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
                block_active = false; block_rows = 0;
                block_cached_tokens.clear(); block_cached = {};
            }
        } else if (ok && h.op == st::Save) {
            ok = !streaming && data.empty() && !h.rows && !h.window && !model.stage_window_pending() &&
                 h.position == model.position() && model.snapshot_save();
            if (ok) saved_position = h.position;
        } else if (ok && h.op == st::Restore) {
            ok = !streaming && data.empty() && !h.rows && !h.window && !model.stage_window_pending() &&
                 h.position == saved_position && h.position == model.snapshot_pos() && model.snapshot_restore();
        } else if (ok && (h.op == st::CheckSave || h.op == st::CheckSaveSnapshot || h.op == st::CheckRestore)) {
            ok = !streaming && !model.stage_window_pending() && !active_window && data.empty() && !h.rows &&
                 h.window >= 1 && h.window <= 256;
            if (ok) {
                const int slot = (int)h.window - 1;
                if (h.op == st::CheckSave) ok = h.position == model.position() && model.ckpt_save(slot, err);
                else if (h.op == st::CheckSaveSnapshot) ok = h.position == model.snapshot_pos() && model.ckpt_save_snapshot(slot, err);
                else ok = model.ckpt_restore(slot, err) && model.position() == h.position;
            }
        } else if (ok && h.op == st::SlotRead) {
            ok = !streaming && !model.stage_window_pending() && !active_window && data.empty() && !h.rows && !h.window &&
                 h.position == model.snapshot_pos() && model.slot_save(slot_bytes, err) > 0;
        } else if (ok && h.op == st::SlotWrite) {
            ok = !streaming && !model.stage_window_pending() && !active_window && !data.empty() && !h.rows && !h.window &&
                 model.slot_load(data, h.position, err);
            if (ok) saved_position = h.position;
        } else if (ok && h.op == st::Prefill) {
            ok = !streaming && !model.stage_window_pending() && !active_window &&
                 h.position == model.position() && h.rows > 0 && h.rows <= st::max_prefill_rows &&
                 h.window == last_window + 1 && data.size() == ((size_t) h.rows + 1)*sizeof(int32_t);
            if (ok) {
                std::vector<int32_t> tokens(h.rows + 1);
                std::memcpy(tokens.data(), data.data(), data.size());
                const int32_t next = tokens.back(); tokens.pop_back();
                ok = model.stage_prefill(tokens, next, nullptr, 0, result.residual, err);
                if (ok) last_window = h.window;
            }
        } else if (ok && h.op == st::BlockStep) {
            st::BlockInput input;
            ok = !streaming && !active_window && h.rows >= 1 && h.rows <= 2 &&
                 h.window == last_window + 1 && data.size() == sizeof(input);
            if (ok) std::memcpy(&input, data.data(), sizeof(input));
            ok = ok && input.guesses >= 0 && input.guesses <= 2 &&
                 h.position <= info.context - h.rows - input.guesses;
            if (ok) for (unsigned i = 0; i < h.rows; ++i)
                ok = ok && input.tokens[i] >= 0 && (uint32_t) input.tokens[i] < info.vocab;
            if (ok) for (int i = 0; i < input.guesses; ++i)
                ok = ok && input.next[i] >= 0 && (uint32_t) input.next[i] < info.vocab;
            int reused = 0;
            if (ok && block_active) {
                ok = input.keep >= 1 && input.keep <= block_rows && h.position == block_position + input.keep;
                if (ok && input.keep < block_rows) {
                    if (model.stage_window_pending()) ok = model.stage_discard(err);
                    block_cached_tokens.clear(); block_cached = {};
                    if (ok) ok = model.stage_block_checkpoint(0, false, err);
                } else if (ok) {
                    while (reused < (int) h.rows && reused < (int) block_cached_tokens.size() &&
                           input.tokens[reused] == block_cached_tokens[(size_t) reused]) ++reused;
                    if (model.stage_window_pending()) {
                        if (reused) {
                            const size_t n = (size_t) reused * info.hc * info.embd;
                            ok = block_cached.residual.size() >= n;
                            if (ok) result.residual.assign(block_cached.residual.begin(), block_cached.residual.begin() + n);
                            if (ok) ok = reused == 2 ? model.stage_block_commit_full(err) : model.stage_commit(reused, err);
                            if (ok && reused == 1 && h.rows == 2) ok = model.stage_block_checkpoint(0, true, err);
                        } else ok = model.stage_discard(err);
                    } else ok = block_cached_tokens.empty();
                }
            } else if (ok) {
                ok = input.keep == 0 && h.position == model.position() && !model.stage_window_pending();
            }
            block_cached_tokens.clear(); block_cached = {};
            if (ok) ok = model.position() == h.position + reused;
            if (ok && reused < (int) h.rows) {
                strata::core::Glm5Model::StageWindowResult rest;
                const int n = (int) h.rows - reused;
                ok = model.stage_window(input.tokens + reused, n, nullptr, 0, rest, err);
                if (ok) ok = n == 2 ? model.stage_block_commit_full(err) : model.stage_commit(n, err);
                if (ok) result.residual.insert(result.residual.end(), rest.residual.begin(), rest.residual.end());
            }
            if (ok) ok = model.position() == h.position + h.rows;
            if (ok) {
                block_active = true; block_position = h.position; block_rows = h.rows;
                block_next.assign(input.next, input.next + input.guesses);
                last_window = h.window;
                st::BlockReply meta; meta.reused = reused;
                if (reused == (int) h.rows) response.flags |= st::cache_hit;
                slot_bytes.resize(sizeof(meta) + result.residual.size() * sizeof(float));
                std::memcpy(slot_bytes.data(), &meta, sizeof(meta));
                std::memcpy(slot_bytes.data() + sizeof(meta), result.residual.data(), result.residual.size() * sizeof(float));
            }
        } else if (ok && h.op == st::BlockForecast) {
            ok = block_active && !streaming && !model.stage_window_pending() && block_cached_tokens.empty() &&
                 h.window == last_window && h.rows >= 1 && h.rows <= 2 && data.size() == h.rows*sizeof(int32_t) &&
                 h.position == model.position() && h.position == block_position + block_rows &&
                 h.position <= info.context - h.rows;
            if (ok) {
                block_next.resize(h.rows); std::memcpy(block_next.data(), data.data(), data.size());
                for (auto t : block_next) ok = ok && t >= 0 && (uint32_t) t < info.vocab;
            }
        } else if (ok && h.op == st::BlockEnd) {
            ok = block_active && !streaming && data.empty() && !h.window && h.rows >= 1 &&
                 h.rows <= (uint32_t) block_rows && h.position == block_position + h.rows;
            if (ok && model.stage_window_pending()) ok = model.stage_discard(err);
            if (ok && h.rows < (uint32_t) block_rows) {
                ok = model.stage_block_checkpoint(0, false, err);
            }
            if (ok) {
                ok = model.position() == h.position && !model.stage_window_pending();
                block_active = false; block_rows = 0; block_cached_tokens.clear(); block_cached = {};
            }
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
        const void* payload = !slot_bytes.empty() ? (const void*)slot_bytes.data() :
                              result.residual.empty() ? nullptr : (const void*)result.residual.data();
        const size_t payload_size = !slot_bytes.empty() ? slot_bytes.size() : result.residual.size()*sizeof(float);
        if (!st::send_packet(fd, response, payload, payload_size, err)) {
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
        if (!block_next.empty()) {
            if (!model.stage_window(block_next.data(), (int) block_next.size(), nullptr, 0, block_cached, err)) {
                std::fprintf(stderr, "stage block prefetch: %s\n", err.c_str()); return 1;
            }
            block_cached_tokens = std::move(block_next);
        }
    }
}
