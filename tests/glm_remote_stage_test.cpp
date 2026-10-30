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
    int64_t position = 0;
    int block_rows = 0; int64_t block_position = 0;
    std::vector<int32_t> block_guesses;
    while (st::receive_packet(fd, h, data, err)) {
        if (h.session != session || h.sequence != ++seq) std::abort();
        h.flags = st::reply;
        if (h.op == st::BlockStep) {
            st::BlockInput input;
            if (data.size() != sizeof(input)) std::abort();
            std::memcpy(&input, data.data(), sizeof(input));
            if (block_rows && (input.keep < 1 || input.keep > block_rows || h.position != block_position + input.keep)) std::abort();
            st::BlockReply meta;
            if (block_rows && input.keep == block_rows)
                while (meta.reused < h.rows && meta.reused < block_guesses.size() &&
                       input.tokens[meta.reused] == block_guesses[meta.reused]) ++meta.reused;
            if (meta.reused == h.rows) h.flags |= st::cache_hit;
            block_position = h.position; block_rows = h.rows;
            block_guesses.assign(input.next, input.next + input.guesses);
            h.position += h.rows;
            std::vector<float> residual((size_t) h.rows * i.hc * i.embd, .625f);
            if (mode == 13) meta.reused ^= 1;
            if (mode == 14) residual.pop_back();
            if (mode == 15) --h.position;
            if (mode == 16) meta.reserved = 1;
            std::vector<uint8_t> payload(sizeof(meta) + residual.size()*sizeof(float));
            std::memcpy(payload.data(), &meta, sizeof(meta));
            std::memcpy(payload.data()+sizeof(meta), residual.data(), residual.size()*sizeof(float));
            if (!st::send_packet(fd, h, payload.data(), payload.size(), err)) break;
        } else if (h.op == st::BlockForecast) {
            if (!block_rows || data.size()!=h.rows*sizeof(int32_t) || h.position!=block_position+block_rows) std::abort();
            block_guesses.resize(h.rows);std::memcpy(block_guesses.data(),data.data(),data.size());
            if (!st::send_packet(fd,h,nullptr,0,err)) break;
        } else if (h.op == st::BlockEnd) {
            if (!block_rows || h.rows < 1 || h.rows > (unsigned) block_rows || h.position != block_position + h.rows) std::abort();
            block_rows = 0; block_guesses.clear();
            if (!st::send_packet(fd, h, nullptr, 0, err)) break;
        } else if (h.op == st::StreamStep) {
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
        } else if (h.op == st::SlotRead) {
            h.position=position;
            const uint8_t blob[4]={1,2,3,4};
            if (!st::send_packet(fd,h,blob,sizeof(blob),err)) break;
        } else if (h.op == st::SlotWrite) {
            if (data.size()!=4 || data.front()!=1 || data.back()!=4) std::abort();
            if (!st::send_packet(fd,h,nullptr,0,err)) break;
        } else if (h.op == st::CheckSave || h.op == st::CheckSaveSnapshot || h.op == st::CheckRestore) {
            if (!data.empty()) std::abort();
            if(h.op!=st::CheckRestore)h.position=position;
            if (!st::send_packet(fd,h,nullptr,0,err)) break;
        } else if (h.op == st::PrefillStream) {
            if(data.size()!=((size_t)h.rows+2)*sizeof(int32_t))std::abort();
            int32_t chunk=0;std::memcpy(&chunk,data.data(),sizeof(chunk));
            const int64_t start=h.position;const uint32_t total=h.rows;bool sent=true;
            for(uint32_t at=0;at<total;) {
                st::Header a=h;a.rows=std::min<uint32_t>(chunk,total-at);a.position=start+at+a.rows;
                a.flags=st::reply|(at+a.rows<total?st::more:0);
                std::vector<float> residual((size_t)a.rows*i.hc*i.embd,.875f);
                if(mode==11)--a.position;
                if(mode==12)residual.pop_back();
                if(!st::send_packet(fd,a,residual.data(),residual.size()*sizeof(float),err)){sent=false;break;}
                at+=a.rows;
            }
            if(!sent)break;
            h.position=start+total;
        } else if (h.op == st::Prefill) {
            if (data.size() != ((size_t)h.rows + 1)*sizeof(int32_t)) std::abort();
            std::vector<float> residual((size_t)h.rows*i.hc*i.embd, .75f);
            h.position += h.rows;
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
            if (h.op == st::Reset) { block_rows = 0; block_guesses.clear(); }
            if (h.op == st::Commit) h.position += h.rows;
            if (!st::send_packet(fd, h, nullptr, 0, err)) break;
        }
        position=h.position;
    }
    ::close(fd);
}
}
int main() {
    std::string err;
    st::Header h;
    for (uint32_t op = st::Hello; op <= st::BlockForecast; ++op) {
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
                std::vector<int32_t> prompt(33, 1);
                require(!client.prefill(1, prompt, -1, residual, err), "prefill rejects wrong position");
                require(!client.prefill(0, prompt, -2, residual, err), "prefill rejects invalid next token");
                require(client.prefill(0, prompt, 2, residual, err), "large prefill RPC");
                require(residual.size() == 33*4*4096 && residual.front() == .75f && residual.back() == .75f,
                        "prefill FP32 payload");
                require(client.verify(33, ids, 1, residual, err) && client.commit(1, err), "decode follows prefill position");
                require(client.save(34,err),"snapshot after prefill and decode");
                require(client.checkpoint(st::CheckSave,0,34,err),"paired live checkpoint");
                require(client.checkpoint(st::CheckSaveSnapshot,1,34,err),"paired snapshot checkpoint");
                std::vector<uint8_t> blob;
                require(client.slot_read(34,blob,err) && blob.size()==4,"export front slot bytes");
                require(client.reset(err), "reset after prefill");
                require(client.slot_write(34,blob,err) && client.restore(34,err),"import front slot establishes snapshot");
                require(client.verify(34,ids,1,residual,err) && client.commit(1,err),"decode after slot import");
                require(client.checkpoint(st::CheckRestore,0,34,err),"restore paired checkpoint position");
                require(client.verify(34,ids,1,residual,err) && client.commit(1,err),"decode after checkpoint restore");
                require(client.reset(err),"reset after parking checks");
                size_t consumed=0;
                require(client.prefill_stream(0,prompt,2,16,[&](size_t at,const std::vector<float>& r,std::string&){
                    require(at==consumed && r.front()==.875f && r.back()==.875f,"ordered exact FP32 streamed chunk");
                    consumed+=r.size()/(4*4096);return true;
                },err) && consumed==33,"multi-frame prefill consumed completely");
                require(client.verify(33,ids,1,residual,err) && client.commit(1,err),"decode after streamed prefill");
                require(client.reset(err),"reset before streaming cancellation");
                require(!client.prefill_stream(0,prompt,2,16,[](size_t,const std::vector<float>&,std::string& e){e="cancelled";return false;},err)
                    && err=="cancelled" && client.ready(),"cancelled consumer drains front without losing connection");
                require(client.reset(err),"reset after streamed cancellation");
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
    for(int mode:{11,12}) {
        int sockets[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0,"streamed fault socketpair");
        std::thread server(peer,sockets[1],mode);GlmRemoteStage client;
        require(client.open_connected(sockets[0],info(),err),"streamed fault hello");
        require(!client.prefill_stream(0,std::vector<int32_t>(33,1),2,16,
            [](size_t,const std::vector<float>&,std::string&){return true;},err) && !client.ready(),"bad streamed frame poisons connection");
        client.close();server.join();
    }
    for (int mode : {0, 13, 14, 15, 16}) {
        int sockets[2]; require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "block socketpair");
        std::thread server(peer, sockets[1], mode); GlmRemoteStage client;
        require(client.open_connected(sockets[0], info(), err), "block hello");
        std::vector<float> residual; int reused = -1;
        int32_t input[2] = {1,2}, guesses[2] = {3,4};
        require(!client.block_commit(1,err), "block commit requires window");
        require(!client.block_step(0,input,3,guesses,2,residual,reused,err), "block rows bounded");
        const bool ok = client.block_step(0,input,2,guesses,2,residual,reused,err);
        if (mode) require(!ok && !client.ready(), "malformed block reply poisons connection");
        else {
            require(ok && reused==0 && residual.size()==2*4*4096 && residual[0]==.625f, "initial FP32 block");
            require(!client.block_end(err), "block end requires local commit");
            require(!client.block_step(2,input,2,guesses,2,residual,reused,err), "no overlapping uncommitted block");
            require(!client.save(2,err), "no snapshot while block speculation exists");
            require(!client.block_commit(0,err) && !client.block_commit(3,err), "block keep bounded");
            require(client.block_commit(2,err), "full local block commit");
            require(!client.block_commit(2,err), "duplicate block commit rejected");
            input[0]=3; input[1]=9; guesses[0]=10; guesses[1]=11;
            require(client.block_step(2,input,2,guesses,2,residual,reused,err) && reused==1, "partial prefix reuse");
            require(client.block_commit(1,err), "partial block commit");
            input[0]=10; input[1]=11;
            require(client.block_step(3,input,2,guesses,2,residual,reused,err) && reused==0, "reject invalidates forecasts even if IDs match");
            require(client.block_commit(2,err) && client.block_end(err), "end drains speculative work");
            require(client.save(5,err), "snapshot after block stream");
            require(client.reset(err), "reset after block stream");
            require(client.block_step(0,input,2,guesses,2,residual,reused,err), "restart block stream");
            require(client.reset(err), "reset before block commit");
            require(client.verify(0,input,1,residual,err) && client.commit(1,err), "ordinary path after block reset");
            require(client.block_begin(1,input,2,nullptr,0,err), "asynchronous block request");
            require(!client.block_begin(1,input,2,nullptr,0,err) && !client.reset(err) && !client.block_end(err), "inflight response must be drained");
            require(client.block_finish(residual,reused,err) && reused==0, "asynchronous block response");
            require(client.block_forecast(guesses,2,err), "lookahead supplied after current response");
            require(!client.block_forecast(guesses,2,err), "duplicate forecast rejected");
            require(client.block_commit(2,err), "async local commit");
            require(client.block_step(3,guesses,2,nullptr,0,residual,reused,err) && reused==2, "forecast feeds following block");
            require(client.block_commit(1,err) && client.block_end(err), "async partial end");
        }
        client.close(); server.join();
    }
    std::printf("glm_remote_stage_test: PASS (%d checks, actual CPU socket transport)\n", checks);
}
