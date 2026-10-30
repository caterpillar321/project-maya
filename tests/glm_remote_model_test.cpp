// Home GPU integration test. Run only with the explicit remote stage configuration and the real unchanged pack.
#include "strata/core/glm_model.hpp"
#include "../src/core/glm_fast_state.hpp"
#include "strata/platform/env_alias.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int checks = 0;
void require(bool ok, const std::string& msg) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "REMOTE MODEL FAIL: %s\n", msg.c_str()); std::exit(1); }
}
int pick(const float* p, size_t n) { return (int) (std::max_element(p, p+n)-p); }
void sequence_difference(const std::vector<int>& a, const std::vector<int>& b, const char* label) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (a[i] != b[i]) { std::fprintf(stderr, "%s first mismatch at %zu: plain=%d speculative=%d\n", label, i, a[i], b[i]); return; }
    if (a.size() != b.size()) std::fprintf(stderr, "%s size mismatch %zu/%zu\n", label, a.size(), b.size());
}
void compare(const float* reference, const float* result, size_t n, const char* label) {
    double max_abs = 0, sum_sq = 0, ref_sq = 0;
    for (size_t i = 0; i < n; ++i) {
        require(std::isfinite(result[i]) && std::isfinite(reference[i]), "finite logits");
        const double d = (double) result[i]-reference[i];
        max_abs = std::max(max_abs, std::abs(d)); sum_sq += d*d; ref_sq += (double) reference[i]*reference[i];
    }
    const double relative = std::sqrt(sum_sq / std::max(ref_sq, 1e-20));
    const bool same_pick = pick(reference, n) == pick(result, n);
    std::printf("compare %s max_abs=%.9g relative_l2=%.9g pick_equal=%d\n", label, max_abs, relative, same_pick);
    std::fflush(stdout);
    require(max_abs <= 0.005 && relative <= 1e-4 && same_pick, std::string("logits mismatch: ")+label);
}
}
namespace strata::core {
// Exercise the actual GPU table/move retirement, not a mock of the admission predicate.
struct GlmStageCacheRegression {
    static void late_promotion(Glm5Model& model) {
        auto* f = model.fast_;
        if (!f->stage_ram_canonical) return;
        std::string err;
        cudaSetDevice(model.dev_);
        require(model.fast_boundary(err), err);
        f->bg_hold = true;
        require(cudaStreamSynchronize(f->copy) == cudaSuccess && model.fast_boundary(err), err);
        require(f->bg.empty() && f->draining.empty(), "idle cache regression setup");
        const int ne = model.g_.n_expert;
        for (int il = model.l0_; il < model.lt_; ++il) {
            auto& p = f->lp[(size_t)il];
            int live = -1, spare = -1, sj = -1;
            for (int s = 0; s < p.n; ++s)
                if (p.st[s] == Glm5Model::FastState::kResident && p.key[s] >= 0 &&
                    f->slot_of[(size_t)il*ne+p.key[s]] == s && f->held_ram_slot(il, il*ne+p.key[s]) >= 0) { live=s; break; }
            for (int j=0;j<kernels::glmf::kSpares;++j) if (p.spare[j]>=0) {spare=p.spare[j];sj=j;break;}
            if (live<0 || spare<0) continue;
            const int key=il*ne+p.key[live], rc=f->layer_rc[il], rs=f->ram_of[key];
            auto& r=f->rc[rc];
            const unsigned long long zero=0;
            require(cudaMemcpy(f->tab+f->spare_key(il,sj), &zero, sizeof(zero), cudaMemcpyHostToDevice)==cudaSuccess,
                    "withdraw simulated background destination from spares");
            p.spare[sj]=-1;p.st[spare]=Glm5Model::FastState::kLanding;p.key[spare]=-1;
            r.st[rs]=Glm5Model::FastState::kRPin;
            cudaEvent_t event;
            require(cudaEventCreateWithFlags(&event,cudaEventDisableTiming)==cudaSuccess,"background copy event");
            require(cudaMemcpyAsync(p.slot_ptr(spare),r.base+(size_t)rs*r.stride,f->L[il].blob,
                                    cudaMemcpyHostToDevice,f->copy)==cudaSuccess,"late duplicate copy");
            require(cudaEventRecord(event,f->copy)==cudaSuccess && cudaEventSynchronize(event)==cudaSuccess,
                    "complete the late promotion after foreground already owns a slot");
            f->bg.push_back({il,spare,key,rc,rs,true,event});
            require(model.fast_boundary(err),err);
            unsigned long long gpu=0;
            require(cudaMemcpy(&gpu,f->tab+f->tab_key(key),sizeof(gpu),cudaMemcpyDeviceToHost)==cudaSuccess,"read live GPU mapping");
            require(f->slot_of[key]==live && p.st[live]==Glm5Model::FastState::kResident &&
                    p.st[spare]==Glm5Model::FastState::kFree && gpu==(unsigned long long)p.slot_ptr(live),
                    "late background promotion cannot replace or duplicate the foreground slot");
            require(r.st[rs]==Glm5Model::FastState::kRHold && f->ram_of[key]==rs,"retain canonical RAM after discarded promotion");
            f->bg_hold=false;
            require(model.fast_boundary(err),err);
            std::puts("LATE BACKGROUND PROMOTION REGRESSION PASS");
            return;
        }
        require(false,"regression needs one resident backed expert and one spare");
    }
};
}
int main(int argc, char** argv) {
    strata::platform::env_aliases();
    // Match the real generation entry point: lazy kernel loading can deadlock a host-submitted DMA lane.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    if (argc < 3 || argc > 4) { std::fprintf(stderr, "usage: glm_remote_model_test PACK PROFILE [CONTEXT]\n"); return 2; }
    char* end = nullptr;
    const long long context = argc == 4 ? std::strtoll(argv[3], &end, 10) : 4096;
    if (context < 64 || context > 262144 || (end && *end)) return 2;
    const bool local_diag = std::getenv("MAYA_LOCAL_PREFILL_DIAG") != nullptr;
    require(local_diag || std::getenv("STRATA_GLM_REMOTE_FRONT") != nullptr, "explicit remote front required");
    strata::core::Glm5Model model;
    model.set_expert_profile(argv[2]);
    std::string err;
    if (std::getenv("MAYA_FRONT_PREFILL_REFERENCE")) {
        require(model.enable_stage_windows(err),err);
        require(model.load_pack(argv[1],context,err,0,0,16),err);
        for (int i=0;i<40;++i) {
            int32_t tok=1000+(i*137)%7000;strata::core::Glm5Model::StageWindowResult r;
            require(model.stage_window(&tok,1,nullptr,0,r,err) && model.stage_commit(1,err),err);
        }
        for (int n : {1,2,16,33}) {
            model.reset();std::vector<int32_t> ids;
            for(int i=0;i<n;++i)ids.push_back(1000+(i*137)%7000);
            require(model.prefill(ids,err,1000+(n*137)%7000),err);
            int32_t tok=1000+(n*137)%7000;strata::core::Glm5Model::StageWindowResult r;
            require(model.stage_window(&tok,1,nullptr,0,r,err) && model.stage_commit(1,err),err);
        }
        std::puts("LEGACY FRONT PREFILL REFERENCE EXPORTED");return 0;
    }
    const char* pf_reference = std::getenv("MAYA_STAGE_PREFILL_REFERENCE");
    if (local_diag && pf_reference) require(model.enable_stage_windows(err), err);
    require(model.load_pack_env(argv[1], context, err), err);
    require((local_diag || model.remote_front_enabled()) && model.mtp_ready(), "remote native MTP must be active");
    const size_t V = model.geometry().n_vocab;
    const auto long_native_greedy = [&] {
        // Actual low-effort Korean chat prompt, including the template's think marker. Cross several
        // recurrent/indexer pool boundaries without forcing the emitted continuation.
        const std::vector<int32_t> prompt = {154822,154824,154826,25062,287,29905,371,25,12035,154827,
            145382,41436,59895,59284,28630,53595,17851,57456,78992,33720,57772,246,13,154828,154841};
        require(context >= 256, "long native test needs at least 256 context tokens");
        std::vector<std::vector<float>> prior_logits;
        std::vector<int> prior_tokens;
        for (int epoch = 0; epoch < 2; ++epoch) {
            std::vector<float> logits;
            std::vector<int> plain;
            model.reset();
            model.observe_stage_helper_usage();
            require(model.forward(std::vector<int32_t>(prompt.begin(), prompt.end()-1), logits, err), err);
            require(model.snapshot_save(), "save real chat prefix before its last token");
            require(model.forward({prompt.back()}, logits, err), err);
            bool same_prefix = true;
            double epoch_max_abs = 0;
            for (int i = 0; i < 160; ++i) {
                const int token = pick(logits.data(), V); plain.push_back(token);
                if (epoch == 0) prior_logits.push_back(logits);
                else if (same_prefix) {
                    double worst = 0;
                    for (size_t j = 0; j < V; ++j) worst = std::max(worst, std::abs((double)logits[j]-prior_logits[i][j]));
                    epoch_max_abs = std::max(epoch_max_abs, worst);
                    if (token != prior_tokens[i]) {
                        std::printf("helper ownership changed greedy at %d: %d -> %d, same-prefix logits max_abs=%g\n",
                                    i, prior_tokens[i], token, worst);
                        same_prefix = false;
                    }
                }
                require(model.forward({token}, logits, err), err);
            }
            // Reuse the same saved prompt twice. The second restore follows block rollback checkpoints,
            // as a server request does; those checkpoints must not overwrite the conversation snapshot.
            for (int replay = 0; replay < 2; ++replay) {
                require(model.snapshot_restore(), "restore real chat prefix after native generation");
                require(model.forward({prompt.back()}, logits, err), err);
                strata::kernels::SamplerParams sampler{}; sampler.greedy = true;
                std::vector<int> speculative;
                int64_t produced = 0;
                require(model.decode_mtp(sampler, 160, [&](int token) { speculative.push_back(token); return true; }, produced, err), err);
                sequence_difference(plain, speculative, "long native Korean greedy");
                require(produced == 160 && plain == speculative, "160-token native Korean MTP equals plain decode");
            }
            if (epoch == 0) {
                prior_tokens = plain;
                require(model.refresh_stage_helper(err), err);
            } else std::printf("helper ownership same-prefix logits max_abs=%g, all greedy equal=%d\n", epoch_max_abs, same_prefix);
        }
        std::puts("LONG NATIVE KOREAN GREEDY PASS (160 unforced tokens)");
    };
    if (std::getenv("MAYA_STAGE_LONG_GREEDY_ONLY")) { long_native_greedy(); return 0; }
    const bool refresh = std::getenv("STRATA_GLM_STAGE_HELPER_REFRESH") != nullptr;
    // Forced valid token IDs isolate state/geometry from tokenizer and sampling changes. The speed harness is separate.
    std::vector<int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back(1000 + (i * 137) % 7000);
    std::vector<std::vector<float>> reference;
    if (refresh) {
        std::vector<float> warm;
        model.observe_stage_helper_usage();
        require(model.forward(tokens, warm, err), err);
        const auto position = model.position();
        require(model.refresh_stage_helper(err), err);
        require(model.position() == position, "helper refresh leaves position unchanged");
        model.reset();
    }
    for (int32_t t : tokens) {
        std::vector<float> logits;
        require(model.forward({t}, logits, err), err);
        reference.push_back(std::move(logits));
    }
    if (!local_diag) strata::core::GlmStageCacheRegression::late_promotion(model);
    if (std::getenv("MAYA_STAGE_PREFILL_DIAG")) {
        for (int n : {1, 2, 16, 33}) {
            model.reset();
            std::vector<float> got;
            const std::vector<int32_t> prompt(tokens.begin(), tokens.begin()+n);
            if (local_diag && pf_reference) {
                std::vector<float> unused;
                require(model.stage_prefill(prompt, tokens[n], nullptr, 0, unused, err), err);
            } else require(model.prefill(prompt, err, tokens[n]), err);
            require(model.forward({}, got, err), err);
            if (pf_reference) {
                const std::string path = std::string(pf_reference)+"-"+std::to_string(n)+".f32";
                if (local_diag) {
                    FILE* f=std::fopen(path.c_str(),"wb"); require(f!=nullptr,"open native prefill reference");
                    require(std::fwrite(got.data(),sizeof(float),V,f)==V,"write native prefill reference");std::fclose(f);
                } else {
                    std::vector<float> ref(V); FILE* f=std::fopen(path.c_str(),"rb"); require(f!=nullptr,"read native prefill reference");
                    require(std::fread(ref.data(),sizeof(float),V,f)==V,"complete native prefill reference");std::fclose(f);
                    double num=0,den=0,worst=0;for(size_t j=0;j<V;++j){double d=got[j]-ref[j];num+=d*d;den+=(double)ref[j]*ref[j];worst=std::max(worst,std::abs(d));}
                    std::printf("PREFILL NATIVE PARITY n=%d worst=%g relative=%g pick=%d/%d\n",n,worst,std::sqrt(num/den),pick(got.data(),V),pick(ref.data(),V));
                }
            }
            double direct_num=0,direct_den=0,direct_worst=0;
            for (size_t j=0;j<V;++j) { double d=got[j]-reference[n-1][j]; direct_worst=std::max(direct_worst,std::abs(d));direct_num+=d*d;direct_den+=(double)reference[n-1][j]*reference[n-1][j]; }
            std::printf("PREFILL DIAG directlogits n=%d worst=%g relative=%g\n",n,direct_worst,std::sqrt(direct_num/direct_den));
            strata::kernels::SamplerParams greedy{}; greedy.greedy=true;
            std::printf("PREFILL DIAG direct n=%d pick=%d/%d\n", n, model.sample_token(greedy,err), pick(reference[n-1].data(),V));
            require(model.forward({tokens[n]}, got, err), err);
            double worst=0, num=0, den=0;
            for (size_t j=0; j<V; ++j) { double d=got[j]-reference[n][j]; worst=std::max(worst,std::abs(d));num+=d*d;den+=(double)reference[n][j]*reference[n][j]; }
            std::printf("PREFILL DIAG n=%d worst=%g relative=%g pick=%d/%d\n",n,worst,std::sqrt(num/den),pick(got.data(),V),pick(reference[n].data(),V));
            std::fflush(stdout);
        }
    }
    if (local_diag) { std::puts("LOCAL PREFILL DIAGNOSTIC COMPLETE (not a correctness gate)"); return 0; }
    model.reset();
    std::vector<float> reset_logits;
    require(model.forward({tokens[0]}, reset_logits, err), err);
    compare(reference[0].data(), reset_logits.data(), V, "T1 reset control");
    for (int prefix : {0, 3, 5, 31}) for (int T : {2, 3}) for (int keep = 1; keep <= T; ++keep) {
        std::printf("case prefix=%d T=%d keep=%d\n", prefix, T, keep); std::fflush(stdout);
        model.reset();
        std::vector<float> logits;
        if (prefix) require(model.forward(std::vector<int32_t>(tokens.begin(), tokens.begin()+prefix), logits, err), err);
        std::vector<int32_t> out((size_t) T);
        require(model.window(tokens.data()+prefix, T, out.data(), err), err);
        if (refresh) {
            require(!model.refresh_stage_helper(err), "refresh refuses a pending verification window");
            err.clear();
        }
        require(model.window_logits(logits, err), err);
        for (int row = 0; row < T; ++row) compare(reference[prefix+row].data(), logits.data()+(size_t)row*V, V, "verify row");
        require(!model.window_commit(0, err), "invalid keep must leave both windows pending");
        err.clear();
        require(model.window_commit(keep, err), err);
        require(model.position() == prefix+keep, "committed position");
        require(model.forward({tokens[prefix+keep]}, logits, err), err);
        compare(reference[prefix+keep].data(), logits.data(), V, "after rejection/acceptance");
    }
    model.reset();
    const std::vector<int32_t> prefix(tokens.begin(), tokens.begin()+33);
    // Keep the original token-window checks exact. Batched GEMM has its own same-GPU legacy reference;
    // comparing it to Q8 token GEMV would conflate pre-existing arithmetic differences with transport bugs.
    const bool batch_pf = std::getenv("STRATA_GLM_STAGE_PREFILL") && std::string(std::getenv("STRATA_GLM_STAGE_PREFILL"))=="1";
    std::vector<float> logits;
    if (batch_pf) require(model.forward(prefix, logits, err), err);
    else require(model.prefill(prefix, err), err);
    require(model.snapshot_save(), "paired snapshot save");
    require(model.forward({tokens[33]}, logits, err), err);
    compare(reference[33].data(), logits.data(), V, "bounded prefill");
    require(model.snapshot_restore(), "paired snapshot restore");
    require(model.forward({tokens[33]}, logits, err), err);
    compare(reference[33].data(), logits.data(), V, "paired prefix restore");

    // Both hosts' checkpoints and self-contained parked slots preserve the original exact token path.
    model.reset();require(model.forward(prefix,logits,err),err);
    require(model.snapshot_save(),"snapshot before paired parking");
    require(model.ckpt_save_snapshot(0,err),err);
    require(model.forward({tokens[33]},logits,err),err);
    require(model.ckpt_save(1,err),err);
    require(model.forward({tokens[34]},logits,err),err);
    require(model.ckpt_restore(1,err),err);
    require(model.forward({tokens[34]},logits,err),err);
    compare(reference[34].data(),logits.data(),V,"paired live checkpoint");
    require(model.ckpt_restore(0,err),err);
    require(model.forward({tokens[33]},logits,err),err);
    compare(reference[33].data(),logits.data(),V,"paired snapshot checkpoint");
    std::vector<uint8_t> parked;
    require(model.slot_save(parked,err)>0,err);
    model.reset();require(model.forward({tokens[8],tokens[9]},logits,err),err);
    require(model.slot_load(parked,33,err),err);
    require(model.forward({tokens[33]},logits,err),err);
    compare(reference[33].data(),logits.data(),V,"paired parked conversation");

    if (batch_pf) {
        model.reset();require(model.prefill(prefix,err,tokens[33]),err);
        std::vector<float> batch_head,batch_next;
        require(model.forward({},batch_head,err),err);
        require(model.snapshot_save(),"snapshot after batched prefill");
        require(model.forward({tokens[33]},batch_next,err),err);
        require(model.snapshot_restore(),"restore batched prefill snapshot");
        require(model.forward({tokens[33]},logits,err),err);
        compare(batch_next.data(),logits.data(),V,"batch prefill snapshot continuation");
        std::vector<uint8_t> batch_slot;
        require(model.slot_save(batch_slot,err)>0,err);
        model.reset();require(model.slot_load(batch_slot,33,err),err);
        require(model.forward({tokens[33]},logits,err),err);
        compare(batch_next.data(),logits.data(),V,"batch prefill parked continuation");
        // Sampling after a batch-prefilled prefix compares against exactly that snapshot, not a rerun
        // whose expert residency could choose another pre-existing MMQ/light arithmetic path.
        require(model.snapshot_restore(),"batch snapshot before sampled reference");
        require(model.forward({tokens[33]},logits,err),err);
        strata::kernels::SamplerParams sp{};sp.seed=2468;sp.temperature=.8f;
        std::vector<int> plain_batch,mtp_batch;
        for(int i=0;i<24;++i){int t=model.sample_token(sp,err);require(t>=0,err);++sp.counter;plain_batch.push_back(t);require(model.forward({t},logits,err),err);}
        require(model.snapshot_restore(),"batch snapshot before sampled pipeline");
        require(model.forward({tokens[33]},logits,err),err);sp.counter=0;int64_t n_out=0;
        require(model.decode_mtp(sp,24,[&](int t){mtp_batch.push_back(t);return true;},n_out,err),err);
        require(n_out==24 && plain_batch==mtp_batch,"sampled pipeline after batched prefill equals its plain continuation");
        if(std::getenv("STRATA_GLM_PREFILL_PIPE")) {
            std::vector<int32_t> longer(257);for(int i=0;i<257;++i)longer[i]=1000+(i*137)%7000;
            model.reset();require(model.prefill(longer,err,4321),err);
            require(model.position()==257 && model.snapshot_save(),"multi-chunk paired prefill position and snapshot");
            require(model.forward({4321},batch_next,err),err);
            require(model.snapshot_restore(),"restore multi-chunk prefix");
            require(model.forward({4321},logits,err),err);
            compare(batch_next.data(),logits.data(),V,"multi-chunk prefill continuation");
            model.reset();model.prefill_progress=[](int64_t,int64_t){return false;};
            require(!model.prefill(longer,err,4321) && err=="cancelled","cancel streamed prefill");
            model.prefill_progress=nullptr;model.reset();
            require(model.forward({tokens[0]},logits,err),err);
            compare(reference[0].data(),logits.data(),V,"reset after streamed prefill cancellation");
        }
    }

    if (refresh && batch_pf) {
        model.reset(); require(model.prefill(prefix, err, tokens[33]), err);
        std::vector<float> before;
        require(model.forward({}, before, err), err);
        const auto position = model.position();
        require(model.refresh_stage_helper(err), err);
        require(model.position() == position, "refresh after batch preserves position");
        require(model.forward({}, logits, err), err);
        compare(before.data(), logits.data(), V, "helper refresh preserves current head");
        require(model.snapshot_save(), "snapshot after batch and helper exchange");
        require(model.forward({tokens[33]}, before, err), err);
        require(model.snapshot_restore(), "restore after batch and helper exchange");
        require(model.forward({tokens[33]}, logits, err), err);
        compare(before.data(), logits.data(), V, "helper exchange snapshot continuation");
    }

    // Front speculation must preserve exact target logits for forced hits, forced misses and cancellation.
    // The first reference epoch may differ after explicit helper refresh, so regenerate for the current one.
    model.reset(); reference.clear();
    for (int32_t t : tokens) {
        require(model.forward({t}, logits, err), err);
        reference.push_back(logits);
    }
    for (int pre : {0, 31}) for (int mode : {0, 1, 2}) {
        model.reset();
        if (pre) require(model.forward(std::vector<int32_t>(tokens.begin(), tokens.begin()+pre), logits, err), err);
        for (int i = 0; i < 8; ++i) {
            const bool wrong = mode == 1 || (mode == 2 && (i & 1));
            const int32_t guess = tokens[pre+i+1] + (wrong ? 1 : 0);
            bool hit = false;
            require(model.stage_stream_token(tokens[pre+i], guess, hit, logits, err), err);
            const bool prior_wrong = mode == 1 || (mode == 2 && ((i-1) & 1));
            require(hit == (i > 0 && !prior_wrong), "worker cache-hit identity");
            compare(reference[pre+i].data(), logits.data(), V, "stream forced hit/miss");
        }
        require(model.stage_stream_finish(err), err); // discards an unused ninth-token guess
        require(model.position() == pre+8, "stream finish preserves committed prefix");
        require(model.forward({tokens[pre+8]}, logits, err), err);
        compare(reference[pre+8].data(), logits.data(), V, "plain after stream end");
    }
    model.reset();
    bool hit = false;
    require(model.stage_stream_token(tokens[0], tokens[1], hit, logits, err), err);
    model.reset(); // reset while the worker is computing a speculative next token
    require(model.forward({tokens[0]}, logits, err), err);
    compare(reference[0].data(), logits.data(), V, "reset during front speculation");

    // Two-row front speculation: exact, partial and rejected blocks at indexer-pool boundaries.
    for (int pre : {0, 3, 31, 32}) for (int mode : {0, 1, 2, 3}) {
        model.reset();
        if (pre) require(model.forward(std::vector<int32_t>(tokens.begin(), tokens.begin()+pre), logits, err), err);
        int p = pre;
        for (int step = 0; step < 3; ++step) {
            const int keep = mode == 3 && (step & 1) ? 1 : 2;
            int32_t input[2] = {tokens[p], tokens[p+1] + (keep == 1 ? 1 : 0)};
            int32_t next[2] = {tokens[p+2] + (mode == 1 ? 1 : 0), tokens[p+3] + (mode == 2 ? 1 : 0)};
            strata::core::Glm5Model::StageWindowResult r;
            require(model.stage_block_window(input, 2, next, 2, r, err, true), err);
            for (int row = 0; row < keep; ++row)
                compare(reference[p+row].data(), r.logits.data()+(size_t)row*V, V, "block stream row");
            require(model.stage_commit(keep, err), err); p += keep;
            require(model.position() == p, "block stream accepted position");
        }
        require(model.stage_block_finish(err), err);
        require(model.forward({tokens[p]}, logits, err), err);
        compare(reference[p].data(), logits.data(), V, "plain after block stream");
    }
    model.reset();
    {
        strata::core::Glm5Model::StageWindowResult r;
        require(model.stage_block_window(tokens.data(), 2, tokens.data()+2, 2, r, err), err);
        require(model.stage_commit(1, err), err);
        require(model.stage_block_finish(err), err); // final partial keep rolls back the already-speculated front
        require(model.forward({tokens[1]}, logits, err), err);
        compare(reference[1].data(), logits.data(), V, "partial block end");
        model.reset();
        require(model.stage_block_window(tokens.data(), 2, tokens.data()+2, 2, r, err), err);
        model.reset(); // reset with both an uncommitted tail and a speculative front
        require(model.forward({tokens[0]}, logits, err), err);
        compare(reference[0].data(), logits.data(), V, "reset during block speculation");
    }

    // Native MTP must match the SAME distributed trunk's ordinary greedy generation, including state after rejection.
    model.reset();
    if (refresh) {
        // Snapshots save sequence state for an extending prompt, not cached logits / last_tok_. Save before
        // the final prefix token and replay it on both branches so MTP cannot sample the old continuation.
        require(model.forward(std::vector<int32_t>(prefix.begin(), prefix.end()-1), logits, err), err);
        require(model.refresh_stage_helper(err), err);
        require(model.snapshot_save(), "save an active prefix after helper refresh");
        require(model.forward({prefix.back()}, logits, err), err);
    } else require(model.forward(prefix, logits, err), err);
    const auto prefix_head = logits;
    std::vector<int> plain;
    for (int i = 0; i < 24; ++i) {
        const int t = pick(logits.data(), V); plain.push_back(t);
        require(model.forward({t}, logits, err), err);
    }
    if (refresh) {
        require(model.snapshot_restore(), "restore the same prefix and helper epoch for MTP");
        require(model.forward({prefix.back()}, logits, err), err);
        compare(prefix_head.data(), logits.data(), V, "restored prefix head before MTP");
    }
    else {
        model.reset();
        require(model.forward(prefix, logits, err), err);
    }
    strata::kernels::SamplerParams sampler{}; sampler.greedy = true;
    std::vector<int> mtp;
    int64_t produced = 0;
    require(model.decode_mtp(sampler, 24, [&](int token) { mtp.push_back(token); return true; }, produced, err), err);
    sequence_difference(plain, mtp, "greedy");
    require(produced == 24 && plain == mtp, "native MTP greedy output equals distributed plain decode");
    model.reset();
    require(model.forward(prefix, logits, err), err);
    strata::kernels::SamplerParams sampled{};
    sampled.seed = 12345; sampled.temperature = 0.8f;
    plain.clear(); mtp.clear();
    for (int i = 0; i < 24; ++i) {
        const int token = model.sample_token(sampled, err);
        require(token >= 0, err); ++sampled.counter; plain.push_back(token);
        require(model.forward({token}, logits, err), err);
    }
    model.reset();
    require(model.forward(prefix, logits, err), err);
    sampled.counter = 0; produced = 0;
    require(model.decode_mtp(sampled, 24, [&](int token) { mtp.push_back(token); return true; }, produced, err), err);
    sequence_difference(plain, mtp, "sampled");
    require(produced == 24 && plain == mtp, "fixed-seed sampled MTP equals distributed plain decode");
    model.reset();
    require(model.forward(prefix, logits, err), err);
    produced = 0; int callbacks = 0;
    require(model.decode_mtp(sampler, 24, [&](int) { return ++callbacks < 7; }, produced, err), err);
    require(produced == 7 && model.position() == (int64_t) prefix.size()+6, "decode callback cancellation drains stages");
    require(model.snapshot_save(), "snapshot allowed after cancelled generation");
    // Failed/pending stages must not restore a checkpoint over an uncommitted remote window.
    model.reset();
    require(model.forward(prefix, logits, err), err);
    const auto checkpoint_head = logits;
    require(model.ckpt_save(0, err), err);
    int32_t pending_picks[2];
    require(model.window(tokens.data(), 2, pending_picks, err), err);
    require(!model.ckpt_restore(0, err) && model.stage_window_pending(),
            "checkpoint restore refuses a pending stage without mutating it");
    model.reset();
    require(model.forward(prefix, logits, err), err);
    compare(checkpoint_head.data(), logits.data(), V, "reset after refused checkpoint restore");
    if (context >= 256) long_native_greedy();
    model.fast_prof_flush();
    std::printf("glm_remote_model_test: PASS (%d checks; T2/T3 every keep, pool boundaries, snapshot, MTP)\n", checks);
}
