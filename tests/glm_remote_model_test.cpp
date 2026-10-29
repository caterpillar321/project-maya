// Home GPU integration test. Run only with the explicit remote stage configuration and the real unchanged pack.
#include "strata/core/glm_model.hpp"
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
int main(int argc, char** argv) {
    strata::platform::env_aliases();
    if (argc < 3 || argc > 4) { std::fprintf(stderr, "usage: glm_remote_model_test PACK PROFILE [CONTEXT]\n"); return 2; }
    char* end = nullptr;
    const long long context = argc == 4 ? std::strtoll(argv[3], &end, 10) : 4096;
    if (context < 64 || context > 262144 || (end && *end)) return 2;
    require(std::getenv("STRATA_GLM_REMOTE_FRONT") != nullptr, "explicit remote front required");
    strata::core::Glm5Model model;
    model.set_expert_profile(argv[2]);
    std::string err;
    require(model.load_pack_env(argv[1], context, err), err);
    require(model.remote_front_enabled() && model.mtp_ready(), "remote native MTP must be active");
    const size_t V = model.geometry().n_vocab;
    const bool refresh = std::getenv("STRATA_GLM_STAGE_HELPER_REFRESH") != nullptr;
    // Forced valid token IDs isolate state/geometry from tokenizer and sampling changes. The speed harness is separate.
    std::vector<int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back(1000 + (i * 137) % 7000);
    std::vector<std::vector<float>> reference;
    if (refresh) {
        std::vector<float> warm;
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
    require(model.prefill(prefix, err), err);
    require(model.snapshot_save(), "paired snapshot save");
    std::vector<float> logits;
    require(model.forward({tokens[33]}, logits, err), err);
    compare(reference[33].data(), logits.data(), V, "bounded prefill");
    require(model.snapshot_restore(), "paired snapshot restore");
    require(model.forward({tokens[33]}, logits, err), err);
    compare(reference[33].data(), logits.data(), V, "paired prefix restore");

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

    // Native MTP must match the SAME distributed trunk's ordinary greedy generation, including state after rejection.
    model.reset();
    require(model.forward(prefix, logits, err), err);
    if (refresh) {
        require(model.refresh_stage_helper(err), err);
        require(model.snapshot_save(), "save an active prefix after helper refresh");
    }
    std::vector<int> plain;
    for (int i = 0; i < 24; ++i) {
        const int t = pick(logits.data(), V); plain.push_back(t);
        require(model.forward({t}, logits, err), err);
    }
    if (refresh) require(model.snapshot_restore(), "restore the same prefix and helper epoch for MTP");
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
    model.fast_prof_flush();
    std::printf("glm_remote_model_test: PASS (%d checks; T2/T3 every keep, pool boundaries, snapshot, MTP)\n", checks);
}
