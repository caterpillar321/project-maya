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
    // Forced valid token IDs isolate state/geometry from tokenizer and sampling changes. The speed harness is separate.
    std::vector<int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back(1000 + (i * 137) % 7000);
    std::vector<std::vector<float>> reference;
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

    // Native MTP must match the SAME distributed trunk's ordinary greedy generation, including state after rejection.
    model.reset();
    require(model.forward(prefix, logits, err), err);
    std::vector<int> plain;
    for (int i = 0; i < 24; ++i) {
        const int t = pick(logits.data(), V); plain.push_back(t);
        require(model.forward({t}, logits, err), err);
    }
    model.reset();
    require(model.forward(prefix, logits, err), err);
    strata::kernels::SamplerParams sampler{}; sampler.greedy = true;
    std::vector<int> mtp;
    int64_t produced = 0;
    require(model.decode_mtp(sampler, 24, [&](int token) { mtp.push_back(token); return true; }, produced, err), err);
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
    require(produced == 24 && plain == mtp, "fixed-seed sampled MTP equals distributed plain decode");
    model.fast_prof_flush();
    std::printf("glm_remote_model_test: PASS (%d checks; T2/T3 every keep, pool boundaries, snapshot, MTP)\n", checks);
}
