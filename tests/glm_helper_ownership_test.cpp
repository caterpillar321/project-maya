// CPU only: verifies the ownership upload's size and router semantics without creating a CUDA context.
#include "strata/core/glm_helper_ownership.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

int main() {
    using strata::core::glm_helper_owners;
    const std::vector<uint8_t> local{0, 1, 0, 1, 0, 0};
    const std::vector<uint8_t> network{0, 0, 2, 0, 4, 0};
    require(glm_helper_owners(6, local, {}) == local, "local-only ownership remains unchanged");
    require(glm_helper_owners(6, local, network) == std::vector<uint8_t>({0, 1, 1, 1, 1, 0}),
            "local and network experts are excluded from primary admission");
    // Regression: copying local owners first previously produced an empty upload in this configuration.
    const auto net_only = glm_helper_owners(6, {}, network);
    require(net_only.size() == 6, "network-only upload covers every model key");
    require(net_only == std::vector<uint8_t>({0, 0, 1, 0, 1, 0}),
            "network helper indices become ownership flags without assigning local experts");
    require(glm_helper_owners(6, {}, {}) == std::vector<uint8_t>(6, 0), "no helpers leave all keys on primary tiers");
    require(glm_helper_owners(6, {1}, {0, 3}) == std::vector<uint8_t>({1, 1, 0, 0, 0, 0}),
            "unlisted keys remain on primary tiers");
    require(glm_helper_owners(2, local, network) == std::vector<uint8_t>({0, 1}),
            "upload cannot exceed the model key count");
    require(glm_helper_owners(0, local, network).empty(), "zero keys have an empty upload");
    require(local == std::vector<uint8_t>({0, 1, 0, 1, 0, 0}) && network == std::vector<uint8_t>({0, 0, 2, 0, 4, 0}),
            "merging leaves source ownership tables intact");
    // A fresh fixed helper previously filled in layer/key order because all usage counters were zero.
    // With a small capacity that strands every later layer on the trunk even when its experts rank highest.
    std::vector<int> candidates{0, 1, 2, 3, 4, 5};
    const std::vector<std::pair<int32_t, int32_t>> profile{{2, 1}, {1, 0}, {0, 1}, {2, 1}, {-1, 0}, {0, 9}};
    strata::core::glm_rank_fixed_helper(candidates, std::vector<uint32_t>(6), profile, 2);
    require(candidates == std::vector<int>({5, 2, 1, 0, 3, 4}), "fresh helper fills globally ranked experts first");
    candidates = {0, 1, 2, 3, 4, 5};
    strata::core::glm_rank_fixed_helper(candidates, {0, 0, 0, 9, 9, 0}, profile, 2);
    require(candidates == std::vector<int>({3, 4, 5, 2, 1, 0}), "learned usage leads, profile breaks ties stably");
    candidates = {4, 1, 3};
    strata::core::glm_rank_fixed_helper(candidates, std::vector<uint32_t>(6), {}, 2);
    require(candidates == std::vector<int>({4, 1, 3}), "absent profile preserves tied candidate order");
    std::puts("glm_helper_ownership_test: OK (network-only upload, local compatibility, routing flags, bounds)");
}
