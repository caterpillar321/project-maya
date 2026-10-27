#include "strata/kernels/copy_lane_policy.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

int main() {
    namespace cl = strata::kernels::copy_lane;
    // Independently enumerate every legal split, including the existing remote
    // CPU work. The chosen split must be optimal and never exceed actual copies.
    for (double remote : {0.18, 0.28, 0.60}) {
        constexpr double pcie = 0.256, trip = 0.35, shared_trip = 0.10;
        const auto plan = cl::make_plan(pcie, remote, trip, shared_trip);
        for (int f = 0; f <= 8; ++f)
            for (int mandatory = 0; mandatory + f <= 8; ++mandatory)
                for (int shared = mandatory > 0 ? 1 : 0; shared <= 1; ++shared)
                    for (int copies = 0; copies <= f; ++copies) {
                        const auto word = plan[(f * 9 + mandatory) * 2 + shared];
                        const int k = (int) (word >> (4 * copies) & 15ull);
                        require(k <= copies, "selected a missing copy");
                        const auto time = [&](int n) {
                            const double gpu = (f - n) * pcie;
                            const double host = (n + mandatory || shared)
                                ? (shared ? shared_trip : trip) + (n + mandatory) * remote : 0;
                            return gpu > host ? gpu : host;
                        };
                        for (int alternative = 0; alternative <= copies; ++alternative) {
                            require(time(k) <= time(alternative) + 1e-9, "nonoptimal split");
                            if (alternative < k)
                                require(time(alternative) > time(k) + 1e-9, "tie should prefer GPU");
                        }
                    }
    }

    // Enumerate every copy mask and requested count. No local-only expert may
    // enter the optional CPU lane; all unchosen entries still require GPU work.
    for (unsigned eligible = 0; eligible < 256; ++eligible)
        for (int take = 0; take <= 8; ++take) {
            const unsigned counts[8] = {9, 1, 3, 1, 8, 2, 5, 1};
            const unsigned chosen = cl::pick_cold(eligible, counts, take);
            require((chosen & ~eligible) == 0, "selected a local-only expert");
            require(std::popcount(chosen) == std::min(take, std::popcount(eligible)), "lost lane work");
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    if ((chosen >> i & 1u) && ((eligible & ~chosen) >> j & 1u))
                        require(counts[i] < counts[j] || (counts[i] == counts[j] && i > j), "wrong cold order");
        }
    const auto plan = cl::make_plan(0.256, 0.28, 0.35, 0.10);
    const auto take = [&](int f, int copies, int mandatory, int shared) {
        return (int) (plan[(f * 9 + mandatory) * 2 + shared] >> (4 * copies) & 15ull);
    };
    require(take(8, 0, 0, 0) == 0, "no-copy regression");
    require(take(4, 4, 0, 0) > take(4, 4, 4, 1), "mandatory remote work not budgeted");
    std::puts("copy_lane_policy_test: OK (split optimality, missing copies, mandatory load, all masks)");
}
