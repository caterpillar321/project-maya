#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace strata::kernels::copy_lane {

// Immutable per-expert metadata for the helper holding the RAM copies.
constexpr unsigned kCopy = 1;
constexpr unsigned kHelper = 2;
constexpr unsigned kHelperRam = 4;
constexpr int kPlanWords = 9 * 9 * 2;
using Plan = std::array<unsigned long long, kPlanWords>;

// Each word packs the choice for 0..8 available copies in four-bit fields.
inline Plan make_plan(double pcie, double remote, double trip, double shared_trip) {
    Plan plan{};
    for (int f = 0; f <= 8; ++f)
        for (int mandatory = 0; mandatory <= 8; ++mandatory)
            for (int shared = 0; shared < 2; ++shared)
                for (int copies = 0; copies <= f; ++copies) {
                    const double rt = shared ? shared_trip : trip;
                    const auto cost = [&](int k) {
                        const double cpu = mandatory + k > 0 ? rt + remote * (mandatory + k)
                                                            : shared ? rt : 0.0;
                        return std::max(pcie * (f - k), cpu);
                    };
                    int take = 0;
                    double best = cost(0);
                    for (int k = 1; k <= copies; ++k) {
                        const double t = cost(k);
                        if (t < best - 1e-9) best = t, take = k;
                    }
                    plan[(f * 9 + mandatory) * 2 + shared] |= (unsigned long long) take << (4 * copies);
                }
    return plan;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_COPY_HD __host__ __device__
#else
#define STRATA_COPY_HD
#endif

// Cold eligible experts go to the CPU; ties keep the legacy preference for the
// later routed entry. Every other expert remains on its original GPU path.
STRATA_COPY_HD inline unsigned pick_cold(unsigned eligible, const unsigned* counts, int take) {
    unsigned chosen = 0;
    for (int t = 0; t < take; ++t) {
        int best = -1;
        for (int i = 0; i < 8; ++i)
            if (((eligible & ~chosen) >> i & 1u) && (best < 0 || counts[i] <= counts[best])) best = i;
        if (best < 0) break;
        chosen |= 1u << best;
    }
    return chosen;
}

#undef STRATA_COPY_HD
}   // namespace strata::kernels::copy_lane
