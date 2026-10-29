#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

// A fixed helper cannot learn a better allocation after startup. Preserve learned usage first, but use
// the supplied global profile to break ties (fresh sessions have zero usage for every expert).
inline void glm_rank_fixed_helper(std::vector<int>& candidates, const std::vector<uint32_t>& usage,
                                  const std::vector<std::pair<int32_t, int32_t>>& profile, int experts) {
    if (experts <= 0) return;
    std::vector<size_t> rank(usage.size(), profile.size());
    for (size_t i = 0; i < profile.size(); ++i) {
        const auto [layer, expert] = profile[i];
        if (layer < 0 || expert < 0 || expert >= experts) continue;
        const uint64_t key = (uint64_t) layer * experts + expert;
        if (key < rank.size()) rank[(size_t) key] = std::min(rank[(size_t) key], i);
    }
    const auto count = [&](int key) { return key >= 0 && (size_t) key < usage.size() ? usage[(size_t) key] : 0u; };
    const auto prior = [&](int key) { return key >= 0 && (size_t) key < rank.size() ? rank[(size_t) key] : profile.size(); };
    std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        if (count(a) != count(b)) return count(a) > count(b);
        return prior(a) < prior(b);
    });
}

// The GPU router reads one ownership byte for every model key, including when only network helpers exist.
// Local ownership is optional; a network entry is a 1-based helper index rather than a boolean.
inline std::vector<uint8_t> glm_helper_owners(size_t keys, const std::vector<uint8_t>& local,
                                             const std::vector<uint8_t>& network) {
    std::vector<uint8_t> owners(keys, 0);
    for (size_t k = 0; k < keys && k < local.size(); ++k) owners[k] = local[k];
    for (size_t k = 0; k < keys && k < network.size(); ++k) owners[k] |= network[k] != 0;
    return owners;
}

}  // namespace strata::core
