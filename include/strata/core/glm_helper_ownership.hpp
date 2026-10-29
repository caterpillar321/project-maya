#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

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
