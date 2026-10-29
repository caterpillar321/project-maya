#pragma once
#include <cstdint>

namespace strata::kernels::glmf {
// External helper residents resolve their key in the helper cache. Fixed CPU residents need the
// pinned RAM pointer in cpu_src. Both are excluded from primary admission and prefetch.
inline constexpr uint8_t kExternalHelperOwner = 1;
inline constexpr uint8_t kFixedCpuOwner = 2;
}
