#pragma once

namespace strata::kernels {

// A routed-expert layer's gate/up type code: the gate's ggml type, plus the up's in bits 8.. when the two differ
// (bartowski's GLM-5.3-Flash IQ4_XS keeps layers 3-8's up in Q5_K / Q4_K under an IQ4_XS gate).  A code below 256 is
// one type for both, as before; native_experts.txt writes the code in its gu_type column.
constexpr int gu_code(int gate, int up) { return gate == up || up < 0 ? gate : gate | (up << 8); }
constexpr bool gu_mixed(int code) { return code >= 256; }
constexpr int gu_gate(int code) { return code >= 256 ? code & 0xff : code; }
constexpr int gu_up(int code) { return code >= 256 ? code >> 8 : code; }

}  // namespace strata::kernels
