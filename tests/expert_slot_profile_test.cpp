#include "strata/core/expert_slot_profile.hpp"
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}
int main() {
    using strata::core::apply_expert_slot_profile;
    const std::vector<int> original{0, 36, 36, 36, 36};
    const std::vector<size_t> stride{0, 20, 10, 10, 10};
    auto slots = original;
    std::string err;
    std::istringstream good("# shrink a larger expert layer, grow a smaller one\n1 30\n2 48 # same bytes\n");
    require(apply_expert_slot_profile(good, slots, stride, 27, 291, 4, err), "valid same-byte profile");
    require(slots == std::vector<int>({0, 30, 48, 36, 36}), "unlisted layers changed");
    for (const char* text : {"", "# comment", "1 26", "1 292", "1 37", "1 36\n1 35",
                             "0 36", "4 36", "5 36", "-1 36", "2 36 extra", "2 nan",
                             "1 30\n2 49", "1 30\n2 999999999999999999999"}) {
        slots = original;
        std::istringstream bad(text);
        require(!apply_expert_slot_profile(bad, slots, stride, 27, 291, 4, err), "invalid profile accepted");
        require(slots == original, "failed profile partially changed layout");
    }
    slots = original;
    std::istringstream broken("1 30"); broken.setstate(std::ios::badbit);
    require(!apply_expert_slot_profile(broken, slots, stride, 27, 291, 4, err), "bad stream accepted");
    require(slots == original, "bad stream modified layout");
    std::puts("expert_slot_profile_test: OK (budget, floors, protected/unlisted layers, atomic failure)");
}
