#include "strata/core/net_copy_profile.hpp"
#include <cstdio>
#include <cstdlib>

static void require(bool ok) { if (!ok) std::exit(1); }
int main() {
    std::vector<double> scores;
    std::string err;
    std::istringstream good("# profile\n1 2 0.75 # demand\n0 0 2\n");
    require(strata::core::read_net_copy_profile(good, 2, 3, scores, err));
    require(scores.size() == 6 && scores[0] == 2 && scores[5] == .75 && scores[1] == 0);
    for (const char* text : {"", "# comment", "2 0 1", "0 3 1", "-1 0 1", "0 0 -1",
                             "0 0 nan", "0 0 1e999", "0 0 1 extra", "0 0 1\n0 0 2", "broken"}) {
        std::istringstream bad(text);
        require(!strata::core::read_net_copy_profile(bad, 2, 3, scores, err));
    }
    std::puts("net_copy_profile_test: OK");
}
