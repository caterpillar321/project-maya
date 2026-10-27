#pragma once

#include <cmath>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

// Copy placement only: never use this profile to change routing or primary tiers.
inline bool read_net_copy_profile(std::istream& in, int layers, int experts,
                                  std::vector<double>& scores, std::string& err) {
    scores.assign((size_t) layers * experts, 0.0);
    std::vector<bool> seen(scores.size(), false);
    std::string line;
    int line_no = 0, entries = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof()) continue;
        int layer = -1, expert = -1;
        double score = -1;
        std::string extra;
        if (!(row >> layer >> expert >> score) || (row >> extra) || layer < 0 || layer >= layers ||
            expert < 0 || expert >= experts || !std::isfinite(score) || score < 0) {
            err = "invalid copy profile row " + std::to_string(line_no);
            return false;
        }
        const size_t key = (size_t) layer * experts + expert;
        if (seen[key]) {
            err = "duplicate copy profile key at row " + std::to_string(line_no);
            return false;
        }
        seen[key] = true;
        scores[key] = score;
        ++entries;
    }
    if (in.bad() || entries == 0) {
        err = "empty or unreadable copy profile";
        return false;
    }
    return true;
}
}   // namespace strata::core
