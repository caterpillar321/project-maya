#pragma once

#include <algorithm>
#include <cstddef>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

// An opt-in partial layout. Unlisted layers and the existing byte budget are
// preserved. Commit only after validating the whole file, including lending
// and scratch requirements supplied by the caller.
inline bool apply_expert_slot_profile(std::istream& in, std::vector<int>& slots,
                                     const std::vector<size_t>& strides, int minimum,
                                     int maximum, int protected_layer, std::string& err) {
    if (slots.size() != strides.size() || minimum <= 0 || maximum < minimum) {
        err = "invalid slot profile geometry";
        return false;
    }
    auto next = slots;
    std::vector<bool> seen(slots.size(), false);
    size_t old_bytes = 0, new_bytes = 0;
    int entries = 0, line_no = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++line_no;
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof()) continue;
        int layer = -1, count = 0;
        std::string extra;
        if (!(row >> layer >> count) || (row >> extra) || layer < 0 ||
            (size_t) layer >= slots.size() || slots[(size_t) layer] == 0 ||
            strides[(size_t) layer] == 0 || count < minimum || count > maximum ||
            layer == protected_layer || seen[(size_t) layer]) {
            err = "invalid slot profile row " + std::to_string(line_no) +
                  " (unique MoE layer, no draft layer, slots " + std::to_string(minimum) +
                  ".." + std::to_string(maximum) + " required)";
            return false;
        }
        seen[(size_t) layer] = true;
        old_bytes += (size_t) slots[(size_t) layer] * strides[(size_t) layer];
        new_bytes += (size_t) count * strides[(size_t) layer];
        next[(size_t) layer] = count;
        ++entries;
    }
    if (in.bad() || entries == 0 || new_bytes > old_bytes) {
        err = entries == 0 ? "empty slot profile" : new_bytes > old_bytes
            ? "slot profile exceeds the existing selected-layer byte budget"
            : "unreadable slot profile";
        return false;
    }
    slots.swap(next);
    return true;
}
}  // namespace strata::core
