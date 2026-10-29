#pragma once

#include <array>
#include <climits>
#include <cstdlib>
#include <string>
#include <utility>

namespace strata::core {

// First split-helper prototype: two trunk parts, one exclusive helper session per physical device.
// This does not multiplex a helperd session and does not enable native multi-row MTP on a split.
struct GlmSplitNetConfig {
    std::string enabled, split, devices, local_helper, global_helper;
    std::string net_copy, copy_aware, net_copy_local, net_proto, fetch_reuse;
    bool net_trace_set = false, cpu_verify_set = false, ram_class_slots_set = false;
};

struct GlmSplitNetPlan {
    bool enabled = false;
    int boundary = 0;
    std::array<int, 2> devices{-1, -1};
    std::array<std::string, 2> endpoints;   // part order; the environment suffix uses the physical device index
};

namespace split_net_detail {
inline std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

inline bool decimal(const std::string& s, int limit, int& value) {
    if (s.empty()) return false;
    int n = 0;
    for (char c : s) {
        if (c < '0' || c > '9' || n > (limit - (c - '0')) / 10) return false;
        n = n * 10 + c - '0';
    }
    value = n;
    return true;
}

// A canonical numeric address avoids hostname aliases accidentally opening two sessions to one daemon.
// Reverse-dial helpers use distinct local listen ports, so they need not be restarted for this prototype.
inline bool endpoint(const std::string& input, std::string& canonical, bool& listen) {
    const std::string s = trim(input);
    const size_t colon = s.find(':');
    if (colon == std::string::npos || s.find(':', colon + 1) != std::string::npos) return false;
    int port = 0;
    if (!decimal(s.substr(colon + 1), 65535, port) || port == 0) return false;
    const std::string host = s.substr(0, colon);
    listen = host == "listen";
    if (!listen) {
        size_t start = 0;
        for (int i = 0; i < 4; ++i) {
            const size_t dot = host.find('.', start);
            if ((i < 3) != (dot != std::string::npos)) return false;
            const std::string octet = host.substr(start, dot == std::string::npos ? dot : dot - start);
            int n = 0;
            if (!decimal(octet, 255, n) || (octet.size() > 1 && octet[0] == '0')) return false;
            start = dot == std::string::npos ? host.size() : dot + 1;
        }
    }
    canonical = host + ':' + std::to_string(port);
    return true;
}
}  // namespace split_net_detail

// helper_for_device(d) returns NET_HELPER<d>. Keeping lookup injected lets the exact runtime policy be tested
// without environment mutation, CUDA initialization, DNS lookup or a network connection.
template<class HelperLookup>
inline bool make_glm_split_net_plan(const GlmSplitNetConfig& c, int layers, HelperLookup helper_for_device,
                                    GlmSplitNetPlan& out, std::string& err) {
    using namespace split_net_detail;
    const auto fail = [&](const std::string& why) {
        err = "GLM split net: " + why;
        return false;
    };
    GlmSplitNetPlan p;
    if (c.enabled.empty() || c.enabled == "0") {
        out = std::move(p);
        return true;   // the existing nonexperimental configuration is unchanged
    }
    if (c.enabled != "1") return fail("MAYA_GLM_SPLIT_NET must be 0 or 1");
    if (!c.local_helper.empty() || !c.global_helper.empty())
        return fail("unset global MAYA_GLM_EXPERT_HELPER and MAYA_GLM_NET_HELPER; use NET_HELPER<device> only");
    // Each part has exactly one private net session. Index 1 therefore means that part's own helper;
    // copies and their routing metadata stay in its FastState, rather than being shared across stages.
    if (!c.net_copy.empty() && c.net_copy != "0" && c.net_copy != "1")
        return fail("MAYA_GLM_NET_COPY must be 0 or 1 (the part's own helper)");
    if (!c.copy_aware.empty() && c.copy_aware != "0" && c.copy_aware != "1")
        return fail("MAYA_GLM_COPY_AWARE must be 0 or 1");
    if (c.copy_aware == "1" && (c.net_copy != "1" || c.net_copy_local != "0"))
        return fail("MAYA_GLM_COPY_AWARE=1 requires MAYA_GLM_NET_COPY=1 and MAYA_GLM_NET_COPY_LOCAL=0");
    if (!c.net_proto.empty() && c.net_proto != "1") return fail("this prototype supports MAYA_GLM_NET_PROTO=1 only");
    if (c.net_trace_set || c.cpu_verify_set)
        return fail("unset MAYA_GLM_NET_TRACE and MAYA_GLM_CPU_LANE_VERIFY (their shared counters are not split-safe)");
    if (!c.fetch_reuse.empty() && c.fetch_reuse != "0")
        return fail("MAYA_GLM_WIN_FETCH_REUSE needs one trunk; set it to 0 for this split");
    if (c.ram_class_slots_set) return fail("unset MAYA_GLM_RAM_CLASS_SLOTS; split RAM uses the shared machine budget");
    if (!decimal(trim(c.split), INT_MAX, p.boundary) || p.boundary < 1 || p.boundary >= layers)
        return fail("give one explicit MAYA_GLM_SPLIT boundary in 1.." + std::to_string(layers - 1));
    const size_t comma = c.devices.find(',');
    if (comma == std::string::npos || c.devices.find(',', comma + 1) != std::string::npos ||
        !decimal(trim(c.devices.substr(0, comma)), INT_MAX, p.devices[0]) ||
        !decimal(trim(c.devices.substr(comma + 1)), INT_MAX, p.devices[1]) || p.devices[0] == p.devices[1])
        return fail("give exactly two distinct physical GPU indices in MAYA_GLM_DEVS");
    bool listen[2]{};
    for (int k = 0; k < 2; ++k) {
        if (!endpoint(helper_for_device(p.devices[k]), p.endpoints[k], listen[k]))
            return fail("MAYA_GLM_NET_HELPER" + std::to_string(p.devices[k]) +
                        " needs one listen:port or canonical IPv4:port endpoint");
    }
    if (listen[0] != listen[1]) return fail("use two listen endpoints or two IPv4 endpoints; mixed modes are unsupported");
    if (p.endpoints[0] == p.endpoints[1]) return fail("each trunk part needs a distinct helper endpoint/session");
    p.enabled = true;
    out = std::move(p);   // failed validation never installs a partial plan
    return true;
}

inline bool read_glm_split_net_plan(const std::string& layer_split, int layers,
                                    GlmSplitNetPlan& out, std::string& err) {
    const auto env = [](const char* name) {
        const char* s = std::getenv(name);
        return s == nullptr ? std::string() : std::string(s);
    };
    GlmSplitNetConfig c;
    c.enabled = env("STRATA_GLM_SPLIT_NET");
    c.split = std::getenv("STRATA_GLM_SPLIT") != nullptr ? env("STRATA_GLM_SPLIT") : layer_split;
    c.devices = env("STRATA_GLM_DEVS");
    c.local_helper = env("STRATA_GLM_EXPERT_HELPER");
    c.global_helper = env("STRATA_GLM_NET_HELPER");
    c.net_copy = env("STRATA_GLM_NET_COPY");
    c.copy_aware = env("STRATA_GLM_COPY_AWARE");
    c.net_copy_local = env("STRATA_GLM_NET_COPY_LOCAL");
    c.net_proto = env("STRATA_GLM_NET_PROTO");
    c.fetch_reuse = env("STRATA_GLM_WIN_FETCH_REUSE");
    c.net_trace_set = std::getenv("STRATA_GLM_NET_TRACE") != nullptr;
    c.cpu_verify_set = std::getenv("STRATA_GLM_CPU_LANE_VERIFY") != nullptr;
    c.ram_class_slots_set = std::getenv("STRATA_GLM_RAM_CLASS_SLOTS") != nullptr;
    return make_glm_split_net_plan(c, layers, [&](int d) {
        return env(("STRATA_GLM_NET_HELPER" + std::to_string(d)).c_str());
    }, out, err);
}

}  // namespace strata::core
