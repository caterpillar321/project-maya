// CPU only. This exercises the policy the loader uses before allocation/session setup, including physical-device
// ordering and reverse-dial endpoints. It does not initialize CUDA or open a socket.
#include "strata/core/glm_split_net_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}
bool same(const strata::core::GlmSplitNetPlan& a, const strata::core::GlmSplitNetPlan& b) {
    return a.enabled == b.enabled && a.boundary == b.boundary && a.devices == b.devices && a.endpoints == b.endpoints;
}
}  // namespace

int main() {
    using strata::core::GlmSplitNetConfig;
    using strata::core::GlmSplitNetPlan;
    using strata::core::make_glm_split_net_plan;
    GlmSplitNetConfig base;
    base.enabled = "1";
    base.split = "18";
    base.devices = "1,0";
    base.net_copy = base.copy_aware = base.fetch_reuse = "0";
    base.net_proto = "1";
    std::map<int, std::string> endpoints{{0, "listen:18801"}, {1, "listen:18800"}};
    const auto lookup = [&](int d) {
        const auto i = endpoints.find(d);
        return i == endpoints.end() ? std::string() : i->second;
    };
    std::string err;
    GlmSplitNetPlan p;
    require(make_glm_split_net_plan(base, 45, lookup, p, err), "valid two-part reverse-dial plan");
    require(p.enabled && p.boundary == 18 && p.devices == std::array<int, 2>{1, 0}, "explicit device order is retained");
    require(p.endpoints[0] == "listen:18800" && p.endpoints[1] == "listen:18801",
            "helper suffix follows CUDA device, not stage ordinal");
    const GlmSplitNetPlan valid = p;
    const auto reject = [&](const GlmSplitNetConfig& c, const char* what) {
        p = valid;
        err.clear();
        require(!make_glm_split_net_plan(c, 45, lookup, p, err), what);
        require(!err.empty() && same(p, valid), "failure explains itself without installing a partial plan");
    };

    for (const char* value : {"", "auto", "0", "45", "-1", "18,30", "9999999999999999999999"}) {
        auto c = base; c.split = value; reject(c, "invalid or implicit split refused");
    }
    for (const char* value : {"", "0", "0,0", "0,1,2", "1,", "-1,0", "2147483648,0", "1,a"}) {
        auto c = base; c.devices = value; reject(c, "requires exactly two distinct valid device indices");
    }
    for (const char* value : {"2", "true", "-1"}) {
        auto c = base; c.enabled = value; reject(c, "explicit opt-in flag is validated");
    }
    {
        auto c = base; c.local_helper = "1"; reject(c, "local helper and split mode conflict");
        c = base; c.global_helper = "listen:18800"; reject(c, "global helper must not be inherited by both parts");
        c = base; c.net_copy = "2"; reject(c, "each part has only one helper so copy index 2 is invalid");
        c = base; c.copy_aware = "1"; reject(c, "copy-aware policy needs a supported copy plan");
        c = base; c.net_proto = "2"; reject(c, "shared protocol2 scratch is not split-safe yet");
        c = base; c.net_trace_set = true; reject(c, "presence-based shared network trace counters refused");
        c = base; c.cpu_verify_set = true; reject(c, "presence-based shared CPU verification counters refused");
        c = base; c.fetch_reuse = "1"; reject(c, "native window reuse remains single-trunk");
        c = base; c.ram_class_slots_set = true; reject(c, "per-class single-trunk RAM override refused");
    }
    for (const char* value : {"2", "-1", "01", "true", "1junk"}) {
        auto c = base; c.net_copy = value; reject(c, "copy index is a strict zero or one");
        c = base; c.copy_aware = value; reject(c, "copy-aware flag is a strict zero or one");
    }
    {
        auto c = base;
        c.net_copy = "1";
        require(make_glm_split_net_plan(c, 45, lookup, p, err) && same(p, valid),
                "legacy per-part copy routing retains exclusive helper sessions");
        c.copy_aware = "1";
        for (const char* value : {"", "1", "false", "00"}) {
            c.net_copy_local = value;
            reject(c, "copy-aware mode requires an explicit disabled local copy lane");
        }
        c.net_copy_local = "0";
        require(make_glm_split_net_plan(c, 45, lookup, p, err) && same(p, valid),
                "copy-aware mode uses each part's own first helper without changing placement");
        c.net_copy = "0";
        reject(c, "copy-aware mode cannot run with copies disabled");
        c.net_copy.clear();
        reject(c, "copy-aware mode cannot run with an unspecified copy helper");
    }
    for (const char* value : {"", "listen:0", "listen:65536", "listen:-1", "listen:1,listen:2", "listen:18801",
                             "listen:018801", "hostname:18800", "192.168.000.76:18800", "256.1.1.1:18800",
                             "192.168.0:18800", "192.168.0.76:18800", "::1:18800"}) {
        endpoints[1] = value;
        reject(base, "missing, malformed, duplicate or mixed endpoint refused before connection");
    }
    endpoints[1] = " listen:018800 ";
    require(make_glm_split_net_plan(base, 45, lookup, p, err) && p.endpoints[0] == "listen:18800",
            "listen ports are canonicalized before duplicate detection and connection");
    endpoints = {{0, "192.168.0.76:18801"}, {1, "192.168.0.76:018800"}};
    require(make_glm_split_net_plan(base, 45, lookup, p, err) && p.endpoints[0] == "192.168.0.76:18800",
            "distinct direct IPv4 endpoints use the same plan");
    endpoints[1] = "192.168.0.76:018801";
    reject(base, "direct endpoint normalization cannot hide a duplicate session");

    // The new policy must not restrict or perform helper lookups for the existing disabled/default paths.
    for (const char* flag : {"", "0"}) {
        GlmSplitNetConfig c;
        c.enabled = flag;
        c.local_helper = "1";
        c.global_helper = "legacy-host:18800,listen:18801";
        c.net_proto = "2";
        c.net_copy = "2";
        c.copy_aware = "legacy";
        bool looked_up = false;
        require(make_glm_split_net_plan(c, 0, [&](int) { looked_up = true; return std::string(); }, p, err),
                "disabled mode leaves existing settings to their original policy");
        require(!p.enabled && !looked_up, "disabled mode neither selects endpoints nor reads experimental routing");
    }
    std::printf("glm_split_net_plan_test: OK (%d checks; ordering, exclusive endpoints, policy, atomic rejection)\n", checks);
}
