# Copy-aware GLM CPU lane (experimental)

The legacy GPU router selects a CPU quota from the number of RAM misses, assuming
that the remote helper has copies. The service thread resolves copies afterward.
On a partially populated copy tier, some selected experts therefore run on the
local CPU even with `MAYA_GLM_NET_COPY_LOCAL=0`: that setting favors remote
copies when available; it does not prohibit local fallback.

The experimental policy selects only RAM misses with a verified remote copy for
the optional CPU lane. RAM misses without copies stay on the existing GPU fetch
path. Experts owned by helpers still run on their helpers. No routed experts are
dropped or reweighted.

## Enable

Add `MAYA_GLM_COPY_AWARE=1` to a **test** server's config `env`, alongside
`MAYA_GLM_NET_COPY`, `MAYA_GLM_NET_HELPER`, and `MAYA_GLM_NET_COPY_LOCAL=0`.
`maya.py` translates these names to `STRATA_GLM_*` for the engine. The native
engine switch is `STRATA_GLM_COPY_AWARE=1`.

The default remains off. A copy-aware policy with local copy computation enabled
is rejected at setup. Existing `NET_COPY_MS`, `NET_COPY_RTT`, and
`NET_COPY_RTT_SHARED` costs are reused. The copy-aware plan supersedes the fixed
`NET_COPY_PLAN` quota; experts within a route are still selected by aged counts,
with the legacy tie order.

The GPU receives immutable per-expert copy/helper flags and a small quota table.
The table considers RAM misses, available copies, and RAM-only experts already
assigned to that same remote helper. It minimizes the estimated slower lane,
preferring the GPU on ties. This is a cost heuristic: it does not model the full
GPU compute path, network jitter, or every helper GPU's execution time.

Different CPU/GPU partitions can change floating-point rounding and generated
text. The change preserves the routed expert IDs and weights, not bitwise output
identity across execution paths.

## Checks

The CPU-only policy test checks every legal small split against exhaustive cost
enumeration and every eight-expert eligibility mask against copy availability
and coldness ordering:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I include \
  tests/copy_lane_policy_test.cpp -o /tmp/copy_lane_policy_test
/tmp/copy_lane_policy_test
```

It is also a CMake/CTest target when `STRATA_BUILD_TESTS=ON`.
Compile the CUDA `maya` target before testing on the serving host. Build tools
and the GGML source must match the intended deployment; `STRATA_PORTABLE=ON`
avoids compiling the GGML CPU backend only for the build host.

For an inference A/B, use separate test configs and the same workload/cache
conditions. Record throughput, local CPU fallback, PCIe fetches, helper wait,
copy hashes, and memory use. Zero local optional-lane fallback is an invariant
of this mode. A larger PCIe count alone is not a regression; end-to-end token
time decides whether shifting work was beneficial. No speed gain has yet been
established.

## Source baseline

`26adf6d` preserves the home `net-helper` working tree over upstream `35bb2b2`,
including mixed-quant support, network helpers, and existing experimental code.
It is a source snapshot, not a claim that every inherited experiment is validated
or that the snapshot is bit-for-bit the source of a particular frozen binary.
The original working tree and production executables are independent of this
branch.
