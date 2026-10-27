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

## First speed comparison: 2026-10-27

Commit `74bad15`, built for sm_120 with CUDA 13.0/GCC 11 and the portable GGML
CPU backend. The existing Claude `qshort.py` and `qset_sets.py` were used
unchanged: the first four chat and code questions, up to 200 generated tokens
each, then the harness's 120-token greedy probe. Both new-policy arms used the
same binary and production settings (IQ4_XS, 256K context, INT8 KV, MTP off,
two network GPU helpers, 54 GiB remote RAM tier). Helpers stayed running.

| Arm | Pass | Chat tok/s | Code tok/s | Chat PCIe fetches/token | Code PCIe fetches/token |
|---|---:|---:|---:|---:|---:|
| Frozen production reference | 1 | 34.4 | 29.2 | 43.0 | 59.0 |
| New binary, policy off | 1 | 34.2 | 29.1 | 43.1 | 59.3 |
| New binary, policy off | 2 | 34.3 | 29.0 | 43.4 | 59.6 |
| New binary, policy on | 1 | 34.1 | 28.6 | 46.4 | 68.3 |
| New binary, policy on | 2 | 33.9 | 28.6 | 47.4 | 67.8 |

There is no demonstrated speed gain. The on/off mean differences are about
-0.7% chat and -1.5% code, within the previously observed approximately 3%
run-to-run variation. This was off then on with two passes per server, not a
randomized multi-start study, so it does not establish a small regression.

The intended placement change did work: optional local CPU fallback fell from
about 4.3-4.6 experts/token in chat and 10.0 in code to zero. Code PCIe fetches
rose about 14.5%. At close the off run checked 3851/3851 remote copies and the on
run checked 3858/3858, all matching. These check weight integrity, not output
equivalence. Greedy hashes differed between the two off passes too, and were
not treated as a correctness gate.

The results do not support treating all local CPU fallback as wasted work.
Some of that work can overlap the GPU, so eliminating it can increase the GPU's
critical path. Any later planner should compare useful local CPU execution,
remote CPU execution and PCIe/GPU work rather than simply banning fallback.
The experimental switch remains off; this candidate is not promoted.

Run identifier: `codex-copy-lane-74bad15-20261027`.
Unmodified qshort harness SHA-256:
`278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.
