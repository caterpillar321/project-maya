# Home RAM backing and NC3 follow-up, 2026-10-27–28

This follow-up keeps the existing IQ4_XS pack, INT8 KV, expert selection and hardware.
The working branch is `codex/ram-backing-probe`. Changes are opt-in. The selected
production configuration is restored between supervised experiments and at completion.

## Implementations and correctness scope

- `d20b7dd`: optional three-column `mv()` / `mvc_launch<T>` specialization for Q8_0/Q4_K/Q6_K.
  The home RTX 5090 passed 60 bitwise comparisons against the original NC4 and
  single-column kernels, including mixed jobs, strides, scales, bias and output guards.
  This validates those kernels; it is not end-to-end model equivalence certification.
- `d02d5fa`: bounded primary-GPU RAM backing, separate reservation and KEEP flags,
  duplicate-first reclamation, prefill lending integration and table-update coalescing.
  The diagnostic compares immutable expert bytes before skipping D2H and checks
  settled GPU/RAM table pointers and the class reservation cap at boundaries.
- `cf9aaad`: optional `MAYA_GLM_RAM_CLASS_SLOTS` startup redistribution. The complete
  CSV is validated before application, with per-class floors/caps and a total-byte
  ceiling equal to the original plan. Thirty-three extracted-source CPU cases passed.

## Small, placement-matched reservation

All arms use `MAYA_GLM_RAM_GB=92`, or 91.9912109375 GiB actually allocated, with
class slots 697/348/6268. The probe reserves 72 slots (918 MiB) in layers [22,24).
The reservation is inside the existing allocation. It shifts 71 cold experts to the
remote helper (467 -> 538) and reduces optional remote copies (3869 -> 3798).
The two KEEP arms have matching initial pool, RAM, cold-share and copy counts.

Each performance arm uses Claude's unchanged `qshort.py` twice: chat4/code4 at a
200-token cap, followed by the original greedy check. It then runs the existing
16,524-token task once and repeats its prefix. The baseline uses the same binary
with backing disabled. TIER_AUDIT is enabled in every arm; byte-verification is
used only in the separate diagnostic arm.

| Arm | Chat tok/s, pass 1 / 2 | Code tok/s, pass 1 / 2 |
|---|---:|---:|
| No reservation | 36.3 / 36.5 | 31.7 / 32.2 |
| Reserve, KEEP=0 | 36.4 / 36.5 | 31.4 / 31.6 |
| Reserve, KEEP=1 | 36.5 / 36.6 | 31.6 / 31.6 |

The small probe does not establish a useful speedup. Relative to the matched
KEEP=0 control, mean reported rates improve only about 0.3%, within the spread
of these short, unseeded runs. Code remains below the unreserved baseline mean.
Main CPU usage is effectively unchanged (about 162% of one core in both KEEP arms).

Across eight chat and eight code requests per arm, each group has 1,592 audited
decode boundaries. KEEP=0 -> KEEP=1 boundary D2H payload falls:

- Chat: 627,332,349,952 -> 566,091,317,248 bytes, about 9.8% less.
- Code: 751,645,229,056 -> 691,389,857,792 bytes, about 8.0% less.

This is enqueued expert payload, not measured PCIe occupancy or exposed transfer
latency. Mapped H2D payload and helper work remain. The speed result does not show
that D2H is universally free; it shows that eliminating this share of it is insufficient.

The diagnostic completed 11 requests including full and repeated long-context
prefill. All 352 sampled expert-blob comparisons and 352 boundary table/cap snapshots
passed; 3798/3798 optional remote copies matched at shutdown. Its slower
35.1/30.4 tok/s rates are excluded from performance comparisons.
Greedy hashes vary in the unmodified baseline too, so their equality is not used
as a full correctness proof.

## Extended experiment

Both extended experiments completed before the user requested shutdown.

| Experiment | Chat tok/s, mean | Code tok/s, mean |
|---|---:|---:|
| MTP private T3, NC3 OFF | 36.20 | 33.25 |
| MTP private T3, NC3 ON | 36.95 | 33.85 |
| 8.17-GiB redistribution, KEEP=0 | 36.80 | 31.65 |
| 8.17-GiB redistribution, KEEP=1 | 37.65 | 33.15 |

NC3 OFF/ON matched all 20 requests' exact observed routing, payload, rounds,
acceptance and replay counters, and matched each pass's greedy hash. Weighted
verification windows fell 63.704 -> 62.401 ms for chat and 76.427 -> 74.982 ms
for code. This supports about 2% execution gain, subject to the limited sequential
OFF-then-ON design (two passes each, without randomized clock/temperature control).

Large backing cut D2H 70.35% for chat and 67.34% for code versus its matched
KEEP=0 control. Mapped fetch+miss payload also changed (-1.34% / -5.62%), so
this is not identical observed work as in the NC3 comparison. The class shift
increased long-first prefill from the earlier plain baseline's 12.0991 s to
13.3839 s, and qshort total wall time from 59.587 s to 65.040 s (+9.15%). Thus
it is not a good default upgrade despite the decode gain. All 704 sampled blob
comparisons and 704 table/cap snapshots in its diagnostic passed.

The NC3 comparison holds fixed-2, stagger=0, private T3, RAM 92 GiB, MV_TILED=1
and MV_MMA=0 constant, changing only MV_NC3. The real model's relevant dense,
attention, shared-expert and output matrices are Q8_0. A full T3 verification
window has 305 eligible matrices (136 KDA, 33 DSA, 9 dense FFN, 126 shared MoE,
1 output head). Routed experts use separate kernels. NC3 does not reduce their
weight payload, helper work or draft cost. Its Q4_K/Q6_K register reductions
must not be used to explain speed on this Q8_0 dense path. The Q8_0 build uses
64 rather than 62 registers; arithmetic reduction is not an automatic speedup.

The 8.16796875-GiB redistribution changes class slots to 289/144/6924. Both old
and new plans total exactly 98,774,810,624 bytes (94,199 MiB). Reserving the 656
new late-class slots (8364 MiB, layers [19,45)) keeps the predicted initial late
unique warm set unchanged. The observed initial cold share remained 467, and both arms retained 3869 optional remote copies. Fewer early-layer RAM
copies may increase prefill pack reads; helper replacement may also leave an
outgoing expert on disk when no RAM copy remains. Existing load fallbacks preserve
weights, but the latency cost must be measured.

## Artifacts

Local artifact root: `/home/caterpillar/Agent/maya-backing-probe-20261027`.
Home run root for the small probe: `~/fn/codex-backing-20261027`.
The supervisor keeps per-arm configs, engine/server logs, harness output, long-context
JSONL, memory/CPU measurements and restoration status. `summarize_probe.py` reads
these artifacts without contacting a live engine.

- Small-probe binary SHA256: `dfddb337cd5e2dbb84bc2dcca29f4490d6f20c4ebf3803c609886007fa7251aa`.
- Extended binary SHA256: `b409ae43554de5d96c5ee04a86e11a8db1347aa5dc93124423f23c9d6032552e`.
- Unchanged harness SHA256: `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.

See [the earlier cycle audit](home-50tps-feasibility-20261027.md) for the baseline
pipeline and MTP round budget, and [the backing design](ram-backing-design-20261027.md)
for reservation, prefill and class-allocation constraints.

## User-requested stop and next resume

At 2026-10-28 00:26 KST the user requested an immediate hold, saving work, and
powering off both home PCs. No automatic continuation or wake-up is scheduled.

Commit `6271113` adds opt-in `MAYA_GLM_WIN_FETCH_REUSE=1`: completed immutable
private scratch weights can be shared by later rows in the same non-staggered
T3 layer. Promotion/helper/CPU policy is preserved, and actual fetch and reuse
payload are separate counters. Pointer sharing also changes expert grouping.
The home 5090 passed route/fetch/admission/guard checks and 1536 bitwise float
output comparisons for each of gu_type 23/3095/3351 with down_type 23.

The new same-binary OFF arm finished two qshort passes (chat36.8/37.2,
code34.0/33.8); its long-context phase was interrupted. **The ON arm did not
run. There is no end-to-end speed claim for WIN_FETCH_REUSE yet.** The latest
binary SHA256 is `6f6dc6c1f11a42bd96c8cfed0943a5e24373b76a8672abf80ff81a4f7e237856`.
The supervisor was stopped and its auto-restored Maya process group was also
stopped for shutdown. The selected production config/launcher was not changed.

See [the Korean resume checkpoint](home-resume-20261028.md) for exact paths and
safe continuation steps. Raw interrupted logs are saved both locally and at
`~/fn/codex-fetch-reuse-20261028` on the main PC.
