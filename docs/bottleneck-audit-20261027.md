# GLM home inference: critical-path audit, 2026-10-27

The dominant measured path is expert movement out of main-PC RAM, followed by
the exposed wait for side computations at each layer. CPU utilization and GPU
utilization alone do not identify those costs. The source and fresh measurements
below distinguish work that overlaps from time that extends token completion.

## Scope and method

- Source: `a926aeb`, based on the preserved home tree and demand-copy experiment.
- Same IQ4_XS pack, 256K context, INT8 KV, MTP off, two local GPUs, two network
  GPU helpers, remote RAM 54 GiB, Ryzen 9600 helper at six threads.
- Baseline and demand-copy/copy-aware candidate: unchanged Claude `qshort.py`
  chat/code harness, one pass per arm; the original long-context task twice.
- Existing `PROF=4`, `NET_TRACE=1`, and route logging; profiling extended to
  token setup, output head/readback, host enqueue/wait/collection and sampling.
- Per-request stream stage sums match their reported total within 0.025 ms.
  Host boundary + enqueue + wait + collection match forward wall time.
- Instrumentation reduces throughput approximately 6%; these diagnostic timings
  are not replacement performance claims. Prior uninstrumented tests measured
  about 34-35 tok/s chat and 29-30 tok/s code.
- Single-client text inference. Browser/Cloudflare, concurrent request queues,
  vision encoding and agent tool execution are not fully benchmarked here.

## Request path

```mermaid
flowchart TD
    U["사용자 · Pi"] --> W["외부 게이트웨이 · SSH 터널"]
    W --> A["Maya API: 큐 · 템플릿 · 토큰화"]
    A --> C["KDA 상태·토큰 체크포인트 확인"]
    C --> P["재사용하지 못한 입력 프리필"]
    P --> D["생성: 45층을 순차 실행"]
    D --> H["출력 헤드 · 다음 토큰 샘플링"]
    H --> S["텍스트 변환 · SSE 전송"]
    S --> U
    H -->|"다음 토큰 입력"| D
```

The model has 34 KDA recurrent layers and 11 DSA layers. The first three layers
have dense FFNs; the other 42 route eight experts each. GPU/CPU helpers distribute
expert computation inside a layer, not independent full layer stacks. The next
layer needs the completed current-layer result.

KDA snapshots are not arbitrary KV-prefix rollback. The production keep-answer
path preserves the state at an answer end; sufficiently long prompt checkpoints
also permit fallback. Reusing a valid state and rereading a prompt have different
first-response costs, independent of steady decode throughput.

## One MoE layer: the two branches and their join

Times on this diagram are **sums over 42 MoE layers for one coding token**,
from the baseline instrumented run; they are not one layer's duration.

```mermaid
flowchart TD
    A["어텐션 · 정규화"] --> R["라우터: 전문가 8개 선택"]
    R --> F["5090: RAM 미스 전문가 가져오기<br/>16.6 ms / token"]
    M["메인컴 RAM"] -->|"가중치 약 13 MB / 전문가"| F
    F --> G["5090: 기존·가져온 전문가 계산<br/>2.6 ms / token"]
    R --> H["중계 스레드: 입력 벡터·ID 전달"]
    H --> P["PRO 4000: VRAM 전문가 계산"]
    H --> L["메인 CPU: 선택적 RAM 전문가 계산"]
    M --> L
    H --> N["2.5GbE → 옆 PC<br/>4070 Ti S · 5060 Ti · 9600"]
    P --> B["부분 결과 합산"]
    L --> B
    N -->|"결과 벡터 반환"| B
    G --> J["두 경로 합류<br/>잔여 대기 5.5 ms / token"]
    B --> J
    J --> Q["다음 층 진행"]
```

The main service sends network requests first, then handles the local helper and
optional local CPU, then finishes network replies and publishes their combined
answer. The main GPU fetches and computes while the side work proceeds.

The source orders RAM fetch and local expert math on the same CUDA stream.
Even already-resident selected experts wait behind that layer's fetch kernel.
The fetch is GPU SM code reading host-mapped memory and writing VRAM. The join
wait is also a GPU kernel. Both contribute to reported GPU busy time without
being dense arithmetic throughput.

Approximate layer critical path:

`attention/routing + max(main fetch + main expert math, side-result latency) + combine`

Side-result latency already overlaps main-GPU work; do not add its full network
or CPU execution time to main-GPU time. Likewise, host `wait_host` includes the
GPU forward and must not be added to the CUDA stage totals.

## Fresh coding-stage measurements

Mean of the four 200-token coding questions, ms/token:

| Exposed stream interval | Baseline | Demand copies + copy-aware |
|---|---:|---:|
| Main RAM → 5090 expert fetch | 16.640 | 16.043 |
| Remaining helper/CPU-result wait | 5.534 | 5.493 |
| 5090 expert gate/up/down math | 2.551 | 2.533 |
| Attention, HC, routing, head, setup and other | 11.770 | 11.770 |
| Total profiled stream | 36.495 | 35.840 |

Fetch plus exposed join wait is about 61% of baseline token time. Eliminating
all measured join wait, with every other cost fixed, would still leave about
31 ms/token in this instrumented path. The current MoE fetch/math/wait subtotal
alone is about 24.7 ms, already above the 20 ms budget for 50 tok/s.
These are diagnostic counterfactuals, not achievable throughput predictions.

Other measured components, baseline coding mean:

- Attention family (KDA, DSA, MLA and KV resolve): 6.48 ms/token.
- Output head + head prep + argmax/readback: 0.443 ms/token.
- Sampled-token selection/readback: 0.085 ms/call.
- Host boundary: 0.421 ms/forward; CUDA enqueue: 5.857 ms/forward.
- Host waiting for GPU completion: 30.223 ms/forward; collecting profiling events:
  0.048 ms/forward. Enqueue overlaps GPU execution; the wait is not extra time.
- Measured disk reads on the four coding requests: 0.00, 0.01, 0.00, 0.01 per
  token. SSD is not the steady-state coding bottleneck in these requests.

## Bytes and placement

Fresh routing logs plus exact expert blob sizes show:

| Coding quantity | Baseline | Candidate |
|---|---:|---:|
| Main RAM expert bytes fetched/token | 0.788 GB | 0.820 GB |
| Effective bytes / fetch interval | 47.4 GB/s | 51.1 GB/s |
| Fraction of fetch bytes in layers 19-44 | 99.83% | 99.88% |

This locates the expensive data path, but does not isolate the hardware limit
among DRAM bandwidth, PCIe transactions and SM-copy implementation. Do not call
it a proven PCIe line-rate ceiling from this measurement alone. The candidate
actually moves more bytes while spending slightly less time fetching, so miss
count alone is not a sufficient optimization objective.

The source's cold-start helper ranking sorts zero-initialized usage counts when
no usage file is provided. The resultant early-layer concentration agrees with
the trace. Primary slots remain nearly uniform at 36-39 per MoE layer. This is a
candidate placement problem, not proof that simply spreading experts uniformly
will be faster: additional remote-active layers incur additional round trips.

Actual links inspected: main GPU slot x16; PRO 4000 x4; both PCs' Ethernet links
2500 Mb/s full duplex. Reported PCIe generation drops at idle, so idle generation
values are not treated as transfer-time link measurements.

## Remote latency and exposed waiting

The candidate trace sampled 43,138 helper routes. Mean net0 and net1 answer
latencies after send were 346 and 604 microseconds respectively. The traced
GPU's remaining waits depended on the class: net0-only about 277 microseconds,
net1-only about 82, both about 389. Slower remote completion is therefore not
automatically a larger exposed wait: net1 commonly overlaps more local GPU work.

These are sampled route classes across all probe requests. The existing trace
collects a lagged record only when a later helper route triggers collection; it
does not sample every route uniformly. Do not multiply these class means by all
routes and claim an exact decomposition of the 5.5 ms stream wait. That stream
interval includes all helpers, publication/dispatch and wait-kernel overhead.

The `glm helper: CUDA1 ... ms` shutdown field uses the whole service-branch
accumulator, including network waits. It is not a standalone PRO 4000 kernel
timer. Per-route local GPU host timing and GPU-stream timing have different
boundaries too. Adding them as independent costs would double-count work.

## Prefill, cache and request transport

The same existing long-context task produced 16,524 prompt tokens (the helper's
input parameter is an approximate character-based target, not an exact token
count).

| Baseline request | Prompt processing | Decode | Outside native engine |
|---|---:|---:|---:|
| First 16,524-token prompt | 12.153 s | 8.366 s / 225 tokens | 15.8 ms |
| Identical prompt repeated | 40.1 ms; 16,523 tokens reused | 7.917 s / 224 tokens | 49.5 ms |

The candidate was similar: 12.128 s first prefill, 37.7 ms repeat prompt,
17.1/50.1 ms outside-native overhead. Context changes attention/indexer cost,
but did not turn attention into the dominant steady-state cost here.

A separate streamed request through B200 localhost:8010 and the SSH tunnel,
after production restoration, had total wall times 5.433/5.311 s. Native prompt
plus generation accounted for all but 24.7/58.8 ms. First text arrived after
1.231/1.211 s; native prompt processing alone took 1.202/1.148 s. These short
requests did not reuse prompt state. Browser/Cloudflare latency was not measured.

## What the next experiment should target

1. Optimize **placement by critical-path cost**: residual PCIe bytes, side-call
   frequency, per-helper latency and VRAM budget, using actual route co-occurrence.
   A frequency-only replay can reduce misses while increasing RPC-active layers.
   The existing copy-profile experiment changes RAM copies, not GPU-helper layout.
2. Test overlap of already-resident expert math with missing-expert fetch. The
   present path serializes both. Preserve selected IDs/weights and arithmetic
   checks; additional streams may contend for SMs, so overlap is not guaranteed.
   Even perfectly hiding the current expert-math interval alone saves only about
   2.6 ms/token, not enough to reach 50 tok/s.
3. Revisit speculative verification only with a budget for **bytes per emitted
   token**. Adjacent-token fetched-expert overlap was only 5-7% in prior traces;
   more accepted tokens per round do not imply proportionally less transfer.

Do not prioritize more CPU threads, sampling or small MLA/indexer kernels as the
main route to 50 tok/s on the current execution path. Memory movement and the
layer join must change substantially; the exact attainable ceiling is unproven.

## Restoration and artifacts

The CPU helper was restored to six threads and 54 GiB before this audit. Original
production was restored at 19:33:31 KST; helper binaries/settings were not changed
during the audit. New instrumentation is opt-in under `MAYA_GLM_PROF`.

Run: `codex-bottleneck-a926aeb-20261027`. Local raw logs, routing traces, context
timings and the machine-readable `audit-summary.json` are retained in the task's
analysis directory. No production promotion was made.


## Kohane helper RAM: user-reported DDR5-3200 → DDR5-5200

After the helper PC reboot, the user reported a stable boot at 5200 MT/s.
No BIOS settings or voltage were changed by this benchmark. A 180-second
`stressapptest -W -M 49152 -m 12` completed with PASS, zero errors and zero
hardware incidents. This is a short check, not proof of long-term stability.

Both helper GPUs had stopped after the reboot. They were restarted using their
previous binaries and GPU assignments; the CPU helper used six threads and
54 GiB of RAM. The original helper launcher still contains an obsolete 86 GiB
RAM limit, so its 18801 service was not started. The saved manual 54 GiB command
was used instead. The main server was restarted for each test arm and restored
to the original production configuration afterward.

Run: `codex-ram5200-cc80982-20261027`. The same cc80982 binary, profile and
unmodified Claude qshort harness were used as in the earlier 3200 run, with
two passes per arm, 3440 generated tokens per arm. The binary SHA-256 is
`73cb5bdb413462d4f638cdd5c12ee1055dff73b4173e20f4a0e8ab438e92ddcd`;
the harness SHA-256 is
`278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.

| Routing | Set | Earlier 3200, tok/s | New 5200, tok/s | Mean change |
|---|---|---|---|---|
| Legacy | Chat | 34.1 / 34.4 | 34.4 / 34.5 | +0.6% |
| Legacy | Code | 29.1 / 29.2 | 29.2 / 29.1 | 0.0% |
| Demand + copy-aware | Chat | 34.9 / 34.8 | 35.0 / 35.1 | +0.6% |
| Demand + copy-aware | Code | 29.7 / 30.0 | 30.3 / 31.0 | +2.7% |

In the demand-aware arm, the recorded service-thread `net answers` stage fell
from 0.416 to 0.362 ms/route (13.0%). The 18801 cumulative wait divided by its
recorded route count fell from 0.565 to 0.482 ms/route (14.8%). These counters
include transport/synchronization and overlapping work; they are not isolated
CPU kernel durations and must not be added directly to token latency. Both
runs kept optional main-CPU experts at zero and all copied experts matched
their hashes at shutdown (3856/3856 earlier, 3855/3855 now).

The direction is consistent with improved helper service latency, but the
end-to-end gain is modest. This is a historical before/after comparison, not
an interleaved RAM-frequency A/B test. Generated outputs, fetch counts and
small startup cache-capacity differences varied, and +2.7% is within the
previous roughly 3% run variation. It does not establish a precise causal
RAM-speed uplift. The primary RAM-to-5090 fetch path was not changed.
Twelve helper threads were not retested in this run.

Raw logs and per-pass CPU statistics are saved under
`/home/caterpillar/Agent/maya-analysis-20261027/runs/codex-ram5200-cc80982-20261027/`.
The accessible kernel journal showed no matching hardware-memory error reports
during the stress test or inference benchmark.


## Small follow-up: remote cost 0.28 → 0.20 ms at DDR5-5200

Run `codex-copycost-5200-20261027` compares the same cc80982 demand-profile plus
copy-aware candidate at six helper threads / 54 GiB. The only routing parameter
changed is `MAYA_GLM_NET_COPY_MS`: 0.28 versus 0.20. Both arms explicitly use
`MAYA_GLM_NET_COPY_RTT=0.35` and `MAYA_GLM_NET_COPY_RTT_SHARED=0.10`; neither uses
a manual plan override. The 0.20 value is a trial scheduling parameter, **not a
measured isolated CPU expert time**. No new engine build was required.

Each arm runs two unchanged Claude qshort chat/code passes (3440 output tokens,
including the greedy probes). The baseline and candidate each start fresh. The
logged generic plan for 0..8 misses changes from `0 0 0 1 1 2 2 3 3` to
`0 0 0 1 2 2 3 3 4`; copy-aware routing further accounts for available copies and
mandatory helper work.

| Quantity | 0.28 ms baseline | 0.20 ms candidate |
|---|---:|---:|
| Chat tok/s, both passes | 34.9 / 35.0 | 35.7 / 35.6 |
| Code tok/s, both passes | 30.2 / 30.4 | 31.6 / 31.3 |
| Mean chat tok/s | 34.95 | 35.65 (+2.0%) |
| Mean code tok/s | 30.30 | 31.45 (+3.8%) |
| Chat RAM fetches/token | 44.00 | 40.45 (-8.1%) |
| Code RAM fetches/token | 60.55 | 52.90 (-12.6%) |
| Optional experts sent to remote RAM copies | 56,589 | 70,006 (+23.7%) |
| Main engine CPU, percent of one logical core | 173.00 | 169.88 |
| Remote helper CPU, percent of one logical core | 450.42 | 451.90 |
| Service-thread net-answer time, ms/route | 0.364 | 0.393 |

More remote work increased recorded side-answer latency while reducing main
RAM fetches enough to improve the measured end-to-end speed. These service
counters overlap GPU work and do not measure the exposed layer-join stall.
Optional main-CPU expert computation stayed at zero in both arms. All remote
copies matched their hashes at shutdown: 3851/3851 and 3863/3863 respectively.

Both candidate passes were faster than both baseline passes. This is still a
small sequential A/B run, not a broad accuracy or performance qualification:
generated text/greedy hashes and startup cache capacities differed. Copy hashes
check weight integrity, not equivalence of model outputs. No claim is made that
0.20 is the calibrated optimum. The original production binary/configuration is
restored after testing; the experimental configuration is retained in
`/home/caterpillar/Agent/maya-analysis-20261027/runs/codex-copycost-5200-20261027/cost020.json`.
Raw logs, helper session summaries, and `comparison.json` are in the same folder.


## Additional code and launch review

- **Calibration transport differs from decode transport.** `fast_cpu_setup`
  measures `cpu_p_ms` using `cudaMemcpyAsync`, while decode's `moe_fetch_kernel`
  reads host-mapped memory with GPU threads. The value is a useful initial
  estimate, but a calibrated copy-engine time does not establish the marginal
  cost of a routed SM fetch. A future automatic scheduler should measure the
  actual routed transfer path for several miss counts, and remote service time
  for both standalone and already-active helper requests.
- **Remote CPU summary is a residual timing.** The helper starts CPU gate/up,
  runs its GPU share, then times `cpu_finish`. The shutdown `ms_cpu` field is
  time past the GPU share, not isolated whole CPU expert execution. Dividing it
  by expert count would not produce the calibration sought above.
- **Layer cache floors also protect borrowed storage.** The floor includes
  selected experts, spare slots and a tail borrowed by prefill/vision, plus
  up to 16 extra slots. Reducing it without respecting the borrowed tail can
  change more than decode capacity. A placement experiment should preserve
  these hard requirements and use residual miss cost after helper ownership,
  rather than only the raw popularity profile.
- **Home launch defaults disagree with the live helper.** On the helper PC,
  `~/fn-bin/helperd-run.sh` still asks for 86 GiB at port 18801, while the live
  six-thread helper explicitly uses 54 GiB on a 64 GB machine. Both helper
  systemd units are disabled/inactive. The main `start-maya.sh` auto-selection
  checks those units, so it selects the non-network configuration even when
  the two manually launched helpers are running. All benchmarks explicitly
  set `MAYA_NET=1`, so these launch issues do not explain the measured decode
  rates. They do explain why a manual/default restart can lose the intended
  setup. No systemd units were enabled as part of the performance experiments.


## Follow-up cost sweep at six threads / DDR5-5200

`codex-copycost-sweep5200-20261027` repeats the previous candidate and compares
0.20, 0.16 and 0.12 ms remote-expert cost assumptions. The same cc80982 binary,
profile, standalone/shared trip assumptions (0.35/0.10 ms), and unchanged
qshort harness are used, with two passes per arm.

| Remote cost assumption | Chat tok/s | Code tok/s | Chat fetches/token | Code fetches/token |
|---|---:|---:|---:|---:|
| 0.20 ms | 35.5 / 35.8 | 31.5 / 31.4 | 40.20 | 52.70 |
| 0.16 ms | 36.2 / 36.6 | 31.7 / 31.6 | 35.45 | 49.35 |
| 0.12 ms | 36.2 / 36.2 | 31.4 / 31.4 | 34.10 | 46.25 |

0.16 is selected for the subsequent fetch-launch experiment. Relative to 0.20,
its average chat rate is +2.1%, code +0.6%, and code fetch count -6.4%. These
small throughput differences need the same variation caveat as prior runs;
they do not establish a universal optimum. Lowering the assumption further to
0.12 reduces traffic but does not improve completion time in either code pass.

The scheduling choice is discrete: with two available copies and no existing
request, the model compares 2*0.256 = 0.512 ms on the main path with
0.35+0.16 = 0.510 ms for sending one expert. A very small estimated benefit can
therefore create a new helper request. Copy routes rose from roughly 43k at
0.20 to 61k at 0.16 per two-pass run. This motivates measuring actual marginal
request costs or adding a measured dispatch margin, rather than continuing to
lower one global constant blindly.

All remote copy hashes matched at shutdown. The original server was restored
before beginning the separately controlled fetch-launch experiment.


## Host-mapped fetch launch-size experiment

Commit `2fb1fbc` adds optional `MAYA_GLM_FETCH_BLOCKS` (1..4096). With the
variable unset, the original one-block-per-SM launch is preserved. Expert
selection, copied bytes and the math/reduction kernels are unchanged. Full
CUDA build passed; both the local and deployed binary SHA-256 are
`93fa19c3168a76d6c4a5bfdd9ece75bc4a7e8c7ab3dc2e035e73f18cbba47ec8`.

Run `codex-fetch-blocks-20261027` holds the remote cost at 0.16 ms and compares
default launch size, 64 blocks and 512 blocks, two unchanged qshort passes each.
The override log is checked for both experimental arms. Profiling is off.

| Fetch launch | Chat tok/s | Code tok/s | Mean code fetches/token |
|---|---|---|---:|
| Default | 36.1 / 36.7 | 31.7 / 31.6 | 49.20 |
| 64 blocks | 36.4 / 36.7 | 31.8 / 31.9 | 48.45 |
| 512 blocks | 36.2 / 36.7 | 31.7 / 31.7 | 49.20 |

The 64-block mean code gain is only 0.6%, accompanied by 1.5% fewer fetches
from a different generated/cache trajectory; it is not isolated launch-size
benefit. Default and 512 have the same aggregate fetch counts and helper-route
counts, with almost identical speed. All copies matched their hashes at shutdown
(3857/3857, 3862/3862, 3857/3857). **Keep the default launch**; the option remains
an opt-in diagnostic and the deployed selected configuration uses the previously
tested cc80982 binary, not this experimental build.

During actual generation, nvidia-smi reported the 5090 at PCIe generation 5 x16,
2880 MHz SM clock, 13801 MHz memory clock, 216 W and 53 C. The PRO 4000 link was
generation 4 x4. This rules out an accidentally negotiated low generation/width
in that sample, not PCIe bandwidth as a bottleneck.

## Selected home configuration and restart path

The selected configuration is demand-ranked RAM copies plus copy-aware routing,
remote cost 0.16 ms, trip/shared-trip 0.35/0.10 ms, six CPU-helper threads and
54 GiB helper RAM. Fetch launch sizing remains default. It trades a small
additional throughput gain and lower main-engine CPU use against more remote
requests; 0.16 is a measured candidate, not a general calibrated optimum.

New separate files on the main home PC:

- `~/fn/maya-config/iq4-kohane5200.json`
- `~/fn/start-maya-kohane5200.sh`
- API log: `~/src/maya-kohane5200-run.log`
- Engine log: `~/src/project-maya/maya-iq4-kohane5200.log`

The new launcher refuses to start a duplicate :18100 server, reuses exactly
matching existing helper processes, and starts missing helpers explicitly with
the tested binaries, GPU assignments and 54 GiB/six-thread CPU settings. It does
not depend on the disabled systemd units or the obsolete 86 GiB wrapper. Different
existing helper configurations are left untouched and reported instead of being
terminated. Available RAM is checked before starting a missing CPU helper. Shell
and embedded Python syntax checks passed, and the running-helper reuse path was
executed successfully. A full reboot/missing-helper startup was not tested here.
The new launcher is manual; no boot services were enabled.

The original `start-maya.sh` and `iq4-prod-net.json` remain unchanged. To revert,
stop the known tuned Maya process group and run `MAYA_NET=1 ~/fn/start-maya.sh`.
The activation controller records the selected server group and request check in
`~/fn/codex-activate-kohane5200-20261027/status.json` and rolls back automatically
if startup or the arithmetic smoke check fails. Local copies of the config,
launcher, experiment supervisors and raw results are retained in the analysis
directory. The tiny smoke request is a wiring check, not a broad quality eval.

## Primary expert-cache placement experiment

The `codex/cache-layout` branch adds an opt-in, byte-budget-checked per-layer
slot layout. Runtime source/build: `3bf036a`; tested binary SHA-256
`0c9d8d796f6165e4d715459b45b468b981c45c47895c0cc4695fbbf3c200314f`.
`8f506b3` subsequently makes the producer's runtime floor explicit and adds
documentation; its only runtime-source change is a comment. See
[expert-slot-layout.md](expert-slot-layout.md) for the format and limitations.
The C++ parser passed ASan/UBSan tests for byte budgets, capacity bounds,
protected/unlisted layers and atomic rejection. The Python producer tests and
full portable CUDA build passed.

Training used the existing code and agent route traces, with chat held out.
The approximate LRU fit is only a filter: it does not reproduce the actual
CPU-offload, admission, promotion or helper-ownership policies. A separate
replay that recomputed helper ownership estimated only 1.6–2.6% fewer misses
for the conservative candidate. This is not a prediction of token-rate gain.

The first run (`codex-slot-layout-20261027`) completed its baseline but rejected
the proposed 28-slot donor layers before candidate allocation. The active
image-enabled configuration requires 34 slots: 15 lendable plus 19 retained.
Although the image encoder requests about 3.32 GiB, `vision_lend()` frees the
entire shared tail, whose prefill requirement makes it 7.92 GiB here. The
guard was retained. The supervisor restored the selected 5200 production
configuration and verified `/health` after that rejected startup.

The corrected run (`codex-slot-layout-v2-20261027`) keeps layers 3–18 at 34
slots and redistributes the remaining budget to layers 19–44. The primary
pool has 1520 total slots in both arms (36–39 versus 34–44 per layer), using
19.12 versus 19.11 GiB. Both arms use the same new binary, 0.16 ms remote-copy
cost, demand copy profile, six-thread/54-GiB helper, 262144 context,
32768 prefill and image support. They run two unchanged qshort passes and
the existing 16,524-token request twice, then restore the selected launcher.

| Layout | Chat tok/s (two passes) | Code tok/s (two passes) | Mean chat fetches/token | Mean code fetches/token |
|---|---|---|---:|---:|
| Default | 36.2 / 36.4 | 31.6 / 31.7 | 35.60 | 49.05 |
| Conservative redistribution | 36.6 / 36.8 | 31.6 / 32.2 | 34.70 | 48.30 |

The candidate means are +1.1% chat and +0.8% code. This does not establish a
reliable speed improvement: the code gain occurs in only one pass, and the
earlier baseline's 36.5/36.7 chat and 31.9/31.5 code overlaps these results.
The main-engine CPU average was 162.6% versus 163.5% of one core, and the
Kohane CPU-helper average was 457.3% versus 456.4%: no meaningful CPU relief.

The long request's first prefill was 12.138 s versus 12.129 s; its repeated
prompt reused 16,523 tokens in both arms (33.2 versus 33.7 ms). Decode was
30.8/32.5 tok/s on first/repeated default requests versus 30.9/32.3 for the
candidate. These generated different output lengths, so total request wall
time is not a fixed-work comparison. No consistent long-context gain emerged.

Remote copy hashes all matched: 3859/3859 default, 3861/3861 candidate. The
first pass's greedy-answer hash matched across layouts, but the second did
not, and baseline hashes also varied between passes. These checks verify
copy integrity and basic functioning, not full output equivalence or a broad
quality evaluation. Actual image requests were not included in this benchmark;
the image lending configuration and its minimum-capacity guard were preserved.

**Decision: keep the selected production configuration without a slot profile.**
The new layout control and fitting tool remain opt-in for further experiments.
The results do not justify replacing the previously tested production binary.
The conservative redistribution neither substantially reduces RAM traffic nor
changes the dominant data path. A more aggressive allocation would require
changing and separately measuring the prefill/lending geometry, or designing
helper placement around both data movement and the added remote-call latency.
Neither was silently changed in this comparison.
