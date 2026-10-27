# GLM-5.3-Flash MTP 디코드 정밀 검토

검토 기준: `codex/cache-layout`의 소스와 2026-10-27 홈 장비의 기존 로그입니다.
원본 모델의 양자화, trunk의 top-8 expert 계산, 최대 문맥을 유지하는 개선만
대상으로 삼았습니다. 역사적 수치는 DDR5-5200 / demand-ranked copies /
copy-aware 0.16 ms 설정의 재측정과 구분했습니다. 아래의 마지막 절에는
2026-10-27 22시 현재 설정으로 새로 측정한 결과와 메모리 원인을 추가했습니다.

기존 Claude 기록은 홈 PC의 `~/src/project-maya-net/PROGRESS-mtpreview.md`이며,
검토에 사용한 로그를 로컬 분석 폴더 `maya-analysis-20261027/mtp-history/`에
복사했습니다. 특히 `maya-test-m5-{off,f1,f2}.log`와
`maya-test-m6-ws1.log`를 직접 읽고 합산했습니다.

## 먼저 구분해야 하는 두 가지 MTP 경로

- 현재 장비는 5090 한 장이 **trunk 전체**를 실행하고 다른 GPU들은 expert만
  계산합니다. 따라서 활성화할 경로는 `glm_mtp.cu::decode_mtp()`입니다.
- `glm_fast_path.cu::decode_spec()`는 trunk를 여러 GPU의 head/tail 그룹으로
  나눈 파이프라인용입니다. 그 경로의 `SPEC_CANCEL`, `SPEC_SAMPLE`을 켠다고
  현재 single-trunk MTP가 개선되지는 않습니다.
- CUDA single-trunk에서 MTP는 기본 off입니다. `MTP_DECODE=1`로 명시적으로
  켜야 하며, 여러 동시 conversation(`SEQS>1`)이나 DFlash와 함께 켜지지
  않습니다. 설치 메타데이터의 `"draft":"mtp"`는 활성화 증거가 아닙니다.

근거: `src/core/glm_mtp.cu:78`의 `mtp_decode_wanted()`,
`src/core/glm_fast_path.cu:7262`의 `spec_ready()`.

## 한 라운드의 실제 실행

```text
이미 확정한 마지막 토큰 y, trunk hidden state h
  ↓
NextN: embedding + h → eh_proj → DSA → MoE → shared norm → output head
  └─ d1, 필요하면 d2 … 제안 (같은 GPU, 대략 draft당 1.1 ms였음)
  ↓
window([y, d1, … dn])
  ├─ tier boundary, KDA 상태 백업
  ├─ trunk 45 layers: 모든 T=n+1개 row를 검증
  ├─ row별 target logits와 target sampler 결과
  └─ scratch 충돌이면 KDA를 복구하고 window 전체를 token-by-token 재실행
  ↓
앞에서부터 target pick과 draft가 일치하는 동안 수락
  └─ 처음 불일치한 target pick 또는 마지막 bonus token까지 출력
  ↓
window_commit(keep): 필요한 KDA rollback/replay
  ↓
NextN KV를 확정된 trunk hidden state로 다시 채움 → 다음 라운드
```

`fast_mtp()`는 단순한 작은 분류기가 아니라 DSA+MoE 블록과 출력 head를
실제로 실행합니다. 다만 기존 fixed-1에서 이 비용은 약 1.15 ms/round여서
수십 ms인 검증에 비하면 작습니다. draft만 더 빠르게 만드는 것으로는
목표까지의 차이를 메울 수 없습니다.

검증에서 dense projection은 여러 row를 함께 계산할 수 있지만 KDA의
conv/recurrent state, DSA cache update/attention, hyper-connection, routing은
각 row의 순서를 지킵니다. “두 토큰을 검증하므로 한 토큰 비용으로 두 개”가
되지 않습니다. 특히 RAM에서 expert를 가져오는 경로는 row마다 호출됩니다.

근거: `src/core/glm_fast_path.cu:6860`의 `fast_mtp()`,
`src/core/glm_mtp.cu:595` 이후 decode loop,
`src/core/glm_window.cu:515` 이후 `fast_moe_window()`.

## 본전이 된 이유와 기존 기록에서 바로잡은 부분

### 1. stagger와 dense batching의 교환 관계

기본 `WIN_STAGGER=1`은 T개 row를 두 그룹으로 나눠 다른 그룹의 attention을
실행하는 동안 helper 응답을 기다립니다. T=2이면 1+1이 되어 dense GEMV도
한 row씩 실행됩니다. `WIN_STAGGER=0`은 T=2의 dense 계산을 묶어 주지만,
그 대신 두 row의 이전 layer 결과가 모두 준비될 때까지 기다려야 합니다.

Claude의 이전 A/B에서:

| 설정 | 대화 tok/s | 코드 tok/s |
|---|---:|---:|
| MTP off | 34.5 | 29.3 |
| fixed-1, 기본 stagger | 30.9 | 25.0 |
| fixed-1, stagger off | 34.6 | 29.7 |

따라서 `WIN_STAGGER=0`은 필요한 출발점이지만 이미 본전까지 회복시킨
변경입니다. 새 발견이나 추가로 얻을 수 있는 속도라고 다시 세면 안 됩니다.
근거: `glm_window.cu`의 `fast_window_layers()` 마지막 stagger 분기,
원본 `PROGRESS-mtpreview.md`의 item 3.

### 2. fixed-2는 수락률보다 scratch 재실행이 먼저 문제였습니다

일반 decode의 공용 scratch는 **8개**입니다. window는 각 row의 expert를
전부 가져온 뒤 계산하므로 여러 row의 miss가 이 8개를 동시에 점유합니다.
각 layer의 여분 슬롯 3개로 승격하지 못하고 scratch도 다 쓰면 sticky clash
bit를 세운 뒤, window 끝에서 **trunk 전체를 다시 실행**합니다.

이 공간은 42개 layer 각각에 별도 할당된 8개가 아니라, layer들이 재사용하는
공용 scratch 8개입니다. 현재 pack 최대 expert blob은 14,680,064 B = 14 MiB:

| 추가 임시 공간 | 정확한 추가량 |
|---|---:|
| row 하나용 8개 | 117,440,512 B = 112 MiB |
| row 두 개용 16개 | 234,881,024 B = 224 MiB |

근거: `glm_fast_path.cu:1027`의 할당,
`src/kernels/cuda/glm_fast.cu:4162`의 claim/fallback,
`glm_window.cu`의 `clash && attempt == 0` 처리.

기존 GPU phase 계측은 재실행 직전에 `W.ev_n=0`으로 첫 pass의 이벤트를
버립니다. 따라서 `glm window prof`는 실패한 첫 pass까지 합한 검증 시간으로
해석할 수 없습니다. 이 때문에 fixed-2의 GPU summary 78 ms와 실제 verify
host wall 131 ms가 크게 달랐습니다. 그 차이를 전부 host enqueue 비용으로
진단하면 잘못된 최적화를 하게 됩니다.

반대로 **실패한 첫 pass의 시간을 정상 batched 검증의 예상 비용으로 쓰는
것도 잘못**입니다. clash 때 scratch pointer를 재사용해 다른 row의 expert
blob을 덮어쓸 수 있고, 그 결과 뒤쪽 layer의 hidden state, routing과 cache
경로도 달라집니다. “131 − 78 = 53 ms이므로 scratch만 늘리면 53 ms” 같은
계산은 성립하지 않습니다. 독립 scratch를 실제로 적용한 정상 실행을
측정하고 비교해야 합니다.

기존 두 로그의 **전체 요청을 round 수로 가중합**한 값입니다. 마지막 요청의
짧은 greedy probe만 대표값으로 사용하지 않았습니다.

| 항목 | fixed-1 | fixed-2 |
|---|---:|---:|
| drafting rounds | 916 | 714 |
| accepted / proposed drafts | 790 / 916 | 994 / 1423 |
| 전체 수락 비율 | 86.24% | 69.85% |
| accepted drafts / round | 0.862 | 1.392 |
| draft host ms/round | 1.150 | 2.243 |
| verify host ms/round, 재실행 포함 | 57.750 | 130.845 |
| commit/cache host enqueue ms/round | 0.038 | 0.247 |
| 기존 GPU phase summary ms/window | 55.940 | 78.069 |

기존 progress 표의 fixed-2 91.7%는 **마지막 요청의** 값입니다. 전체 수락률이
아닙니다. 2.86 tokens/round도 전체 워크로드를 대표하지 않습니다.

### 3. MTP 자체가 trunk cache를 줄입니다

기존 로그에서 off는 총 1520 slots, MTP는 1485 slots였습니다. MTP 1485에는
draft layer의 34 slots도 포함됩니다. 따라서 trunk는 1451 slots로,
실제 trunk 감소분은 **69 slots, 약 4.5%**입니다. “전체 슬롯 35개 감소”만
보면 trunk에 미치는 영향을 과소평가합니다.

VRAM pool은 19.12 → 18.70 GiB였고, MTP 활성화 시 prefill borrowed requirement도
7824 → 9671 MiB로 커졌습니다. 이 변화는 기존 max-context/prefill 설정에서
측정한 것이며, MTP가 공짜로 올라가는 상태가 아닙니다.

그래서 `MTP_DECODE=1, MTP_FIXED=0`이 유용한 대조군입니다. MTP 메모리 배치를
확보하되 평범한 token path로 실행하여, 추측 자체와 메모리 배치 비용을
분리할 수 있습니다. 단 이 모드는 NextN cache 유지도 수행하므로 완전히
“메모리만” 달라지는 대조군은 아닙니다.

### 4. promotion 감소와 WIN_SPARES 실패 원인은 별개로 다뤄야 합니다

window당 boundary가 한 번이라 layer별 spare 3개의 보충 주기가 emitted token
기준으로 길어집니다. 기존 fixed-1은 promotion이 약 20–21/token으로,
off의 약 33–34/token보다 적었습니다. 미승격 scratch fetch가 이후 다시
miss하는 악순환을 만들 수 있습니다.

그러나 이전 `WIN_SPARES=1` 실험은 0.8–3.8 tok/s로 크게 회귀했습니다. 원본
메모의 “local CPU fallback 증가가 원인일 수 있다”는 가설은 **확정 사실이
아닙니다**. 정상 fixed-1도 15,548 locally-computed experts, 실패 실험은
15,447였고, 실패 실험의 CPU 계산은 2–10 ms/token인데 전체 시간은
264–1212 ms/token이었습니다. 수백 ms stall은 별도 동기화/서비스 의존
문제까지 조사해야 합니다. 개선된 copy-aware guard만으로 이 문제가 모두
해결됐다고 단정할 수 없습니다.

## 현재 copy-aware 정책은 MTP window에도 적용됩니다

`fast_moe_window()`는 `md=F->md`에서 시작하고 row별 `tok_dev()`는 plan과
answer 포인터만 바꾸므로 `copy_kind`와 `copy_lane_plan`을 유지합니다.
batched 경로와 token-by-token fallback 모두 같은 `gf::moe_route()`를 씁니다.

`glm_fast.cu:4094–4114`의 copy-aware 분기에서는 복사본이 있는 RAM expert만
`copy_eligible`에 들어갑니다. 그러므로 새로 demote된 expert에 복사본이
없으면 optional local CPU lane으로 보내지 않고 GPU fetch 경로에 남깁니다.
mandatory helper 소유 expert는 기존 helper로 갑니다. 0.16 ms budget 정책은
현재 MTP에도 적용되지만, 그 정책은 **row 단위** 비용 모델입니다. window의
held requests와 누적 helper 작업량을 공동 최적화하지는 않습니다.

`NET_WIN_HOLD`는 기본 on으로, row의 요청은 바로 보내고 마지막 row가 route된
뒤 응답들을 회수합니다. 이는 네트워크 요청 자체를 하나로 합치는 것과
다릅니다. 현재 `NET_PROTO=2` 경로는 `!in_win` 조건 때문에 window에는
적용되지 않습니다. 그렇다고 FP16 응답이나 더 낮은 precision으로 바꾸는
방법을 이 검토의 무손실 개선으로 취급하지 않았습니다.

## 50 tok/s를 위한 정확한 조건

제안 n개, 조건부 수락률 a1…an, 라운드 전체 비용 Cn이라면:

```text
E[emitted tokens / round] = 1 + a1 + a1*a2 + … + a1*…*an
MTP ms / emitted token    = Cn / E
본전 조건                 Cn < plain_ms * E
50 tok/s 조건             Cn <= 20 ms * E
```

Cn에는 draft, 검증, scratch 재실행, 상태 복구, NextN cache 갱신이 모두
포함돼야 합니다. 동기화를 기다리지 않는 commit의 host enqueue 시간만
device 실행 비용으로 해석해서는 안 됩니다. 일부 비용은 다음 라운드의
draft sync/boundary에 청구됩니다. 충분히 긴 구간의 전체 wall / emitted
tokens가 최종 기준입니다.

역사적 fixed-1 E≈1.862에서는 전체 round가 **37.25 ms 이하**여야 50 tok/s가
됩니다. 측정된 draft+verify+commit enqueue 합 약 58.94 ms보다 약 36.8%
낮아야 합니다. acceptance만 100%로 높여도 2/1.862 = **최대 약 7.4%**
속도 상승이라서 이 차이를 해결하지 못합니다.

fixed-2는 실제 accepted/round로 E≈2.392이며, 전체 round가 **47.84 ms 이하**여야
합니다. 69.85%라는 aggregate acceptance를 a1=a2처럼 넣어 `1+a+a²`를
계산하면 안 됩니다. `spec_steps`는 도달하지 않은 뒤쪽 draft도 proposed에
포함하기 때문입니다.

이 수치는 현재 DDR5-5200/copy-aware 결과의 예측이 아니라, 목표를 판정할
계산법과 기존 실패 규모를 보여 줍니다. 새 환경의 실제 E와 C로 갱신해야 합니다.

## 실행 가치가 있는 순서

1. **현재 설정의 off / MTP fixed-0 / fixed-1 재측정.** fixed-1은
   `MTP_DECODE=1`, `MTP_FIXED=1`, `WIN_STAGGER=0`; `WIN_SPARES=0`을 유지합니다.
   기존 Claude 하네스를 그대로 쓰고, 속도 측정과 상세 `WIN_PROF` 측정을
   구분합니다. 새 copy-aware 정책으로 local fallback은 없어진 상태인지,
   helpers의 throughput/대기 시간이 달라졌는지 확인합니다.
2. **재실행 계측 후 row별 scratch 분리.** fixed-1/fixed-2에서 replay 비율이
   높으면 이미 `glm_rows.cu:488–501`에 있는 `rows_scr + 8*i` 방식을 MTP에도
   적용하는 후보가 가장 직접적입니다. T=2/3에 필요한 112/224 MiB를 pool
   계산 전에 예약하고, 최대 T를 명확히 제한해야 합니다. kernel의 8-bit
   claim 구조를 무작정 확장할 필요는 없습니다. 늘어난 scratch만큼 줄어드는
   cache와 동일 expert의 중복 fetch도 실제 측정해야 합니다.
3. **그 후 promotion/배치 비용을 개선.** mid-window spare 갱신은 기존
   실패 원인을 계측한 뒤 별도 실험으로 다룹니다. 또는 window 전체의
   mandatory+optional helper 작업량으로 비용 모델을 만들어 row별 과다
   offload를 피합니다. 아무 조건 없이 draft 길이를 늘리는 것은 우선순위가
   아닙니다.

fixed-1의 이득이 확인돼도 기존 adaptive를 그대로 켜면 기본 시작 길이 2를
시험합니다(`mtp_choose()`, `MTP_START` 기본 2). scratch 문제가 남아 있다면
첫 답변에서 느린 길이를 탐색하는 비용이 큽니다. 그 경우 최대 draft를 1로
제한한 뒤 0/1 사이에서만 선택하는 adaptive가 운영 후보입니다. 이는 속도
이득의 증명이 아니라, 확인된 나쁜 경로로 들어가지 않게 하는 제한입니다.

CUDA graph부터 다시 구현하는 것은 우선순위가 낮습니다. 기존 nsys 검토에서
verify 중 GPU가 host enqueue를 기다려 빈 시간이 거의 없었습니다. 보고서의
36 ms host enqueue를 GPU 작업 시간에 더하거나 이를 전부 없앨 수 있다고
계산하면 안 됩니다. 기존 nsys 결론은 현재 설정에서도 필요한 경우 짧은
timeline으로 재확인할 수 있습니다.

수락률을 올리는 speculative sampling이나 다른 drafter는 후속 후보입니다.
현재 single-trunk 경로는 argmax draft와 target sampler의 exact match를
비교하고, `SPEC_SAMPLE`의 p/q 방식은 다른 split-trunk 경로에만 있습니다.
이를 옮기려면 각 depth의 q와 target p, rejection residual, RNG 처리를
검증해야 하며 단순 flag 변경이 아닙니다. 확률분포 보존을 검증해야 하고,
낮은 양자화나 trunk expert 생략으로 얻은 속도를 목표 달성으로 세면 안 됩니다.

## 정확성에 필요한 검증

KDA는 rejected suffix 이전의 recurrent/conv state로 되돌린 뒤 kept rows의
저장된 입력을 replay합니다. DSA는 버려진 위치를 다음 실행에서 다시 쓰며
현재 query는 유효한 prefix만 읽습니다. NextN KV도 draft hidden이 아닌
확정된 trunk hidden으로 다시 채웁니다. 이 세 가지를 빠뜨리는 최적화는
속도가 빨라도 올바른 MTP가 아닙니다.

기존 여러-column GEMV 검사는 130,986 columns / 488,279,744 values에 대해
bit difference 0을 보고했습니다. 이는 해당 dense kernel의 근거이며 MTP
전체의 parity 증명은 아닙니다. 별도 생성의 greedy hash는 off-vs-off도
달라져 충분하지 않았습니다. 같은 입력·초기 state·고정 route/residency에서
token path와 window path의 logits/state를 비교하고, rejection 위치마다
KDA/DSA/NextN 상태와 후속 토큰까지 비교하는 in-engine 검사가 더 적합합니다.
프롬프트 cache 재사용 및 keep-answer snapshot과의 결합도 포함해야 합니다.

## 이번에 추가한 최소 진단

`MAYA_GLM_WIN_AUDIT=1`은 `window()`의 기존 동기화 뒤에서만 host timestamp와
scratch sticky bit를 읽습니다. 새 kernel, memcpy, GPU sync를 추가하지 않습니다.
`WIN_PROF`와 독립적이고 기본 off입니다. 요청 끝 `win_prof_report()`에서:

- window/row/attempt 수, scratch replay 수와 비율, 그룹별 clash 수
- 모든 attempt를 포함한 host ms/window와 setup / discarded attempts /
  final attempt / other
- T별 windows/replays

를 출력하고 요청 누계를 초기화합니다. `rows`는 원래 window row 수이며
재실행한 row를 중복 포함하지 않습니다. replay 실행량은 T별 replay 수를
곱해서 별도로 구합니다. `discarded attempts`는 첫 full window/head 실행을
포함하고, 복구 host enqueue는 other, 비동기 복구의 device 실행은 final
attempt에 포함될 수 있습니다. 이 host 구간을 GPU phase 시간과 합산하면
중복 계산입니다. draft/accept/commit은 window 외부라 전체 MTP round 시간은
별도로 봐야 합니다. layer별 scratch high-water는 현재 기록하지 않습니다.

## 2026-10-27 22시 재측정: 본전에서 소폭 개선, 아직 50과는 차이가 있습니다

원본: 홈 PC `~/fn/codex-cycle-20261027/`의 `plain`, `mtp-loaded-fixed0`,
`mtp-fixed1` 설정/engine 로그/pass 결과입니다. 완료된 파일을 로컬
`maya-analysis-20261027/mtp-cycle/`에도 보관했습니다. 기존 Claude qshort
하네스로 pass당 chat 4×200, code 4×200, greedy 120 tokens를 생성했습니다.
plain/fixed-1은 두 pass, fixed-0은 한 pass입니다. 같은 서버의 두 pass라
서버 재시작 간 분산을 평가한 결과는 아닙니다.

| 설정 | chat tok/s, pass 1 / 2 | code tok/s, pass 1 / 2 | RAM fetches/emitted token, chat / code |
|---|---:|---:|---:|
| plain | 36.3 / 36.5 | 31.7 / 31.7 | 약 35.4 / 49.0 |
| MTP loaded, fixed-0 | 35.4 / 미실행 | 31.0 / 미실행 | 36.4 / 50.4 |
| MTP fixed-1, stagger=0 | 38.0 / 38.1 | 32.1 / 33.4 | 약 40.9 / 57.3 |

fixed-1의 두 pass 평균은 plain보다 chat 약 4.5%, code 약 3.3% 빠릅니다.
수락률이 높아져서만 얻은 개선은 아닙니다. fixed-1은 fetch/emitted token이
여전히 더 많고, 시작부터 scratch replay 경고가 있습니다. 새 WIN_AUDIT
진단과 row별 scratch probe의 결과로 이 비용을 구분해야 합니다.

### 요청 수가 아닌 round 수로 가중한 실제 비용

| 항목 | chat 8요청 | code 8요청 | greedy 포함 짧은 18요청 | 긴 문맥 2요청 |
|---|---:|---:|---:|---:|
| drafting rounds | 870 | 845 | 1837 | 233 |
| accepted drafts | 718 | 742 | 1574 | 209 |
| accepted/proposed | 82.53% | 87.81% | 85.68% | 89.70% |
| E = 1 + accepted/round | 1.8253 | 1.8781 | 1.8568 | 1.8970 |
| draft host ms/round | 1.110 | 1.075 | 1.090 | 1.161 |
| verify host ms/round | 47.031 | 56.587 | 51.581 | 55.905 |
| commit/cache host enqueue ms/round | 0.026 | 0.023 | 0.024 | 0.020 |
| 위 host phase 합 ms/round | 48.168 | 57.684 | 52.696 | 57.086 |
| 현재 E에서 50 tok/s인 round 상한 | 36.506 | 37.562 | 37.137 | 37.940 |
| phase 합에서 필요한 절감 비율 | 24.2% | 34.9% | 29.5% | 33.5% |

로그의 phase 평균은 소수 둘째 자리로 반올림된 값이고, commit은 비동기
enqueue이므로 위 합은 정확한 GPU 실행 시간 분해가 아닙니다. 그래도 충분한
round의 host 비용 규모를 보여 줍니다. 최종 속도는 API의 predicted_ms /
실제 emitted token으로 판단해야 합니다. 요청 첫 토큰, 끝에서 남은 fixed-0
round, 종료 조건 때문에 `전체 요청 토큰/전체 round`와 `1+accepted/drafting
round`는 약간 다릅니다. 후자는 지속적인 drafting round의 break-even 식에
사용했습니다.

16,524-token prompt의 첫 요청은 prefill 12.774초(1,293.6 tok/s), decode
32.6 tok/s였습니다. 같은 prompt 재사용은 16,523 tokens를 cache에서 복구하고
decode 34.2 tok/s였습니다. 짧은 질문에서 얻은 속도를 긴 문맥 전체에 그대로
적용할 수 없습니다.

### cold RAM이 483개에서 793개로 늘어난 이유

MTP를 켜면 단지 작은 draft activation이 생기는 것이 아니라 **NextN MoE
한 층, expert 288개가 모델 저장 계층에 추가**됩니다. `load_pack()`은
`load_mtp()` 뒤 `lt_=l1_+1`로 바꾸고, `load_mtp()`는 blk.45 expert를
`pack_layers_[45]`에 등록합니다. 이후 VRAM/RAM/helper/cold/copy 배치 loop는
모두 `il < lt_`를 순회합니다. 로그의 “trunk's RAM-tier experts”라는 문구도
실제 loop 범위에는 이 draft 층을 포함합니다.

GGUF 5번째 shard의 실제 tensor metadata를 읽어 확인한 크기는 다음과 같습니다.
gate/up은 `[4096,2048,288]`, down은 `[2048,4096,288]`, 세 tensor 모두
type 2(Q4_0)입니다. Q4_0의 32값/18bytes에 따라 **expert당 13.5 MiB,
288개 합 4,076,863,488 bytes = 3.796875 GiB**입니다. 기존 IQ4_XS pack에
원래 들어 있던 NextN의 형식을 읽은 것이며 새로 양자화를 낮추지 않았습니다.

| 시작 시 배치 | plain | MTP fixed-0 | MTP fixed-1 |
|---|---:|---:|---:|
| pool 전 5090 VRAM, GiB | 10.81 | 11.23 | 11.23 |
| 5090 expert pool, GiB | 19.12 | 18.70 | 18.70 |
| 총 pool slots | 1520 | 1485 | 1485 |
| 그중 draft층 slots | 0 | 34 | 34 |
| trunk pool slots | 1520 | 1451 | 1451 |
| main pinned expert RAM, GiB | 91.75 | 91.62 | 91.64 |
| main RAM slots | 7294 | 7275 | 7277 |
| helper 소유와 중복되어 반환/재충전한 main RAM slots | 1636 | 1580 | 1582 |
| cold remote RAM experts | 483 | 795 | 793 |
| cold remote RAM, GiB | 6.01 | 9.97 | 9.94 |
| remote 선택적 복사본 수 | 3853 | 3536 | 3538 |
| remote 선택적 복사본, decimal GB | 51.51 | 47.27 | 47.30 |
| copy 로그의 RAM-tier expert 수 | 7282 | 7094 | 7096 |

local helper GPU 1674개, remote GPU 각 1198개의 소유 expert 수는 유지됩니다.
하지만 MTP 가중치가 추가되고 main VRAM/RAM 저장 공간도 조금 줄어,
remote CPU가 담당하는 cold 원본이 약 3.93 GiB 늘어납니다.

`glm_fast_path.cu`의 cold-share loop는 VRAM/main RAM/GPU helper에 없는
원본 expert부터 remote RAM에 넣습니다. 다음 copy-share loop가 같은 helper의
`ram_available()` 안에서 선택적 복사본을 채웁니다. 따라서 cold 원본 증가가
선택적 복사본 용량을 직접 잠식합니다. 51.51→47.30 decimal GB 감소는
약 3.92 GiB로 cold 증가와 맞습니다. `NET_COPY_GB=84`는 상한일 뿐,
실제 helper의 54 GiB RAM 예산을 늘려 주지는 않습니다.

이 때문에 “MTP는 135MB라 RAM 영향이 작다”는 설명은 틀립니다. 135MiB는
NextN **dense tensor만** 센 로그입니다. 3.797GiB routed expert와 상태/검증
버퍼는 별도입니다. slot 개수의 변화에는 size class와 중복 배치도 섞이므로,
모든 extra byte를 단순히 `288 + 줄어든 slots`로 더해 정확히 맞추면 안 됩니다.

### main GPU/CPU의 추가 버퍼를 분리하면

실제 GGUF geometry는 E=4096, KDA inner=8192, recurrent 34층, trunk DSA 11층,
NextN DSA 1층입니다. 소스의 allocation 식으로 계산한 값은 다음과 같습니다.

| 추가 저장 공간 | 크기 | 위치/근거 |
|---|---:|---|
| NextN dense tensor | 로그상 135 MiB | GPU, `glm_model.cu::load_mtp()` |
| NextN DSA state | 57.0625 MiB | GPU, latent17 + key/gate8.0625 + pooled index32 |
| verify window arena | 85.3958 MiB | GPU, `glm_window.cu::win_alloc(16)`; head banks는32 |
| KDA 상태 backup | 145.5625 MiB | GPU, `spec_kda_bytes()`의34층 |
| KDA commit replay workspace | 1.25 MiB | GPU,8층×5×8192 float |
| trunk capture rows | 0.25 MiB | GPU,16×4096 float |
| NextN cache-update row buffers | 1.0078 MiB | GPU, `mtp_cache_setup()` |
| draft hidden/logits/token 등 | 약0.6153 MiB | GPU, `fast_setup()` |
| draft head의 추가437행과 ID | 약1.8154 MiB | GPU, 기존 output head의 일부만 별도 gather |
| NextN full latent cache | 136 MiB | main pinned RAM,262144×544bytes |

GPU의 위 명시적 allocation 합은 dense 로그의 반올림을 포함해 약428MiB로,
시작 시 VRAM 사용량 약0.42GiB 증가와 일치하는 규모입니다. CUDA allocation
정렬과 driver 부수 비용까지 일치하는 실측 byte 회계는 아닙니다. main RAM은
이136MiB latent와 작은 host row/answer buffers 외에 메모리 측정 시점 차이도
반영되므로 17/19개 slot 감소 전부를 하나의 버퍼 탓으로 확정하지 않았습니다.

draft vocab 로그의 “410 of 643 MB a draft”는 head **읽기량**입니다. 기존
output.weight 앞부분을 재사용하므로 410MiB 전체가 추가 할당되는 것은
아닙니다. 전체 token embedding을 복제하는 `MTP_CHAIN`도 이 장비에서는
기본 room 조건을 만족하지 않으며 실제 로그에 chained 표시가 없습니다.

prefill activation 5304→7151MiB, borrow7824→9671MiB 증가는 별개입니다.
`glm_prefill.cu:1473`은 NextN cache fill용 scratch를
`T*(6*E+kv_lora+2*idx_key)*4`로 잡습니다. T32768에서3168MiB인 이 값이
기존 mixer/MoE union보다 커져 추가1847MiB를 예약합니다. **이 공간은
prefill 중 expert pool에서 빌리며 계속 점유하는 decode VRAM이 아닙니다.**
실제 짧은 prompt는 `prefill_carve(Tn)`가 필요한 만큼만 빌립니다.

또한 이 scratch 상한에는 과대 예약 후보가 있습니다. 실제 cache-fill
Carve(`glm_prefill.cu:4249` 부근)는 h(E float), emb(E float),
cat16(2E uint16), hid(E float), kv/ik/ig이므로 float 환산 **4E+kv+2idx**입니다.
`hgemm_q`/`sgemm_bf16`은 별도 weight/workspace를 사용합니다. 독립 소스
검토에서도 추가 `S->uni` 소비자를 찾지 못했습니다. 측정과 실행에 공통
`carve_mtp()`를 사용하도록 정리하면 32K chunk에서 최대1GiB, 16K에서
약512MiB의 불필요한 차용을 줄일 후보입니다. 다른 mixer/MoE scratch가
더 크면 실제 절감은 줄어듭니다. 아직 수정하거나 GPU로 검증하지 않았습니다.

지속적인 decode 개선은 먼저 재실행/배치/승격 비용을 줄여야 합니다.
MTP window 최대치16을 실제2/3에 맞게 작게 예약하거나 prefill cache fill
scratch를 정리하는 작업은 용량 개선 후보지만, 288개 draft expert의 기본
가중치나 KDA rollback 상태를 제거하지는 못합니다. 추가 scratch112/224MiB
probe의 평가에도 줄어든 pool/copy capacity를 포함해야 합니다.

## WIN_AUDIT 실측: fixed-1은 재실행이 드물었습니다

원본 `~/fn/codex-window-audit-20261027/fixed1-audit-engine.log`, 현재5200/.16
설정, fixed-1/stagger0, 기존 shared scratch, WIN_AUDIT만 켠 첫 진단입니다.
하네스 결과는 chat38.0/code33.4 tok/s로 앞선 비계측 pass 범위 안입니다.
이 비교만으로 계측 overhead가 정확히0이라고 단정하지는 않았습니다.

| 항목 | chat4요청 | code4요청 | greedy 포함9요청 |
|---|---:|---:|---:|
| windows | 433 | 422 | 916 |
| scratch replays | 1 | 7 | 8 |
| replay 비율 | 0.231% | 1.659% | 0.873% |
| inclusive host ms/window | 47.385 | 55.451 | 51.215 |
| setup ms/window | 0.446 | 0.473 | 0.462 |
| discarded attempts ms/window | 0.166 | 1.061 | 0.567 |
| final attempt ms/window | 46.773 | 53.916 | 50.184 |

T는 전부2이며 총924 attempts입니다. replay된8건의 버려진 첫 attempt는
평균 약64.98ms였지만, 916 windows 전체로 나누면0.567ms입니다.
**현재 fixed-1의 주병목은 scratch 재실행이 아닙니다.** T2 row-private
scratch는 이 소수의 재실행과 token fallback을 없앨 후보이며, 대신112MiB의
expert pool을 사용하므로 실측 이득이 작거나 상쇄될 수 있습니다.

fixed-2는 별도 진단이 필요합니다. 또한 discarded/final의 차를 빼서 정상
batched 비용이라고 추정해서는 안 됩니다. scratch가 부족한 첫 pass는
expert pointer alias와 달라진 route/cache 경로를 포함할 수 있습니다.
재실행 없이 올바르게 계산하는 probe의 전체 emitted-token 속도로 판단해야
합니다. WIN_PROF를 켠 다음 arm은 더 무거운 계측이므로 이 비계측에 가까운
WIN_AUDIT arm의 속도와 직접 우열을 정하면 안 됩니다.

같은 설정에 WIN_PROF를 추가한 `fixed1-stages`는 chat36.9/code32.4 tok/s,
918 windows 중7회 replay(0.763%)였습니다. round 가중 최종 attempt의 GPU
구간은51.149ms/window로 다음과 같습니다.

| WIN_PROF 구간 | ms/window | 해석 |
|---|---:|---|
| moe_routes | 25.017 | routing/fetch를 포함하는 공급 구간 |
| moe_cpu_wait | 9.069 | CPU/local·remote helper 결과를 기다리는 구간 |
| moe_experts | 4.095 | batched routed-expert 계산 |
| moe_mv | 1.642 | MoE router/shared projection |
| 나머지 | 약11.325 | KDA/DSA/dense/hyper-connection 등 |

`moe_routes + moe_cpu_wait`만 최종 GPU summary의 약66.6%입니다. 새 설정에서도
fixed-1의 중심 과제는 expert 공급과 helper 응답입니다. profile의 기존
제약대로 버려진 attempt는 위 GPU 합계에 들어 있지 않습니다. 같은 run의
inclusive host52.799ms는 setup0.454 + discarded0.498 + final51.815 +
other0.033으로 확인됐습니다. GPU 합계에 discarded host를 기계적으로 더해
새 GPU phase 합계를 만들지는 않았습니다.

## fixed-2에는 재실행 비용이 크게 남아 있습니다

같은 진단 run의 `fixed2-audit`는 chat28.2/code22.1 tok/s였습니다. 설정은
fixed-2, stagger0, 기존 shared scratch이며 WIN_AUDIT만 켰습니다.

| 항목 | chat4요청 | code4요청 | greedy 포함9요청 |
|---|---:|---:|---:|
| windows | 344 | 310 | 700 |
| scratch replays | 75 | 138 | 217 |
| replay 비율 | 21.80% | 44.52% | 31.00% |
| inclusive host ms/window | 79.918 | 114.165 | 94.619 |
| setup ms/window | 0.465 | 0.509 | 0.484 |
| discarded attempts ms/window | 16.155 | 36.482 | 24.519 |
| final attempt ms/window | 63.287 | 77.150 | 69.599 |
| draft host ms/round | 2.109 | 2.077 | 2.093 |
| commit/cache host enqueue ms/round | 0.268 | 0.178 | 0.223 |
| accepted/proposed | 449/686 | 485/619 | 1007/1396 |
| E = 1+accepted/round | 2.3052 | 2.5645 | 2.4386 |
| 현재 E에서50 tok/s인 round 상한 | 46.105ms | 51.290ms | 48.771ms |

700 windows는 원래2096 rows(T3×696 + T2×4)이며 총917 attempts를 수행했습니다.
weighted acceptance는72.13%입니다. 첫 phase의 fixed-1보다 더 많은 draft를
수락해도, round가 약96.94ms에 달하므로 emitted-token 비용이 커집니다.

여기서는 scratch 충돌이 주요 원인이라는 가설이 계측으로 뒷받침됩니다.
그러나 discarded24.519ms를 빼고 남은 final69.599ms에는 정상 batch와
token-by-token fallback이 섞여 있습니다. 반대로 discarded pass 역시
잘못된 scratch alias의 영향을 받아 정상 batch의 비용을 대표하지 않습니다.
row-private T3 실측이 재실행 제거 후의 검증 비용과 수락률·cache 배치까지
함께 확인하는 필수 다음 단계입니다.

## row-private scratch 실험

`~/fn/codex-window-scratch-20261027/`의 별도 실험 binary로 측정했습니다.
`WIN_SCRATCH_ROWS=2/3`은 기존 row scratch allocation을 재사용하고, 각
window row에8개씩 절대 row index로 분리한 포인터를 줍니다. shared claim
word를 넘기지 않으므로 다른 row가 같은 임시 expert blob을 덮지 않습니다.
top-8 계산, 원본 가중치, KDA rollback, DSA/NextN cache 갱신은 유지합니다.
기본 off이며 GPU 수학 parity 전체를 검증한 운영 변경으로 간주하지 않습니다.

### T2 / fixed-1

| 항목 | 기존 shared fixed-1 | row-private T2 |
|---|---:|---:|
| chat tok/s,2 passes | 38.0 / 38.1 | 37.6 / 38.5 |
| code tok/s,2 passes | 32.1 / 33.4 | 33.3 / 33.6 |
| 긴 문맥 first/repeat decode | 32.6 / 34.2 | 32.6 / 34.4 |
| 짧은18요청 drafting rounds | 1837 | 1837 |
| accepted drafts | 1574 | 1575 |
| E | 1.8568 | 1.8574 |
| draft host ms/round | 1.090 | 1.087 |
| verify host ms/round | 51.581 | 51.050 |
| commit/cache enqueue ms/round | 0.024 | 0.025 |

private T2는 짧은1837 + 긴231 windows에서 scratch replay가0입니다.
shared의 별도 WIN_AUDIT arm에서는8/916이었습니다. 비용은 약112MiB의
추가 VRAM이며 pool1485→1476 slots(18.70→18.59GiB), cold793→806개,
remote copy3538→3525개(47.30→47.13 decimal GB)로 변했습니다.

T2 실험은 재실행 제거의 기전이 동작함을 보여 주지만 큰 속도 개선을
보이지 않았습니다. chat 평균은38.05로 동일하고 code 평균은32.75→33.45로
약2.1% 차이입니다. acceptance/출력/배치 상태가 매번 같지 않고 독립 서버
반복도 적으므로 이2.1%를 확정된 순수 scratch 효과로 주장하지 않았습니다.
약52.16ms인 round 비용은 E1.8574에서50 tok/s에 필요한37.15ms보다 여전히
약15ms 깁니다.

정상 T3 경로가 확보된 뒤에는 dense kernel의 작은 낭비도 검토할 수 있습니다.
`glm_fast.cu:1811`의 `mvc_launch()`는 ncol3을NC4로 반올림합니다.
`mvc_kernel_t`는 네 번째 열에 마지막 실제 입력을 복제하고 dot/reduction을
계산한 뒤 저장만 생략합니다. Q8_0/Q4_K/Q6_K에NC3 specialization을 추가하면
해당 padding 계산을 피할 수 있습니다. weight 읽기 자체는 줄지 않고 전체
round의 dense 부분에만 영향을 주므로, 예상 개선을25% 전체 가속으로
해석하면 안 됩니다. 기존 `MV_COLS_CHECK`와 실측이 필요한 미구현 후보입니다.

### T3 / fixed-2: 재실행은 없어졌지만 정상 검증이 여전히 비쌉니다

| 항목 | 기존 shared fixed-2 | row-private T3 |
|---|---:|---:|
| chat tok/s | 28.2,1 pass | 36.0 / 36.3,2 passes |
| code tok/s | 22.1,1 pass | 33.5 / 32.8,2 passes |
| RAM fetches/emitted token, chat | 62.5 | 49.6 / 49.4 |
| RAM fetches/emitted token, code | 92.4 | 62.0 / 63.9 |
| 짧은 windows | 700 | 1370 |
| scratch replays | 217/700,31.0% | 0/1370 |
| 긴 문맥 windows/replays | 미측정 | 176/0 |
| GPU pool | 1485 slots,18.70GiB | 1468 slots,18.49GiB |
| cold remote RAM | 793개,9.94GiB | 809개,10.14GiB |
| remote optional copies | 3538개,47.30GB | 3522개,47.09GB |

T3에서 필요한 추가224MiB를 실제로 예약했고, scratch 충돌/replay는 모든
짧은·긴 요청에서0이 되었습니다. shared의 심한 회귀가 크게 회복되어 원인을
수정했다는 근거가 되지만, fixed-1/T2보다 더 빠른 운영 설정은 아니었습니다.

| T3 private,round 가중 | chat8요청 | code8요청 | greedy 포함18요청 | 긴 문맥2요청 |
|---|---:|---:|---:|---:|
| drafting rounds | 663 | 620 | 1370 | 176 |
| accepted/proposed | 923/1325 | 970/1234 | 2042/2733 | 272/352 |
| E | 2.3922 | 2.5645 | 2.4905 | 2.5455 |
| draft ms/round | 2.108 | 2.077 | 2.092 | 2.275 |
| verify inclusive host ms/round | 64.105 | 75.431 | 69.380 | 75.724 |
| commit/cache enqueue ms/round | 0.247 | 0.171 | 0.207 | 0.190 |
| 위 phase 합 ms/round | 66.459 | 77.679 | 71.679 | 78.187 |
| 현재 E에서50 tok/s인 상한 | 47.843 | 51.290 | 49.810 | 50.909 |
| 필요한 round 비용 절감 | 28.0% | 34.0% | 30.5% | 34.9% |

긴16,524-token prompt는 prefill12.760초(1294.9 tok/s), decode31.4 tok/s였고,
재사용 시16,523 tokens cache hit 후 decode34.1 tok/s였습니다. T2의32.6/34.4보다
빠르지 않습니다.

새 T3의 정상 검증69.38ms는 이전 shared의 `final attempt`69.60ms와 우연히
가깝지만, 그것을 소스만으로 미리 정확히 예측할 수 있었다는 뜻은 아닙니다.
정상 경로의 전체 실측이 있어야만 판단할 수 있는 수치였습니다. 이제는
**replay를 없앤 뒤에도 검증 자체가 약69ms/round**라는 것이 확인됐습니다.

짧은 요청의 현재 phase 합71.68ms를 유지한다고 가정하면, 수락률이100%라도
한 round 최대3토큰이므로41.9 tok/s 정도입니다. 따라서 acceptance 개선만으로
50에 도달할 수 없습니다. 현재 E2.4905에서는 round49.81ms 이하로 줄여야
하므로 정상 검증/공급 경로에서 약21.9ms의 추가 절감이 필요합니다.
모델 양자화 하향이나 trunk expert 생략 없이50이 가능하다고 입증한 상태는
아니며, 지금 확보한 실측 최선은 대화 약38, 코드 약33 tok/s 수준입니다.

## 후속 helper T행 batch의 구체적 범위

현재 `NET_WIN_HOLD`는 row별 begin을 먼저 보내고 레이어 마지막 row에서 finish를
순서대로 회수합니다. host가 row마다 응답까지 기다리는 동기 RPC는 아닙니다.
다만 helperd는 kCompute 하나의 계산·응답을 마친 뒤 다음 메시지를 처리합니다.
row0 helper 계산은 이미 row1/2의 main fetch와 겹칩니다. 모든 행을 모은 뒤
한 번 보내면 이 시작 시각이 늦어질 수 있습니다.

`GROUP_EXPERTS`는 여러 sequence의 RowsState 경로이고 현재 single-sequence
MTP에 연결되지 않습니다. `win_cpu_merged()`도 main RAM blob 주소와 로컬
CPU pool을 가정하므로 helper 구성에서 조건 하나만 풀어 사용하면 안 됩니다.
기존 `kRows/kTokRows`는 FP16 입출력이며 CPU를 key 순서로 더하므로, 현재
FP32 decode의 정밀도와 합산 순서를 그대로 보존하는 대체가 아닙니다.

작은 출발점은 FP32 decode-batch envelope와
`GlmExpertHelper::compute_tok()`의 재사용입니다. 기존 quantize→gate/up→down과
token별 expert 순서는 유지하고 upload/download/sync를 묶을 수 있습니다.
`compute_rows()`의 무가중 expert별 결과를 host에서 다시 더하거나, prefill용
`compute_tok_mmq()`를 그대로 쓰면 다른 반올림·합산 경로가 섞입니다.

CPU 쪽은 `native_*_rows_split()`의 exact gate/fallback을 사용하고 결과를
원래 `(token, expert 순번)`에 저장한 뒤 기존 순서로 합산하는 후보입니다.
현재 CPU gate/up→GPU→CPU down의 겹침도 보존해야 합니다. route IDs·weights·
ownership·copy 결정을 그대로 재생하는 검증을 먼저 하고, batch 수집으로
늦어진 시작 시간과 절감한 계산·동기화 시간을 측정해야 합니다. 마지막 row에
helper 일이 없는 경우에도 batch를 닫고 disk miss는 즉시 처리해야 합니다.

소스: `glm_fast_path.cu::fast_service()`(NET_WIN_HOLD),
`glm_window.cu::fast_moe_window()`(row별 route/fetch),
`net_helper.hpp`(kRows/kTokRows), `helperd.cpp`(각 요청 처리와 CPU 합산),
`remote_experts.cpp::compute_tok()`, `native_expert.cpp`(multi-row exact 검사).
아직 구현·GPU parity·성능 검증을 실행한 결과는 아닙니다.
