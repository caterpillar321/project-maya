# 두 호스트 MTP 엔진: 구현 결정과 진행 기준

2026-10-29. 기존 엔진 코드 `c5e0b83`의 검토, pack 전문가 크기 계산, 기존 T2 trace 재집계,
새 TCP 전송 측정을 합친 설계입니다. 원격 trunk/native split MTP 구현은 아직 없습니다.
모델 서버·GPU 작업·방화벽 설정은 변경하지 않았습니다. 분석은 로컬 CPU로 수행했습니다.

## 결정 요약

사용자 제안대로 보조컴은 4070 Ti SUPER trunk+5060 Ti 상주 전문가,
메인컴은 5090 trunk/head/NextN+PRO 4000 상주 전문가로 구성하는 후보를 유지합니다.
고정 전문가 배치는 모델의 top-8 routing을 바꾸지 않습니다. 양자화·하드웨어·클럭을 유지합니다.

**첫 구현의 분할 후보는 cut16/cut18로 넓힙니다.** 이전 cut10~11은 helper RAM 중복을
제거한다는 가정에서의 용량 출발점이었습니다. 기존 backing을 유지하는 첫 구현에서는
cut16~18 쪽이 양쪽 RAM에 담길 여지가 더 있습니다. 실제 최적 경계와 성능은 미측정입니다.

Native T2/T3 전체 묶음을 유지한 직렬 두-stage 실행을 정확성 기준으로 만듭니다.
MTP의 초안·채택 판단과 sampler는 메인에 두고, 각 stage가 자기 상태의 검증·복원을 담당합니다.
행별 pipeline과 T3의 2+1 microbatch는 실제 stage 시간과 batching 이득을 확인한 후 선택합니다.

## 새로 얻은 수치

### TCP 경계 전송

2026-10-29 16:18 KST에 실제 메인/보조컴 사이에서 측정했습니다. 연결은 기존 helperd와
동일하게 보조컴이 메인으로 dial하며, 방화벽에 이미 허용돼 있고 사용하지 않던 18802를
사용했습니다. 임의 포트는 양방향에서 EHOSTUNREACH로 거부됐습니다. 방화벽은 수정하지 않았습니다.

16-byte 요청 → 보조컴의 payload → 메인의 16-byte commit → 8-byte ACK를 한 persistent
TCP_NODELAY 연결에서 반복했습니다. 크기별 warm-up 10회 뒤 100회(1 MiB는 30회)를 측정했습니다.

| payload | 요청부터 payload 완료 p50 | commit ACK 포함 p50 | commit ACK 포함 p95 |
|---|---:|---:|---:|
| 64 KiB, T1 크기 | 0.303 ms | 0.363 ms | 0.383 ms |
| 128 KiB, T2 크기 | 0.530 ms | 0.596 ms | 0.621 ms |
| 192 KiB, T3 크기 | 0.751 ms | 0.819 ms | 0.840 ms |
| 1 MiB | 3.668 ms | 3.756 ms | 3.776 ms |

이는 **TCP 모양만 재현한 테스트**입니다. 실제 window protocol, GPU 복사·계산·KDA commit,
동시 추론 부하는 포함하지 않았습니다. 실제 엔진 통신 비용의 대체값으로 확정하지 않습니다.
디코드 경계 전송이 수십 ms를 필연적으로 차지한다는 우려를 줄이는 근거입니다.
테스트 프로세스의 정상 종료와 18802 listener 해제를 확인했습니다.
원자료: [TCP 결과](home-dual-host-link-20261029.json).

### 층별 무게와 RAM

Native expert 원자료에서 trunk 152.15625 GiB를 합산했습니다. 기존 형식의 NextN experts
3.796875 GiB를 tail에 추가했습니다. 아래는 **상주 전문가에 실제로 쓸 수 있다고 가정한**
4070 10.5 GiB / 5060 14.5 GiB / 5090 20.5 GiB / PRO 21.5 GiB와 RAM 54/92 GiB의 용량 모형입니다.
GPU 숫자는 새 설정의 실측 할당량이 아닙니다. 작업공간·scratch·spare를 제외한 유일 가중치 용량이라는
가정이며, 경계별 실제 dense/KV/window 비용과 클래스별 단편화는 별도로 검증해야 합니다.

| 경계 [0,k) | 보조 experts | 메인 experts+NextN | helper RAM 중복 유지 시 보조 RAM 최소 | 같은 조건의 메인 RAM 최소 |
|---|---:|---:|---:|---:|
| 10 | 26.65 GiB | 129.30 GiB | 16.15 GiB | 108.80 GiB |
| 11 | 30.23 GiB | 125.72 GiB | 19.73 GiB | 105.22 GiB |
| 16 | 48.16 GiB | 107.79 GiB | 37.66 GiB | 87.29 GiB |
| 18 | 55.34 GiB | 100.62 GiB | 44.84 GiB | 80.12 GiB |
| 20 | 62.51 GiB | 93.45 GiB | 52.01 GiB | 72.95 GiB |

cut11의 메인 RAM은 PRO 복사본을 제거하는 이상적 배치에서는 약 83.72 GiB까지 낮아지지만,
이를 유지하면 약 105.22 GiB입니다. 따라서 dedup 구현 없이 cut11을 그대로 시작하면 안 됩니다.
cut18은 양쪽 RAM 예산에 약 9.16/11.88 GiB의 합산 여유가 생기는 후보입니다.
cut20은 보조 RAM 여유가 약 1.99 GiB라 실제 할당 오차에 더 민감합니다.
이 여유는 전문가 예산 내부의 값이며 시스템 전체의 MemAvailable을 뜻하지 않습니다.

첫 구현은 기존 local helper의 RAM backing을 보존해 prefill 경로를 재사용하고, 실제 메모리
manifest를 검증합니다. 이후 중복 제거가 필요하면 prefill 공급 경로와 함께 변경합니다.
현재 prefill의 `disk_only()`는 일반 local helper VRAM을 독립적인 가중치 공급처로 보지 않으므로,
RAM 원본만 해제하면 prefill SSD 읽기가 늘 수 있습니다 (`glm_prefill.cu:2306`).
local helper에는 `compute_tok_mmq()`가 있어 향후 prefill 전문가 실행을 맡길 기반은 있지만,
이는 현재 trunk prefill에 연결되어 있다는 뜻이 아닙니다.

### 앞부분 이동만으로는 부족한 이유

기존 **단일 5090 trunk** T2 trace의 chat/code 각 28개 표본을 다시 집계했습니다.

| 경계 | 기존 대화 앞층 / 나머지 span | 기존 코드 앞층 / 나머지 span |
|---|---:|---:|
| 11 | 8.69 / 41.19 ms | 7.94 / 48.10 ms |
| 16 | 13.11 / 36.77 ms | 12.43 / 43.61 ms |
| 18 | 14.83 / 35.05 ms | 14.16 / 41.87 ms |

이 값은 **4070/5090 새 배치의 실행 시간 예측이 아닙니다.** 기존 배치의 helper 대기와
fetch를 포함한 구간 분포입니다. 앞쪽의 짧은 구간만 이전하면 tail 병목이 남는다는 판단에 사용합니다.
tail의 hot experts를 PRO/5090에 어떻게 배치해 fetch를 줄일지도 설계의 일부여야 합니다.
보조컴 RAM/CPU를 front 전용으로 바꾸면 기존 tail CPU offload가 사라지므로, 이것도 비용에 포함합니다.

표본 fetch는 대화/코드 20.83/27.57 ms, 노출 helper join은 7.70/8.60 ms입니다.
join은 순수 LAN 시간이 아니며 전부 없어지는 항목도 아닙니다. 가중치 공급의 지연을 줄이는
것이 중요합니다. 표본 span과 전체 host-round verify는 분모가 다르므로 그대로 차감하지 않습니다.
계산 원자료: [용량과 기존 trace](home-dual-host-budget-20261029.json).

## 코드의 변경 경계

일반 expert RPC와 원격 trunk는 다른 계약이므로 기존 helperd에 토큰 몇 개를 추가해서
trunk로 취급하지 않습니다. 기존 CUDA 연산은 재사용하고 아래 제어 경계를 분리합니다.

| 구성 | 기존 기반 | 필요한 변경 |
|---|---|---|
| 층 범위 로더 | `Glm5Model::load_pack(...,l0,l1)`, `glm_model.cu:2857` | front/tail 역할, NextN/head/embedding 및 window 예약을 명시화 |
| 로컬 검증 stage | `fast_window_layers()`, `glm_window.cu:1160` | 준비·snapshot, 입력/출력 residual, 층 실행, commit을 stage API로 추출 |
| MTP coordinator | `decode_mtp()`, `glm_mtp.cu:538` | sampler/채택 로직은 유지하고 verify 실행체와 tail hidden 소유자를 분리 |
| 원격 front worker | 새 프로그램/transport 필요 | 모델을 한 번 적재하고 PREFILL/VERIFY/COMMIT/RESET 처리 |
| local expert helper | `GlmExpertHelper` | 각 호스트가 자기 GPU helper만 소유; 원래 양자화와 routing 유지 |
| prefix 재사용 | snapshot/checkpoint API | 양쪽 position·prefix identity를 묶어서 저장/복원 |

`mtp_ready()`의 guard만 제거하지 않습니다. `fast_setup()`은 현재 `l0==0`이고 NextN이
있는 단일 모델에만 native window/cache를 예약합니다 (`glm_fast_path.cu:1025`).
front는 NextN 없이 window/KDA backup이 필요하고, tail은 `l0>0`이어도 native draft/cache가
필요합니다. Native draft vocabulary 설정도 현재 단일 모델 조건에 묶여 있습니다.
전역 분할 수를 속여서 동작시키지 않고 stage의 기능과 local GPU 수를 구분해야 합니다.

batch reuse/route-ahead의 `n_parts==1` 조건도 새 stage의 private buffer·범위를 확인하는
조건으로 명시적으로 바꿉니다. 계산을 행별 `fast_layers()`로 바꿔서 검증을 통과시키는 것은
native T2/T3를 보존한 구현으로 인정하지 않습니다.

## 최소 원격 계약과 상태 수명

초기 버전은 한 요청, 한 창만 진행합니다. GPU는 PCI bus/UUID로 확인하고 각 호스트의
CUDA 번호를 전역 장치 번호로 혼동하지 않습니다. 모델 manifest, 형상, dtype, 층 범위,
문맥 한도와 프로토콜 버전을 연결 시 대조합니다.

1. `PREFILL`: front가 토큰 chunk를 처리해 FP32 residual을 보내고 tail이 같은 위치를 처리합니다.
2. `VERIFY(session,window,p0,T,tokens)`: 양쪽이 각자의 창 시작 상태를 저장하고 T행을 계산합니다.
   front residual을 tail이 받아야 하며, 단계 완료는 payload 수신과 CUDA 완료를 구분합니다.
3. 메인이 logits와 요청 sampler로 채택 수를 결정합니다. NextN은 tail의 실제 최종 hidden과
   embedding을 사용하며 초안 생성 때 front를 다시 실행하지 않습니다.
4. `COMMIT(window,keep)`: 양쪽이 동일한 position까지 상태를 확정하고 NextN cache를 갱신합니다.
   미채택 행의 입력/백업은 commit이 끝날 때까지 재사용하지 않습니다.
5. `RESET/CANCEL`: 진행 중 작업을 drain합니다. 연결이 끊긴 창은 양쪽 상태를 확인하지 않고
   재실행하거나 계속 생성하지 않습니다. 초기 버전은 요청 실패 후 재프리필로 회복합니다.

거절 시 KDA/convolution은 저장된 입력으로 각 호스트에서 재생하며 전체 FFN을 재계산하거나
KV 전체를 전송할 필요가 없습니다. 중복 commit의 같은 ID/keep은 재적용하지 않고,
다른 keep이나 오래된 generation은 거부합니다. 첫 구현은 commit 확인까지 명시적으로 기다리고,
이후에만 stream ordering을 입증하여 다음 요청에 piggyback할 수 있습니다.
sampler counter, EOS/출력 길이, 취소 시 실제 방출 토큰 수와 commit 위치도 함께 검증합니다.

## Prefill과 대화 재사용은 별도 성능 항목

FP32 residual을 모두 보내면 16,384토큰은 **1 GiB**, 32,768토큰은 **2 GiB**입니다.
2.5Gb/s에서 payload만의 선로 점유 시간은 약 **3.44/6.87초**입니다. 262,144토큰은 16 GiB,
약 54.98초입니다. chunk pipeline으로 일부를 숨길 수 있으므로 전부 추가 TTFT라고 단정하지 않습니다.
한 chunk 전체를 기다린 뒤 전송하는 첫 구현은 정확성 기준이며, 긴 prompt에는 chunk 중첩이 필요합니다.
양자화나 residual 정밀도를 낮춰 이 비용을 감추지 않습니다.

Prefill은 trunk expert pool을 빌리므로 반환 직후와 안정된 decode의 residency를 분리해 기록합니다.
또한 main checkpoint만 복원하면 front의 KDA가 다른 prefix에 남을 수 있습니다. 양쪽의 prefix
identity·position이 맞는 pair로 저장/복원해야 합니다. 이를 지원하기 전에는 cache capability를
비활성화하고 양쪽 모두 재프리필합니다. 같은 TTFT 비교에서도 양쪽 cache 조건을 맞춥니다.

## 단계별 진행과 기각 조건

1. **Stage API 추출과 로컬 검증.** 모델 arithmetic을 바꾸지 않고 부분 층 window/prefill을
   호출 가능하게 만듭니다. 같은 prefix의 reference residual과 logits를 비교합니다.
2. **원격 plain + full-window MTP 연결.** cut16/18부터 실제 VRAM/RAM class별 할당,
   helper 소유권, cold 전문가 공급처를 출력합니다. startup manifest가 예산을 넘으면 실패시킵니다.
   일반 분할 decode와 native T2/T3를 같은 초기 상태에서 비교합니다.
3. **채택·거절 계약 검증.** T2 첫 거절, T3 첫/둘째 거절·전부 채택, 다음 라운드의 hidden과
   KDA/DSA 상태, EOS, 취소, prefix 복원, 지연·중복 메시지를 확인합니다. 불일치는 최초 logits와
   경로 차이부터 조사하며 해시가 다르다는 이유만으로 수치 오차라고 넘기지 않습니다.
4. **시간 예산 판정.** 원래 qshort를 수정 없이 사용하고 별도 고정-prefix correctness 검사를
   수행합니다. A2/A3/B2/B3, boundary 전송, commit, draft, 실제 채택 토큰을 같은 창 ID로 기록합니다.
   서로 다른 GPU/호스트의 절대 timestamp를 빼지 않습니다. main wall time과 장치별 elapsed를 구분합니다.
5. **필요한 최적화 선택.** 전체 묶음과 행별/2+1 후보의 실측 비용을 비교합니다. 빈 GPU를
   채우는 것 자체를 목표로 하지 않고 출력 토큰당 시간을 기준으로 선택합니다.

과거 yield와 draft/commit 비용을 유지하는 50 tok/s용 verify 예산은 T2 대화 약 36.0 ms,
T2 코드 약 36.7 ms, T3 코드 약 50.1 ms입니다. 실제 새 통신/commit·채택률을 반영해 다시 계산합니다.
기존 stage 행별 시간이 a/b라면 이상적 pipeline도 `a+b+(T-1)max(a,b)`이므로, 20 ms짜리
stage 두 개를 만들었다고 단일 요청 50 tok/s가 되는 것은 아닙니다.

성공 판정은 기존 단일 trunk의 대화 39.4/코드 34.7 tok/s 및 별도 T3 코드 35.8을 넘고,
정확성과 긴 입력 TTFT를 함께 확인하는 것입니다. 초기 직렬 stage가 이를 못 넘더라도 원인별
시간을 얻는 목적은 있지만, 개선 가능한 예산 없이 후속 pipeline을 무조건 개발하지 않습니다.

## 재현

```bash
python3 tools/analyze_dual_host_budget.py \
  --experts /home/caterpillar/Agent/maya-analysis-20261027/native_experts.txt \
  --trace-analysis /home/caterpillar/Agent/maya-engine-50tps-20261029/trace-on-analysis.json \
  --output docs/home-dual-host-budget-20261029.json
```

TCP probe 스크립트와 원자료는 `/home/caterpillar/Agent/maya-dual-host-design-20261029/`에 있습니다.
결과 JSON에 스크립트 SHA256을 기록했습니다. 모델 프로세스를 중단하거나 기존 helper 연결을
재사용하지 않고 측정했으며, stage worker의 실제 GPU 실행 시간은 아직 없습니다.
