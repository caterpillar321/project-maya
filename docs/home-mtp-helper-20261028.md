# 집 GLM: MTP 길이 비교와 helper 대기열 계측 — 2026-10-28

## 결과와 결정

추가 구매·가중치 양자화 변경 없이, 교체한 코하네 RAM 5200 설정에서 측정했습니다.
현재 최선은 채팅의 fixed1 **38.75 tok/s**, 코드의 fixed2+fetch reuse **35.20 tok/s**입니다.
50 tok/s는 달성하지 못했습니다. 각 최선에서 토큰당 시간을 약22.5%/29.6% 더 줄여야 합니다.

이미 도착한 helper 요청을 묶어 같은 CPU expert를 공유하는 후보는 실제 재사용량이
**0.146%**로 작았습니다. 따라서 microbatch 실행 경로는 구현하지 않았습니다.
기존 adaptive MTP도 이번 조건에서는 최선의 고정 설정을 따라잡지 못해 기본값으로
채택하지 않았습니다. 모든 실험 후 기존 운영 binary/config/helper로 복원했습니다.

## 1. 동일한 최신 engine으로 MTP OFF / loaded0 / fixed1 / fixed2 비교

Claude의 원래 `qshort.py`를 수정하지 않고 조건마다 두 번 실행했습니다.
채팅4개×200, 코드4개×200, greedy120 한 요청이 한 회입니다.
각 조건에서 기존 16,524-token context 하네스의 첫 입력과 prefix 반복도 실행했습니다.

| 조건 | 채팅 1/2회 | 채팅 평균 | 코드 1/2회 | 코드 평균 |
|---|---:|---:|---:|---:|
| MTP 완전히 OFF | 36.4 / 36.4 | 36.40 | 31.6 / 31.8 | 31.70 |
| NextN 로드, fixed0 | 35.7 / 35.8 | 35.75 | 30.9 / 31.1 | 31.00 |
| fixed1, private scratch | 38.8 / 38.7 | **38.75** | 34.0 / 33.4 | 33.70 |
| fixed2, private scratch + fetch reuse | 36.7 / 37.2 | 36.95 | 34.5 / 35.9 | **35.20** |

plain 대비 fixed1 채팅은+6.46%, fixed2 코드는+11.04%입니다.
loaded0의 손실은 채팅−1.79%, 코드−2.21%입니다. loaded0에는 NextN의 캐시 유지와
private scratch 예약도 있으므로 이를 순수한 가중치 메모리 점유 비용이라고 부르지 않습니다.

| 시작 배치 | plain | loaded0 / fixed1 / fixed2 |
|---|---:|---:|
| 메인 GPU expert pool | 1,519 slots / 19.11 GiB | 1,466 / 18.46 GiB |
| 메인 RAM tier | 7,312 slots / 91.98 GiB | 7,302 / 91.95 GiB |
| 원격 cold expert | 468 / 5.83 GiB | 788 / 9.88 GiB |
| 원격 optional RAM copy | 3,868 / 51.71 decimal GB | 3,543 / 47.37 decimal GB |

NextN을 올리는 것이 expert 배치에도 영향을 줍니다. 초안 생성 시간을 절약하는 것만으로
loaded0의 손실이 모두 없어지는 구조가 아닙니다.

| 16.5K context | 첫 프리필 s | 첫 decode tok/s | 반복 decode tok/s | 첫/반복 출력 토큰 |
|---|---:|---:|---:|---:|
| plain | 12.0534 | 31.2 | 32.5 | 225 / 225 |
| loaded0 | 12.7428 | 30.0 | 31.7 | 219 / 220 |
| fixed1 | 12.7290 | 33.2 | 34.3 | 225 / 225 |
| fixed2 + reuse | 12.7625 | 33.2 | 34.7 | 219 / 219 |

같은 입력이지만 출력 토큰열까지 고정한 replay가 아닙니다. 길이가 다른 출력의 전체
wall time을 곧바로 같은 작업의 가속률로 해석하지 않습니다.

### 디코드 라운드에서 시간을 쓰는 곳

각 그룹8요청의 drafting round 수로 가중한 host 시간입니다. 겹쳐 실행되는 GPU/CPU
stage를 독립 비용처럼 모두 합한 값이 아닙니다.

| 조건·작업 | 초안 채택률 | draft ms/round | verify ms/round | commit/cache enqueue ms/round |
|---|---:|---:|---:|---:|
| fixed1 채팅 | 85.75% | 1.096 | 47.084 | 0.024 |
| fixed1 코드 | 89.03% | 1.055 | 55.281 | 0.023 |
| fixed2 채팅 | 65.41% | 2.103 | 60.404 | 0.274 |
| fixed2 코드 | 81.01% | 2.063 | 72.446 | 0.173 |

scratch replay는0이었습니다. 이제 주된 비용은 정상적인 검증 창의 실행입니다.
초안 생성 약1~2ms를 더 줄이는 일보다 verify의 가중치 공급·expert 계산·helper 합류를
개선하는 일이 50 tok/s까지의 차이에 더 큰 영향을 줍니다.

## 2. helper의 이미 도착한 FP32 요청을 묶을 가치가 있는가

`HELPERD_QUEUE_PROBE=1`을 기본 OFF인 진단 옵션으로 추가했습니다.

- 현재 `kCompute`의 FP32 payload를 받은 직후, 다음 최대2개 frame을
  `MSG_PEEK | MSG_DONTWAIT`로 봅니다. 기다리거나 소비하거나 응답 순서를 바꾸지 않습니다.
- 완전한 연속 `kCompute` frame만 보고 타입·down offset·limit가 같은 경우만 묶음 후보로 셉니다.
  부분 frame, 다른 op, 비정상 expert 수·중복 key 등에서 멈춥니다.
- 후보로 센 다음 요청은 다시 후보의 시작으로 세지 않습니다. 따라서 같은 요청을
  겹치는 snapshot 여러 개로 중복 계산하지 않습니다.
- 같은 global expert key가 RAM tier에 실제로 있을 때만 CPU 재사용으로 셉니다.
  바이트는 gate/up/down의 논리 가중치 크기이며 DRAM traffic 측정값이 아닙니다.
- 64개 FP32 요청당 하나를 골라 pool 작업 함수 안의 **현재 스레드 CPU 시간**을 기록합니다.
  FP32 wire, 연산, expert 합산 순서, helper 배치는 바꾸지 않습니다.

원래18800 helper는 유지하고 CPU/RAM helper18801만 같은 진단 binary로 OFF/ON 비교했습니다.
각 조건 qshort2회이며 긴 context는 이 비교에 넣지 않았습니다.

| helper 진단 | 채팅 평균 | 코드 평균 | 메인 CPU % | 코하네 helper CPU % |
|---|---:|---:|---:|---:|
| OFF | 37.50 | 33.90 | 177.90 | 459.45 |
| ON | 37.55 | 33.90 | 178.48 | 462.16 |

CPU는 한 코어=100%, 프리필 포함 qshort 전 구간입니다. 이 표는 MTP 비교에서 사용한
원래 helper와 다른 binary를 쓰므로 위 표와 합쳐 단일 flag의 효과처럼 계산하지 않습니다.
OFF/ON18요청 모두 route/payload/reuse/MTP 계측값이 같았고 회차별 greedy hash도 같았습니다.
모든 출력 float의 bitwise 검증을 수행했다는 뜻은 아닙니다.

| 대기열 결과, ON2회 합계 | 값 |
|---|---:|
| FP32 요청 | 107,400 |
| 1요청 후보 묶음 | 94,497 |
| 2요청 후보 묶음 | 6,450 |
| 3요청 후보 묶음 | 1 |
| 2~3요청 후보에 들어간 요청 비율 | 12.014% |
| CPU expert 사용 횟수 / 묶음 안의 중복 | 144,243 / **211** |
| CPU expert 중복 비율 | **0.14628%** |
| 논리 CPU 가중치 / 중복 부분 | 1,929,959,964,672 / 2,820,931,584 bytes |
| 가중치 중복 비율 | **0.14617%** |
| snapshot 계측 시간 합 | 36.070 ms |

대부분의 묶음 후보가 같은 CPU expert를 공유하지 않습니다. 이는 **이미 도착한 요청만
즉시 묶는 정책**의 결과입니다. 기다려서 모으는 정책이나 모든 검증 창에서의 전체 expert
반복률까지 측정한 것은 아닙니다. 그럼에도 이번에 제안한 가중치 재사용형 microbatch를
우선 구현할 근거로는 부족하므로 실행 경로를 추가하지 않았습니다.

### CPU 부하 해석과 제외한 계측값

1,678개 표본 요청의 service wall time 합은561.934ms, pool 작업 스레드 CPU 시간
합은2,791.343ms였습니다. 표본 service 구간에 평균 약4.97개 코어분의 pool 연산이
있었습니다. 여러 스레드의 CPU 시간을 더하므로 wall time보다 커지는 것은 정상입니다.
요청을 실제 처리할 때 CPU가 거의 놀고 있다는 해석은 맞지 않습니다.

이 표본에는 socket 대기 중 worker의 busy spin, GPU 대기, 직렬 양자화/합산 등은
pool 작업 시간으로 포함되지 않습니다. 따라서 전체 프로세스 CPU 약460%를 전부
유용한 expert 연산으로 보거나, 그 차이를 전부 특정 종류의 spin으로 단정하지 않습니다.

초기 진단의 sub-ms `CLOCK_PROCESS_CPUTIME_ID` 차이를 합한 항목은 561.934ms 구간에
17,047.535 CPU-ms를 보고해 이 CPU의 병렬도와 맞지 않았습니다. **그 항목은 폐기했습니다.**
원시 로그에는 남겼지만 결론에 사용하지 않고 최종 코드에서도 제거했습니다.
원인을 특정하지 않았습니다. 최종 binary는 이 잘못된 항목만 제거해 빌드 검사했으며,
별도로 홈에 배포·재측정하지 않았습니다. 대기열/현재 스레드 계측 코드는 같습니다.

## 3. 기존 adaptive MTP 제한 실험

`glm_mtp.cu`에는 이미 길이별 측정 시간과 조건부 채택률로 예상 tokens/ms를 비교하는
제어기가 있습니다. 새 제어기를 만들지 않고 `MTP_DRAFT=2`, `MTP_START=1`,
`MTP_ADAPT=1`, private3/reuseON으로 시험했습니다. 원래 두 helper를 복원한 뒤 실행했습니다.

| 작업 | 1회 | 2회 | 평균 |
|---|---:|---:|---:|
| 채팅 | 37.3 | 36.9 | 37.10 |
| 코드 | 34.8 | 33.1 | 33.95 |

긴 입력은 프리필12.7673s, decode첫33.7/반복34.9tok/s였습니다.
하지만 두 번째 qshort 회차에서 채팅은 draft2를325라운드, draft1을25라운드 선택했고,
코드는 draft0을108라운드, draft1을215라운드, draft2를114라운드 선택했습니다.
고정 조건에서 빨랐던 방향과 달랐으며, 기본 설정으로 승격하지 않았습니다.

길이별 시간 EMA와 재탐색 간격은 요청 사이에 남아 있고, 한 길이에서 관측한 비용이
다른 작업에서도 현재 비용을 잘 대표하는지는 아직 검증하지 않았습니다. 이것은 후속
가설이며 지금의 자료만으로 원인을 확정하지 않습니다. 다음에는 선택 시점의 길이별
시간·예상 토큰수·점수·마지막 관측 시점을 기록해 왜 그 선택을 했는지 확인하는 것이 맞습니다.

## 4. 다음 구현 우선순위

1. **fixed1의 T2 검증 창에도 fetch reuse를 확장할 수 있는지 검증.** 현재 구현은
   fullT3만 허용합니다. T2도 private scratch의 lifetime·순서가 안전한지 단위 검사하고,
   같은 binary의 OFF/ON으로 비교해야 합니다. 이번에는 확장하지 않았습니다.
2. **adaptive의 오래된 비용 추정과 draft0 선택을 계측.** 전환 비용과 탐색 비용도 함께
   보며, 채택률만으로 길이를 고르거나 단순히 자동 제어기를 다시 쓰지 않습니다.
3. **265K는 교체 실험으로 판단.** 이미 가진 CPU/보드라는 조건을 유지합니다.
   코하네6스레드가 실제 계산 중에도 비어 있다는 근거는 없으므로 추가 코어 수만으로
   가속률을 예측하지 않습니다. 같은 RAM5200에서 P코어6/8개, 그 뒤 P+E를 비교합니다.

## 검증·재현·제약

- engine SHA256: `6f6dc6c1f11a42bd96c8cfed0943a5e24373b76a8672abf80ff81a4f7e237856`.
- qshort SHA256: `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.
- 측정 helper SHA256: `4efc4d9e43d604d83fbbfeb9e4058d04d6cddd8de359c23b4c96744cc5346694`.
- 최종 helper 빌드 SHA256: `27405fc1ec75aa8ed72de2a257db5c7f41d2186ee8813a3247161322946e8fa1`.
- IQ4_XS, INT8 KV, context262144, prefill32768, RAM92GiB, helper54GiB/6스레드,
  copy-aware0.16ms와 기존 demand profile을 유지했습니다. B200 GPU는 사용하지 않았습니다.
- MTP 네 조건80요청, helper OFF/ON36요청, adaptive20요청 모두 route audit가 완결됐습니다.
  이 합계는 전체136요청이며 qshort의 모델 출력 내용을 별도로 모두 보존한 것은 아닙니다.
- `helper_queue_probe_test`는 부분 header/payload, FIFO 장벽, 잘못된 크기·key·형식,
  빈 socket의 비차단 동작과 peek 후 원본 바이트 보존을 검사합니다. CTest1/1 PASS.
- 로컬 기본 GCC11은 기존 helper의 `_Float16`을 컴파일하지 못해 helper translation unit만
  임시 디렉터리에 푼 GCC12.3으로 빌드했습니다. 시스템 compiler 설치·교체는 하지 않았습니다.
  기존 portable AVX2/런타임 선택 CPU 라이브러리와 CUDA sm120 라이브러리를 연결했습니다.
- 순차 실행·두 회 평균입니다. qshort의 채팅/코드는 seed를 명시하지 않으며, 같은 조건의
  회차 간 greedy hash도 일부 달랐습니다. 무작위 순서/열조건 통제나 고정 토큰 replay가 아닙니다.
- 사용자께서 다른 에이전트의 SPD 조회 영향을 질문하셨습니다. 16:35:47에 두 PC에서
  지속 실행 중인 SPD 조회/메모리 테스트를 찾지 못했습니다. 읽기만 했다면 RAM 타이밍을
  바꾸는 작업이 아닙니다. SPD는 별도 NVRAM을 I2C 장치의 EEPROM 속성으로 읽습니다
  ([Linux 문서](https://docs.kernel.org/hwmon/spd5118.html)). 과거 조회 시각은 확인하지
  않았으므로 측정 순간과의 겹침까지 배제하지 않습니다.

상세 수치: [측정 JSON](home-mtp-helper-measurements-20261028.json).
로컬 raw logs·plan·supervisor·분석 스크립트·두 helper binary는
`/home/caterpillar/Agent/maya-mtp-helper-20261028`에 있습니다.
메인 run은 `~/fn/codex-mtp-matrix-20261028`, `~/fn/codex-helper-probe-20261028`,
`~/fn/codex-mtp-adaptive-20261028`입니다. 운영은 기존 `start-maya-kohane5200.sh`와
`maya-copy-profile-cc80982`로 복원했으며 최적화·계측 flag를 기본값으로 올리지 않았습니다.
