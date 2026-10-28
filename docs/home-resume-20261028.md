# 집 GLM 작업 재개 메모 — 2026-10-28 후속 실험 완료

## 최신 상태 — 16:56:45 기존 운영 복원 완료

사용자의 `ㄱㄱㄱㄱㄱㄱ` 승인으로 MTP 네 조건, helper 계측 OFF/ON,
기존 adaptive MTP(cap2/start1)까지136요청 측정을 완료했습니다.
세 supervisor 모두 `complete`, `restored=true`, error없음입니다.
운영 `/health` 정상, 메인 engine PID284053/PGID284050,
코하네 helper18800 PID5151 / 18801 PID11615입니다. 두 PC 모두 켜져 있습니다.
아래 15:56의 PID는 과거 기록이며, 실제 제어 전 /proc argv를 다시 확인합니다.

- MTP 네 조건: 메인 `~/fn/codex-mtp-matrix-20261028`, supervisor261174 완료.
  채팅/코드 평균: plain36.40/31.70, loaded0 35.75/31.00,
  fixed1 38.75/33.70, fixed2+reuse36.95/35.20tok/s.
- helper 계측 run `~/fn/codex-helper-probe-20261028` 완료.
  18801만 진단 binary로 교체했다가 원래 것으로 복원했습니다. 현재 상태는 코하네
  `~/fn-bin/kohane5200-helper-state.json`에서 읽어 실제 argv를 확인합니다.
  새 binary `maya-helperd-queue-probe-20261028`, SHA256
  `4efc4d9e43d604d83fbbfeb9e4058d04d6cddd8de359c23b4c96744cc5346694`.
  현재 실행 중인 것은 원래 helper `maya-helperd-20261026`입니다.
  107,400요청에서 즉시 묶을 후보의 CPU expert중복은211/144,243회(0.146%).
  microbatch 실행 경로는 구현하지 않았습니다. 진단 옵션만 기본OFF로 추가했습니다.
- adaptive `~/fn/codex-mtp-adaptive-20261028` 완료. 채팅37.10/코드33.95tok/s.
  두 번째 채팅은 draft2를 주로, 코드는 draft0/1을 자주 골라 최적 고정값에 못 미쳤습니다.
  새로운 제어기는 작성하지 않았고 기본값으로 승격하지 않았습니다.
- 최종 진단 코드에서는 비현실적인 sub-ms 프로세스 CPU-clock 항목만 제거했습니다.
  최종 binary SHA `27405fc1ec75aa8ed72de2a257db5c7f41d2186ee8813a3247161322946e8fa1`,
  빌드 검사만 했고 추가 배포하지 않았습니다. 원시 측정 binary와 최종 binary는 로컬에 보존했습니다.
- 로컬 원본·실행 스크립트·분석:
  `/home/caterpillar/Agent/maya-mtp-helper-20261028`.
- SPD 조회가 벤치에 영향을 줬는지 사용자께서 질문하셨습니다. 16:35:47에
  두 호스트에서 지속 실행 중인 SPD 조회/메모리 테스트 후보를 찾지 못했습니다.
  다른 에이전트의 과거 실행 시각은 확인하지 않았으므로 겹침을 배제하지 않습니다.
- 다음 후보: fixed1의 T2 fetch reuse 확장 검증, adaptive의 길이별 시간 추정·재탐색 계측.
  265K+Z890은 이미 가진 부품이지만 아직 실제 교체하지 않았습니다.

상세: [후속 측정 보고서](home-mtp-helper-20261028.md),
[측정 JSON](home-mtp-helper-measurements-20261028.json).

## 15:56 완료 시점 기록

## 현재 상태

사용자께서 코하네 RAM 교체·복구 후 작업 재개를 요청하셨습니다.
중단됐던 WIN_FETCH_REUSE OFF/ON 측정을 새 폴더에서 완료했고, **15:56:29에
기존 운영 서버의 /health 정상 및 복원을 확인했습니다. 두 PC 모두 켜져 있습니다.**
새 실험 flag를 운영 기본값으로 승격하지 않았습니다. 양자화도 그대로입니다.

- 저장소 `/home/caterpillar/Agent/project-maya`, branch `codex/ram-backing-probe`.
- 운영 시작: 메인컴 `MAYA_NET=1 ~/fn/start-maya-kohane5200.sh`.
- 운영 binary: `~/fn/bin-test/maya-copy-profile-cc80982`, config `~/fn/maya-config/iq4-kohane5200.json`.
- 코하네 helper: 18800 PID5151, 18801 PID5164 (재부팅 시 바뀜).
  18801은 RAM54GiB/CPU6스레드입니다. **옛 86GiB wrapper를 쓰지 않습니다.**
- 현재 helper PID는 `~/fn-bin/kohane5200-helper-state.json`에서 읽고 /proc argv를 검증합니다.
  `~/codex-helper-threads-20261027/active.json`에는 재부팅 전 PID가 남아 있습니다.

## 완료한 개선과 결과

| 실험 | 결과 |
|---|---|
| NC3 Q8_0/Q4_K/Q6_K, `d20b7dd` | 이전 측정 약2% 개선. 집5090 bitwise60개 검사 통과 |
| 작은 RAM backing, `d02d5fa` | D2H8~10% 감소, 유용한 속도 개선 입증 못함 |
| RAM class 재배정, `cf9aaad` | D2H67~70% 감소·decode소폭 개선, prefill악화로 기본 승격하지 않음 |
| WIN_FETCH_REUSE, `6271113` | 오늘 OFF/ON 완료. 채팅35.95→36.85, 코드34.15→35.15tok/s |

오늘 같은 binary, fixed2 MTP, stagger0, private3, NC3=1, RAM92GiB에서
WIN_FETCH_REUSE만 바꿨습니다. 각 조건 qshort2회 + 16.5K context첫/반복입니다.

- 채팅 H2D expert payload6.78%, 코드8.47% 감소, D2H변화0.
- 20요청 모두 rounds/출력수/채택수/route수/재실행수 일치. scratch replay0.
- 모든 요청에서 OFF fetch = ON fetch + reuse (count와bytes 모두).
- greedy hash는 OFF/ON 각 회차 일치. 전체 출력의 bitwise 증명은 아닙니다.
- CPU 부하율은 main약178%, helper약462%로 거의 같았습니다(한코어100%,prefill포함).
- 긴 입력 프리필12.7960→12.7553초, decode32.1→33.2, 반복33.5→34.6tok/s.
- 시작 배치 수량도 일치: pool1466/RAM7302slots, cold788개, optionalcopies3543개.
- 순차 두 회 실험이며 무작위·열조건 통제 실험은 아닙니다. 50tok/s는 미달입니다.

상세: [측정 보고서](home-window-fetch-reuse-20261028.md),
[비교 JSON](home-window-fetch-reuse-measurements-20261028.json).

## 다음 후보

1. 현재 개선은 fixed2 privateT3 안에서 유효합니다. 새 plain/MTP1 기준과 직접
   비교해 운영 후보를 고르는 작업은 아직 하지 않았습니다.
2. helper의 이미 도착한 FP32 kCompute 요청1~3행을 묶는 microbatch를 검토합니다.
   먼저 묶일 수 있는 실제 요청 비율·같은 expert 재사용량을 계측해야 합니다.
   timeout으로 기다리게 하거나 FP16 wire format으로 바꾸는 것은 별도 변경입니다.
   행별CPU expert원래순서→CPUsubtotal→GPUsubtotal한번합산을 보존하고,
   CPU GU를 먼저 시작해 GPU와 겹쳐 실행해야 합니다. 아직 구현하지 않았습니다.
3. 사용자께 **265K + Z890 AORUS ELITE 보드가 이미 있습니다.** 코하네의 9600을
   이것으로 바꾸는 성능 질문을 하셨지만, 실제 교체는 아직 하지 않았습니다.
   교체 비교 시 RAM5200/GPU/모델/하네스부터 고정하고 P코어6/8개와 P+E를
   따로 측정합니다. 9600의6→12스레드 실험은 SMT이며 추가 물리코어의 효과와 다릅니다.
   265K에 AVX-512는 없지만 현재IQ4 단일행의 GGML 경로도 있으므로 그것만으로
   느리다고 단정하지 않습니다. 헬퍼 binary의 CPU 호환성도 확인해야 합니다.

## 실행·원본 경로

- 최신 test binary: 메인 `~/fn/bin-test/maya-window-reuse-20261028`.
  SHA256 `6f6dc6c1f11a42bd96c8cfed0943a5e24373b76a8672abf80ff81a4f7e237856`.
- 오늘 wrapper: 메인 `~/fn/bench_fetch_reuse_resume.py`.
  `~/fn/fetch-reuse-plan.json`은 OFF/ON 두 조건, 각2회+context입니다.
  wrapper는 정상 운영 서버에서 시작하고 끝나면 기존 운영을 복원합니다.
- 오늘 메인 로그: `~/fn/codex-fetch-reuse-resume-20261028`.
- 오늘 로컬 원본/환경/분석/계측: `/home/caterpillar/Agent/maya-fetch-reuse-resume-20261028`.
- 이전 원본: `/home/caterpillar/Agent/maya-backing-probe-20261027`.
- Claude `~/fn/qshort.py`는 변경하지 않았습니다.
  SHA256 `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.
- 재실험은 기존 RUN 폴더를 덮어쓰지 않고 새 폴더를 지정합니다.

## 기기와 RAM

- 메인: `ssh sekai`,192.168.0.4,사용자caterpillar,hostname `Caterpillar-Desktop`.
  9950X3D2, RAM128GB/4800 MT/s, 5090(백석역)32GB + PRO4000(천룡인)24GB.
  ADATA32GB두장+KLEVV32GB두장, 모두SPD DRAM제조사SK Hynix입니다.
- 코하네: `ssh b650`,192.168.0.76,사용자user,hostname `sekai-pc`.
  native Debian, Ryzen9600, 4070TiSUPER(쪼꼬미)16GB + 5060Ti(코하네)16GB.
  교체 RAM은TeamGroup UD5-4800 32GB두장/각dualrank/SK Hynix,
  SPD steppingFF여서 A/M다이 구분은 못 했고, SMBIOS설정5200MT/s확인했습니다.
- 교체후 stressapptest48GiB/12스레드/90초 PASS, 오류0.
  테스트 로그는코하네 `/tmp/codex-ram-check-20261028.log`이며 오늘 로컬 환경기록에도
  결과를 저장했습니다. 장기 안정성을 보장하는 검사는 아닙니다.
- B200 GPU/idle keeper는 집 실험에서 사용하거나 중단하지 않았습니다.

## 이전 중단 기록

00:26 KST에 사용자 요청으로 저장·종료했습니다. 당시 메인 전원은 종료됐으나
코하네 전원은 sudo인증 문제로 끄지 못했고 helper만 중단했습니다.
당시 WIN_FETCH_REUSE는 OFF두회만 완료했고ON은 시작하지 않았습니다.
그 미완료 상태는 오늘의 새 OFF/ON 측정으로 해소됐습니다.
