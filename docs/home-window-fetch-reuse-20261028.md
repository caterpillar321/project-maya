# 집 GLM: MTP 검증 창의 expert 재사용 측정 — 2026-10-28

## 결과

RAM 교체 후 새로 측정한 OFF/ON 비교에서 `MAYA_GLM_WIN_FETCH_REUSE=1`은
채팅 약 2.5%, 코드 약 2.9%의 디코드 개선을 보였습니다. 긴 입력의 디코드도
개선됐으며 프리필 시간은 거의 같았습니다. 50 tok/s에는 아직 도달하지 못했습니다.
운영 설정으로 자동 승격하지 않고, 측정 후 기존 운영 서버를 복원합니다.

| 항목 | OFF | ON |
|---|---:|---:|
| 채팅 tok/s, 1 / 2회 | 35.6 / 36.3 | 36.6 / 37.1 |
| 코드 tok/s, 1 / 2회 | 33.4 / 34.9 | 34.5 / 35.8 |
| 채팅 평균 tok/s | 35.95 | 36.85 (+2.50%) |
| 코드 평균 tok/s | 34.15 | 35.15 (+2.93%) |
| 16,524-token 첫 입력: API 프리필 | 12.7960 s | 12.7553 s |
| 같은 요청 디코드 | 32.1 tok/s | 33.2 tok/s |
| 긴 입력 prefix 반복 디코드 | 33.5 tok/s | 34.6 tok/s |
| qshort 전체 wall 평균, 프리필 포함 | 58.307 s | 57.005 s |

각 조건은 순차 OFF → ON, qshort 두 회와 긴 입력 첫/반복 두 요청입니다.
순서를 무작위화하거나 온도·클록을 통제한 실험은 아니므로 개선율의 정밀한
신뢰구간이나 다른 부하에서의 동일한 효과를 주장하지 않습니다.

## 무엇이 같고 무엇이 줄었는가

조건당 20개 요청의 **모든 요청별** 다음 계측값이 일치했습니다.

- MTP rounds, emitted tokens, accepted/proposed drafts.
- 검증 창의 windows/rows/attempts, T별 창 수, scratch replay 0.
- expected/observed/trunk/draft route 수, metadata_missing/service_pending 0.
- D2H, background DMA, host-resolution miss 및 prefetch payload.
- OFF의 RAM fetch count/bytes = ON의 RAM fetch count/bytes + window reuse count/bytes.

즉, 이번 비교에서는 계측한 처리량과 채택률이 달라져서 생긴 개선이 아닙니다.
다만 route **수**가 같다는 것이 모든 expert ID와 전체 출력을 비교했다는 뜻은 아닙니다.
greedy 출력 해시는 각 회차의 OFF/ON이 같았고, 1회와 2회의 해시는 서로 달랐습니다.
이는 종단 간 수치 동등성의 완전한 증명이 아닙니다. 이전 GPU 단위 검사에서는
route/fetch 경계 조건과 세 가중치 형식 각각 1,536개 float의 bitwise 비교를 통과했습니다.

| 디코드 계측, 조건당 합계 | 채팅 8요청 | 코드 8요청 | 전체 20요청 |
|---|---:|---:|---:|
| 회피한 H2D expert 복사 횟수 | 5,619 | 8,727 | 18,305 |
| 회피한 expert payload, bytes | 75,122,343,936 | 116,674,265,088 | 244,725,841,920 |
| mapped fetch + miss 감소 | 6.78% | 8.47% | 8.13% |
| D2H 변화 | 0% | 0% | 0% |
| MTP 채택률, OFF = ON | 65.41% | 81.01% | 73.88% |

이는 논리 expert payload/전송 요청 계측입니다. 전체 PCIe 트래픽이나 완료 시점,
노출된 전송 지연을 직접 측정한 값이 아닙니다. 같은 scratch pointer를 공유하면
기존 expert grouping도 같은 expert를 묶을 수 있으므로 이 실험은 **복사 회피와
그룹 연산 변화가 결합된 효과**를 측정합니다.

| 요청별 창 수로 가중한 시간 | OFF | ON |
|---|---:|---:|
| 채팅 verify window, host ms | 61.808 | 60.482 |
| 코드 verify window, host ms | 74.687 | 72.545 |
| 채팅 drafting round, host ms | 64.421 | 62.862 |
| 코드 drafting round, host ms | 77.008 | 74.784 |

drafting round는 draft + verify + commit/cache enqueue의 host 계측 합입니다.
출력 첫 토큰은 decode loop 전에 샘플링되므로 emitted/rounds를 그대로 쓰지 않습니다.
이번 로그는 모든 그룹에서 emitted = 요청 수 + rounds + accepted drafts를 만족합니다.
상세 JSON은 전체 loop 및 drafting round의 토큰 수를 별도로 기록합니다.

## CPU 부하와 다음 병목

qshort 전체 구간의 한 코어 = 100% 기준 평균입니다. 프리필도 포함되어 있습니다.

| 프로세스 | OFF | ON |
|---|---:|---:|
| 메인 engine CPU | 178.67% | 178.21% |
| 코하네 CPU/RAM helper | 462.08% | 461.68% |

**CPU 부하율이 의미 있게 줄었다고 볼 결과는 아닙니다.** 같은 작업을 끝내는
시간은 조금 줄었습니다. ON 종료 시 optional RAM copies는 3,543/3,543 hash 일치입니다.
검증 창 helper 계측에는 held flush 63,124회가 남아 있습니다. 다음 후보는
이미 도착한 FP32 요청을 묶어 CPU expert 가중치를 재사용하는 microbatch입니다.
timeout을 추가해 요청을 기다리게 하기 전에, 실제 묶을 수 있는 요청 비율과
같은 expert의 반복 정도를 먼저 측정해야 합니다.

현재 ON 평균에서 50 tok/s까지는 토큰당 시간을 채팅 약 26%, 코드 약 30% 더
줄여야 합니다. 이번 개선을 더해도 50 tok/s의 달성을 보장하지 않습니다.

## 조건과 재현

- 메인: 9950X3D2, RAM 128GB/4800 MT/s, 5090 + PRO 4000.
- 코하네: Ryzen 9600, 교체한 TeamGroup 32GB 두 장(SK Hynix 칩),
  BIOS/SMBIOS 설정 5200 MT/s, 4070 Ti SUPER + 5060 Ti.
- 코하네 RAM 교체 후 stressapptest 48GiB/12스레드/90초 PASS, 오류 0.
  이것을 장시간 안정성 인증으로 해석하지 않습니다.
- 같은 IQ4_XS pack, INT8 KV, context 262144, prefill 32768.
- fixed2 MTP, stagger0, private scratch3, NC3=1, MV_TILED=1, MV_MMA=0.
- RAM_GB=92, backing OFF, TIER_AUDIT/WIN_AUDIT/SPEC_PROF ON.
- copy-aware/profile 유지, cost 0.16 ms, RTT 0.35/0.10 ms.
- CPU helper 54GiB/6스레드. 기존 두 helper binary를 그대로 사용했습니다.
- 바뀐 engine flag는 `MAYA_GLM_WIN_FETCH_REUSE` 0 → 1뿐입니다.
- 두 arm의 시작 배치 수량: primary pool 1,466 slots/18.46GiB,
  RAM 7,302 slots/91.95GiB, remote cold 788개/9.88GiB,
  optional copies 3,543개/47.37 **decimal GB**로 같았습니다.
- CPU별 초기 calibration 로그는 조금 다르지만 CPU plan 및 copy 비용은 고정했습니다.
- B200의 GPU와 idle keeper는 사용하거나 중단하지 않았습니다.

binary: `~/fn/bin-test/maya-window-reuse-20261028` (commit `6271113`)

SHA256: `6f6dc6c1f11a42bd96c8cfed0943a5e24373b76a8672abf80ff81a4f7e237856`

Claude의 `qshort.py`는 변경하지 않았습니다.
SHA256: `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`

재부팅 후 CPU 계측이 오래된 PID를 읽는 문제는 별도 resume wrapper에서 수정했습니다.
현재 launcher가 쓴 helper-state의 PID를 읽고 실제 argv의 54GiB/6스레드를 검증합니다.
긴 입력 client도 종료 시 정리할 수 있도록 supervisor가 추적합니다.

- 메인 실행 폴더: `~/fn/codex-fetch-reuse-resume-20261028`.
- 메인 wrapper: `~/fn/bench_fetch_reuse_resume.py`.
- 로컬 원본·환경·분석 스크립트: `/home/caterpillar/Agent/maya-fetch-reuse-resume-20261028`.
- 상세 비교: [측정 JSON](home-window-fetch-reuse-measurements-20261028.json).
- 이전 실험: [RAM backing/NC3](home-backing-probe-20261027.md).

운영 binary는 기존 `maya-copy-profile-cc80982`, launcher는
`MAYA_NET=1 ~/fn/start-maya-kohane5200.sh`입니다. 이 실험은 새로운 기본 설정을
선정한 것이 아니며, fresh plain/MTP1와의 비교는 아직 하지 않았습니다.
