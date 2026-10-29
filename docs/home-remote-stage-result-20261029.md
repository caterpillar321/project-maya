# 원격 두 트렁크 구현 결과와 재개 지점

2026-10-29. 브랜치 `codex/engine-50tps`. **기능 검증 통과, 성능 목표 실패**입니다.

- 보조컴 4070 Ti SUPER가 [0,18), 메인 5090이 [18,45)와 NextN/head를 처리합니다.
- 각 호스트의 5060 Ti/PRO 4000은 고정 전문가를 담당합니다. 두 호스트 모두 CPU expert lane은 0입니다.
- `MAYA_GLM_REMOTE_FRONT=listen:18802`, `MAYA_GLM_REMOTE_CUT=18`로 명시적으로 켭니다.
  보조컴의 `maya-stage-worker --dial 192.168.0.4:18802 --cut 18`가 연결합니다.
- 양쪽은 native T2/T3 묶음 계산을 유지하지만 **트렁크 실행은 직렬**입니다. 경계는 FP32입니다.
- 전문가 자동 재배치를 끄기 전에는 reset 후 결과가 달랐습니다. stage 모드에서는 소유권을 고정합니다.
- 독립 호스트 RAM 예산을 기존 로컬 분할 비율로 줄이던 문제를 수정했습니다.
- 미사용 대용량 prefill staging을 예약하면서 보조 워커가 OOM 종료된 실험이 있었습니다.
  stage 모드에서 해당 예약을 제거하고 pinning 전에 MemAvailable와 headroom을 검사하도록 수정했습니다.
- 유휴 상태에서 워커가 socket timeout으로 종료되지 않도록 request 대기와 활성 I/O timeout을 분리했습니다.

## 검증

CPU transport 79개 검사(실제 socketpair, ASan/UBSan), 관련 CTest 4개가 통과했습니다.
실제 두 호스트, 262144 문맥 설정 및 INT8 KV streaming에서 다음을 검증했습니다.

- T2/T3의 모든 keep 길이, prefix 0/3/5/31, 거절 후 다음 토큰, reset 및 양쪽 snapshot 복원.
- 전체 어휘 logits 비교 75회: 최대 절대 차이 **0**.
- 분산 plain decode와 native MTP: greedy 24토큰 및 고정 seed sampling 24토큰 일치.
- 짧은 입력의 검사입니다. 긴 문맥에서 KV eviction과 모든 취소/장애 상황을 검증한 것은 아닙니다.

## 성능

원래 Claude `qshort.py LOG chat code`를 변경 없이 사용했습니다. 각 행은 9요청 한 회입니다.

| 구성 | 대화 tok/s | 코드 tok/s |
|---|---:|---:|
| 과거 단일 트렁크 T2 기준 | 39.4 | 34.7 |
| 원격 직렬 T2 첫 회 | 24.3 | 21.0 |
| 같은 구성, 검증 구간 계측 | 24.5 | 21.1 |

두 원격 실행의 마지막 greedy MD5는 `bde9193a1a03c9647115830774e85fa6`입니다.
계측 실행의 MTP round와 local window audit 수가 일치하는 것을 확인하고 창 수로 가중 평균했습니다.

| T2 검증 구간 | 대화 | 코드 |
|---|---:|---:|
| 전체 verify | 74.595 ms | 89.438 ms |
| 메인 tail local span | 26.907 ms | 36.034 ms |
| 차이: front + 통신 + coordinator overhead | 47.688 ms | 53.405 ms |

차이를 순수 front GPU 시간으로 부르지 않습니다. 현재 비용을 유지하면 front만으로도
50 tok/s용 T2 verify 예산 약 36~37 ms를 넘으므로, pipeline 추가만으로 목표를 달성한다고 주장하지 않습니다.
앞선 대화 45/코드 40 예측은 실측으로 뒷받침되지 않았습니다.

## 제한과 현재 상태

- 큰 tensor-core remote prefill은 미구현입니다. 작은 window 기반 prefill 함수는 검증했으나,
  현재 API의 `prefill_chunk()`는 0을 보고하므로 API prompt는 T1 경로로 처리합니다. TTFT 개선을 주장하지 않습니다.
- 양쪽 단일 snapshot은 지원하지만 persisted conversation slots/checkpoint slots는 거부합니다.
- 전송 식별은 metadata/geometry/KV 형식 대조입니다. 모델 weights 전체의 암호학적 hash 검사는 아닙니다.
- 기존 운영 구성은 복구하지 않았습니다. 원래 helperd PID 5829/16021은 종료했습니다.
- 현재 개발 API: 메인 PID **255800**, 보조 stage worker PID **44147**.
  설정과 PID 기록은 메인 `/home/caterpillar/fn/remote-stage-speed-20261029e/`에 있습니다.
  유휴 시간 이후 실제 짧은 생성 응답까지 확인했습니다. 그 짧은 응답 속도는 벤치마크에 포함하지 않습니다.
- 메인 binary: `/home/caterpillar/fn/bin-test/maya-remote-stage-profile-20261029`.
  보조 binary: `/home/user/fn-bin/maya-stage-worker-20261029`.
- 실제 모델 테스트: 메인 `remote-stage-model-test-20261029e`, 속도: `remote-stage-speed-20261029c`와 `d`.
- 로컬 원자료/실행 스크립트: `/home/caterpillar/Agent/maya-dual-host-design-20261029/`.
  `home/`에 검증 및 성능 로그 사본을 저장했습니다. 향후 제어 전에 PID/argv를 다시 확인합니다.

다음 판단은 front 작업량/전문가 배치/CPU 활용을 바꿔 이 구간 자체를 줄일 수 있는지입니다.
새 구성에서 CPU lane을 켜거나 소유권을 변경하면 해당 구성의 정확성을 다시 검증해야 합니다.
