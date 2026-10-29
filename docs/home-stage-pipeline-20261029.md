# 50 tok/s: 원본 RAM 유지와 두 호스트 파이프라인

2026-10-29. 원래 qshort 하네스, 같은 모델 가중치와 KV 양자화입니다.

| 개발 구성 | 대화 tok/s | 코드 tok/s |
|---|---:|---:|
| 최초 원격 cut18 | 24.5 | 21.1 |
| 고정 CPU를 보조에만 사용, cut12 | 35.9 | 29.8 |
| 프롬프트 뒤 helper 갱신 | 35.4 | 31.1 |
| window prefetch | 36.1 | 30.0 |
| prefetch16 blocks | 36.1 | 30.1 |
| main canonical RAM, cut12, T2 | **41.5** | **35.6** |
| 같은 구성 T3 | 40.2 | 35.8 |

canonical RAM은 고정 local helper의 RAM 중복을 제거하고 primary GPU expert 원본을 RAM에 유지합니다.
실제 모델 검사에서 helper 중복22.56GiB 제거,21.90GiB의 원본 추가,99.57GiB 유효 원본,
전체 pinned RAM101.13GiB였습니다. 예산을 늘리지 않았습니다. T2/T3 출력 비교75회 max_abs0,
greedy/고정 seed sampled MTP 검사 통과. qshort greedy 해시도 직전과 일치합니다.
디코드 계측의 boundary/background D2H count와 bytes는0입니다.

옵션은 `MAYA_GLM_STAGE_RAM_CANONICAL=1`; stage+discrete fixed local helper에 한정합니다.
재배치(`STAGE_HELPER_REFRESH`)와 함께 쓰지 않습니다. class별 슬롯 수가 모자라면 변경 전에 거부합니다.
cut12 main은 RAM_GB106/headroom10/RAM_CLASS_SLOTS=7800,304를 사용했습니다.

선행 전송은 `MAYA_GLM_WIN_PREFETCH=1`, private T2/T3, unstaggered 한 trunk에서만 동작합니다.
다음 층 router를 현재 입력에 적용한 후보를 별도 스트림에서 복사합니다. 실제 router/weights는 그대로입니다.
레이어 parity별 목록과 완료 event로 수명을 관리하며, 실제 사용 전 복사를 기다립니다.
`WIN_PREFETCH_T1=1`은 단일 행도 묶음 커널에서 실행하여 같은 전송 선행 경로를 사용합니다.
이는 별도 opt-in이며 T1/T2/T3 및 rollback 정확성 검사를 통과했습니다.

prefetch는 요구 시 fetch를 줄였지만 expert math/helper wait를 늘려 이득이 상쇄됐습니다.
SM 점유량16 blocks만으로는 해결되지 않았고, canonical RAM을 더한 뒤 실제 가속이 나왔습니다.
기존 RAM backing 유지 경로를 재사용하며, fixed helper는 VRAM에, 나머지는 모두 RAM 원본을 갖습니다.

## 실제 중첩을 위한 시간 예산

cut16, 양쪽 canonical RAM, main CPU0/front CPU6, front fixed CPU50%, 양쪽 T1 prefetch16blocks.
main RAM92/headroom20/class slots6600,304; front RAM44/headroom8.
이 구성도 실제 모델 정확성 검사를 통과했습니다.
`STAGE_TIMING=1`은 decode audit 구간에서 front RPC와 local tail을 따로 계측합니다.

원래 하네스, MTP_FIXED0으로 확인한 단일 행 시간:

| 항목 | front RPC ms | tail ms |
|---|---:|---:|
| 대화4개 평균 | 15.380 | 13.558 |
| 코드4개 평균 | 16.012 | 16.308 |

현재는 합쳐서 기다립니다(직렬 대화34.5/코드30.9tok/s). 실제 파이프라인에서는 느린 단계에
추가 drafting/통신/오예측 복구 비용이 붙습니다. 이 수치는 파이프라인에서50이 나왔다는 뜻이 아닙니다.
다음 구현은 보조가 다음 토큰의 front를 추측해서 계산하고, 메인이 현재 토큰의 tail을 계산하는
단일 토큰 파이프라인입니다. 틀린 front 추측은 recurrent state를 복원하고 정확한 token으로 재실행합니다.
최종 출력은 항상 main verifier의 샘플만 사용합니다.

원자료: 메인 `/home/caterpillar/fn/remote-stage-*20261029*`, 로컬
`/home/caterpillar/Agent/maya-dual-host-design-20261029/home/`.
주요 실행은 canonical-test/speed-20261029k, canonical-t3-speed-20261029l,
timing16-test-20261029m, timing16-t1-speed-20261029m입니다.
