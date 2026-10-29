# MTP 검증 창의 헬퍼 요청 선발송 — 2026-10-29

홈 5090의 CUDA 단위검사와 동일 엔진 OFF/ON 비교를 완료했습니다. **대화 38.90→39.35, 코드 33.90→34.30 tok/s**로 약 1.2% 개선을 관측했습니다. 50 tok/s는 아직 달성하지 못했습니다. 두 회씩 순차 측정이므로 개선폭의 장기 재현성을 단정하지 않습니다.

| 항목 | OFF | ON |
|---|---:|---:|
| chat pass 1 / 2, tok/s | 38.9 / 38.9 | 39.4 / 39.3 |
| code pass 1 / 2, tok/s | 34.2 / 33.6 | 34.6 / 34.0 |
| chat verify ms/window | 46.846 | 46.338 |
| code verify ms/window | 54.904 | 54.238 |
| 메인 engine CPU, 한 코어=100% | 174.82% | 163.39% |
| 원격 CPU helper CPU, 한 코어=100% | 459.00% | 458.84% |

CPU 비율은 프리필을 포함한 qshort 전체 구간의 두 회 평균이며 전력 측정이 아닙니다. 모든 20개 요청 쌍에서 MTP rounds/승인 수, window 수, tier payload/route 수가 같았습니다. RAM→GPU 및 boundary D2H 바이트 변화는 0%, 요청 작업량 차이는 0건입니다. 대응하는 두 greedy 해시도 일치했습니다. 전체 모델의 모든 출력 float를 비교한 것은 아닙니다.

단위검사는 T2/T3의 RAM/disk/staging/admission/reuse/stale-sequence와 세 가중치 조합의 grouped arithmetic bitwise 비교를 통과했습니다(0 failures). 실제 모델은 조건당 qshort 18요청 + 16.5K context 2요청을 완료했고 양쪽 audit가 완결됐습니다. 초기 전문가/메모리 배치도 같습니다.

원자료는 로컬 `maya-early-route-20261029/`, 메인 `~/fn/codex-early-route-20261029/`에 있습니다. `comparison.json`에는 요청별 작업량 비교와 시간 집계가 있고 `binary-sha256.txt`, `cuda-unit.txt` 및 원본 로그를 보관했습니다. 운영 반영 상태와 후속 비용 조정 결과는 별도 재개 기록에서 확인합니다.

## 줄이려는 대기

현재 fixed1 MTP는 두 행(T2)을 검증합니다. `fast_moe_window()`는 한 레이어에서 다음 순서로 일합니다.

```text
row0 route → disk wait → RAM/disk fetch
row1 route → disk wait → RAM/disk fetch
두 행의 전문가 계산 → CPU/헬퍼 응답 합류
```

원격 헬퍼 요청은 각 route를 서비스 스레드가 읽었을 때 전송됩니다. 따라서 row1의 요청은 row0의 RAM→GPU 복사가 끝날 때까지 발행되지 않습니다. 헬퍼 내부 CPU/GPU 혼합 요청을 겹치는 이전 실험과는 다른 구간입니다. 이번 변경은 입력 행의 route를 먼저 발행하여 원격 계산을 더 일찍 시작할 수 있게 합니다. 전문가 선택, 가중치, 정밀도, 최종 합산 순서는 바꾸지 않습니다.

## 구현과 활성 조건

설정은 `MAYA_GLM_WIN_ROUTE_AHEAD=1`, 엔진 내부 이름은 `STRATA_GLM_WIN_ROUTE_AHEAD`입니다. 기본값은 OFF입니다.

활성 조건은 다음을 모두 만족하는 경우입니다.

- 단일 trunk, full T2 또는 T3 window
- row별 private scratch가 실제 창 전체를 수용
- batched expert 경로, `t0=0`, 전체 행을 처리하는 `part=0`
- non-staggered, non-pipelined window

조건을 만족하지 않는 호출은 원래 경로를 유지합니다. 활성화 시 한 번 다음 로그가 나옵니다.

```text
glm window route ahead: CUDA0 active for private T2; RAM fetches deferred, disk rows fetched before the next route
```

지원 조건의 새 순서는 다음과 같습니다.

```text
row0 route → disk wait → 이 행에 disk miss가 있으면 모든 fetch 즉시 수행
row1 route → disk wait → 이 행에 disk miss가 있으면 모든 fetch 즉시 수행
disk miss가 없던 row0의 fetch
disk miss가 없던 row1의 fetch
두 행의 전문가 계산 → CPU/헬퍼 응답 합류
```

T3에서는 같은 규칙을 세 행에 적용합니다. GPU에서 행별 disk 여부를 판단하므로 별도 host 동기화는 추가하지 않습니다. 대신 행마다 조건부 fetch kernel launch가 하나 늘어납니다. disk miss가 없는 일반 행에서는 앞쪽 fetch kernel이 바로 반환하고, 뒤쪽에서 실제 복사합니다. 이 추가 launch 비용을 포함한 종단 간 측정이 필요합니다.

`MoeFetchPhase::{All, HasDiskMiss, NoDiskMiss}`를 추가했습니다. 기존 호출은 `All` 기본 인자로 원래 복사를 수행하며, `All` kernel specialization에는 disk 조건 분기를 넣지 않았습니다. 새 경로는 `W.cpu_seq_w`의 행당 64-byte 예약 공간에서 첫 uint 다음의 미사용 셀을 행별 `wait_seq`로 사용합니다. CPU 답변 sequence 셀은 그대로입니다. 별도 GPU/RAM 버퍼나 전문가 슬롯 예약은 추가하지 않습니다.

## 수명과 순서 검토

**디스크 staging은 즉시 소비해야 합니다.** `fast_service()`는 비어 있는 RAM 슬롯이 없으면 `F->stage[m]`에 디스크 전문가를 읽습니다. 다음 행도 같은 staging 슬롯을 쓸 수 있습니다. 모든 route를 먼저 실행한 뒤 모든 fetch를 실행하는 단순 변경은 앞 행의 원본을 덮어쓸 수 있어 사용하지 않았습니다.

새 경로에서 disk miss가 하나라도 있는 행은 RAM 부분까지 포함하여 기존처럼 route→wait→fetch를 완료합니다. 다음 행의 route가 동일 CUDA stream에서 그 뒤에 실행되므로 서비스 스레드가 다음 요청을 보고 staging을 덮기 전에 현재 행 복사가 끝납니다. 뒤쪽 fetch는 그 행을 건너뜁니다.

**RAM tier source는 window 중 재활용되지 않습니다.** 서비스의 disk resolve는 `kRFree`만 새 쓰기 대상으로 선택합니다. RAM에서 읽는 source는 `kRHold`이며 promotion은 이를 `kRRelease`로 표시할 뿐 즉시 재사용하지 않습니다. `ram_victim()`과 release→free 처리는 `fast_boundary()` 내부에 있고, boundary는 compute stream을 동기화한 뒤 실행합니다. Lookahead도 `kRFree`만 예약합니다. `glm_win_spares()`는 완료된 demotion을 `kRHold`로 만들고 VRAM spare를 채우며 RAM victim을 선택하지 않습니다. 따라서 뒤 행의 disk resolve가 앞 행의 deferred RAM source를 희생 슬롯으로 덮는 경로는 찾지 못했습니다.

**Scratch reuse도 모든 소비보다 앞서 완료됩니다.** 기존 fetch reuse는 같은 레이어의 앞 행 private scratch 포인터만 공유합니다. 새 경로에서는 그 포인터의 실제 복사가 아직 지연되어 있을 수 있지만, 모든 행의 delayed fetch가 끝난 뒤에만 grouped expert 연산을 시작합니다. promotion으로 다음 행이 앞 행의 VRAM 슬롯을 hit한 경우도 같은 순서로 보호됩니다. 다음 레이어는 같은 stream에서 현재 전문가 계산 뒤에 오므로 scratch를 먼저 덮지 않습니다.

**`wait_seq`는 행별로 보존합니다.** 기존 route의 전역 `wait_seq`를 그대로 쓰면 다음 행이 앞 행의 disk 여부를 덮습니다. 새 경로는 행별 셀을 사용하고 route가 매번 0 또는 자신의 sequence를 씁니다. `moe_wait()`는 이를 읽고 응답을 반영하지만 지우지 않습니다. 값을 지우는 `moe_wait_fetch()`로 바꾸면 이 설계가 깨지므로 새 경로에서 사용하지 않습니다.

관련 코드: `src/core/glm_window.cu::fast_moe_window()`, `src/kernels/cuda/glm_fast.cu::moe_fetch_kernel()`, `src/core/glm_fast_path.cu::fast_service()/fast_boundary()/fast_ahead_route()`.

## 검사 범위와 실행법

기존 `tests/glm_window_fetch_reuse_test.cu`를 확장했고 홈 5090에서 실행하여 통과했습니다.

- T2/T3, fetch reuse OFF/ON에서 원래 순서와 새 순서의 전문가 bytes, route weights, pointer group 수, fetch/reuse/CPU/promotion/disk/선택 결정을 비교합니다.
- RAM-only, CPU-only, helper-only, disk-only, RAM/disk 혼합, promotion 및 promotion threshold를 포함합니다.
- 각 행의 disk 응답이 **같은 staging 슬롯에 서로 다른 전문가 bytes를 다시 쓰는** fixture를 사용하여 staging lifetime을 검사합니다.
- RAM-only 경로에서는 뒤쪽 fetch 전까지 scratch sentinel이 유지되는지 확인해 실제 복사 지연을 검사합니다.
- 다음 레이어의 retained plan storage 및 중단된 route의 이전 disk sequence를 검사합니다.
- IQ4_XS gate/down, up의 IQ4_XS·Q4_K·Q5_K 조합에서 grouped float 출력을 bitwise 비교합니다. RAM-only와 shared disk staging을 쓰는 혼합 경로를 포함합니다.

이 fixture는 실제 모델 전체나 네트워크 연결을 실행하지 않습니다. full-window 활성 조건과 서비스 스레드의 실제 overlap, 처리량은 별도의 모델 하네스로 검증해야 합니다.

빌드 대상은 기존 `maya`, `glm_window_fetch_reuse_test`입니다.

```sh
cmake --build build-copy-aware --target maya glm_window_fetch_reuse_test -j8
```

생성된 단위검사 binary를 홈 머신으로 옮겨 메인 서버가 점유한 GPU를 안전하게 해제한 뒤, 홈 5090만 선택하여 실행합니다. **로컬 B200 GPU에서는 실행하지 않습니다.** GPU가 없는 경우 테스트는 77로 종료하며 PASS가 아닙니다.

실제 A/B는 같은 새 engine으로 OFF 2회 / ON 2회 수행했습니다. `WIN_SCRATCH_ROWS=3`을 포함한 기존 fixed1/T2 reuse 설정을 유지하며, Claude의 원래 qshort 하네스를 수정하지 않습니다. scratch 용량2 실험은 별개입니다. 비교 항목은 chat/code tok/s, MTP 채택률·rounds, scratch replay, tier/route/payload 수, greedy 해시 및 실제 활성 로그입니다. 출력 변화나 실패가 있으면 성능 개선으로 채택하지 않습니다.

측정한 개선폭은 약 1.2%였습니다. RAM fetch의 총 bytes와 CPU 전문가의 총 계산량을 줄이는 변경이 아니라 원격 요청 시작 시각을 앞당기는 변경이며, 헬퍼 대기와 겹치는 비율에 따라 효과가 달라집니다.
