# T3 선발송 후속 비교 — 2026-10-29

## 결과와 선택

초안 2개를 검증하는 T3에서도 요청 선발송의 효과를 확인했습니다. 다만 대화에는 T2가 더 빨라 **기본 운영은 fixed1/T2 선발송 ON을 유지**합니다. T3는 코드용 선택 설정으로 저장했으며 자동 적용하지 않았습니다. 50 tok/s는 미달입니다.

| 조건, 실행 순서 | chat tok/s | code tok/s |
|---|---:|---:|
| T3 선발송 OFF, 1회 | 36.5 | 34.5 |
| T3 선발송 ON, 1회차 | 37.1 | 35.1 |
| T3 선발송 ON, 2회차 | 37.7 | 36.5 |
| T3 ON 두 회 평균 | **37.4** | **35.8** |
| 현재 T2 ON, 마지막 재확인 1회 | **39.3** | **34.6** |

T3의 두 번째 회차는 캐시 상태가 이어집니다. OFF 첫 회와 ON 두 회 평균을 비교해 선발송만의 가속률로 주장하지 않습니다. 같은 첫 회끼리의 개선은 chat +1.64%, code +1.74%입니다. 마지막 T2는 단일 재확인 측정입니다. 이전 동일 엔진 T2 ON 두 회 평균은 39.35/34.30이었습니다.

## 같은 작업에서 줄어든 시간

첫 OFF/ON의 9개 대응 요청에서 rounds, MTP 승인 수, tier/route payload, reuse 수가 모두 같았고 greedy 해시도 같습니다. scratch replay와 누락된 route/대기 service는 0입니다.

| 첫 회 대응 비교 | T3 OFF | T3 ON |
|---|---:|---:|
| chat verify ms/window | 60.354 | 59.507 |
| code verify ms/window | 73.306 | 71.906 |
| chat round 전체 host ms | 62.744 | 61.899 |
| code round 전체 host ms | 75.544 | 74.141 |

전송량을 줄인 효과가 아니라 기존 작업을 겹친 효과입니다. 전체 모델의 모든 float 출력 동일성을 검사한 것은 아닙니다. 같은 바이너리는 직전 T2/T3 CUDA 단위검사에서 RAM/disk 임시 버퍼 및 grouped arithmetic bitwise 검증을 이미 통과했으므로 이번에 같은 단위검사를 다시 실행하지 않았습니다.

## 왜 대화 기본값은 T2인가

T3 ON 두 회의 대화 평균은 drafting round당 2.304토큰, draft+verify+commit 합계 약62.012ms였습니다. 마지막 T2의 대화는 1.873토큰/47.803ms였습니다. 토큰을 더 얻더라도 라운드가 더 오래 걸려 T3가 느립니다.

코드는 T3 2.616토큰/73.408ms, T2 1.904토큰/55.217ms여서 T3가 유리한 방향입니다. 이 값은 host 단계 계측이며 API decode tok/s와 정확히 같은 분모는 아닙니다. 하네스 전체의 성능 결과를 우선합니다.

16.5K context의 T3 첫/반복 decode는 33.8/35.2 tok/s, 프리필12.7492초였습니다. 이전 T2는 33.8/35.0이었으나 출력 길이가 각각219/219와225/225로 달라 같은 토큰열의 replay 비교가 아닙니다. 긴 입력에서 뚜렷한 우위를 입증하지 못했습니다.

## 설정과 재현

- 새 코드 변경 없이 `maya-early-route-20261029` 동일 바이너리 사용.
- SHA256 `b59380b34f7311c8225f6e93430327d132f634c274b3dc40a13208acfec4b2d9`.
- fixed2/T3, scratch3, fetch reuse ON, 기존 CPU/복사 비용 .16/.35/.10 유지.
- 같은 양자화·클럭·CPU helper 6스레드/54GiB·프로파일·원래 Claude qshort 하네스.
- T3 OFF9 + ON20 + T2재확인9, 총38개 모델 요청의 audit가 완결됐습니다.
- 선택 설정: 메인 `~/fn/maya-config/iq4-kohane5200-code-t3.json`. 검증한 T3 ON 설정에서 로그 경로만 운영용으로 분리했습니다. 기본 파일을 대체하지 않았으며 별도 동시 서버용 설정이 아닙니다.
- 기본 설정은 계속 `~/fn/maya-config/iq4-kohane5200.json`, fixed1 및 route ahead ON입니다.
- 메인 원자료: `~/fn/codex-t3-route-20261029/`.
- 로컬: `/home/caterpillar/Agent/maya-t3-route-20261029/`의 plan, wrapper, results.json, comparison.json, 원본 로그와 선택 설정 사본.

검증기의 T2 고정 문자열은 실행 전에 `T{fixed+1}`로 고쳤습니다. 초기 draft0이나 출력 종료 직전 T2 fallback은 정상이며, 실제 T3 활성 로그와 window audit를 함께 확인했습니다.

측정 종료 후 기본 T2 설정으로 복원하고 `/health` 및 32토큰 실제 생성(drafts16/accepted15)을 확인했습니다. `production-smoke.json`에 저장했습니다. 실험은 종료됐으며 기본 서버가 정상 실행 중입니다.
