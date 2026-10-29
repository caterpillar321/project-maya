# 원격 MTP의 고정 CPU 전문가 및 묶음 계산

2026-10-29. 모델 가중치 양자화, KV 형식, 기존 Claude `qshort.py`는 변경하지 않았습니다.

## 정확성 문제와 구현

이전 CPU6 구성은 reset 후 동일 입력의 logits가 달랐습니다. 기존 CPU lane은 캐시 상태 및
route별 quota에 따라 전문가의 CPU/GPU 실행 경로를 바꾸며, 두 경로의 산술은 완전히 동일하지 않습니다.
cut12에서 RAM 전문가 전부를 CPU로 보내고 promotion을 끈 진단 구성은 full-vocabulary 비교75회가
모두 max_abs=0으로 통과했습니다. 이 진단은 이전 cut18 실패와 배치도 다르므로 원인 하나만을
독립적으로 증명한 실험으로 취급하지 않습니다.

`MAYA_GLM_STAGE_CPU_PERCENT=1..100`은 로드 시 local GPU helper/primary VRAM에 없는
RAM 전문가 후보 중 지정 비율을 레이어별 profile 순서에 고르게 배정합니다.
CPU 소유권은 요청마다 바뀌지 않습니다. 기존 quota는 끄고, GPU 담당 전문가의 캐시는 계속 적응합니다.

- 외부 helper 소유권1과 CPU 소유권2를 구분합니다. CPU 경로에는 보호된 RAM 주소를 전달합니다.
- CPU의 RAM 슬롯은 기존 `kRHelper` 보호 상태를 사용해 eviction/promotion에서 제외합니다.
- source 주소 또는 소유권이 어긋나면 기존 helper 실패 경로로 요청을 종료합니다.
- stage+discrete local helper+활성 CPU pool에서만 지원합니다. network helper 조합은 거부합니다.
- 런타임 PCIe quota 변경은 이 모드에서 거부합니다. 기존 단일 트렁크 및 기본 CPU 정책은 유지합니다.

`MAYA_GLM_STAGE_CPU_BATCH=1`은 추가 opt-in입니다. local helper의 합을 실제 answer slot에 두고,
CPU 요청만 복사해 기존 `win_cpu_merged`에서 계산한 뒤 더합니다. 두 기여분이 모두 끝나기 전에는
GPU가 보는 answer sequence를 공개하지 않습니다. 마지막 row에 CPU 일이 없거나 route 오류가 있어도
보류한 요청을 처리합니다. 이 모드의 기본값은0입니다.

## 검증

관련 CTest4개 통과. CPU 소유권 태그, external helper 우선권, 레이어별 비율 및 경계값을 검사합니다.
실제 두 호스트에서 고정 CPU 모드와 묶음 모드 각각 full-vocabulary logits75회 비교 max_abs=0,
T2/T3 모든 keep, prefix0/3/5/31, reset, snapshot, greedy/고정 seed sampled MTP 일치를 통과했습니다.
짧은 문맥 검사이며262K 문맥 전체와 모든 장애/취소 상황을 검증한 것은 아닙니다.

cut12에서 mainCPU8/frontCPU6, 양쪽 CPU비율50%:
메인3043/6107개 후보(37.89GiB), 보조322/649개 후보(4.17GiB)를 CPU에 고정했습니다.
이 메모리는 기존 RAM tier 안의 보호 영역이며 추가로 복제한 용량이 아닙니다.

## 성능

모든 행은 원래 하네스9요청 1회입니다. CPU는 전체 하네스 동안 process CPU time / wall time,
논리 코어 하나=100%로 기록했습니다. sampled 요청의 출력 경로는 구성 사이에 완전히 고정되지 않습니다.

| cut12 T2 구성 | 대화 tok/s | 코드 tok/s | 메인 CPU | 보조 CPU |
|---|---:|---:|---:|---:|
| 양쪽 CPU0 (이전) | 33.2 | 28.2 | 미계측 | 미계측 |
| 고정 CPU50%, 메인8/보조6 | 34.3 | 30.5 | 859% | 378% |
| 같은 구성 + CPU 묶음 계산 | 34.0 | 30.3 | 결과 JSON 참고 | 결과 JSON 참고 |

묶음 전후120토큰 greedy MD5는 `838b887bd6e3b4518dbff6a15c98dfc3`로 일치했습니다.
CPU0 구성과는 다릅니다. 검증한 것은 해당 실행 배치 내 plain/window/MTP의 일치이며,
CPU/GPU 배정 변경 전후의 완전 동일성을 주장하지 않습니다.

고정 CPU의 구간별 가중 평균(대화/코드): verify52.682/61.480ms,
tail38.254/47.171ms, 차이(front+전송+coordinator)14.428/14.308ms.
CPU 묶음 계산은 verify53.107/61.894ms, tail38.572/47.255ms로 전체 시간을 줄이지 못했습니다.
중복 expert evaluation은 감소했지만 묶음을 기다리는 비용과 겹침의 변화가 있어 기본값은 계속 끕니다.
메인 CPU 추가 부하 대비 tail 이득이 작아, 보조 CPU만 쓰는 구성을 별도로 비교합니다.

## 원자료 및 실행

메인 `/home/caterpillar/fn/`, 로컬 `/home/caterpillar/Agent/maya-dual-host-design-20261029/home/`:

- `remote-stage-staticcpu-test-20261029a`: 기존 binary, RAM 전부 CPU 및 promotion OFF 진단.
- `remote-stage-fixedcpu-test-20261029b`: 고정 CPU, GPU cache 적응 유지, 정확성 통과.
- `remote-stage-fixedcpu-speed-20261029b`: 메인8/보조6, CPU비율50%, T2.
- `remote-stage-fixedcpu-batch-test-20261029c`: CPU 묶음 계산 정확성 통과.
- `remote-stage-fixedcpu-batch-speed-20261029c`: 같은 CPU 배정, 묶음 계산 T2.

실행 환경은 각 폴더의 `environment.json`/`front-environment.json`/`config.json`에 기록됩니다.
모델/워커 binaries의 이름에는 `fixedcpu[-batch]-20261029`가 들어갑니다.
기존 운영 구성은 복구하지 않았습니다.
