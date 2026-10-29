# 원격 MTP 고정 전문가 배치 수정 및 부하 재분배

2026-10-29. 양자화 및 원래 Claude `qshort.py`는 변경하지 않았습니다.

## 확인된 문제와 변경

stage 모드는 정확성 때문에 helper 소유권을 고정하지만, 초기 helper 선발은 `F->usage`만 정렬했습니다.
사용 이력 파일을 지정하지 않은 새 세션에서는 모두 0이어서, 입력의 레이어/전문가 ID 순서대로
VRAM을 채웠습니다. 트렁크가 사용한 `--expert-profile`의 전역 우선순위가 helper에는 적용되지 않았습니다.

고정 stage helper에서는 usage 우선순위를 유지하고 동률을 profile 순위로 정렬하도록 수정했습니다.
일반 단일 트렁크의 기존 정책은 유지합니다. 로드 시 레이어별 helper 전문가 수를 출력합니다.
CPU 회귀검사는 새 세션의 전역 선발, 학습된 usage 우선, profile 누락 시 안정 순서를 확인합니다.
관련 CTest 4개가 통과했습니다.

## 실험

| 구성 | 대화 tok/s | 코드 tok/s | 판정 |
|---|---:|---:|---|
| 기존 cut18, 양쪽 CPU0 | 24.5 | 21.1 | 이전 비교 기준 |
| 기존 cut18, 앞단 CPU6 | 측정 안 함 | 측정 안 함 | T1 reset 정확성 검사 실패 |
| profile 순위 수정, cut18, CPU0 | 25.7 | 22.9 | 정확성 통과 후 원래 하네스 1회 |
| profile 순위 수정, cut12, CPU0, T2 | 33.2 | 28.2 | 정확성 통과 후 원래 하네스 1회 |
| profile 순위 수정, cut12, CPU0, T3 | 31.4 | 28.5 | 같은 배치, 원래 하네스 1회 |

CPU6은 자동 보정에서 전문가당 CPU 0.243ms / PCIe 0.550ms를 관측했지만,
T1 reset 비교에서 max_abs=2.06681919, relative_l2=0.127159742였습니다.
이는 속도 개선으로 채택할 수 있는 결과가 아닙니다. CPU/GPU 실행 배정과 캐시 이동에 따른
산술 경로 변화가 후보 원인이며, 이 실행만으로 원인을 확정하지 않습니다.

수정 cut18은 전체 어휘 logits 75회 비교에서 최대 절대 차이 0, T2/T3 모든 keep,
paired snapshot, greedy 및 고정 seed sampled MTP 일치를 통과했습니다.
속도 하네스의 마지막 greedy MD5도 이전과 동일한 `bde9193a1a03c9647115830774e85fa6`입니다.

수정 cut18 MTP window 수와 audit 수를 대조한 가중 평균:

| 구간 | 대화 ms | 코드 ms |
|---|---:|---:|
| 전체 verify | 70.449 | 83.460 |
| tail local span | 23.424 | 31.374 |
| 차이: front + 통신 + coordinator | 47.025 | 52.086 |

대부분의 이득은 tail에서 발생했습니다. front 병목은 여전히 남습니다.
표의 속도는 각 구성 1회 실행이며 sampled 요청의 출력 길이/경로는 동일하게 고정되지 않았습니다.

## cut12 재분배

앞단은 [0,12), 뒷단은 [12,45)+NextN입니다. 메인 RAM 예산106GiB/headroom10GiB,
보조 RAM 예산54GiB/headroom8GiB입니다. 미사용 대형 prefill staging 제거 후 가능한 배치입니다.
실제 속도 실행 메인 RAM tier는103.02GiB, 보조 트렁크 전문가 VRAM은10.90GiB입니다.
모델 weights/KV 양자화, context262144, scratch3rows는 그대로입니다.

cut12도 전체 어휘 logits 비교75회 최대 절대 차이0 및 greedy/sampled MTP 검사를 통과했습니다.
그러나 하네스의120토큰 greedy MD5는 `8a689b1f560a3b1e6f2c4f2edc2021af`로 cut18과 다릅니다.
**검증한 것은 동일 배치 내 plain/window/MTP의 일치이며, 배치 변경 전후 출력의 완전 동일성이 아닙니다.**
GPU 및 expert 소유권 변경에 따른 산술 경로 차이 가능성이 있지만, cross-cut 차이의 원인은 아직 분리하지 않았습니다.

| cut12 T2 구간 | 대화 ms | 코드 ms |
|---|---:|---:|
| 전체 verify | 53.957 | 66.200 |
| tail local span | 36.238 | 48.143 |
| 차이: front + 통신 + coordinator | 17.720 | 18.057 |

front 비용이 크게 줄었지만 tail이 길어졌습니다. 완전한 stage overlap을 가정해도 코드의48ms tail은
50tok/s의 T2 예산을 초과합니다. 단순 pipeline 추가만으로 목표 달성을 보장할 수 없습니다.

T3에서는 전체 verify가 대화71.465ms/코드88.853ms, tail이47.949ms/65.023ms로 늘었습니다.
토큰 수 증가가 그 비용을 충분히 상쇄하지 못했습니다. T2/T3의120토큰 greedy MD5는 일치했습니다.

## 다음 개발 지점과 실행 상태

다음 대상은 tail의 전문가 처리/전송 비용과 CPU 참여 시 산술 경로의 안정성입니다.
CPU6을 켜기만 하면 reset 검사를 통과하지 못하므로, 고정 expert 실행 배정 또는 동일한 산술 경로를
먼저 구현/검증해야 합니다. 현재 결과는 기존 단일 트렁크39.4/34.7보다 느립니다.
50tok/s 달성이나 분산 구성이 단일 트렁크보다 빠르다는 결론은 아직 낼 수 없습니다.

현재 개발 서버는 `remote-stage-ranked12-speed-20261029b`의 cut12/T3입니다.
이것은 마지막 측정 구성으로, 대화 기준 최선은 cut12/T2입니다.
PID/argv는 해당 폴더의 `status.json`을 조회한 뒤 제어합니다. 이전 운영 구성은 복구하지 않았습니다.

## 원자료

메인 `/home/caterpillar/fn/` 및 로컬 `/home/caterpillar/Agent/maya-dual-host-design-20261029/home/`:

- `remote-stage-cpu6-test-20261029a`: 기존 binary, CPU6 검사 실패.
- `remote-stage-ranked18-test-20261029a`: 수정 binary, cut18 정확성 통과.
- `remote-stage-ranked18-speed-20261029a`: 수정 binary, cut18 원래 하네스.
- `remote-stage-ranked12-test-20261029a`: 수정 binary, cut12 정확성 통과.
- `remote-stage-ranked12-speed-20261029a`: 수정 binary, cut12 T2 원래 하네스.
- `remote-stage-ranked12-speed-20261029b`: 수정 binary, cut12 T3 원래 하네스.

수정 binaries는 이름에 `ranked-20261029`를 포함합니다. 외부 실행 스크립트
`run_model_variant.py`, `run_speed_variant.py`는 cut/CPU/RAM을 기록하고,
속도 실행이 해당 구성의 성공한 정확성 검사 환경을 읽도록 했습니다.
이전 운영 구성을 복구하지 않았습니다.
