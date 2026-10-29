# 50 tok/s 엔진 개발: 측정 결과와 재개 지점

2026-10-29. 코드 기준 `26497c1`, 브랜치 `codex/engine-50tps`.
개발 경로는 `/home/caterpillar/Agent/project-maya-50tps`입니다.
압축 후에는 이 문서를 먼저 읽고 필요한 원자료만 확인합니다.

## 목표와 사용자 제약

- GLM-5.3-Flash 단일 요청 decode 50 tok/s. IQ4_XS와 KV INT8을 유지합니다.
- 새 하드웨어 구매와 CPU 오버클럭은 제외합니다.
- 50 tok/s 전까지 기존 운영 구성을 다시 띄울 필요가 없습니다. 개발에 집중합니다.
- 실제 모델/GPU 검증은 `ssh sekai`의 집 장비에서만 합니다. 로컬 GPU와 idle keeper는 건드리지 않습니다.
- 원래 Claude 하네스 `~/fn/qshort.py`를 그대로 사용합니다.
- 사용자에게는 항상 한국어 존댓말을 사용합니다.

## 구현·검증 완료 범위

1. 로컬 PRO helper 없이도 원격 expert helper를 사용할 수 있도록 초기화와 dispatch를 분리했습니다.
2. 기본 OFF인 T2/T3 window 계측과 CPU 검증 파서를 추가했습니다.
3. 메인컴의 PRO/5090 두 trunk part가 서로 다른 원격 helper endpoint를 사용하도록 연결했습니다.
4. 동일한 메모리 예약 상태에서 plain/continuous pipeline을 비교하는 `MAYA_GLM_SPEC_DECODE` 스위치를 추가했습니다.

현재 분할은 기존 연속 `decode_spec()`입니다. **native batched T2/T3를 split에 연결한 상태가 아닙니다.**
CUDA 빌드와 세 CPU 테스트 타깃이 통과했습니다. 소유권 8개, split 계획 125개,
trace 파서 21개 검사를 포함합니다. 정확성 검증 범위는 아래 한계를 함께 봐야 합니다.

## 실제 하네스 결과

각 arm은 qshort 1회, 9요청입니다. 첫 계측 실험 27요청과 아래 분할 54요청을 합쳐 81요청입니다.
장기 평균이나 긴 문맥 검증 결과는 아닙니다.

| 구성 | 대화 tok/s | 코드 tok/s | 비교 근거 |
|---|---:|---:|---|
| 기존 단일 trunk T2, trace OFF | 39.4 | 34.7 | 이번 개발의 비교 기준 |
| 같은 구성, trace ON | 39.2 | 34.5 | 9요청 route/tier/MTP 수 일치 |
| 원격 helper만 사용, PRO helper 제외 | 31.5 | 27.0 | 연결 확인, 성능 채택 아님 |
| cut18, plain → pipeline | 13.2 → 15.1 | 11.1 → 13.2 | 마지막 greedy hash 일치 |
| cut11 앞층 GPU 상주·tail RAM 복사, plain → pipeline | 33.9 → 38.4 | 29.9 → 32.9 | **greedy hash 불일치 미해결** |
| cut18 앞층 CPU 6스레드, plain → pipeline | 25.3 → 35.7 | 23.0 → 30.7 | 마지막 greedy hash 일치 |

각 split 쌍의 초기 장치별 할당은 같았으며 tier audit가 완결됐습니다.
마지막 greedy 요청 이전의 8개 요청은 샘플링하므로 캐시 이력이 통제된 새 프로세스의
첫 요청 비교와 다릅니다. 한 번의 hash 일치는 전체 정확성 증명이 아니며,
cut11 불일치를 무해한 반올림 차이나 확정된 상태 버그 어느 쪽으로도 단정하지 않습니다.
기존 별도 T3 코드 평균 35.8 tok/s도 이번 split 최고보다 빨랐습니다.

## 병목에서 확인한 것

- 단일 trunk의 표본 T2 window에서 fetch 구간은 대화 20.8 / 코드 27.6 ms,
  노출된 helper 합류 대기는 7.7 / 8.6 ms였습니다. 전문가 계산은 약 4 ms입니다.
  CUDA event 구간에는 host enqueue 지연도 들어가며 순수 PCIe 서비스 시간은 아닙니다.
- cut18은 PRO의 Gen4×4로 많은 전문가 가중치를 옮기면서 느려졌습니다.
- cut11은 앞쪽 8개 MoE 층의 2,304개 전문가를 PRO 1,405개 + 4070 899개로 모두
  GPU에 상주시켰습니다. 앞부분의 mapped fetch는 plain 9요청 합계 약 338 MB까지 줄었습니다.
- 해당 pipeline 프로파일의 front 약 7 ms, 5090 tail 약 26 ms가 남았습니다.
  이 시간은 API 출력 토큰 시간과 집계 범위가 다르지만 남은 tail 병목을 보여 줍니다.
- CPU-front pipeline은 메인 CPU 약 754%(한 코어=100%)를 사용하면서도 더 느렸습니다.
- 논리적 disk-miss payload가 남아 있습니다. 해당 카운터를 물리 SSD 읽기량으로 해석하지 않습니다.

## 마지막 확인한 실행 상태

측정 종료 후 개발 엔진을 유지했습니다. **현재 실행 여부와 PID는 재개 시 다시 확인합니다.**

- 메인 `sekai`, API 18100.
- 설정: `/home/caterpillar/fn/codex-split-net-cpu-20261029/cpu-front-pipe.json`.
- 바이너리: `/home/caterpillar/fn/bin-test/maya-split-net-copy-20261029`.
- SHA256: `8a368792a86427d9ddcb5a9ff446d80b12dfec76f99857695851f54a8886bc69`.
- cut18, PRO 앞부분+메인 CPU 6스레드, 5090 뒷부분, optional copies OFF, continuous pipeline ON.
- 이는 마지막 실험 설정이며 최고 성능 설정으로 채택한 것이 아닙니다. 텍스트 전용 prototype입니다.
- 보조컴 `user@192.168.0.76`의 두 helper는 유지했으며 재시작하지 않았습니다.
- 원래 qshort SHA256: `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64`.

## 최신 사용자 제안과 다음 설계 판단

느린 슬롯에는 상주 전문가를, 빠른 슬롯에는 trunk를 맡기는 방향을 검토합니다.
고정 전문가란 모델의 routing을 바꾸는 것이 아니라 해당 가중치의 GPU 배치를 고정하는 뜻입니다.

| 호스트 | trunk 후보 | 상주 전문가 후보 |
|---|---|---|
| 보조컴 9600, RAM 64GB | 4070 Ti SUPER, Gen4×16, 앞쪽 연속 층 | 5060 Ti, Gen4×4 |
| 메인컴, RAM 128GB | 5090, Gen5×16, 뒤쪽 연속 층 + head/NextN | PRO 4000, Gen4×4 |

이 후보는 기존 같은 호스트 PRO→5090 split과 다릅니다. 각 호스트가 자기 층의 RAM과
전문가 계산을 맡으면 층 경계의 hidden 전달로 매층 원격 전문가 호출을 줄일 수 있습니다.
단, tail의 전문가를 보조 RAM에 계속 의존하면 매층 왕복이 다시 생깁니다. 전체 가중치
배치, RAM 예산, prefill, KV/작업공간부터 계산해야 합니다. native T2/T3의 배치 재사용과
MTP 상태 복구를 보존하는 원격 stage 프로토콜도 필요합니다. **아직 구현·실측되지 않은 후보입니다.**

상주 고정 층도 가능하지만 그 층의 전체 전문가와 상태가 준비되어야 합니다. 단지 층 번호만
고정하고 매 토큰 RAM 가중치를 읽으면 cut18의 병목을 되풀이합니다. 50 tok/s 달성은 미확정입니다.

## 근거 파일

- [첫 계측 실험](home-engine-50tps-first-20261029.json)
- [세 분할 실험 전체 수치](home-split-engine-measurements-20261029.json)
- [연결 구현과 MTP 상태 계약](engine-50tps-pipeline-design.md)
- 원자료: `/home/caterpillar/Agent/maya-engine-50tps-20261029/`,
  `/home/caterpillar/Agent/maya-split-net-20261029/`,
  `/home/caterpillar/Agent/maya-split-net-copy-20261029/`,
  `/home/caterpillar/Agent/maya-split-net-cpu-20261029/`.
