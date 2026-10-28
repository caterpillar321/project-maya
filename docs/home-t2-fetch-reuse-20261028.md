# 집 GLM: fixed1 MTP의 T2 가중치 재사용 — 2026-10-28

## 결과와 다음 단계

초안 하나를 검증하는 두 행(T2)에도 private scratch의 가중치 재사용을 확장했습니다.
같은 binary에서 OFF/ON을 비교한 결과, 채팅 38.80 → 39.00 tok/s(+0.52%),
코드 33.75 → 34.00 tok/s(+0.74%)였습니다. 가중치 복사 감소는 확인했지만,
두 회씩의 순차 측정으로 1% 미만의 속도 차이가 반복될 것이라고 단정하지 않습니다.

이번에 승인받은 추가 소프트웨어 실험은 여기까지 완료했습니다. 다음 비교 대상은
사용자께서 이미 보유한 265K + Z890 AORUS ELITE입니다. 물리 교체는 아직 하지
않았습니다. 이 변경만으로 50 tok/s에 도달하지 못했으며, CPU 교체의 효과도
측정 전에는 알 수 없습니다.

**17:41:22 KST에 기존 운영 서버를 복원했습니다.** 17:45에 `/health` 정상,
기존 engine PID 306379 / PGID 306367 및 운영 환경 변수를 확인했습니다.
17:46에는 코하네 helper 18800 PID 5151 / 18801 PID 11615의 실제 argv를
확인했습니다. 둘 다 기존 binary이며, CPU helper는 54GiB/6스레드입니다.
두 PC는 켜져 있습니다. 아래 PID는 기록일 뿐이며 제어 전 실제 argv를 다시 확인합니다.

## 속도와 CPU

| 항목 | 재사용 OFF | 재사용 ON |
|---|---:|---:|
| 채팅 tok/s, 1 / 2회 | 38.8 / 38.8 | 39.1 / 38.9 |
| 코드 tok/s, 1 / 2회 | 34.1 / 33.4 | 34.3 / 33.7 |
| 채팅 평균 tok/s | 38.80 | 39.00 |
| 코드 평균 tok/s | 33.75 | 34.00 |
| 긴 입력 첫 프리필 | 12.7140 s | 12.7287 s |
| 같은 요청 디코드 | 33.3 tok/s | 33.5 tok/s |
| 긴 입력 prefix 반복 디코드 | 34.5 tok/s | 34.7 tok/s |
| qshort 전체 wall 평균 | 57.029 s | 56.781 s |
| 메인 engine CPU 평균 | 175.30% | 175.01% |
| 코하네 CPU/RAM helper CPU 평균 | 459.15% | 458.50% |

tok/s 평균은 원래 하네스가 소수 한 자리로 보고한 회차별 값의 산술평균입니다.
CPU는 한 코어 100% 기준이며 qshort 전체 구간의 프리필과 대기를 포함합니다.
CPU 부하가 의미 있게 줄었다고 볼 결과는 아닙니다. OFF 두 회 다음 ON 두 회를
측정했으며 무작위 교차 순서나 온도·클록 통제 실험은 아닙니다.

## 실제로 줄인 일과 남은 시간

| 디코드 계측 | 채팅 8요청 | 코드 8요청 | 전체 20요청 |
|---|---:|---:|---:|
| mapped fetch + miss payload 감소 | 2.23% | 2.87% | 2.72% |
| D2H payload 변화 | 0% | 0% | 0% |
| MTP 채택률, OFF = ON | 85.75% | 89.03% | 88.02% |
| verify host ms/창, OFF | 46.989 | 55.166 | 51.369 |
| verify host ms/창, ON | 46.759 | 54.769 | 51.055 |

전체에서 5,304회, 70,911,000,576 bytes의 expert 복사를 회피했습니다.
이는 논리 expert payload 계측이며 전체 PCIe 트래픽이나 노출된 전송 지연의
실측값은 아닙니다. 같은 pointer를 공유하면서 기존 expert grouping도 달라지므로
복사 회피와 그룹 연산 변화의 효과가 함께 포함됩니다.

T2에서 재사용 가능한 복사는 전체 fetch+miss의 약 2.7%였습니다. 이전 T3 실험의
약 8.1%보다 작습니다. 초안 하나의 채택률은 높지만 이 변경으로 줄일 수 있는
작업량 자체가 작았습니다. 초안 생성은 여전히 약 1.1 ms/round이고 verify는
약 47~55 ms/창입니다. 절약한 verify 시간은 약 0.23~0.40 ms/창입니다.
이번 수치만으로 verify 중 CPU, GPU 연산, 통신 중 하나를 단독 병목으로
확정할 수는 없습니다.

이번 ON 평균을 50 tok/s로 높이려면 토큰당 시간을 채팅 약 22%, 코드 약 32%
더 줄여야 합니다. 이전 fixed2 코드 평균 35.2 tok/s를 기준으로 해도 약 30%가
필요합니다. 작은 재사용 최적화만으로 메울 수 있는 차이는 아닙니다.

## 정확성과 계측 검증

- 집 5090에서 실제 CUDA route/wait/fetch 및 grouped expert 단위 검사 PASS.
  T2는 복사 16 → 12회, T3는 24 → 17회인 fixture를 확인했습니다.
- 실제 두 행과 세 행 크기로 각각 할당했습니다. CPU/helper/disk 경로,
  promotion 정책, 잘못된 창 크기·expert ID·이전 layer plan,
  현재/미래 행 pointer와 중단된 route의 guard를 검사했습니다.
- gate/up IQ4_XS, mixed up Q4_K, mixed up Q5_K의 세 형식에서 down IQ4_XS를
  사용했습니다. T2의 1,024개 × 3 및 T3의 1,536개 × 3, 총 7,680개 float 출력이
  OFF/ON bitwise 일치했습니다.
- 하네스는 조건당 qshort 18요청 + 긴 입력 2요청, 총 40요청을 완료했습니다.
  모든 20개 요청 쌍에서 rounds, emitted tokens, accepted/proposed drafts,
  window/row/attempt 수, route 수 및 복사 외 tier payload가 일치했습니다.
- 조건당 T2 검증 창 2,053개, scratch replay 0, route 누락 및 pending 0입니다.
  모든 요청에서 `OFF RAM fetch = ON RAM fetch + reuse`가 횟수와 bytes 모두
  정확히 성립했습니다. `emitted = 요청 수 + rounds + accepted`도 성립합니다.
- greedy 해시는 OFF/ON의 각 회차 쌍이 같습니다. 두 회차 사이의 해시는 다르며,
  전체 출력이나 모든 expert ID를 비교한 것은 아닙니다.
- 분석기의 warning/error는 없고, 양쪽 시작 pool/RAM/cold/copy 수량이 같습니다.

## 구현 범위와 재현 조건

`MAYA_GLM_WIN_FETCH_REUSE=1`은 이제 private scratch 2 또는 3행을 허용합니다.
실제 full T2/T3 window, stagger0, one trunk, batched 실행에서만 활성화합니다.
같은 layer의 앞선 행이 같은 CUDA stream에서 fetch를 완료한 private scratch만
공유하며 CPU/helper/promotion 선택 정책과 연산 정밀도는 그대로입니다.
기본값은 OFF입니다.

측정은 두 조건 모두 **fixed1 MTP / scratch 용량 3 / 실제 창 T2**입니다.
이전 fixed1 기준과 메모리 배치를 맞추기 위해 scratch 용량까지 동시에 바꾸지
않았습니다. 용량 2 자체의 모델 전체 성능은 이번에 측정하지 않았습니다.

- 같은 IQ4_XS pack, INT8 KV, context 262144, prefill 32768.
- `WIN_STAGGER=0`, `WIN_BATCH=1`, `MV_TILED=1`, `MV_MMA=0`, `MV_NC3=1`.
- `RAM_GB=92`, RAM backing OFF, TIER_AUDIT/WIN_AUDIT/SPEC_PROF ON.
- copy-aware/profile 유지: cost 0.16 ms, RTT 0.35 ms / shared 0.10 ms.
- CPU helper 54GiB/6스레드. `HELPERD_QUEUE_PROBE=0`.
- 시작 pool 1,466 slots/18.46GiB, main RAM 7,302 slots/91.95GiB,
  remote cold 788개/9.88GiB, optional copies 3,543개/47.37 decimal GB.
- 두 조건의 engine flag 차이는 `WIN_FETCH_REUSE`뿐입니다.
- Claude의 `qshort.py`와 긴 입력 하네스는 수정하지 않았습니다.
- B200의 GPU 및 idle keeper는 사용하거나 중단하지 않았습니다.

| 파일 | SHA256 |
|---|---|
| test engine `maya-t2-reuse-20261028` | `8f9c377fcbf067d4b9ed0365fe4e9c3f7d04ff19eb1e86e3221df156859ba0e6` |
| CUDA unit binary | `a43784a04a6013bf8bab007b724ca10bdef574f8df6a9f557b55be62bd7a4e55` |
| Claude `qshort.py` | `278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64` |
| copy demand profile | `654d083ac0bd196307a4a2b71f5aa753f91eb9bfb44a7b400f5514da30dbbea7` |

메인 실행 폴더는 `~/fn/codex-t2-reuse-20261028`, wrapper는
`~/fn/bench_t2_reuse.py`, plan은 `~/fn/t2-reuse-plan.json`입니다.
로컬 원본·분석·binary·환경·복원 파일은
`/home/caterpillar/Agent/maya-t2-reuse-20261028`에 보존했습니다.
[비교 JSON](home-t2-fetch-reuse-measurements-20261028.json),
[이전 MTP/helper 측정](home-mtp-helper-20261028.md)도 함께 참조합니다.

## 265K 교체 비교 기준

교체 전 코하네는 Ryzen 9600, 64GB/5200 MT/s, 4070 Ti SUPER + 5060 Ti입니다.
메인은 9950X3D2, 128GB/4800 MT/s, 5090 + PRO 4000입니다.
CPU, OS, GPU UUID·PCI bus·driver·전력 제한, governor, binary/profile hash는
로컬 `main-before-swap.txt`, `kohane-before-swap.txt`에 저장했습니다.
PCIe current generation은 순간값이며, 특히 코하네의 idle Gen1을 추론 중 링크
속도로 해석하면 안 됩니다. 이 기록을 위해 SPD 조회나 RAM 부하를 반복하지 않았습니다.

교체 후에는 다음 순서로 비교합니다.

1. 같은 RAM을 우선 5200 MT/s로 두고 GPU 역할·model 경로·네트워크를 확인합니다.
   18800은 4070 Ti SUPER, 18801의 RAM/CPU helper는 5060 Ti입니다.
   보드 교체로 PCI 번호나 GPU index가 바뀔 수 있으므로 저장한 UUID로 확인합니다.
2. 기존 helper binary의 CPU 호환성과 실제 실행 경로를 확인합니다. 재빌드가
   필요하면 별도 변경으로 기록하고 같은 quantization과 작업 배치를 유지합니다.
3. 같은 T2 test engine/plan/하네스로 먼저 P코어 6개, 그다음 P코어 8개,
   마지막으로 P+E를 비교합니다. 실제 CPU topology에서 P코어 번호를 확인한 뒤
   affinity를 설정하며 번호를 가정하지 않습니다. RAM helper 예산은 54GiB입니다.
4. 각 조건의 tok/s뿐 아니라 시작 expert 배치, MTP 채택률, 전송량, CPU 시간을
   함께 비교합니다. PCIe 배치나 링크 변화가 있으면 CPU만의 효과로 해석하지 않습니다.

운영 복원 경로는 메인의 `MAYA_NET=1 ~/fn/start-maya-kohane5200.sh`,
config는 `~/fn/maya-config/iq4-kohane5200.json`입니다. 이 launcher는 helper를
6스레드/54GiB로 맞춥니다. P코어 8개 등의 실험은 별도 wrapper에서 수행해야 합니다.
운영 binary는 기존 `maya-copy-profile-cc80982`이며, 이번 실험 설정을 기본값으로
승격하지 않았습니다. 보드 교체로 네트워크 주소나 GPU index가 바뀌면 launcher도
실제 장치에 맞춰야 합니다. 옛 86GiB systemd wrapper는 사용하지 않습니다.
