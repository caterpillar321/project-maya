# 50 tok/s 작업 재개 기록 — 2026-10-29

**13:39 KST에 검증한 요청 선발송 엔진을 운영에 반영하고 실제 생성을 확인했습니다.** 50 tok/s는 아직 달성하지 못했습니다. 동일 엔진 A/B 두 회 평균은 chat 38.90→39.35, code 33.90→34.30 tok/s였습니다. 뒤 대조에서도 ON 39.3/34.7을 확인했습니다. 약 1.2% 개선이며 장기 반복 검증 결과는 아닙니다.

메인 engine CPU 사용량은 qshort 전 구간 평균 약174.82→163.39%(한 코어=100%), 원격 CPU helper는 약459%로 비슷했습니다. 전력이나 전체 시스템 효율을 측정한 수치는 아닙니다.

## 최종 운영 상태

- 메인 API 18100, 모델 로드 및 응답 정상.
- 설정: 메인 `~/fn/maya-config/iq4-kohane5200.json`
- 시작: `MAYA_NET=1 ~/fn/start-maya-kohane5200.sh`
- 엔진: `~/fn/bin-test/maya-early-route-20261029`
- SHA256: `b59380b34f7311c8225f6e93430327d132f634c274b3dc40a13208acfec4b2d9`
- MTP fixed1, scratch3, fetch reuse ON, **route ahead ON**.
- NET_COPY_MS=0.16, RTT=0.35, shared RTT=0.10 유지.
- 원래 9600 CPU helper 6스레드/54GiB, 두 GPU helper 바이너리 유지.
- 양자화, CPU/메모리 클럭 및 전압 변경 없음.
- 원본 설정 백업: `~/fn/codex-route-production-20261029/original-config.json`.
- 운영 확인: 32토큰 생성, drafts16/accepted15, 실제 T2 route-ahead 활성 로그 확인. 이 짧은 응답의 39.2 tok/s는 별도 성능 평균으로 사용하지 않습니다.

실험은 종료됐고 운영 서버가 켜져 있습니다. 자동으로 다음 최적화가 진행되는 상태는 아닙니다.

## 이번 실험에서 남긴 것

| 실험 | 결과 | 결정 |
|---|---|---|
| scratch3→2 | chat 39.0→38.0, code34.2 동일 | 제외 |
| T2 helper 요청 선발송 | chat38.90→39.35, code33.90→34.30 | 운영 반영 |
| CPU 비용 .16→.20 | 36.65/33.35 | 제외 |
| CPU 비용 .16→.24 | 35.55/31.60 | 제외 |
| RTT .35→.20 | 39.20/34.50, 긴 입력 일관 개선 없음 | 기존 유지 |

143개 모델 하네스 요청과 운영 smoke1회를 완료했습니다. 새 코드의 CUDA T2/T3 단위검사는 0 failures이며 RAM/disk 임시 버퍼 수명, promotion/reuse 및 세 가중치 조합의 grouped arithmetic bitwise 비교를 포함합니다. 새 코드 OFF/ON의 20개 대응 요청에서 route/tier payload/MTP 승인 수가 같고 greedy 해시도 같습니다. 전체 모델의 모든 float 출력을 비교했다는 뜻은 아닙니다.

B650의 현재 부팅 커널 로그에서 PCIe Bus Error, AER error, NVIDIA Xid를 찾지 못했습니다. AER 카운터를 0으로 측정했다는 뜻은 아닙니다. 로컬 B200 GPU와 idle keeper는 건드리지 않았습니다.

## 다음 판단에 필요한 근거

헬퍼 작업을 덜 보내는 비용 .20/.24는 메인 RAM→GPU fetch를 늘리고 성능을 떨어뜨렸으며, .16 복귀 시 원래 속도가 돌아왔습니다. 더 적극적인 작은 RTT 조정도 복사량 감소만큼 속도가 개선되지는 않았습니다. 단순 분배 상수만으로 남은 약21%/31%의 토큰당 시간 차이를 메울 근거는 없습니다.

초기 helper 배치는 USAGE가 없으면 앞쪽 층에 집중되며 자동 local swap도 같은 층 안에서만 움직입니다. 다만 전층 인기 전문가로 단순 재배치한 offline replay는 held-out chat을 악화시켰습니다. 보수적인 broad-profile 재배치는 generic miss를 약1~6% 줄였지만 local 호출 층과 RAM 중복/원격 cold가 늘 수 있습니다. 실측 속도 예측으로 사용하지 않았으며 적용하지 않았습니다. 후속 배치 변경은 원격 소유권, main warm, unique 저장 용량, 새 호출 비용까지 함께 보존·계측해야 합니다.

## 재현 자료

- [요청 선발송 설계 및 측정](home-window-route-ahead-20261029.md)
- [scratch 용량 비교](home-scratch-capacity-20261029.md)
- [CPU/호출 비용 비교](home-mtp-routing-cost-20261029.md)
- [전체 요약 JSON](home-50tps-progress-20261029.json)
- 로컬 `maya-early-route-20261029/`: 엔진/단위검사, build log, plan, wrapper, comparison.json, 실제 운영 설정과 smoke.
- 로컬 `maya-scratch2-20261029/`, `maya-mtp-copycost-20261029/`, `maya-mtp-rtt-20261029/`: 각 실험 원본 로그와 결과.
- 로컬 `maya-placement-next-20261029/`: 배치 replay 스크립트, 결과 및 가정.

소스 구현 커밋은 `7572f4a`입니다. 실험은 `codex/ram-backing-probe` 브랜치에 기록했습니다.
