# 집 GLM 작업 재개 메모 — 2026-10-28

사용자께서 00:26 KST에 **즉시 중단·저장 후 메인컴과 코하네컴 종료**, 다음 날 재개를
요청하셨습니다. 자동 재개/기상 예약은 하지 않습니다. 양자화와 하드웨어는 그대로입니다.

## 저장된 코드

- 저장소: `/home/caterpillar/Agent/project-maya`, branch `codex/ram-backing-probe`.
- `d20b7dd`: NC3 Q8_0/Q4_K/Q6_K 커널. 집5090의60개 bitwise 검사 통과.
- `d02d5fa`: 선택적 RAM 원본 보존·prefill/table 처리·진단.
- `cf9aaad`: RAM class별 슬롯 재배정. 기존 pinned bytes 초과 금지,33개 CPU 검증 통과.
- `6271113`: private T3의 같은 층에서 이전 행의 임시 가중치를 재사용합니다.
  GPU route/fetch/guard 및3형식 각각1536 float bitwise 검사 통과입니다.
- 모든 새 동작은 기본 OFF입니다. 운영 launcher/config는 원래 설정 그대로입니다.

## 확정된 측정 결과

| 구성 | 채팅 tok/s | 코드 tok/s | 판단 |
|---|---:|---:|---|
| 고정92GiB 기본 |36.4|31.95|기준,2회 평균|
| 918MiB 예약 KEEP0 → KEEP1 |36.45→36.55|31.50→31.60|D2H8~10% 감소, 유의미한 속도 이득 입증 못함|
| MTP T3 NC3 OFF → ON |36.20→36.95|33.25→33.85|약2% 이득.20요청의 exact routes/bytes/acceptance 일치|
| 8.17GiB 재배정 KEEP0 → KEEP1 |36.80→37.65|31.65→33.15|D2H67~70% 감소. 재배정 전보다 전체 하네스는 약9% 느림|

큰 재배정은 총98,774,810,624bytes로 동일합니다. 슬롯은697/348/6268 →289/144/6924,
후반[19,45) 예약8364MiB입니다. 큰 재배정은 프리필 비용 때문에 기본 승격하지 않습니다.
50tok/s는 아직 달성하지 못했습니다.

## 중단 지점: 다음에는 이것부터

`WIN_FETCH_REUSE`의 실제 ON 성능은 **미측정**입니다. GPU unit만 통과했습니다.
같은 최종 binary의 OFF qshort2회는 끝났습니다(채팅36.8/37.2, 코드34.0/33.8).
그다음16.5K context 도중 중단했으며 ON arm은 시작하지 않았습니다.
다음 날에는 새 실행 폴더에서 OFF/ON을 다시 비교하는 편이 깔끔합니다.

- 최신 binary: 메인컴 `~/fn/bin-test/maya-window-reuse-20261028`.
- SHA256: `6f6dc6c1f11a42bd96c8cfed0943a5e24373b76a8672abf80ff81a4f7e237856`.
- wrapper/plan: 메인컴 `~/fn/bench_fetch_reuse.py`, `~/fn/fetch-reuse-plan.json`.
- 각 arm: MTP fixed2, stagger0, private3, NC3=1, RAM92GiB, TIER_AUDIT1;
  WIN_FETCH_REUSE만0/1. qshort2회+기존context첫/반복입니다.
- 먼저 두PC가 켜지고, 기존 helper2개와 운영 서버가 정상인지 확인하세요.
  운영 시작은 메인컴 `MAYA_NET=1 ~/fn/start-maya-kohane5200.sh`입니다.
- wrapper는 기존 운영 서버 정상 상태를 요구하고 종료/실패 시 운영을 복원합니다.
  기존 RUN 폴더를 덮어쓰지 말고 새로운 이름으로 실행하세요.
- GPU unit: `~/fn/bin-test/glm-window-fetch-reuse-check-20261028`.
  CUDA_VISIBLE_DEVICES=0 및 CUDA13.4 lib 경로로 실행했고0fail이었습니다.
- 새 reuse 로그는 `summarize_probe.py`가 backing과 별도로 집계합니다.
- 확인할 것: H2D 감소, same-window reuse, expert grouping 변화, 채택률/출력,
  재실행0, 실제round/출력tok 비용. pointer공유는H2D와group계산양쪽에영향을줍니다.

## 경로와 기기

- 로컬(B200): `/home/caterpillar/Agent/maya-backing-probe-20261027`에 전체 로그/요약/빌드/테스트 저장.
- 메인컴: `ssh sekai`,192.168.0.4,사용자caterpillar,hostname `Caterpillar-Desktop`.
- 코하네컴: `ssh b650`,192.168.0.76,사용자user,hostname `sekai-pc`.
  현재 native Debian이며4070TiSUPER+5060Ti입니다. WSL 종료 명령을 쓰지 않습니다.
- 홈 원본 로그: `~/fn/codex-backing-20261027`, `~/fn/codex-extended-20261027`,
  `~/fn/codex-fetch-reuse-20261028`. 마지막 status는paused-for-user-shutdown입니다.
- Claude 원본 `~/fn/qshort.py`는 변경하지 않았습니다.
- 코하네 helper 원래설정:18800 구binary20261025c,18801 binary20261026/RAM54GiB/CPU6threads.
  종료 전 PID는2107/2096이었으나 재부팅 후에는 다시확인해야합니다.
- B200의GPU/idle keeper는 이 작업에서 사용하거나 중단하지 않았습니다.

## 추가 설계 메모

MTP helper의 기존FP32 kCompute 요청 중 이미 완전히 도착한 인접1~3행만 묶는
내부microbatch가 다음 후보입니다. 기다리는timeout/FP16프로토콜변경은 피하고,
행별CPU expert 원래순서로 subtotal을 만든 뒤 GPU subtotal에한번더해야합니다.
CPU GU를먼저시작해GPU와overlap을유지해야합니다. 아직구현하지않았습니다.

## 실제 종료 처리 결과

- 메인컴: `sudo -n systemctl poweroff`가 exit0으로 접수되었습니다.
- 코하네컴: 일반 `systemctl poweroff`는 Interactive authentication required,
  sudo 종료 명령은 password required로 거절되었습니다. 원격 root key도 허용되지 않습니다.
  따라서 **PC 전원 종료는 미완료**이며 사용자께 현지 `sudo poweroff`가 필요하다고 안내했습니다.
- 코하네의 이 작업 helper PID2096/2107은 확인 후 SIGTERM으로 중단했습니다.
- 재개 시 코하네 전원 상태 및 helper 기동부터 다시 확인해야 합니다.
