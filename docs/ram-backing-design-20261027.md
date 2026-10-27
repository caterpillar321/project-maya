# 선택적 RAM 원본 보존: 용량·시작 배치 설계

2026-10-27, `codex/ram-backing-probe`의 설계 검토입니다. 이 문서 작성에서는
runtime 소스·홈 설정·프로세스를 변경하지 않았습니다. 기존 IQ4_XS/INT8 KV,
실제 top-8, helper의 기존 FP32 응답을 유지하는 범위입니다.

## 권장 첫 단계

첫 probe는 **기존 RAM arena에서 primary GPU의 유효한 복사본만 선택적으로
유지하는 방식**이 적절합니다. 1–4 GiB는 반드시 채우는 목표가 아니라 상한이어야 합니다.
이 probe에서 로컬 helper의 RAM 원본, arena 크기, helper 소유권은 건드리지 않습니다.
실제 free 슬롯과 primary GPU 중복을 우선 사용하고, RAM의 유일한 사본을 더 내보내야
한다면 보존을 포기해야 합니다.

약15 GiB의 로컬 helper 중복을 후반부 backing으로 전환하는 방안은 다음 단계입니다.
용량은 충분하지만, prefill·local helper 교체·RAM class 배분을 함께 바꿔야 합니다.
단순히 원본을 지우고 다른 class로 쓰는 안전한 한 줄 변경은 없습니다.

## 구현 담당자가 선택한 첫 대조: reserve를 같게 두고 KEEP만 바꿉니다

첫 구현안은 `MAYA_GLM_RAM_BACKING_MIB`로 late class의 초기 여유를 확보하고,
KEEP0/1로 원본 유지와 D2H reuse만 비교하는 것입니다. 이 방식은 두 arm이 같은
physical RAM budget과 같은 초기 main/remote 배치를 가진다는 조건에서,
원본 유지 자체의 효과를 분리하기 좋습니다. 전면15-GiB class 재배분은 분리합니다.

기존 총 RAM budget에서1–2 GiB를 reserve하면 그만큼 main RAM의 초기 unique
warm 수가 줄어듭니다. Cold share가 이를 remote mandatory RAM으로 옮기면 전체
coverage는 유지할 수 있지만, optional remote copy 예산과 소유자가 바뀝니다.
따라서 다음 두 비교를 구분해야 합니다.

- **동일 reserve의 KEEP0 대 KEEP1:** 원본 보존/퇴출 재사용의 인과적 대조입니다.
- **현재 production 대 새 reserve 구성:** 초기 cold/copy 배치 변화와 RAM 예산의
  비용까지 포함한 실제 사용자 이득입니다.

메인 headroom을20→18 GiB로 줄여 두 arm에 같은2 GiB를 더 pin하는 대안도 있지만,
현재 `MemAvailable`과 실제 pin 성공치를 확인해야 합니다. 자동 headroom 방식은
시작 시 가용 메모리 변동을 남깁니다. 실험 재현에는 같은 `MAYA_GLM_RAM_GB`를
명시하고 class별 슬롯 수·primary/helper 소유·cold/copy key 집합이 같은지 확인하는
편이 더 명확합니다. 총2 GiB 증분은 class별 가중 배분을 거치므로 late class에
정확히2 GiB가 추가되는 것과 같지는 않습니다.

### 현재 구현을 검토한 첫 probe 조건

구현 담당자는 `MAYA_GLM_RAM_GB=92`, `MAYA_GLM_RAM_BACKING_MIB=1024`,
`FROM=22`, `TO=24`를 두 arm에 동일하게 적용하고 KEEP만0/1로 바꾸는 구성을
선택했습니다. 소스의 `RAM_GB`는 이름과 달리 decimal GB가 아니라 **GiB**입니다
(`std::atof(rg) * 1073741824.0`). Reservation은 새 할당이 아니라 이 arena 안에서
초기 warm을 제외하는 용량입니다. [22,24)의 GPU pool은72 slots이며, 한 slot이
12.75 MiB이므로 요청1024 MiB는 **72 slots, 918 MiB**로 제한됩니다.

아래는 실측이 아닌 **소스 재현 예상치**입니다. 기존 `baseline-slots.json`과
`profile-mix4agent.bin`, `RAM_SLACK=0`, `n_main=n_slots-15`, primary spare3,
local/remote GPU helper1674/2396개가 같고 pinning fallback이 없다는 조건입니다.

| 고정92-GiB arena | 무예약 대조 계산 | 72-slot 예약 계산 |
|---|---:|---:|
| Class slots:14/13/12.75 MiB | 697 / 348 / 6268 | 697 / 348 / 6268 |
| 실제 pinned arena | 91.99121094 GiB | 91.99121094 GiB |
| Primary warm experts | 1394 | 1394 |
| Main RAM 초기 warm experts | 7301 | 7229 |
| Local helper와 겹친 RAM | 1136 | 1135 |
| Net GPU helper RAM 해제/재충전 | 1636 / 1636 | 1617 / 1617 |
| Remote cold share로 넘길 experts | 467 | 538 |
| 해당 cold payload | 5.81469727 GiB | 6.69873047 GiB |
| Class별 true free slots | 4 / 4 / 4 | 4 / 4 / 76 |

Cold가72가 아닌71개 늘어나는 것은 warm의 local-helper 중복이1개 줄어드는 효과가
함께 있기 때문입니다. 따라서 운영 구성과의 차이를 “918 MiB backing만 추가”라고
해석하면 안 됩니다. KEEP0/1 사이에서는 위 배치가 같아야 하며, 시작 로그의 실제
class별 수량·cold/copy 집합을 확인해야 합니다. 자동 RAM budget을 쓴 과거7307,
7299-slot 실측과도 구분합니다.

계산은 class별
`weight = sum((288-(n_slots-15)+3+2)*stride)`,
`n = floor(92*2**30 * weight/sum(weight) / stride)`를 사용합니다. 이어서 기존
profile 순서로 primary warm(`n_slots-3`)과 class round-robin RAM warm을 재현하며,
RAM 종료 조건만 `4+reserve`로 바꿉니다. GPU helper 후보는 primary warm을 제외한
layer/expert 순서로 배정하고, net GPU와 겹친 RAM slots만 원래 코드대로 재충전한
뒤 전체12096 experts에서 coverage union을 뺍니다. 입력 파일과 원형 재현 코드는
[`placement-roofline-audit-20261027.md`](placement-roofline-audit-20261027.md)의
마지막 재현 절에 있습니다.

### Reserve가 소모되는 소스 지점

- `fast_warm()`의 round-robin은 수정본에서 class별 `4+backing_slots`를 남깁니다.
  KEEP0/1 모두 같은 수량을 제외하므로 원본 보존 여부가 초기 warm 배치를 바꾸지 않습니다.
- `fast_helper_setup()`의 net-helper RAM 해제/재충전(기존 약2043–2105행)은
  **이미 net GPU owner가 보유한 kRHold slots만** `freed[class]`에 모으고 그 목록에서만
  Fill합니다. Warm이 비워둔 reserve는 이 목록에 없으므로 현재 구조에서는 직접
  침범하지 않습니다. Reserve를 warm 뒤에 계산하거나 Fill을 전체 free 탐색으로
  바꾸는 경우에는 다시 확인해야 합니다.
- `fast_boundary` step4b RAM_REFILL은 수정본에서
  `nfree-keep_free-max(0,backing_slots-backing_count)`를 사용합니다.
  Step4는 원래 landing-room floor만 강제하고, optional reserve를 다시 만들기 위해
  unique RAM을 더 퇴출하지 않습니다.
- Host-miss landing, AHEAD read, ordinary/background demotion, prefill park도 free slot을
  잡습니다. Unique source 확보와 in-flight copy가 backing보다 우선이며, 단순히
  free 총수만으로 arena의 안전한 여유를 계산해서는 안 됩니다.

**Backing과 free reserve는 같은 예산입니다.** 상한 B slots 중 Q개를 실제 backing으로
점유했다면 남길 backing용 free는 대략 `max(0,B-Q)`이며, 여기에 기존 landing 여유를
더합니다. KEEP1에서도 free floor를 landing+B로 고정하면 backing B와 free B를
이중 예약할 수 있습니다. Pending load/demotion/pin은 별도 accounting이 필요합니다.
이 probe에서 원본을 계속 유지하려고 unique RAM victim까지 늘려서는 안 됩니다.

### 검토 중 보완된 prefill/퇴출 경계

초기 diff에서는 prefill park가 unique RAM을 먼저 밀어낼 수 있었으므로, 수정본은
canonical RAM `kRHold`와 유효한 primary `kResident`가 함께 있는 중복을 먼저
회수합니다. 이미 park한 key는 GPU slot을 빌려 준 뒤에도 RAM이 살아 있어야 하므로
별도로 보호합니다. 아직 park하지 않은 key의 backing을 회수했다가 같은 lending
배치에서 다시 park할 수도 있어, 테이블 갱신은 `upd_gen/upd_at`으로 key별 마지막
값을 병합하도록 수정되었습니다. 이로써 병렬 table-update kernel에 같은 key의
서로 다른 값이 동시에 들어가는 경로를 막습니다.

또한 ordinary D2H 중인 expert를 다른 GPU spare로 재승격하면, demotion retire에서
새 backing이 생길 수 있습니다. 수정본은 이 retire 시점에도 cap을 검사합니다.
Warm 제외·net helper refill·RAM_REFILL의 예약 회계는 소스상 일관됩니다. 이 검토는
정적 검토이며, 반복 prefill 뒤 table/weight 검증과 실제 성능 결과를 대신하지 않습니다.

## 무엇이 실제 용량인가요?

새 plain-profile의 7307-slot 시작 로그를 재현한 값입니다. 후속 plain-tier7299,
prefetch7298 slots와 시점을 섞지 않습니다.

| RAM class | 전체 slots | Local PRO 4000과 중복된 slots | 중복 크기 |
|---|---:|---:|---:|
| 14 MiB | 696 | 692 | 9.46094 GiB |
| 13 MiB | 348 | 344 | 4.36719 GiB |
| 12.75 MiB | 6263 | 99 | 1.23267 GiB |
| 합계 | 7307 | 1135 | **15.06079 GiB** |

14/13-MiB class의 held RAM은 초기 상태에서 모두 local helper 중복이고,
각 class에 남는4개는 warm에서 비워 둔 슬롯입니다. 12.75-MiB class에는
9층의 local helper 중복99개가 있습니다. 후반19–44층은 모두12.75 MiB입니다.

`layer_rc`와 `ram_of[key]`는 현재 층마다 하나의 고정된 RAM class를 가정합니다.
따라서 14/13-MiB arena에서 공간을 비웠다고 후반층 RAM 슬롯이 자동으로 늘지는 않습니다.
바이트를 더 작은 전문가에 담을 수 있다는 사실과, 현재 bookkeeping이 그 위치를
지원한다는 사실은 별개입니다.

또한 로그의 `ram_slots - ram_used`는 실제 free 수가 아닙니다. `fast_stats()`는
`kRHold/kRNew/kRHelper`만 used에 넣습니다. `kRLoad/kRDemote/kRPin/kRRelease`는
여전히 사용 중이거나 예약된 메모리인데 used 합계에서 빠집니다. Probe 시작 시에는
**class별 kRFree 수, pending load/demotion/pin, hold 중 유일 사본/primary 중복/helper 중복**을
직접 분류해야 합니다.

## 중복 우선 회수로 unique coverage를 지킬 수 있나요?

가능합니다. 다만 **중복 우선 회수 자체가 새로운 backing 용량을 만들지는 않습니다.**
현재 primary GPU에 유효한 사본이 있는 RAM A를 회수해 B의 원본에 쓰면, A의 데이터는
GPU에 남아 있으므로 그 순간 유일한 데이터가 사라지지 않습니다. 그러나 기존 RAM/GPU
중복 하나를 다른 중복 또는 RAM-only 데이터로 교환한 것입니다.

고정된 용량에서 `unique coverage = GPU 집합 ∪ RAM 집합 ∪ mandatory helper 집합`입니다.
중복을 늘리려면 실제 빈 공간이나 다른 안전한 중복이 필요합니다. RAM-only 전문가를
victim으로 삼아1–4 GiB를 억지로 채우면 unique coverage가 줄 수 있습니다.
기존 `TIER_GC` 주석에는 중복 누적으로 약12%가 어느 tier에도 없게 되고 disk reads가
늘었던 사례가 있습니다. 새로운 보존 정책은 이 회귀를 피해야 합니다.

Optional remote `copy_of`는 현재 main RAM이 없어지면 자동 fallback이 되는 소유권이
아닙니다. 그러므로 이 집합을 mandatory helper coverage로 세면 안 됩니다.

### 안전한 회수 대상

첫 probe에서는 다음 조건을 모두 만족하는 RAM 사본만 중복 우선 victim으로 삼습니다.

- RAM 상태가 안정적인 `kRHold`이고 `R.key[rs] == key`, `ram_of[key] == rs`입니다.
- `slot_of[key]`가 실제 존재하는 primary GPU slot을 가리키고, slot key가 같은 expert입니다.
- 해당 GPU slot은 안정적인 `kResident`입니다. `kDraining/kLent`나 pending background
  이동을 단순히 `slot_of >= 0`만으로 유효 사본이라고 간주하지 않습니다.
- Main compute가 끝나고 service metadata도 따라온 boundary에서 회수합니다.
  Service thread가 promotion metadata를 먼저 본 순간에는 GPU fetch가 RAM을 읽는 중일
  수 있으므로, 그 자리에서 원본을 바로 재사용하면 안 됩니다.
- 초기 probe는 integrated helper를 제외하며, local/remote helper의 원본을 회수하지 않습니다.

Backing을 잃은 primary resident는 GPU에 남습니다. 나중에 퇴출될 때 기존 D2H 경로를
사용하면 됩니다. 반대로 **GPU eviction victim 자신의 유효한 RAM 원본이 있으면**
그 RAM을 unique 사본으로 전환하고 GPU slot을 즉시 spare로 반환할 수 있습니다.
이때는 D2H가 필요 없습니다.

## 작은 bounded probe의 구성

권장 범위는 먼저19–44층입니다. 현재 mapped 전송의99.88%가 이 구간이고 모두 같은
12.75-MiB class라 cross-class 위치 변경을 피할 수 있습니다.

| 상한 | 12.75-MiB 전문가 수 | 전체26층에 균등할 경우 |
|---|---:|---:|
| 1 GiB | 80개 | 층당 약3개 |
| 2 GiB | 160개 | 층당 약6개 |
| 4 GiB | 321개 | 층당 약12개 |

기본 layer pool은 대개36 slots입니다.1 GiB로 두 층의 전체72 slots를 보존하면
약0.89648 GiB입니다. 따라서 작은 용량을 모든 층에 얇게 배분하는 방식과,
22/23층처럼 fetch가 큰 두 층에 집중하는 방식을 구분할 가치가 있습니다.
어느 방식이든 **실제로 안전한 RAM 여유가 있을 때만** 이 상한에 접근합니다.

### 필요한 정책 순서

1. 일반 RAM fetch의 기존 원본만 보존합니다. 처음에는 host-miss staging,
   prefetch, background promotion까지 동시에 확대하지 않아도 됩니다.
2. GPU victim에 유효한 backing이 있으면 D2H 없이 퇴출합니다. Backing은 이제
   unique RAM이므로 duplicate budget에서 빠집니다.
3. 다른 GPU victim의 D2H에 RAM이 필요하면 `kRFree`, 그다음 primary GPU 중복을
   우선 사용합니다. Unique RAM을 내보내면서 backing을 지키지는 않습니다.
4. Class별 landing 여유, 기존 RAM_REFILL과 in-flight 요청이 먼저입니다.
   Unique coverage를 복원할 free 공간이 필요한 경우도 backing보다 우선해야 합니다.
5. 상한 초과 또는 압력이 생기면 backing을 버리고 기존 exclusive 동작으로 돌아갑니다.
   Backing이 적게 생기거나 거의 남지 않는 결과도 정상적인 probe 결과입니다.

초기 GPU의 hot resident는 오래 머물러 backing을 가지고 있어도 D2H를 거의 절약하지
않을 수 있습니다. 반대로 곧 퇴출될 GPU LRU 쪽 backing이 살아 있으면 효용이 큽니다.
RAM 중복을 보통의 RAM-LRU로만 회수하면, 오래 보존한 원본을 GPU eviction 직전에
버리는 역효과가 있을 수 있습니다. 첫 구현은 일관된 간단한 정책으로 correctness를
확인하되, **보존 기간·회수 뒤 GPU 퇴출까지 시간·backed eviction 수**를 기록해야 합니다.

### 프리필과 helper adapt를 보존하는 최소 경계

가장 단순한 실험 격리는 `lend_tail` 진입 시 기존 DMA/service를 완전히 retire한 뒤,
primary GPU에 여전히 있는 bounded backing만 해제하여 기존 exclusive 상태로
되돌리고 원래 lending을 실행하는 것입니다. 추가 byte 복사 없이 metadata만 정리할
수 있습니다. 이 경우 짧은 새 요청마다 backing이 다시 warm되는 비용까지 포함됩니다.
Weight cache를 유지한 반복 decode와 새 prefill 뒤 decode를 모두 측정해야 합니다.

더 완성된 방식은 `lend_tail::park`가 이미 유효한 backing을 발견하면 D2H 없이
그 원본을 쓰도록 하는 것입니다. 이때 `parked` 보호, 한 key당 table update,
GPU tail→main relocation, RAM victim 선택을 모두 다뤄야 합니다. 단순 probe의
첫 성능 결론을 내기 전에 꼭 동시에 구현해야 하는 범위는 아닙니다.

Local helper 원본을 건드리지 않으면 기존 `fast_helper_adapt`의 소유권 전환을
유지하기 쉽습니다. 이 함수는 primary에 있는 key를 candidate에서 제외하고,
helper 교체도 같은 층 안에서 수행합니다. Primary에서 backing을 가진 채 퇴출된
key는 정상 RAM-only 항목이 되어 기존 incoming helper source로 사용할 수 있습니다.
Backing의 식별은 GPU slot 번호만이 아니라 **key와 유효한 RAM 위치**를 기준으로
해야 slot 재사용에 안전합니다.

## 약15 GiB 재배정의 최소 아키텍처

이 안은 bounded probe에서 D2H 감소가 실제 지연 개선으로 이어진 뒤 진행하는 편이
좋습니다. 후반 GPU 풀 전체11.71655 GiB를 보존할 원시 용량은 확보할 수 있지만,
현재 단일 `layer_rc` 체계에서 완전히 작은 변경은 아닙니다.

### 안 A: 시작 시 helper 소유권을 먼저 확정하고 RAM class를 재배분

권장하는 장기 경로입니다.

1. Primary pool/profile로 초기 resident를 확정합니다. Discrete local helper의
   capacity/ownership을 먼저 계획하거나 reserve하여 기존1674개 배치를 유지합니다.
2. Local helper는 자기 GPU에 가중치를 적재하고 완료를 확인합니다. Main RAM을
   모든 local owner의 영구 원본으로 배정하는 전제를 제거합니다.
3. 초기 RAM warm 후보에서 local GPU owners를 제외하고, 남은 class별 unique
   수요와 명시적인 late backing 예산으로 arena를 배분합니다.
4. Mandatory remote cold set과 optional copy profile도 초기 ownership 계획에 맞춰
   확정합니다. 빈 RAM을 전부 cold unique expert로 채우면 backing 예산이 사라지므로,
   baseline의 remote cold 소유권을 유지할지 별도 정책으로 정해야 합니다.
5. 기존 `R.base + slot * R.stride`, `layer_rc` 가정을 유지하면서12.75-MiB arena를
   늘립니다. 이 점에서 runtime cross-class relocation보다 구조가 단순합니다.

현재 `GlmExpertHelper::open`은 discrete GPU일 때 host-region mapping을 만들지 않고,
`reserve()`는 helper의 free VRAM에서512MiB를 남깁니다. 따라서 primary RAM allocation
이전에 discrete helper capacity를 계획하는 것은 원리상 가능하지만, 초기화 순서와
실패 정리 경로의 변경이 필요합니다. Helpers가 계획한 용량을 실제로 확보했는지도
반드시 확인해야 합니다.

재시작 시 실제 예산은 가용 메모리에 따라 변하므로15.06 GiB를 하드코딩하지 않습니다.
계획한 각 class의 unique expert 수, backing 수, 실제 pin 바이트를 시작 로그에 남깁니다.

### 안 B: 기존 arena를 적재 뒤 재활용하여 별도 backing pool로 사용

기존14/13-MiB arena의 local-helper 사본을 비우고 그 메모리를 late backing에
쓰는 방식입니다. 물리적인 pinned allocation을 재활용하면 일시적으로15 GiB를
추가 pin할 필요는 줄어듭니다. 하지만 위치 표현이 복잡해집니다.

- 지금 `ram_of[key]`는 해당 층의 `layer_rc` 안의 index입니다. 다른 class의 큰 슬롯에
  late key를 넣고 rtab 포인터만 바꾸면 service·CPU lane·background·lending이 잘못된
  주소를 계산합니다.
- 별도 backing은 GPU resident 동안만 숨겨 두어도, GPU를 퇴출한 순간 normal RAM-only
  source가 되어야 합니다. 그 전환을 위해 class+slot의 per-key locator 또는 segmented
  RAM arena를 여러 경로에 도입해야 합니다.
- 원래14/13-MiB 슬롯을 그대로 쓰면 정렬은 맞지만1.25/0.25MiB씩 여유를 낭비합니다.
  더 촘촘히 재단하려면 기존 class free list와 lifecycle까지 바꿔야 합니다.

따라서 모든 runtime 주소 경로를 바꾸는 안 B보다, 시작 시 재배분하여 기존 class
가정을 유지하는 안 A가 검증 범위를 줄이기 쉽습니다. 기존 큰 arena를 일단 free하고
새 arena를 pin하는 변형은 peak 메모리는 낮출 수 있지만, 그 순간 helper의 유일한
원본 수명과 실패 시 복구가 명확해야 합니다.

### 두 안에 공통으로 필요한 prefill/adapt 변경

- Prefill은 현재 local helper owners의 main RAM 복제를 읽습니다. `net_pf`는
  `net_of`를 처리할 뿐 local helper를 자동 호출하지 않습니다. 기존
  `compute_rows/compute_tok/compute_tok_mmq`를 local prefill에 연결하고, 필요한 scratch와
  결과 정밀도를 검증해야 합니다. 단순히 없어진 원본을 SSD에서 읽으면 decode 개선과
  별개로 TTFT가 악화할 수 있습니다.
- Local helper `replace(out,in)`은 기존 GPU slot을 incoming으로 덮습니다. Out의 main
  RAM 사본도 없앴다면 **덮기 전에 out의 새 backing을 확보**해야 합니다. 같은 layer의
  incoming RAM slot을 그대로 overwrite하면 incoming H2D가 끝나기 전 source를 망가뜨릴 수
  있습니다. 순서와 별도 landing slot을 보장해야 합니다.
- 최소 검증에서는 adaptive swap을 잠시 제한하는 별도 실험을 할 수는 있지만, 그것을
  production과 동일한 경로라고 주장하면 안 됩니다. 최종 구현에는 ownership transfer가
  필요합니다.
- Lookahead의 읽기 제외 조건에도 주의해야 합니다. 현재 `fast_ahead_route`는
  `net_of`를 제외하지만 local `helper.owns()`를 별도로 검사하지 않습니다. Local RAM을
  없애는 확장에서는 불필요한 SSD 읽기가 생기지 않도록 auxiliary ownership 전체를 다뤄야 합니다.

## 실험과 채택 기준

처음에는 현재 plain decode, prefetch off, MTP off를 기준으로 하는 편이 원인을
분리하기 쉽습니다. 다음 순서는 구현을 검토하기 위한 제안이며 실행하지 않았습니다.

1. 동일 pinned RAM bytes/class/초기 ownership인 baseline과 bounded probe를 비교합니다.
   작은 layer subset 또는 작은 상한부터 시작합니다. 보존할 여유가 없으면 성공적으로
   fallback하는지 먼저 확인합니다.
2. 짧은 기존 harness를 반복하고, 긴 첫 prefill→decode→동일 prefix 반복을 비교합니다.
   작은 forced plan이나 고정 experts에서 weight blob/output을 확인해, 기존 plain끼리도
   달라지는 전체 greedy hash만으로 correctness를 판단하지 않습니다.
3. `backing_peak_bytes`, class별 true free 수, duplicate 회수, backed eviction/절약 D2H,
   ordinary D2H, host-resolved miss, 실제 reader 호출, uncovered key 수를 남깁니다.
   RAM_REFILL이 복원해야 하는 unique key를 backing 때문에 놓치지 않는지도 확인합니다.
4. D2H가 줄어도 H2D·helper wait·SSD read가 늘면 tok/s와 TTFT로 판단합니다.
   가능한 경우 GPU pending drain 수와 실제 event wait를 같이 측정합니다.
5. 효과가 있으면 큰 용량 재배정과 MTP를 각각 별도 단계로 연결합니다. MTP의 window
   admission은 ordinary decode와 달라, plain의 backing 이득을 MTP 개선에 단순 합산하지 않습니다.

채택의 필수 조건은 양자화/실제 expert 선택을 유지하면서 중복·유일 사본·진행 중인
복사를 구분하고, 반복 prefill/lending/helper 교체에서 소유권이 깨지지 않는 것입니다.
성능상으로는 단순 D2H byte 감소를 넘어 실제 출력 지연이 줄고 TTFT/reader 비용이
불리하게 바뀌지 않는 근거가 필요합니다.

## 첫 구현의 CPU 관리 비용과 후속 최적화

이 절은 소스의 반복 횟수 분석이며, CPU 실행 시간 실측은 아닙니다. Plain decode의
`fast_token()`은 한 forward마다 `fast_boundary()`를 한 번 호출합니다. Prefill lending과
MTP window에는 별도 경계가 있으므로 emitted token 수와 무조건 같다고 보면 안 됩니다.

현재 `[22,24)`, KEEP1에서는 `backing_count()`가12.75-MiB class의 **6268 RAM slots**를
훑습니다. 나머지 class는 reserve0, KEEP0은 전체 class에서 즉시 반환합니다.

| 호출 위치 | KEEP1의 전체 RAM scan 빈도 |
|---|---|
| Boundary step1, promotion 원본 보존 | Boundary당1회이며, 각 promotion은 로컬 `backed`만 증가시킵니다. |
| Step4b RAM_REFILL | 활성화되어 있으면 boundary당1회입니다. |
| Step2 D2H retire | 대상층에서 재승격되어 primary resident이고 canonical RAM 위치도 맞는 retire마다1회입니다. |
| Step5 background promotion retire | 대상층의 canonical `kRPin` 원본을 보존할 때마다1회입니다. |
| Request 종료 TIER_AUDIT snapshot | Request당1회입니다. |
| `RAM_BACKING_VERIFY>0` | 설정한 횟수의 boundary마다 추가1회이며, 별도 table 검사도 실행합니다. |

따라서 일반 설정의 기본 비용은 **2×6268=12,536 slot 검사/forward**이고, 재승격된
demotion/background retire가 더해집니다. 일반 route promotion마다6268개를 새로 세는
구조는 아닙니다. 두 층에서 ordinary demotion을 새로 발행할 수 있는 양은 평상시
boundary당 최대6개(층당 spare3개)지만, 이전 DMA가 몰려 retire할 수 있으므로 그 숫자를
특정 boundary의 모든 추가 scan 상한으로 쓰면 안 됩니다.

`ram_victim()`에도 KEEP1 전용 중복 우선 scan이 추가되었습니다. Free RAM이 없는
ordinary demotion, 또는 landing-room floor를 복원할 때 호출되며, 중복이 없으면
기존 victim scan도 이어집니다. 실제 호출 횟수는 free 상태에 달려 있습니다. Prefill의
duplicate-first park 역시 같은 종류의 추가 탐색이지만 decode 토큰마다 실행되지는 않습니다.

가장 작은 후속 최적화는 **추가 상태 없이 GPU pool에서 역으로 세는 방식**입니다.

- `backing_count`: 선택한 층의 GPU slots만 순회합니다. `[22,24)`에서는 **72개**로,
  현재6268개보다 약87배 적습니다. `kResident`, slot key, `slot_of[key]==slot`,
  해당 RAM class, canonical `ram_of[key]`, RAM key와 `kRHold`를 모두 확인하면
  현재 count와 같은 predicate를 유지합니다. 선택층의288 expert keys씩576개를
  확인하는 대안도 가능하지만, 이 구성에서는 GPU slots를 보는 편이 더 작습니다.
- `ram_victim`의 중복 우선 단계: 현재 predicate는 보존 대상층 밖의 primary 중복도
  포함하므로 **같은 class의 모든 GPU pools**를 봐야 합니다.12.75-MiB class는
  **1304 GPU slots**로,6268개보다 약4.8배 적습니다. 같은 canonical 조건과 기존의
  최대 RAM tick 선택, 동률 시 작은 RAM slot 우선까지 유지해야 정책이 같아집니다.
  중복이 없을 때는 기존 unique victim 탐색을 그대로 실행합니다.

이때 “87배”는 검사 후보 수의 비율이며 토큰 속도 향상률이 아닙니다. Metadata는
작고 cache에 머물 수 있으며, 현재 비용은 아직 별도 계측하지 않았습니다. 과거 다른
배치의 plain profile에서 전체 `boundary_host`는 약0.29–0.38 ms/forward였지만, 이는
sync·event 처리·기존 GC까지 포함한 값입니다. 새로운 KEEP1의 CPU 비용으로 그대로
사용할 수 없습니다.

영구 counter는 count 질의를 O(1)로 만들 수 있으나 더 많은 수명 관리가 필요합니다.
RAM retain/free, backing 재사용 퇴출, GPU 재승격/D2H retire, background 이동,
prefill park/relocation, canonical pointer 교체 모두에서 count가 바뀔 수 있습니다.
어떤 한 경로를 빠뜨리면 cap 또는 refill slack이 틀어집니다. 먼저 GPU-pool 순회로
줄이고, 그래도 필요할 때 경계별 로컬 counter와 명시적인 transition 갱신을 도입하는
편이 검증 범위를 줄입니다. 최적화 시 기존 RAM 전체 scan과 count/victim 결과를
진단 모드에서 대조할 수 있습니다.

현재 VERIFY32 arm은 request마다 최대32 expert blobs를 추가 D2H하여 byte 비교하고,
32 boundary table snapshots도 확인합니다. 이 arm의 처리량은 correctness 진단용이며
성능 판정에서는 제외해야 합니다. 일반 KEEP0/1 비교에서는 wall time, `boundary_host`,
scan 호출/방문 slots, D2H·H2D payload, reader 호출을 함께 보면 관리 비용과 배치·전송
변화를 구분하기 좋습니다. 전체 engine CPU 사용률만으로는 service 대기와 전송 변화가
섞여 count scan의 비용을 분리할 수 없습니다.
