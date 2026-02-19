# RMAPPO MaxProp++ 구현 체크리스트

원본 명세/메모: `RMAPPO_MAXPROP_IMPL_KR.txt`

## 진행 로그
- 2026-02-12: 체크리스트 생성

## 0) 실험 시나리오(고정)
- [ ] `realistic_event_heavy_energy_43200_maxprop_settings.txt` (baseline)
- [ ] 비교군: `realistic_event_heavy_energy_43200_spraywait_settings.txt`
- [ ] 비교군: `realistic_event_heavy_energy_43200_prophet_settings.txt`

## 1) 액션(파라미터) 스펙 확정 + 클램프
- [ ] `λ_cost` 범위 확정
- [ ] `τ_age` 범위 확정
- [ ] `β_x` 범위 확정
- [ ] `k_x` 범위 확정
- [ ] `m_relay` 범위 확정
- [ ] (옵션) `{aggressive, balanced, conservative}` 이산 모드 지원

## 2) MaxProp++(파라미터 제어) 라우터 구현
- [x] Dijkstra cost에 에너지 리스크 패널티 반영 (`λ_cost`, `τ_age`)
- [x] `avgTransferredBytes(x)` 추정 EWMA 지원 (`β_x`)
- [x] threshold 스케일링 (`k_x`)
- [x] 릴레이 마진 게이팅 (`m_relay`: `Δcost >= m_relay`일 때만 relay)
- [x] 상태/디버깅용 getter 추가: `x_ewma`, `threshold_cur`, `m_relay` 등

## 3) 이웃 정보 공유 최소 확장(에너지-인지 + aging)
- [x] `E_nb` 공유/저장
- [x] `r_nb`(energy/sec) 추정 + 공유/저장
- [x] `t_last` 공유/저장
- [x] staleness `Δt_info` 계산 + `w(Δt)` 적용(aging)

## 4) RMAPPO 브릿지(서버/시뮬레이터 통신) 구현
- [x] state: `[SELF]` 토큰 1개
- [x] state: `[CONTACT]` 토큰 1개(대표 이웃 선택 규칙 포함)
- [x] state: `[MSG_i]` 토큰 32개(정렬/선택 규칙 포함)
- [x] padding/mask 처리(32개 미만)
- [x] step당 모든 호스트 전송/적용(`maxHostsPerStep=0`)
- [x] host-dest 해상도(v2): `host#dest` 키로 `m_relay` 오버라이드(`hostDestMode=true`, `maxDestsPerHost`)
- [x] prev_transition: step 리워드 계산용 카운터(Delivery/Overhead/Energy)
- [x] action: 파라미터 벡터 수신/적용(호스트 단위)
- [x] 학습 업데이트는 에피소드 종료 시점에만 트리거(`mode=episode_end`)
- [x] (임시) `toolkit/drl_server_rmappo_maxprop.py`에 `rmappo_maxprop_v1` 프로토콜 처리(휴리스틱 액션)

## 5) Reward(곱셈형) 구현
- [x] `R = S * (1-O) * (1-E) * (1-D)` 스텝 보상 스켈레톤(서버 측 계산/로그)
- [x] step 단위 `S`(전달성공률), `O`(오버헤드), `E`(에너지소모율), `D`(드랍율) 카운터 전송(브릿지→서버)
- [ ] (튜닝) `E` 정규화(`MP_V1_ENERGY_RATE_NORM`) 시나리오별 보정
- [ ] (옵션) TTL 가중 전달 성공률/힌드사이트 크레딧(creation-step reward) 적용

## 6) CTDE critic 글로벌 피처 확장
- [x] `global_alive_frac`
- [x] `global_energy_frac`
- [x] `global_drop_rate_ema`
- [x] `global_abort_rate_ema`
- [x] `global_relay_rate_ema`
- [ ] actor 입력과 critic 입력 분리(CTDE 유지)

## 7) 평가(논문용) 최소 보고 지표
- [ ] delivery_prob (전체/phase별)
- [ ] overhead_ratio (전체/phase별)
- [ ] 차량 그룹 생존(energy 0 도달 시각 분포)
- [ ] aborted/dropped/relayed 추이(phase별)
- [ ] (권장) Phase C delivery 유지 여부
- [ ] (권장) trade-off 곡선(오버헤드 감소 vs delivery 유지)
