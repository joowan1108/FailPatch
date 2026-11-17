# FailPatch on RoboCasa365 — RLDX-1 / Xiaomi-Robotics-1

논문 *FailPatch: Failure Residual Patching for Vision-Language-Action Models* (arXiv 2609.34175)을
두 개의 RoboCasa365 체크포인트에 구현한 코드입니다. 공식 코드는 아직 공개되지 않아 논문과 부록만 보고 구현했습니다.

| 모델 | 체크포인트 | 보정 모듈을 끼운 위치 (액션 은닉 표현, 1024차원) |
|---|---|---|
| `xr1` | `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` | DiT 출력 → `action_output_layer` 직전 |
| `rldx` | `RLWRLD/RLDX-1-FT-RC365` | MSAT 출력 → `action_decoder` 직전 |

태스크: `CoffeeSetupMug`, `GetToastedBread`, `SteamInMicrowave` (모델마다 세 태스크를 함께 학습).

## 방법

VLA 는 전부 얼리고, 액션 헤드 직전에 **Null-gated Residual Expert Bank** 만 학습합니다
(전문가 8개 + "그대로 두기" 경로, 학습 파라미터 1,731만 개 — 논문과 같은 수).

1. **배치 버퍼**: 원본 정책을 굴려 성공 50 / 실패 50 롤아웃을 모으고, 청크마다 액션 은닉 표현을 저장
2. **오프라인 주석**: VLM 이 실패 롤아웃의 실패 시작 청크와, 성공 롤아웃의 신뢰 여부를 판정
3. **학습 (SFT)**: 사람 시연으로 보정 방향을 배우고(액션 지도), 버퍼로 언제 개입할지를 배움
   - Preserve: 정상 구간에서는 Null 경로를 고르고 보정 크기를 억제
   - Redirect: 실패 구간의 표현을 가까운 성공 표현 20개의 중앙값(성공 프로토타입) 쪽으로 이동
   - Trust: 개입 크기 제한
4. **추론**: 청크마다 라우터가 Top-1 경로를 골라 그대로 두거나 보정

하이퍼파라미터는 논문 표 5 그대로입니다(Null 워밍업 5k, 배치 32, lr 2.5e-5, λ_P=λ_R=0.2, λ_T=0.05 등). 학습 길이만 30k 고정 대신 수렴할 때까지입니다(아래).

## 논문과 다른 점

| 항목 | 논문 | 이 구현 | 이유 |
|---|---|---|---|
| 모델 / 벤치마크 | π0.5, RoboTwin | XR-1·RLDX-1, RoboCasa365 | 요청된 실험 |
| 시연 분기의 VLA 순전파 | 학습 루프 안에서 매 스텝 | 학습 전에 한 번 추출해 저장 (샘플당 노이즈·시간 16쌍) | VLA 가 전부 얼어 있어 같은 목적함수이고 비용이 훨씬 작음. 이미지 증강은 없음 |
| 디노이징 스텝 | 명시 없음 | 각 스텝(XR-1 5회, RLDX-1 4회)마다 라우팅, 프로토타입도 같은 스텝끼리 검색 | 두 모델 모두 스텝마다 액션 투영을 호출 |
| 접촉 시트 | 청크당 1장, 최대 24청크 | 원본 256px 그대로, 균등 간격 최대 96장을 8x6 시트 1~2장으로 (96청크 이하면 청크당 1장) | RoboCasa 에피소드는 11~188 청크. 처음엔 160px 로 줄여 24장만 보냈는데 VLM 확신도가 0.42~0.52 로 낮아 바꿈 |
| 주석 VLM | GPT-5.5 | `OPENAI_API_KEY` 가 있으면 GPT-5.5, `ANTHROPIC_API_KEY` 가 있으면 Claude Opus 5.5 (effort medium), 둘 다 없으면 VLM 없이 휴리스틱(실패 롤아웃의 뒤 50%) | 키가 없어도 돌아가게 함. 어떤 방식이 쓰였는지 결과 표에 기록 |
| 보정 크기 ‖Δh‖² 의 단위 | 은닉 표현 그대로 | 은닉 표현의 평균 크기로 나눈 값 (`--energy-scale auto`) | XR-1 의 은닉 표현 크기는 약 65 라서, 논문 값(λ_u, m)을 그대로 쓰면 Redirect 가 항상 \"보정 안 함\"만 고르게 됨 (실제로 확인). `--energy-scale 1` 이 논문 그대로 |
| Redirect 문턱값 (m, m_r) | 0.1 고정 | `VARIANT=adapted_sum`: 태스크마다, 실패 시작 직후 5청크와 성공 표현의 거리 차이의 절반 (`--margins task --margin-onset-chunks 5`). H- 자체는 논문대로 실패 시작부터 끝까지. 고쳐야 할 실패 직후 상태는 H- 평균보다 성공에 가까워(Coffee 0.032 vs 0.062), H- 전체로 재면 그 상태들이 문턱을 못 넘어 Null 로 학습되기 쉬움. `adapted` 는 세 태스크 평균 차이의 절반 하나 | 예전 XR-1 버퍼에서 성공 프로토타입까지 거리 차이가 Coffee 0.07, Toast 0.04, Steam 0.01 로 태스크마다 7배 다름(평균 0.041). 0.1 은 실패 상태가 프로토타입에서 0.17~0.28 떨어져 있어 도달은 가능하지만, 실패와 성공을 가르는 거리보다 커서 처음에 넘는 전문가가 없음. 평균 하나를 쓰면 Steam 은 성공 상태만큼 옮겨도 문턱을 못 넘음 |
| Redirect 의 당기는 신호 | 문턱을 넘는 전문가가 하나라도 있는 상태에서만(그때는 8개 전문가 모두가 효용 비율대로), 가중치 0.01 | `adapted`, `adapted_sum`: 실패 구간의 모든 상태에서 8개 전문가 모두가 효용 비율대로, 가중치 1.0 | 논문 방식으로는 어떤 전문가도 문턱을 넘지 못해 신호가 시작되지 않음. 실제 버퍼로 확인: 논문 설정은 실패 구간 개입 0%, 수정 설정은 실패 구간 17% / 정상 구간 1% |
| 클래스 균형 CE | 식 없음 | 경로별 타깃 질량의 역수로 가중 | 논문에 정의가 없어 해석 |
| 학습 길이 | 30k 스텝 고정 | 수렴할 때까지 (최소 30k, 최대 150k). `--schedule fixed` 가 논문 방식 | 30k 고정으로 돌린 `adapted_sum` 은 마지막 1만 스텝에도 시연 액션 오차가 0.65 → 0.47 로 계속 내려가 수렴하지 않았음. 아래 "학습 길이" 참고 |
| 학습률 스케줄 | 최댓값(2.5e-5)만 명시 | 워밍업 1k → 최댓값 유지 → 수렴 판정 뒤 코사인 감쇠 | 끝나는 스텝을 미리 모르므로 감쇠를 수렴 뒤로 미룸. `--schedule fixed` 는 전체 길이에 걸친 코사인 |

### 학습 길이 (`--schedule converge`, 기본)

- 시연은 논문처럼 전부(태스크당 50개) 학습에 씁니다. 검증용으로 떼어 두지 않습니다. 로봇 학습에서 오프라인 손실은
  롤아웃 성공률과 잘 맞지 않아서, 모델이 좋은지는 성공률로만 판단합니다. 수렴 판단은 "최적화가 아직 진행되는가"만 봅니다.
- 1,000 스텝마다 고정된 모니터링 세트에서 목적함수를 잽니다: 학습 시연 샘플 10개 중 1개(노이즈 2개)와 H+, H- 에서 고정으로
  뽑은 태스크당 1,024 개. 온도는 마지막 값 T_d = 0.2, Null 은 켠 상태라서 미니배치 잡음이 없고 온도 스케줄에 따라 값이
  바뀌지 않습니다. (`--val-demo-fraction 0.1` 을 주면 시연 10% 를 떼어 두는 방식도 쓸 수 있습니다.)
- T_d 는 처음 30k 스텝 동안 1.0 → 0.2 로 내려가고 그 뒤로는 0.2 입니다. 30k 전에는 멈추지 않습니다.
- 수렴 조건 (모두 만족): 최근 10k 스텝 동안 ① 모니터링 세트의 목적함수와 ② 최선 전문가 액션 오차가 그 이전의
  최고값보다 1% 넘게 좋아지지 않았고, ③ H+ / H- 에서 Null 로 가는 비율이 10k 스텝 전과 비교해 0.02 넘게 바뀌지 않았음.
  한 번의 평가가 아니라 1만 스텝(평가 10번) 구간 전체의 최고값끼리 비교하므로 일시적인 정체에서는 멈추지 않습니다.
- 수렴하면 학습률을 max(3k, 지금까지 스텝의 10%) 동안 코사인으로 2.5e-6 까지 낮춘 뒤 멈춥니다.
- 결과: `reb.pt` (마지막 상태, 평가에 사용), `reb_best.pt` (모니터링 목적함수가 가장 낮았던 상태),
  `checkpoints/reb_step*.pt` (1만 스텝마다, 성공률로 비교할 때 사용), `val_log.jsonl` (모니터링 기록). `train_summary.json` 의 `stop_reason`,
  `converged_at_step`, `best_val_step` 에 언제 왜 멈췄는지 남습니다.

## 측정 항목

`scripts/summarize.py` 가 `summary.csv`, `gpu_summary.csv` 와 표를 만듭니다.

| 항목 | 정의 |
|---|---|
| base SR / FailPatch SR | 같은 50개 시드(같은 장면)에서 원본 정책과 보정 모듈을 붙인 정책의 성공률 |
| SFT time (s) | 시연 특징 추출 + 프로토타입 계산 + 수렴할 때까지의 최적화(모니터링 평가 포함)의 벽시계 시간 (모델 로딩, 버퍼 수집, 주석 제외 — 각각 따로 기록). 스텝 수와 멈춘 이유를 같이 적음 |
| GPU / GPU hours | `nvidia-smi` 의 GPU 이름과, SFT 에 쓴 시간·파이프라인 전체에 쓴 시간. 예: `SFT took 1800 s = 0.50 h on 1x NVIDIA GeForce RTX 4090` |
| mean GPU util / peak memory | 10초 간격으로 기록한 값 |
| patched chunks % | 평가 중 Null 이 아닌 경로가 선택된 청크 비율 |

## 실행 (Vast.ai)

Docker 이미지는 DSRL 실험과 같은 `joowan1108/dsrl-robocasa365:v1` 을 씁니다(추가 패키지 없음).
템플릿 값은 [docker/VAST_TEMPLATE.md](docker/VAST_TEMPLATE.md) 참고.

```bash
cd /workspace/FailPatch && tmux new -s failpatch
bash setup/check.sh
# 주석용 키 (없으면 휴리스틱 주석). 파일로 두면 run_all.sh 가 읽습니다.
echo 'export ANTHROPIC_API_KEY=...' > /workspace/.api_env && chmod 600 /workspace/.api_env

# 짧은 확인 (모델 하나, 태스크 하나)
MODELS=xr1 TASKS=CoffeeSetupMug EVAL_EPISODES=4 N_SUCCESS=3 N_FAIL=3 MAX_EPISODES=16 DEMO_SAMPLES=64 \
  EXTRA_TRAIN_ARGS="--schedule fixed --steps 600 --warmup-steps 100 --draws 4" RESULTS=/workspace/failpatch_smoke bash scripts/run_all.sh

# 전체 실험 (결과: /workspace/failpatch_results)
bash scripts/run_all.sh                  # VARIANT=adapted (기본)
VARIANT=adapted_sum bash scripts/run_all.sh   # 합 액션 오차 + 태스크별 문턱값 (주 실험)
VARIANT=paper bash scripts/run_all.sh    # 논문 값 그대로. 같은 버퍼를 다시 쓰고 학습·평가만 추가

# GPU 두 장이면 모델별로 나눠서
GPU=0 PORT=10086 MODELS=xr1  bash scripts/run_all.sh
GPU=1 PORT=10087 MODELS=rldx bash scripts/run_all.sh
```

주요 환경변수: `MODELS`, `TASKS`, `N_ENVS`, `EVAL_EPISODES`, `N_SUCCESS`/`N_FAIL`/`MAX_EPISODES`(버퍼 크기),
`NUM_DEMOS`, `DEMO_SAMPLES`, `ANNOTATOR`(`auto|openai|anthropic|heuristic`), `EXTRA_TRAIN_ARGS`.
`EXTRA_TRAIN_ARGS="--demo-only"` 는 논문의 Demo-Only REB 기준선입니다.

## 구조

```
failpatch/reb.py            보정 모듈(REB)과 Preserve-Redirect-Trust 손실
failpatch/adapters.py       모델별 코드: 공식 추론 경로, 은닉 표현 훅, 시연 특징 추출
failpatch/server.py         정책 서버 (--reb 로 보정 모듈 부착)
failpatch/collect.py        배치 버퍼 수집
failpatch/annotate.py       VLM/휴리스틱 주석 (논문 부록의 프롬프트)
failpatch/prepare_demos.py  RoboCasa365 사람 시연(LeRobot) → 학습 샘플
failpatch/train.py          FailPatch 학습 + SFT 시간 측정
failpatch/eval.py           평가
scripts/run_all.sh          전체 파이프라인
```

## 알아둘 점

- 평가 조건은 `split=pretrain`, RoboCasa 공식 horizon, 고정 시드로 DSRL 실험과 같습니다.
- 시연은 체크포인트가 학습된 것과 같은 RoboCasa365 pretrain 사람 시연에서 태스크당 50개를 씁니다.
- RLDX-1 체크포인트는 비상업 라이선스(RLWRLD Model License v1.0)입니다.
