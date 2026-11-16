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

하이퍼파라미터는 논문 표 5 그대로입니다(30k 스텝, 워밍업 5k, 배치 32, lr 2.5e-5, λ_P=λ_R=0.2, λ_T=0.05 등).

## 논문과 다른 점

| 항목 | 논문 | 이 구현 | 이유 |
|---|---|---|---|
| 모델 / 벤치마크 | π0.5, RoboTwin | XR-1·RLDX-1, RoboCasa365 | 요청된 실험 |
| 시연 분기의 VLA 순전파 | 학습 루프 안에서 매 스텝 | 학습 전에 한 번 추출해 저장 (샘플당 노이즈·시간 16쌍) | VLA 가 전부 얼어 있어 같은 목적함수이고 비용이 훨씬 작음. 이미지 증강은 없음 |
| 디노이징 스텝 | 명시 없음 | 각 스텝(XR-1 5회, RLDX-1 4회)마다 라우팅, 프로토타입도 같은 스텝끼리 검색 | 두 모델 모두 스텝마다 액션 투영을 호출 |
| 접촉 시트 | 청크당 1장, 최대 24청크 | 에피소드가 길어 균등 간격 24장 | RoboCasa 에피소드는 수십~수백 청크 |
| 주석 VLM | GPT-5.5 | `OPENAI_API_KEY` 가 있으면 GPT-5.5, `ANTHROPIC_API_KEY` 가 있으면 Claude Opus 5.5 (effort medium), 둘 다 없으면 VLM 없이 휴리스틱(실패 롤아웃의 뒤 50%) | 키가 없어도 돌아가게 함. 어떤 방식이 쓰였는지 결과 표에 기록 |
| 클래스 균형 CE | 식 없음 | 경로별 타깃 질량의 역수로 가중 | 논문에 정의가 없어 해석 |

## 측정 항목

`scripts/summarize.py` 가 `summary.csv`, `gpu_summary.csv` 와 표를 만듭니다.

| 항목 | 정의 |
|---|---|
| base SR / FailPatch SR | 같은 50개 시드(같은 장면)에서 원본 정책과 보정 모듈을 붙인 정책의 성공률 |
| SFT time (s) | 시연 특징 추출 + 프로토타입 계산 + 30k 스텝 최적화의 벽시계 시간 (모델 로딩, 버퍼 수집, 주석 제외 — 각각 따로 기록) |
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
  EXTRA_TRAIN_ARGS="--steps 600 --warmup-steps 100 --draws 4" RESULTS=/workspace/failpatch_smoke bash scripts/run_all.sh

# 전체 실험 (결과: /workspace/failpatch_results)
bash scripts/run_all.sh

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
