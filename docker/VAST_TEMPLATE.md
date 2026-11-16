# Vast.ai 템플릿

Docker 이미지는 DSRL 실험용으로 만든 `joowan1108/dsrl-robocasa365:v1` 을 그대로 씁니다
(이 저장소는 추가 파이썬 패키지가 필요 없습니다). 기존 템플릿 `joowan-dsrl-robocasa365-v1` 을 복제해
On-start Script 만 아래로 바꾸면 됩니다.

| 항목 | 값 |
|---|---|
| Template Name | `joowan-failpatch-robocasa365-v1` |
| Image Path:Tag | `joowan1108/dsrl-robocasa365:v1` |
| Docker Options | `-e NVIDIA_DRIVER_CAPABILITIES=all` |
| Environment Variables | `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONUNBUFFERED=1` |
| Launch Mode | SSH |
| Extra Filters | `cpu_arch in ['amd64'] cuda_max_good>=12.8 gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=16 cpu_ram>=64 inet_down>=300 reliability>0.99 verified=true` |
| Docker Repository Authentication | `joowan1108` + Docker Hub Read-only 토큰 |
| Disk Space | 150 GB |
| Visibility | Private |

On-start Script (`setup/onstart.sh`):

```
cd /workspace
if [ ! -d FailPatch-RC365 ]; then
  git clone "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/joowan1108/FailPatch-RC365.git"
fi
cd FailPatch-RC365 && git pull --ff-only || true
nohup bash setup/download.sh > /workspace/failpatch_download.log 2>&1 &
```

- 주석용 API 키(`OPENAI_API_KEY` 또는 `ANTHROPIC_API_KEY`)는 템플릿이 아니라 Vast 계정 환경변수에 넣거나,
  접속 후 `export` 합니다. 없으면 휴리스틱 주석으로 돌아갑니다.
- DSRL 실험을 돌린 인스턴스를 그대로 써도 됩니다. `git clone` 후 `bash setup/download.sh` 를 실행하면
  이미 받은 에셋·체크포인트는 건너뛰고 시연 데이터만 추가로 받습니다.
- GPU 는 RTX 4090 기준입니다. GPU 두 장 인스턴스(`num_gpus=2`)면 README 의 두 줄 명령으로 모델별 병렬 실행이 됩니다.
  RTX 5090 은 이 이미지의 RLDX-1·시뮬레이터 환경(CUDA 12.6 빌드 torch)이 지원하지 않습니다.
