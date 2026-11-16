"""Turn RoboCasa365 human demonstrations (LeRobot v2.1: parquet + h264 mp4) into raw training
samples that both model adapters understand.

For `--num-demos` episodes of the task, picks evenly spaced anchor steps t and stores
  frames   observation at t + frame_deltas (clipped at 0), in the gym-env key format
  actions  a_t .. a_{t+H-1} in env order [eef_pos, eef_rot, gripper, base_motion, control_mode]
so that a sample looks exactly like what the policy sees in a rollout.
Output: <out>/demos_<task>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_KEYS = ("end_effector_position_relative", "end_effector_rotation_relative", "gripper_qpos",
              "base_position", "base_rotation")
ACTION_KEYS = ("end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode")
FRAME_DELTAS = (-6, -4, -2, 0)
HORIZON = 16


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1])  # BGR -> RGB
    cap.release()
    return np.ascontiguousarray(np.stack(frames))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-demos", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000, help="anchor steps for this task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from robocasa.utils.dataset_registry_utils import get_ds_path

    root = Path(get_ds_path(args.task, source="human", split=args.split))
    if not (root / "meta").exists():
        raise FileNotFoundError(f"{root}: run setup/download.sh (demo datasets) first")
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(episodes), size=min(args.num_demos, len(episodes)), replace=False).tolist())
    usable = {e: max(0, episodes[e]["length"] - HORIZON + 1) for e in chosen}
    total = sum(usable.values())
    sl = lambda group, key: slice(modality[group][key]["start"], modality[group][key]["end"])  # noqa: E731

    out = {f"video.{c}": [] for c in CAMERAS}
    out.update({f"state.{k}": [] for k in STATE_KEYS})
    actions, instructions, episode_ids = [], [], []
    for e in chosen:
        ep = episodes[e]
        n = max(1, round(args.samples * usable[e] / total)) if usable[e] else 0
        if n == 0:
            continue
        anchors = np.unique(np.linspace(0, usable[e] - 1, n).round().astype(int))
        chunk = ep["episode_index"] // info["chunks_size"]
        table = pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=ep["episode_index"]))
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(table["action"].to_numpy()).astype(np.float32)
        action = np.concatenate([action[:, sl("action", k)] for k in ACTION_KEYS], axis=1)
        videos = {c: read_video(root / info["video_path"].format(
            episode_chunk=chunk, video_key=f"observation.images.{c}", episode_index=ep["episode_index"]))
            for c in CAMERAS}
        length = min(len(state), min(len(v) for v in videos.values()))
        for t in anchors:
            if t + HORIZON > length:
                continue
            idx = [max(0, t + d) for d in FRAME_DELTAS]
            for c in CAMERAS:
                out[f"video.{c}"].append(videos[c][idx])
            for k in STATE_KEYS:
                out[f"state.{k}"].append(state[idx][:, sl("state", k)])
            actions.append(action[t : t + HORIZON])
            instructions.append(ep["tasks"][0])
            episode_ids.append(ep["episode_index"])
    arrays = {k: np.stack(v) for k, v in out.items()}
    path = Path(args.out) / f"demos_{args.task}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, actions=np.stack(actions), instructions=np.array(instructions),
             episode_ids=np.array(episode_ids), frame_deltas=np.array(FRAME_DELTAS), **arrays)
    print(f"[demos] {args.task}: {len(actions)} samples from {len(set(episode_ids))} demonstrations -> {path}")


if __name__ == "__main__":
    main()
