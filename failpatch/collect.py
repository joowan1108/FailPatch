"""Collect the fixed deployment buffer of FailPatch: rollouts of the frozen base policy until
`--n-success` successful and `--n-fail` failed episodes are stored (or `--max-episodes` is hit).

Per episode: <out-dir>/buffer/ep_<seed>.npz with
  hidden   (T, S, d) float16  action hidden representation per chunk and denoising step
  thumbs   (n, H, W, 3)       left agent-view frames at native camera resolution (256x256) at
                              `thumb_idx`: every chunk, or `--max-frames` evenly spaced chunks
  success, steps, instruction
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from failpatch.rollout import add_common_args, connect, run_episodes


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--n-success", type=int, default=50)
    parser.add_argument("--n-fail", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=96, help="frames kept per episode for the annotator")
    args = parser.parse_args()

    out = Path(args.out_dir) / "buffer"
    out.mkdir(parents=True, exist_ok=True)
    counts = {True: 0, False: 0}
    for f in out.glob("ep_*.npz"):  # resume
        counts[bool(np.load(f)["success"])] += 1
    target = {True: args.n_success, False: args.n_fail}
    state = dict(started=sum(counts.values()), finished=sum(counts.values()))
    done = lambda: all(counts[k] >= target[k] for k in target) or state["finished"] >= args.max_episodes  # noqa: E731

    policy, pool, exec_steps = connect(args)
    if policy.info["patched"]:
        raise RuntimeError("the deployment buffer must be collected with the base policy (no --reb)")

    def next_seed():
        if done() or state["started"] >= args.max_episodes:
            return None
        state["started"] += 1
        return args.seed + state["started"] - 1

    def on_episode(ep):
        state["finished"] += 1
        key = ep["success"]
        if counts[key] < target[key]:  # surplus episodes of the already-full class are dropped
            counts[key] += 1
            idx = np.unique(np.linspace(0, ep["chunks"] - 1, min(args.max_frames, ep["chunks"])).round().astype(int))
            thumbs = np.stack([ep["frames"][i] for i in idx])  # native resolution, no resizing
            np.savez_compressed(out / f"ep_{ep['seed']:07d}.npz", hidden=ep["hidden"], thumbs=thumbs,
                                thumb_idx=idx, success=key, steps=ep["steps"], instruction=ep["instruction"])
        print(f"[collect] {args.model}/{args.task}: success {counts[True]}/{args.n_success}, "
              f"fail {counts[False]}/{args.n_fail} ({state['finished']} episodes)", flush=True)

    tic = time.time()
    try:
        run_episodes(pool, policy, exec_steps, next_seed, on_episode, record=True, stop=done)
    finally:
        pool.close()
    meta = dict(model=args.model, task=args.task, split=args.split, layout=args.layout, style=args.style, exec_steps=exec_steps,
                n_success=counts[True], n_fail=counts[False], episodes_run=state["finished"],
                collect_wall_time_s=time.time() - tic)
    (Path(args.out_dir) / "buffer_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] done: {meta}")


if __name__ == "__main__":
    main()
