"""Evaluate the policy currently served (base, or base + REB) on one task.
Base and FailPatch runs use the same seeds, i.e. the same scenes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from failpatch.rollout import EVAL_SEED_OFFSET, add_common_args, connect, run_episodes, summarize


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--first-episode", type=int, default=0,
                        help="index of the first evaluation scene; with --first-episode 50 --episodes 50 the scenes "
                             "50-99 are run, so an earlier 50-scene result can be extended (scripts/merge_evals.py)")
    parser.add_argument("--tag", default=None, help="output stem; default: base or failpatch")
    args = parser.parse_args()

    policy, pool, exec_steps = connect(args)
    tag = args.tag or ("failpatch" if policy.info["patched"] else "base")
    seeds = iter([args.seed + EVAL_SEED_OFFSET + i for i in range(args.first_episode, args.first_episode + args.episodes)])
    episodes: list[dict] = []
    tic = time.time()
    try:
        run_episodes(pool, policy, exec_steps, lambda: next(seeds, None), episodes.append)
    finally:
        pool.close()
    result = summarize(episodes)
    result.update(model=args.model, task=args.task, split=args.split, tag=tag, horizon=pool.horizon,
                  exec_steps=exec_steps, eval_wall_time_s=time.time() - tic,
                  mean_vla_latency_s=policy.infer_time / max(1, policy.n_queries))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[eval:{tag}] {args.model}/{args.task}: {result['successes']}/{result['num_episodes']} = "
          f"{100 * result['success_rate']:.1f}% (+-{100 * result['success_rate_stderr']:.1f}), "
          f"patched chunks {100 * result['mean_patched_fraction']:.1f}%")


if __name__ == "__main__":
    main()
