"""Merge evaluation files of disjoint scene ranges into one, e.g. eval_base.json (scenes 0-49) and
eval_base_ext.json (scenes 50-99) -> eval_base100.json.

    python scripts/merge_evals.py OUT.json IN1.json IN2.json [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from failpatch.rollout import summarize


def main():
    out, *ins = sys.argv[1:]
    parts = [json.loads(Path(p).read_text()) for p in ins]
    episodes = [e for part in parts for e in part["episodes"]]
    seeds = [e["seed"] for e in episodes]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("overlapping scenes (seeds) between the inputs")
    result = summarize(episodes)
    keep = ("model", "task", "split", "tag", "horizon", "exec_steps")
    result.update({k: parts[0][k] for k in keep if k in parts[0]})
    result.update(eval_wall_time_s=sum(p.get("eval_wall_time_s", 0) for p in parts), merged_from=[str(p) for p in ins])
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[merge] {out}: {result['successes']}/{result['num_episodes']} = {100 * result['success_rate']:.1f}% "
          f"(+-{100 * result['success_rate_stderr']:.1f}), seeds {min(seeds)}-{max(seeds)}")


if __name__ == "__main__":
    main()
