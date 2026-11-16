"""Sim-side policy handle and the episode loop (one decision = one VLA query + exec_steps env steps)."""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from failpatch.envs import EnvPool
from failpatch.wire import RpcClient

EVAL_SEED_OFFSET = 1_000_000  # evaluation scenes never overlap with the deployment-buffer seeds


class Policy:
    def __init__(self, host: str, port: int):
        self.rpc = RpcClient(host, port)
        self.info = self.rpc.call({"cmd": "info"})
        self.name = self.info["model"]
        self.frame_deltas = [int(d) for d in self.info["frame_deltas"]]
        self.horizon = int(self.info["horizon"])
        self.default_exec_steps = int(self.info["default_exec_steps"])
        self.infer_time, self.n_queries = 0.0, 0

    def act(self, obs, reset, session_ids):
        response = self.rpc.call({
            "cmd": "act",
            "obs": [{"frames": o["frames"], "instruction": o["instruction"]} for o in obs],
            "reset": reset, "session_ids": session_ids,
        })
        self.infer_time += float(response["infer_time"])
        self.n_queries += len(obs)
        return response["actions"], response["hidden"], response["routes"]


def add_common_args(parser) -> None:
    parser.add_argument("--model", required=True, choices=("xr1", "rldx"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--split", default="pretrain", choices=("pretrain", "target"))
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=None, help="default: robocasa get_task_horizon")
    parser.add_argument("--exec-steps", type=int, default=None)
    parser.add_argument("--out-dir", required=True)


def connect(args):
    policy = Policy(args.host, args.port)
    if policy.name != args.model:
        raise RuntimeError(f"server on port {args.port} serves {policy.name!r}, not {args.model!r}")
    exec_steps = args.exec_steps or policy.default_exec_steps
    pool = EnvPool(n_envs=args.n_envs, task=args.task, split=args.split, seed=args.seed,
                   frame_deltas=policy.frame_deltas, horizon=args.horizon)
    return policy, pool, exec_steps


def run_episodes(pool: EnvPool, policy: Policy, exec_steps: int,
                 next_seed: Callable[[], int | None],
                 on_episode: Callable[[dict], None],
                 record: bool = False,
                 stop: Callable[[], bool] = lambda: False) -> None:
    """Runs episodes until `next_seed` is exhausted or `stop()` is true.
    With record=True each finished episode carries the per-chunk hidden representations,
    routes and the left agent-view frame observed when the chunk was queried."""
    pool.drain()
    ready, inflight, episodes = {}, {}, {}

    def start_episode(i):
        seed = next_seed()
        if seed is None:
            return
        episodes[i] = dict(seed=int(seed), steps=0, first=True, t0=time.time(), hidden=[], routes=[], frames=[])
        pool.send_reset(i, int(seed))

    for i in range(pool.n):
        start_episode(i)
    while True:
        if ready and not stop():
            ids = sorted(ready)
            obs = [ready.pop(i) for i in ids]
            actions, hidden, routes = policy.act(obs, [episodes[i]["first"] for i in ids], [f"env{i}" for i in ids])
            for j, i in enumerate(ids):
                ep = episodes[i]
                ep["first"] = False
                ep["routes"].append(routes[j])
                if record:
                    ep["hidden"].append(hidden[j].astype(np.float16))
                    ep["frames"].append(obs[j]["frames"][-1]["video.robot0_agentview_left"])
                inflight[i] = True
                pool.send_step(i, actions[j, :exec_steps])
        if not pool.busy:
            break
        for i, kind, data in pool.recv():
            if kind == "reset":
                ready[i] = data
                continue
            inflight.pop(i)
            ep = episodes[i]
            ep["steps"] += data["n_steps"]
            if data["success"] or data["timeout"]:
                routes = np.stack(ep["routes"])
                result = dict(seed=ep["seed"], success=bool(data["success"]), steps=ep["steps"],
                              chunks=len(ep["routes"]), wall=time.time() - ep["t0"],
                              patched_fraction=float((routes > 0).mean()))
                if record:
                    result.update(hidden=np.stack(ep["hidden"]), frames=ep["frames"],
                                  instruction=data["instruction"])
                on_episode(result)
                if not stop():
                    start_episode(i)
            else:
                ready[i] = data


def summarize(episodes: list[dict]) -> dict:
    episodes = sorted(episodes, key=lambda e: e["seed"])
    n = len(episodes)
    wins = sum(e["success"] for e in episodes)
    rate = wins / n
    win_steps = [e["steps"] for e in episodes if e["success"]]
    return dict(num_episodes=n, successes=wins, success_rate=rate,
                success_rate_stderr=math.sqrt(rate * (1 - rate) / n),
                mean_episode_steps=float(np.mean([e["steps"] for e in episodes])),
                mean_steps_to_success=float(np.mean(win_steps)) if win_steps else None,
                mean_patched_fraction=float(np.mean([e["patched_fraction"] for e in episodes])),
                episodes=episodes)
