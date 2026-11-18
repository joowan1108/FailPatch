"""Policy server for one VLA, optionally with a trained Residual Expert Bank attached.

Run inside the model's own environment:
  python -m failpatch.server --model xr1 --model-path ... --xr1-repo ... [--reb reb.pt]
"""

from __future__ import annotations

import argparse
import time

import torch

from failpatch.adapters import build
from failpatch.reb import ResidualExpertBank
from failpatch.wire import serve


def load_reb(path: str, device) -> ResidualExpertBank:
    ckpt = torch.load(path, map_location="cpu")
    reb = ResidualExpertBank(**ckpt["config"])
    reb.load_state_dict(ckpt["state_dict"])
    return reb.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("xr1", "rldx"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--xr1-repo", default=None)
    parser.add_argument("--reb", default=None, help="trained REB checkpoint; omit for the base policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    args = parser.parse_args()

    adapter = build(args.model, args)
    if args.reb:
        adapter.hook.reb = load_reb(args.reb, "cuda")
    info = {
        "model": adapter.name,
        "patched": bool(args.reb),
        "frame_deltas": adapter.frame_deltas,
        "horizon": adapter.horizon,
        "hidden_dim": adapter.hidden_dim,
        "default_exec_steps": adapter.exec_steps,
    }
    print(f"[{adapter.name}] ready: {info}", flush=True)

    def handle(request: dict) -> dict:
        if request["cmd"] == "info":
            return info
        if request["cmd"] != "act":
            raise ValueError(f"unknown cmd {request['cmd']!r}")
        tic = time.time()
        actions, hidden, routes = adapter.act(request["obs"], request["reset"], request["session_ids"],
                                              noise_seeds=request.get("noise_seeds"))
        torch.cuda.synchronize()
        return {"actions": actions, "hidden": hidden, "routes": routes, "infer_time": time.time() - tic}

    serve(handle, args.host, args.port)


if __name__ == "__main__":
    main()
