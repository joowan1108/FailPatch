"""FailPatch training for one VLA on several tasks jointly (run in the model's environment).

Stage 1 (feature extraction): for every demonstration sample, run the frozen VLA once and the
  frozen action head for `--draws` random (noise, time) pairs; keep the action-token hidden
  states and the flow-matching targets. The paper runs the frozen backbone inside the training
  loop instead; because every weight of the VLA is frozen, doing it once up front optimises the
  same objective (without image augmentation) at a fraction of the cost.
Stage 2 (optimisation): 30k steps of the Preserve-Redirect-Trust objective on the REB only.

Reported in <out>/train_summary.json:
  sft_time_s = feature_extraction_s + optimization_s   (model loading is reported separately)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from failpatch.adapters import build
from failpatch.reb import (ResidualExpertBank, demo_loss, local_success_prototypes, preserve_loss,
                           redirect_trust_loss)


def load_buffer(task_dir: Path):
    """-> reliable-success pool, H+ and H- representations, each (N, S, d) float32."""
    labels = json.loads((task_dir / "labels.json").read_text())
    pool, plus, minus = [], [], []
    for stem, lab in labels.items():
        hidden = np.load(task_dir / "buffer" / f"{stem}.npz")["hidden"].astype(np.float32)
        if lab["success"]:
            if lab["reliable_success"]:
                pool.append(hidden)
                plus.append(hidden)
        else:
            onset = lab["failure_onset_chunk"]
            plus.append(hidden[:onset])
            minus.append(hidden[onset:])
    cat = lambda xs: torch.from_numpy(np.concatenate(xs)) if xs and sum(len(x) for x in xs) else None  # noqa: E731
    return cat(pool), cat(plus), cat(minus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("xr1", "rldx"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--xr1-repo", default=None)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--results", required=True, help="results/<model>: contains <task>/buffer, labels.json")
    parser.add_argument("--demos", required=True, help="directory with demos_<task>.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000, help="Null-disabled warm-up")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-5, help="peak learning rate")
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-final", type=float, default=2.5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--bottleneck", type=int, default=704)
    parser.add_argument("--k-nn", type=int, default=20)
    parser.add_argument("--lambda-rd", type=float, default=1.0)
    parser.add_argument("--lambda-p", type=float, default=0.2)
    parser.add_argument("--lambda-r", type=float, default=0.2)
    parser.add_argument("--lambda-t", type=float, default=0.05)
    parser.add_argument("--lambda-u", type=float, default=0.01)
    parser.add_argument("--fail-temperature", type=float, default=0.2)
    parser.add_argument("--null-margin", type=float, default=0.1)
    parser.add_argument("--redirect-margin-weight", type=float, default=0.01)
    parser.add_argument("--redirect-margin", type=float, default=0.1)
    parser.add_argument("--demo-only", action="store_true", help="Demo-Only REB baseline of the paper")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tic = time.time()
    adapter = build(args.model, args)
    load_s = time.time() - tic
    print(f"[train] {args.model} loaded in {load_s:.0f}s", flush=True)

    # ---- stage 1: frozen features of the demonstrations ---------------------------------
    tic = time.time()
    demo = {}
    for task in args.tasks:
        data = np.load(Path(args.demos) / f"demos_{task}.npz")
        if list(data["frame_deltas"]) != list(adapter.frame_deltas):
            raise RuntimeError(f"demo frames {list(data['frame_deltas'])} != model frames {adapter.frame_deltas}")
        frame_keys = [k for k in data.files if k.startswith(("video.", "state."))]
        arrays = {k: data[k] for k in frame_keys}
        zs, targets, masks = [], [], []
        for i in range(len(data["actions"])):
            sample = {
                "frames": [{k: arrays[k][i, j] for k in frame_keys} for j in range(len(adapter.frame_deltas))],
                "instruction": str(data["instructions"][i]),
                "actions": data["actions"][i],
            }
            z, target, mask = adapter.demo_features(sample, args.draws)
            zs.append(z.half())
            targets.append(target)
            masks.append(mask)
            if i % 100 == 0:
                print(f"[train] features {task}: {i}/{len(data['actions'])} ({time.time() - tic:.0f}s)", flush=True)
        demo[task] = (torch.cat(zs), torch.cat(targets), torch.cat(masks))
    feature_s = time.time() - tic
    n_draws = sum(len(v[0]) for v in demo.values())
    print(f"[train] demo features: {n_draws} (sample, noise, time) draws in {feature_s:.0f}s", flush=True)

    # ---- deployment buffer: H+, H- and local success prototypes -------------------------
    tic = time.time()
    plus, minus, protos, buffer_stats = {}, {}, {}, {}
    if not args.demo_only:
        for task in args.tasks:
            pool, h_plus, h_minus = load_buffer(Path(args.results) / task)
            buffer_stats[task] = dict(pool=0 if pool is None else len(pool), plus=0 if h_plus is None else len(h_plus),
                                      minus=0 if h_minus is None else len(h_minus))
            if h_plus is not None:
                plus[task] = h_plus.flatten(0, 1)
            if h_minus is not None and pool is not None:
                # prototypes are retrieved among representations of the same denoising step
                steps = h_minus.shape[1]
                proto = torch.stack([local_success_prototypes(h_minus[:, s].to(device), pool[:, s].to(device), args.k_nn).cpu()
                                     for s in range(steps)], dim=1)
                minus[task], protos[task] = h_minus.flatten(0, 1), proto.flatten(0, 1)
        print(f"[train] deployment buffer chunks: {buffer_stats}", flush=True)
    prototype_s = time.time() - tic

    # ---- stage 2: optimise the REB --------------------------------------------------------
    reb = ResidualExpertBank(adapter.hidden_dim, args.n_experts, args.bottleneck).to(device)
    n_params = sum(p.numel() for p in reb.parameters())
    opt = torch.optim.AdamW(reb.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def learning_rate(step: int) -> float:
        """Linear warm-up to the peak, then cosine decay. The paper only states the peak value
        (2.5e-5); warm-up length and final value are the OpenPI defaults its code base uses."""
        if step <= args.lr_warmup_steps:
            return args.lr * step / max(1, args.lr_warmup_steps)
        progress = (step - args.lr_warmup_steps) / max(1, args.steps - args.lr_warmup_steps)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * progress))

    for p in adapter.model.parameters():
        p.requires_grad_(False)

    def sample(store: dict, n: int):
        """Task-balanced minibatch from {task: tensor or tuple of tensors}."""
        tasks = list(store)
        picks = [tasks[i % len(tasks)] for i in torch.randperm(n).tolist()]
        parts = []
        for task in tasks:
            count = picks.count(task)
            if count == 0:
                continue
            item = store[task]
            first = item[0] if isinstance(item, tuple) else item
            idx = torch.randint(0, len(first), (count,))
            parts.append(tuple(t[idx] for t in item) if isinstance(item, tuple) else (item[idx],))
        return tuple(torch.cat(col).to(device).float() for col in zip(*parts))

    minus_store = {t: (minus[t], protos[t]) for t in minus}
    tic = time.time()
    log = (out / "train_log.jsonl").open("w")
    for step in range(1, args.steps + 1):
        temperature = 1.0 + (0.2 - 1.0) * (step - 1) / max(1, args.steps - 1)  # T_d: 1.0 -> 0.2
        z, target, mask = sample(demo, args.batch_size)
        l_expert, l_router, stats = demo_loss(reb, z, target, mask, adapter.project, temperature,
                                              null_enabled=step > args.warmup_steps)
        loss = l_expert + args.lambda_rd * l_router
        stats.update(expert=float(l_expert), router_demo=float(l_router))
        if plus:
            l_p = preserve_loss(reb, sample(plus, args.batch_size)[0])
            loss = loss + args.lambda_p * l_p
            stats["preserve"] = float(l_p)
        if minus_store:
            h, proto = sample(minus_store, args.batch_size)
            l_r, l_t, extra = redirect_trust_loss(reb, h, proto, args.lambda_u, args.fail_temperature,
                                                  args.null_margin, args.redirect_margin_weight, args.redirect_margin)
            loss = loss + args.lambda_r * l_r + args.lambda_t * l_t
            stats.update(redirect=float(l_r), trust=float(l_t), **extra)
        for group in opt.param_groups:
            group["lr"] = learning_rate(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reb.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            stats.update(step=step, loss=float(loss), elapsed_s=time.time() - tic)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            if step % 2000 == 0 or step == 1:
                print("[train] " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()), flush=True)
    optimization_s = time.time() - tic
    log.close()

    torch.save(dict(config=reb.config(), state_dict=reb.state_dict(), args=vars(args)), out / "reb.pt")
    vla_params = sum(p.numel() for p in adapter.model.parameters())
    summary = dict(
        model=args.model, tasks=args.tasks, demo_only=args.demo_only,
        gpu=torch.cuda.get_device_name(0),
        sft_time_s=feature_s + prototype_s + optimization_s,
        feature_extraction_s=feature_s, prototype_s=prototype_s, optimization_s=optimization_s,
        model_load_s=load_s, steps=args.steps, demo_draws=n_draws,
        trainable_params=n_params, trainable_fraction=n_params / vla_params,
        buffer_chunks=buffer_stats, peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        final=stats,
    )
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "final"}, indent=2))


if __name__ == "__main__":
    main()
