"""FailPatch training for one VLA on several tasks jointly (run in the model's environment).

Stage 1 (feature extraction): for every demonstration sample, run the frozen VLA once and the
  frozen action head for `--draws` random (noise, time) pairs; keep the action-token hidden
  states and the flow-matching targets. The paper runs the frozen backbone inside the training
  loop instead; because every weight of the VLA is frozen, doing it once up front optimises the
  same objective (without image augmentation) at a fraction of the cost.
Stage 2 (optimisation): the Preserve-Redirect-Trust objective on the REB only.
  --schedule fixed     `--steps` steps, cosine learning rate, T_d 1.0 -> 0.2 over the whole run.
  --schedule converge  (default) train until the objective on a fixed monitoring set stops improving:
    * monitoring set: a fixed subset of the training demonstrations (every `--monitor-every`-th
      sample; all demonstrations are trained on, as in the paper; `--val-demo-fraction` > 0 holds
      whole demonstrations out instead) plus fixed subsets of H+ and H-. It measures whether the
      optimisation still progresses, not generalisation: in robot learning offline losses track
      rollout success poorly, so success rates are the only quality measure. Evaluated every
      `--eval-every` steps at the final temperature (T_d = 0.2) with Null enabled, so the numbers
      carry no minibatch noise and do not move with the temperature schedule;
    * learning rate: warm-up, then constant at the peak; T_d anneals 1.0 -> 0.2 over `--min-steps`
      (the paper's 30k) and then stays at 0.2;
    * converged once step >= `--min-steps` and, over the last `--patience` steps, neither the
      monitored objective nor the monitored best-expert action error improved on their earlier best
      by more than `--min-rel-improvement`, and the share of H+ / H- states routed to Null moved by
      less than `--max-route-change`;
    * then the learning rate decays (cosine) to `--lr-final` over max(`--decay-min-steps`,
      `--decay-fraction` x steps so far) and training stops. `--max-steps` bounds the whole run
      (decay included); hitting it is reported as stop_reason = "max_steps".
    reb.pt is the final state; reb_best.pt the state with the lowest monitored objective.

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
                           prototype_distance_gap, redirect_trust_loss)


def load_buffer(task_dir: Path):
    """-> reliable-success pool, H+ and H- representations, each (N, S, d) float32, and for every
    H- row its chunk offset from the failure onset (0 = onset chunk)."""
    labels = json.loads((task_dir / "labels.json").read_text())
    pool, plus, minus, offset = [], [], [], []
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
            offset.append(np.arange(len(hidden) - onset))
    cat = lambda xs: torch.from_numpy(np.concatenate(xs)) if xs and sum(len(x) for x in xs) else None  # noqa: E731
    return cat(pool), cat(plus), cat(minus), (np.concatenate(offset) if offset else np.zeros(0, int))


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
    parser.add_argument("--center-features", default="none", choices=("none", "task"),
                        help="task: subtract the per-task, per-denoising-step mean of the deployment representations "
                             "before l2-normalising them for prototype retrieval and Redirect distances (about 90%% of "
                             "|h|^2 of XR-1 lies along that shared direction, which compresses all normalised "
                             "distances). none: the paper's space (normalise only)")
    parser.add_argument("--demo-target", default="draw", choices=("draw", "scene"),
                        help="draw: responsibilities from each (noise, time) draw, as in eq. 6-7. scene: from the "
                             "loss averaged over the cached draws of the same demonstration sample")
    parser.add_argument("--demo-samples", type=int, default=0,
                        help="use this many evenly spaced samples of each demos_<task>.npz (0 = all of them)")
    parser.add_argument("--feature-cache", default=None,
                        help="directory to store/reuse the frozen demonstration features across variants")
    parser.add_argument("--schedule", default="converge", choices=("converge", "fixed"))
    parser.add_argument("--steps", type=int, default=30_000, help="--schedule fixed: number of steps")
    parser.add_argument("--min-steps", type=int, default=30_000,
                        help="--schedule converge: no stop before this; also the length of the T_d annealing")
    parser.add_argument("--max-steps", type=int, default=150_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--patience", type=int, default=10_000)
    parser.add_argument("--min-rel-improvement", type=float, default=0.01)
    parser.add_argument("--max-route-change", type=float, default=0.02)
    parser.add_argument("--decay-fraction", type=float, default=0.1)
    parser.add_argument("--decay-min-steps", type=int, default=3_000)
    parser.add_argument("--val-demo-fraction", type=float, default=0.0,
                        help="> 0: hold out this share of each task's demonstrations (whole demonstrations) for the "
                             "convergence test; 0 (default): train on all and monitor a fixed subset of them")
    parser.add_argument("--monitor-every", type=int, default=10,
                        help="with --val-demo-fraction 0: every N-th demonstration sample is in the monitoring set")
    parser.add_argument("--val-draws", type=int, default=2, help="(noise, time) draws per held-out sample")
    parser.add_argument("--val-buffer-size", type=int, default=1024, help="H+ and H- rows per task for validation")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
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
    parser.add_argument("--action-loss", default="sum", choices=("sum", "mean"),
                        help="candidate action loss: squared L2 norm as in the paper (sum) or per-entry mean")
    parser.add_argument("--fail-temperature", type=float, default=0.2)
    parser.add_argument("--null-margin", type=float, default=0.1)
    parser.add_argument("--redirect-margin-weight", type=float, default=0.01)
    parser.add_argument("--redirect-margin", type=float, default=0.1)
    parser.add_argument("--margins", default="paper", choices=("paper", "relative", "task"),
                        help="paper: --null-margin / --redirect-margin as given (0.1). relative: both are set to "
                             "--margin-fraction of the measured gap between failure and success representations, "
                             "averaged over all tasks. task: the same, but measured and applied per task")
    parser.add_argument("--margin-fraction", type=float, default=0.5)
    parser.add_argument("--fail-temperature-scaling", default="none", choices=("none", "margin"),
                        help="margin: the failure routing temperature of every task is --fail-temperature x "
                             "(task margin / --reference-margin), i.e. scaled with the margin so that the "
                             "paper's ratio T/m = 0.2/0.1 is kept when the margins are rescaled")
    parser.add_argument("--reference-margin", type=float, default=0.1, help="the paper's margin m")
    parser.add_argument("--margin-onset-chunks", type=int, default=0,
                        help="relative/task margins: measure the failure/success gap on the first N chunks after the "
                             "failure onset only (0 = every H- chunk). H- itself is unchanged")
    parser.add_argument("--redirect-bootstrap", action="store_true",
                        help="push experts towards the success prototype before any of them clears the Null threshold")
    parser.add_argument("--energy-scale", default="auto",
                        help="norm that residual energies ||delta h||^2 are measured against; 'auto' = mean norm of "
                             "the cached action hidden representations, '1' = raw energies as written in the paper")
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
    cache_dir = Path(args.feature_cache) if args.feature_cache else None
    feature_cache_hits = 0

    def demo_index(task: str) -> np.ndarray:
        """Indices of the demonstration samples used (evenly spaced, so every demonstration is kept)."""
        n = len(np.load(Path(args.demos) / f"demos_{task}.npz")["episode_ids"])
        if args.demo_samples <= 0 or args.demo_samples >= n:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, args.demo_samples).round().astype(int))

    demo_idx = {task: demo_index(task) for task in args.tasks}
    suffix = f"_n{args.demo_samples}" if args.demo_samples > 0 else ""
    for task in args.tasks:
        cache_file = cache_dir / f"demo_features_{task}_draws{args.draws}{suffix}.pt" if cache_dir else None
        if cache_file is not None and cache_file.exists():
            # The frozen features do not depend on the training variant, so variants share them.
            demo[task] = torch.load(cache_file)
            feature_cache_hits += 1
            print(f"[train] features {task}: loaded from {cache_file}", flush=True)
            continue
        data = np.load(Path(args.demos) / f"demos_{task}.npz")
        if list(data["frame_deltas"]) != list(adapter.frame_deltas):
            raise RuntimeError(f"demo frames {list(data['frame_deltas'])} != model frames {adapter.frame_deltas}")
        frame_keys = [k for k in data.files if k.startswith(("video.", "state."))]
        arrays = {k: data[k] for k in frame_keys}
        zs, targets, masks = [], [], []
        for n_done, i in enumerate(demo_idx[task]):
            sample = {
                "frames": [{k: arrays[k][i, j] for k in frame_keys} for j in range(len(adapter.frame_deltas))],
                "instruction": str(data["instructions"][i]),
                "actions": data["actions"][i],
            }
            z, target, mask = adapter.demo_features(sample, args.draws)
            zs.append(z.half())
            targets.append(target)
            masks.append(mask)
            if n_done % 100 == 0:
                print(f"[train] features {task}: {n_done}/{len(demo_idx[task])} ({time.time() - tic:.0f}s)", flush=True)
        demo[task] = (torch.cat(zs), torch.cat(targets), torch.cat(masks))
        if cache_file is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(demo[task], cache_file)
    feature_s = time.time() - tic

    # Monitoring set for the convergence test. By default it is a fixed subset of the TRAINING
    # demonstrations (every `--monitor-every`-th sample, first `--val-draws` draws): all 50
    # demonstrations are trained on, as in the paper, and the test only asks whether the
    # optimisation still makes progress. Success rates in rollouts are the only quality measure.
    # `--val-demo-fraction` > 0 instead holds out whole demonstrations (never trained on).
    # The cached features are stored sample by sample, `--draws` rows each, in the order of
    # demos_<task>.npz (sorted by demonstration).
    val_demo, val_info = {}, {}
    if args.schedule == "converge":
        for task in args.tasks:
            z, target, mask = demo[task]
            episode_ids = np.load(Path(args.demos) / f"demos_{task}.npz")["episode_ids"][demo_idx[task]]
            if len(z) != len(episode_ids) * args.draws:
                raise RuntimeError(f"{task}: {len(z)} feature rows != {len(episode_ids)} samples x {args.draws} draws")
            first = torch.from_numpy(np.tile(np.arange(args.draws) < args.val_draws, len(episode_ids)))
            episodes = sorted(set(episode_ids.tolist()))
            if args.val_demo_fraction > 0:
                n_val = max(1, round(args.val_demo_fraction * len(episodes)))
                held = np.isin(episode_ids, episodes[-n_val:])
                rows = torch.from_numpy(np.repeat(held, args.draws))
                demo[task] = (z[~rows], target[~rows], mask[~rows])
                val_info[task] = dict(source="held-out demonstrations", val_demonstrations=n_val,
                                      train_demonstrations=len(episodes) - n_val, val_samples=int(held.sum()))
            else:
                held = np.arange(len(episode_ids)) % args.monitor_every == 0
                rows = torch.from_numpy(np.repeat(held, args.draws))
                val_info[task] = dict(source="fixed subset of the training demonstrations",
                                      train_demonstrations=len(episodes), val_samples=int(held.sum()))
            val_demo[task] = (z[rows & first], target[rows & first], mask[rows & first])
            val_info[task]["val_rows"] = int((rows & first).sum())
        print(f"[train] convergence monitoring set: {val_info}", flush=True)
    n_draws = sum(len(v[0]) for v in demo.values())
    print(f"[train] demo features: {n_draws} (sample, noise, time) draws in {feature_s:.0f}s", flush=True)

    # ---- deployment buffer: H+, H- and local success prototypes -------------------------
    tic = time.time()
    plus, minus, protos, buffer_stats, gaps, gaps_all, centers = {}, {}, {}, {}, {}, {}, {}
    if not args.demo_only:
        for task in args.tasks:
            pool, h_plus, h_minus, onset_offset = load_buffer(Path(args.results) / task)
            buffer_stats[task] = dict(pool=0 if pool is None else len(pool), plus=0 if h_plus is None else len(h_plus),
                                      minus=0 if h_minus is None else len(h_minus))
            if h_plus is not None:
                plus[task] = h_plus.flatten(0, 1)
            if h_minus is not None and pool is not None:
                # prototypes are retrieved among representations of the same denoising step
                steps = h_minus.shape[1]
                if args.center_features == "task":
                    allh = torch.cat([x for x in (h_plus, h_minus) if x is not None])
                    mu = allh.mean(dim=0)  # (S, d): one mean per denoising step
                else:
                    mu = torch.zeros(h_minus.shape[1:])
                c_minus, c_pool = h_minus - mu, pool - mu
                proto = torch.stack([local_success_prototypes(c_minus[:, s].to(device), c_pool[:, s].to(device), args.k_nn).cpu()
                                     for s in range(steps)], dim=1)
                minus[task], protos[task] = h_minus.flatten(0, 1), proto.flatten(0, 1)
                centers[task] = mu[None].expand(len(h_minus), -1, -1).flatten(0, 1).half()
                gaps_all[task] = [prototype_distance_gap(c_minus[:, s].to(device), c_pool[:, s].to(device), args.k_nn)
                                  for s in range(steps)]
                # The margin is calibrated on the states right after the onset (where a patch should act);
                # H- itself still holds the onset chunk and every later one, as in the paper.
                early = torch.from_numpy(onset_offset < args.margin_onset_chunks) if args.margin_onset_chunks > 0 else None
                if early is not None and early.any():
                    gaps[task] = [prototype_distance_gap(c_minus[early, s].to(device), c_pool[:, s].to(device), args.k_nn)
                                  for s in range(steps)]
                else:
                    gaps[task] = gaps_all[task]
        print(f"[train] deployment buffer chunks: {buffer_stats}", flush=True)
    geometry, task_margins = {}, {}
    if gaps:
        d_fail = float(np.mean([g[0] for task in gaps for g in gaps[task]]))
        d_succ = float(np.mean([g[1] for task in gaps for g in gaps[task]]))
        geometry = dict(dist_fail_to_prototype=d_fail, dist_success_to_prototype=d_succ, gap=d_fail - d_succ)
        if args.margins == "relative":
            args.null_margin = args.redirect_margin = max(1e-4, args.margin_fraction * (d_fail - d_succ))
        for task in gaps:
            t_fail = float(np.mean([g[0] for g in gaps[task]]))
            t_succ = float(np.mean([g[1] for g in gaps[task]]))
            all_fail = float(np.mean([g[0] for g in gaps_all[task]]))
            geometry.setdefault("per_task", {})[task] = dict(
                dist_fail_to_prototype=t_fail, dist_success_to_prototype=t_succ, gap=t_fail - t_succ,
                margin_onset_chunks=args.margin_onset_chunks, dist_all_h_minus_to_prototype=all_fail,
                gap_all_h_minus=all_fail - t_succ)
            if args.margins == "task":
                m = max(1e-4, args.margin_fraction * (t_fail - t_succ))
                task_margins[task] = dict(null=m, redirect=m)
            else:
                task_margins[task] = dict(null=args.null_margin, redirect=args.redirect_margin)
            task_margins[task]["temperature"] = (
                args.fail_temperature * task_margins[task]["null"] / args.reference_margin
                if args.fail_temperature_scaling == "margin" else args.fail_temperature)
        if args.margins == "task":
            print(f"[train] per-task margins: { {t: round(v['null'], 4) for t, v in task_margins.items()} }, "
                  f"failure routing temperatures: { {t: round(v['temperature'], 4) for t, v in task_margins.items()} }",
                  flush=True)
        margin_text = ("margins per task (above)" if args.margins == "task" else
                       f"null margin {args.null_margin:.4f}, redirect margin {args.redirect_margin:.4f} ({args.margins})")
        print(f"[train] prototype distances: failure {d_fail:.3f}, success {d_succ:.3f}, gap {d_fail - d_succ:.3f} -> "
              + margin_text, flush=True)
    prototype_s = time.time() - tic

    # ---- stage 2: optimise the REB --------------------------------------------------------
    # The paper's energy penalties and utility threshold (lambda_u, lambda_T, m) assume the scale of
    # pi_0.5's hidden states. XR-1's have norm ~65, so with raw energies moving a representation by
    # 0.1 in the normalised space would cost lambda_u * 42 = 0.42 > the gain: the Redirect target
    # would always be Null. Measuring the residual relative to the typical hidden norm makes the
    # objective independent of the model's hidden scale.
    if args.energy_scale == "auto":
        norms = [t.norm(dim=-1).mean() for t in list(plus.values()) + list(minus.values())]
        if not norms:
            norms = [z.float().mean(dim=1).norm(dim=-1).mean() for z, _, _ in demo.values()]
        scale = float(torch.stack(norms).mean())
    else:
        scale = float(args.energy_scale)
    print(f"[train] hidden-norm scale for residual energies: {scale:.2f}", flush=True)
    reb = ResidualExpertBank(adapter.hidden_dim, args.n_experts, args.bottleneck).to(device)
    n_params = sum(p.numel() for p in reb.parameters())
    opt = torch.optim.AdamW(reb.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    converge = args.schedule == "converge"
    anneal_steps = args.min_steps if converge else args.steps
    decay = dict(start=None, length=None, reason=None)

    def temperature_at(step: int) -> float:
        """T_d: 1.0 -> 0.2 linearly over the annealing horizon, then constant."""
        return 1.0 + (0.2 - 1.0) * min(1.0, (step - 1) / max(1, anneal_steps - 1))

    def learning_rate(step: int) -> float:
        """Linear warm-up to the peak, then cosine decay (fixed: over the whole run; converge: only
        after convergence). The paper only states the peak value (2.5e-5); warm-up length and final
        value are the OpenPI defaults its code base uses."""
        if step <= args.lr_warmup_steps:
            return args.lr * step / max(1, args.lr_warmup_steps)
        if converge:
            if decay["start"] is None:
                return args.lr
            progress = min(1.0, (step - decay["start"]) / decay["length"])
        else:
            progress = (step - args.lr_warmup_steps) / max(1, args.steps - args.lr_warmup_steps)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * progress))

    for p in adapter.model.parameters():
        p.requires_grad_(False)

    def sample_scenes(store: dict, n: int):
        """Task-balanced batch of whole demonstration samples: n // draws samples, all their draws
        (rows are stored sample by sample, `--draws` rows each)."""
        g = args.draws
        tasks = list(store)
        picks = [tasks[i % len(tasks)] for i in torch.randperm(max(1, n // g)).tolist()]
        parts = []
        for task in tasks:
            count = picks.count(task)
            if count == 0:
                continue
            z, target, mask = store[task]
            scenes = torch.randint(0, len(z) // g, (count,))
            rows = (scenes[:, None] * g + torch.arange(g)).reshape(-1)
            parts.append((z[rows], target[rows], mask[rows]))
        return tuple(torch.cat(col).to(device).float() for col in zip(*parts))

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

    # every H- row carries its task's Null and Redirect margins
    minus_store = {t: (minus[t], protos[t],
                       torch.full((len(minus[t]),), task_margins[t]["null"] if t in task_margins else args.null_margin),
                       torch.full((len(minus[t]),), task_margins[t]["redirect"] if t in task_margins else args.redirect_margin),
                       torch.full((len(minus[t]),), task_margins[t]["temperature"] if t in task_margins else args.fail_temperature),
                       centers[t])
                   for t in minus}

    # fixed validation subsets of the deployment buffer (the same rows at every evaluation)
    g = torch.Generator().manual_seed(args.seed)
    pick = lambda n: torch.randperm(n, generator=g)[: args.val_buffer_size]  # noqa: E731
    val_plus = {t: plus[t][pick(len(plus[t]))] for t in plus}
    val_minus = {}
    for t in minus_store:
        idx = pick(len(minus_store[t][0]))
        val_minus[t] = tuple(x[idx] for x in minus_store[t])

    def chunks(item, size=256):
        first = item[0] if isinstance(item, tuple) else item
        for i in range(0, len(first), size):
            yield tuple(x[i : i + size].to(device).float() for x in (item if isinstance(item, tuple) else (item,)))

    @torch.no_grad()
    def evaluate() -> dict:
        """Validation metrics, averaged over chunks within a task and then over tasks (task-balanced
        like the minibatches). The total uses the training weights at T_d = 0.2 with Null enabled."""
        acc = {}

        def add(task, key, value, n):
            acc.setdefault(key, {}).setdefault(task, []).append((float(value), n))

        for task, item in val_demo.items():
            for z, target, mask in chunks(item):
                l_e, l_r, st = demo_loss(reb, z, target, mask, adapter.project, 0.2, True, reduction=args.action_loss,
                                         group=args.val_draws if args.demo_target == "scene" else 1)
                add(task, "demo_expert", l_e, len(z)); add(task, "demo_router", l_r, len(z))
                for k in ("loss_best", "loss_null", "demo_null_target"):
                    add(task, "demo_" + k.replace("demo_", ""), st[k], len(z))
        for task, item in val_plus.items():
            for (h,) in chunks(item):
                add(task, "preserve", preserve_loss(reb, h, scale), len(h))
                add(task, "null_top1_plus", (reb.log_probs(h).argmax(dim=1) == 0).float().mean(), len(h))
        for task, item in val_minus.items():
            for h, proto, m_null, m_red, t_fail, ctr in chunks(item):
                l_r, l_t, st = redirect_trust_loss(reb, h, proto, args.lambda_u, t_fail, m_null,
                                                   args.redirect_margin_weight, m_red, scale,
                                                   bootstrap=args.redirect_bootstrap, center=ctr)
                add(task, "redirect", l_r, len(h)); add(task, "trust", l_t, len(h))
                for k in ("redirect_null_target", "progress_best", "energy_best"):
                    add(task, k, st[k], len(h))
                add(task, "null_top1_minus", (reb.log_probs(h).argmax(dim=1) == 0).float().mean(), len(h))
        m = {k: float(np.mean([sum(v * n for v, n in xs) / sum(n for _, n in xs) for xs in per_task.values()]))
             for k, per_task in acc.items()}
        m["objective"] = (m.get("demo_expert", 0) + args.lambda_rd * m.get("demo_router", 0) + args.lambda_p * m.get("preserve", 0)
                          + args.lambda_r * m.get("redirect", 0) + args.lambda_t * m.get("trust", 0))
        return {"val_" + k: v for k, v in m.items()}

    history = []  # (step, metrics)
    best = dict(objective=float("inf"), step=None)  # lowest validation objective -> reb_best.pt

    def converged(step: int) -> tuple[bool, dict]:
        """Long-window test: compare the best value of the last `patience` steps with the best value
        before that window, and the routing now with the routing `patience` steps ago."""
        old = [(s_, m) for s_, m in history if s_ <= step - args.patience]
        new = [(s_, m) for s_, m in history if s_ > step - args.patience]
        if step < args.min_steps or not old or not new:
            return False, {}
        info = {}
        ok = True
        for key in ("val_objective", "val_demo_loss_best"):
            if key not in new[-1][1]:
                continue
            best_old, best_new = min(m[key] for _, m in old), min(m[key] for _, m in new)
            rel = (best_old - best_new) / max(abs(best_old), 1e-12)
            info[key + "_rel_improvement"] = rel
            ok &= rel < args.min_rel_improvement
        ref = old[-1][1]
        for key in ("val_null_top1_plus", "val_null_top1_minus"):
            if key in ref:
                change = abs(new[-1][1][key] - ref[key])
                info[key + "_change"] = change
                ok &= change < args.max_route_change
        return ok, info

    ckpt_dir = out / "checkpoints"
    tic = time.time()
    eval_s = 0.0
    log = (out / "train_log.jsonl").open("w")
    val_log = (out / "val_log.jsonl").open("w")
    step = 0
    while True:
        step += 1
        temperature = temperature_at(step)
        z, target, mask = sample_scenes(demo, args.batch_size) if args.demo_target == "scene" else sample(demo, args.batch_size)
        l_expert, l_router, stats = demo_loss(reb, z, target, mask, adapter.project, temperature,
                                              null_enabled=step > args.warmup_steps, reduction=args.action_loss,
                                              group=args.draws if args.demo_target == "scene" else 1)
        loss = l_expert + args.lambda_rd * l_router
        stats.update(expert=float(l_expert), router_demo=float(l_router))
        if plus:
            l_p = preserve_loss(reb, sample(plus, args.batch_size)[0], scale)
            loss = loss + args.lambda_p * l_p
            stats["preserve"] = float(l_p)
        if minus_store:
            h, proto, m_null, m_red, t_fail, ctr = sample(minus_store, args.batch_size)
            l_r, l_t, extra = redirect_trust_loss(reb, h, proto, args.lambda_u, t_fail,
                                                  m_null, args.redirect_margin_weight, m_red, scale,
                                                  bootstrap=args.redirect_bootstrap, center=ctr)
            loss = loss + args.lambda_r * l_r + args.lambda_t * l_t
            stats.update(redirect=float(l_r), trust=float(l_t), **extra)
        lr = learning_rate(step)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reb.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            stats.update(step=step, loss=float(loss), lr=lr, temperature=temperature, elapsed_s=time.time() - tic)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            if step % 2000 == 0 or step == 1:
                print("[train] " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()), flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(dict(config=reb.config(), state_dict=reb.state_dict(), args=vars(args), step=step),
                       ckpt_dir / f"reb_step{step:06d}.pt")

        if not converge:
            if step >= args.steps:
                decay.update(reason="fixed_steps")
                break
            continue
        if decay["start"] is not None and step >= decay["start"] + decay["length"]:
            break
        if step % args.eval_every == 0 or step == args.max_steps:
            t_eval = time.time()
            metrics = evaluate()
            eval_s += time.time() - t_eval
            history.append((step, metrics))
            if metrics["val_objective"] < best["objective"]:
                best.update(objective=metrics["val_objective"], step=step)
                torch.save(dict(config=reb.config(), state_dict=reb.state_dict(), args=vars(args), step=step),
                           out / "reb_best.pt")
            info = {}
            if decay["start"] is None:
                done, info = converged(step)
                length = max(args.decay_min_steps, round(args.decay_fraction * step))
                if done or step + length >= args.max_steps:
                    decay.update(start=step, length=min(length, max(1, args.max_steps - step)),
                                 reason="converged" if done else "max_steps", converged_at=step if done else None)
                    print(f"[train] {decay['reason']} at step {step} ({info}); learning-rate decay over "
                          f"{decay['length']} steps", flush=True)
            val_log.write(json.dumps(dict(step=step, lr=lr, temperature=temperature, **metrics, **info)) + "\n")
            val_log.flush()
            print("[val] step=%d %s" % (step, " ".join(f"{k[4:]}={v:.4g}" for k, v in metrics.items())), flush=True)
    final_val = evaluate() if converge else {}
    optimization_s = time.time() - tic
    log.close()
    val_log.close()

    torch.save(dict(config=reb.config(), state_dict=reb.state_dict(), args=vars(args)), out / "reb.pt")
    vla_params = sum(p.numel() for p in adapter.model.parameters())
    summary = dict(
        model=args.model, tasks=args.tasks, demo_only=args.demo_only,
        gpu=torch.cuda.get_device_name(0),
        sft_time_s=feature_s + prototype_s + optimization_s,
        feature_extraction_s=feature_s, features_from_cache=feature_cache_hits, prototype_s=prototype_s, optimization_s=optimization_s,
        model_load_s=load_s, steps=step, schedule=args.schedule, stop_reason=decay["reason"],
        converged_at_step=decay.get("converged_at"), decay_start_step=decay["start"], decay_steps=decay["length"],
        evaluation_s=eval_s, validation=val_info, final_val=final_val, best_val_objective=best["objective"] if best["step"] is not None else None,
        best_val_step=best["step"], demo_draws=n_draws,
        trainable_params=n_params, trainable_fraction=n_params / vla_params,
        buffer_chunks=buffer_stats, energy_scale=scale, geometry=geometry, action_loss=args.action_loss,
        margins=args.margins, null_margin=args.null_margin, redirect_margin=args.redirect_margin, task_margins=task_margins,
        redirect_bootstrap=args.redirect_bootstrap, peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        final=stats,
    )
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "final"}, indent=2))


if __name__ == "__main__":
    main()
