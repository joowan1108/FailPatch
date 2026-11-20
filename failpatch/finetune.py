"""Fine-tune the XR-1 action expert on demonstrations (the VLM stays frozen).

Used to build a base policy for one fixed kitchen from demonstrations of that kitchen, the way the paper's base is
trained on the demonstrations that FailPatch later reuses.

Trainable: the action expert (DiT, state / action / timestep projections, attention sink, action output layer).
Frozen: the VLM (Qwen3-VL); its KV cache is computed without gradients and shared by `--repeat` (noise, t) draws
of the same sample, like `training_repeat` of the official trainer.

Loss: the flow-matching part of the official post-training loss (xr1/mibot/models/VLA/XR1.py), without the
VLM-side choice / score losses (the VLM is frozen) and without the asynchronous action prefix:
    t = (1 - Beta(1.5, 1)) * 0.999,  x_t = (1 - t) noise + t a,  target = a - noise
    loss = 0.5 * masked MSE(pred, target) + freq_coefficient * masked |rfft(pred) - rfft(target)|
Optimiser and schedule: the official trainer config (AdamW betas (0.9, 0.95), weight decay 0.1, cosine schedule
with 500 warm-up steps from 5e-7 to 2e-5 and a 5e-6 floor, gradient clipping at 1.0).

Output: <out>/action_expert.pt (+ checkpoints/action_expert_stepXXXXXX.pt) with the fp32 action-expert weights,
loadable with `--action-expert` (server / train) or XR1_ACTION_EXPERT=<file> in scripts/model_env.sh.
Run inside the xr1 environment:
  python -m failpatch.finetune --model-path ... --xr1-repo ... --demos <dir with demos_<task>.npz> --task T --out D
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from failpatch.adapters import XR1Adapter

FREQ_EXCLUDED_DIMS = (17, 18, 19)  # official default (xr1 posttrain config)


def flow_loss(pred, target, mask, freq_coefficient):
    """compute_flow_loss of the official trainer with unit weights (no prefix)."""
    pred, target = pred.float(), target.float()
    m = mask.bool()
    mse = F.mse_loss(pred, target, reduction="none")[m].mean()
    freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
    fmask = m[:, : freq.shape[1]].clone()
    dims = [d for d in FREQ_EXCLUDED_DIMS if d < fmask.shape[-1]]
    fmask[:, :, dims] = False
    return 0.5 * mse + freq_coefficient * freq[fmask].mean(), mse


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.warmup_lr + (args.lr - args.warmup_lr) * step / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, adapter: XR1Adapter):
        self.a = adapter
        self.model = adapter.model
        self.model.requires_grad_(False)
        self.params = []
        for name in adapter.ACTION_EXPERT:
            module = getattr(self.model, name)
            module.float().requires_grad_(True)  # fp32 master weights, bf16 compute under autocast
            self.params += list(module.parameters())

    def condition(self, data: dict, repeat: int):
        """Frozen VLM pass and the DiT attention layout of MiBoTForActionGeneration.forward, for `repeat` draws
        of the same observation (the VLM cache is shared; the DiT attention does not broadcast it, so it is expanded)."""
        model = self.model
        state, action_mask = data.pop("state"), data.pop("action_mask")
        with torch.no_grad():
            vlm = model.vlm(**data, use_cache=True)
        pkv = [(vlm.past_key_values[i][0].expand(repeat, -1, -1, -1), vlm.past_key_values[i][1].expand(repeat, -1, -1, -1))
               for i in range(len(vlm.past_key_values))]
        mask_r = action_mask.expand(repeat, -1, -1)
        length = action_mask.shape[1]
        query = length + state.shape[1] + 1
        position_ids = (torch.arange(0, query, device=action_mask.device).view(1, 1, -1).repeat(3, repeat, 1)
                        + vlm.position_ids.max(dim=-1)[0][..., None] + 1)
        position_embeds = model.rotary_emb(mask_r, position_ids)
        dit_mask = torch.tril(torch.ones((repeat, query, query), device=action_mask.device))
        cache_mask = vlm.attention_mask[:, None, :].expand(repeat, query, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        return state, action_mask, mask_r, position_embeds, pkv, attn_mask

    def sample_loss(self, data: dict, actions: np.ndarray, repeat: int, freq_coefficient: float, gen=None):
        model = self.model
        state, action_mask, mask_r, position_embeds, pkv, attn_mask = self.condition(data, repeat)
        clean = self.a.normalize_actions(actions, action_mask[0]).to(action_mask.device)
        mask = action_mask[0].float()
        noise = torch.randn((repeat, *clean.shape), device=clean.device, generator=gen).to(torch.bfloat16).float()
        if gen is None:  # training: t = (1 - Beta(1.5, 1)) * 0.999
            t = (1 - torch.distributions.Beta(1.5, 1.0).sample((repeat,))).to(clean.device) * 0.999
        else:  # validation metric: fixed noise (generator) and evenly spaced t
            t = torch.linspace(0.05, 0.95, repeat, device=clean.device)
        t = t.view(-1, 1, 1)
        noisy = (1 - t) * noise + t * clean[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state_embed = model.state_projector(state).expand(repeat, -1, -1)
            pred = model.dit_forward(noisy.to(torch.bfloat16), t.to(torch.bfloat16), mask_r, state_embed,
                                     position_embeds, pkv, attn_mask)
        target = clean[None] - noise
        m = mask[None].expand(repeat, -1, -1)
        loss, mse = flow_loss(pred, target, m, freq_coefficient)
        # the FailPatch demo-loss metric: squared error summed over horizon and valid dims (mean over draws)
        sum_metric = ((pred.float() - target) ** 2 * m).sum(dim=(1, 2)).mean()
        return loss, mse, sum_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--xr1-repo", required=True)
    parser.add_argument("--demos", required=True, help="directory with demos_<task>.npz (failpatch.prepare_demos)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=8, help="demonstration samples per optimiser step")
    parser.add_argument("--repeat", type=int, default=4, help="(noise, t) draws per sample (official training_repeat)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-lr", type=float, default=5e-7)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freq-coefficient", type=float, default=1.0)
    parser.add_argument("--val-samples", type=int, default=200, help="fixed demo samples for the loss metric")
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    tic0 = time.time()
    adapter = XR1Adapter(args.model_path, args.xr1_repo)
    trainer = Trainer(adapter)
    n_params = sum(p.numel() for p in trainer.params)
    print(f"[finetune] trainable action expert: {n_params / 1e6:.1f}M parameters", flush=True)

    data = np.load(Path(args.demos) / f"demos_{args.task}.npz")
    if list(data["frame_deltas"]) != list(adapter.frame_deltas):
        raise RuntimeError(f"demo frames {list(data['frame_deltas'])} != model frames {adapter.frame_deltas}")
    keys = [k for k in data.files if k.startswith(("video.", "state."))]
    arrays = {k: data[k] for k in keys}
    actions, instructions = data["actions"], data["instructions"]
    n = len(actions)
    print(f"[finetune] {args.task}: {n} samples from {len(set(data['episode_ids'].tolist()))} demonstrations", flush=True)

    lock = threading.Lock()  # adapter.model_inputs keeps per-call state on the adapter

    def inputs(i):
        with lock:
            return adapter.model_inputs(sample(i))

    def sample(i):
        return {"frames": [{k: arrays[k][i, j] for k in keys} for j in range(len(adapter.frame_deltas))],
                "instruction": str(instructions[i])}

    val_idx = np.unique(np.linspace(0, n - 1, args.val_samples).round().astype(int))

    @torch.no_grad()
    def validate():
        gen = torch.Generator(device="cuda")
        tot = dict(loss=0.0, mse=0.0, sum=0.0)
        for i in val_idx:
            gen.manual_seed(int(i))
            loss, mse, s = trainer.sample_loss(inputs(i), actions[i], 4, args.freq_coefficient, gen)
            tot["loss"] += float(loss); tot["mse"] += float(mse); tot["sum"] += float(s)
        return {k: v / len(val_idx) for k, v in tot.items()}

    # preprocessing (image processor on the CPU) runs ahead in a thread
    order = queue.Queue(maxsize=4 * args.batch)

    def producer():
        try:
            while True:
                for i in rng.permutation(n):
                    order.put((int(i), inputs(int(i))))
        except BaseException as e:  # surface the error in the training loop instead of hanging
            order.put((None, e))
            raise

    opt = torch.optim.AdamW(trainer.params, lr=args.warmup_lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=args.weight_decay, fused=True)
    log = open(out / "finetune_log.jsonl", "a")
    v = validate()
    print(f"[finetune] step 0 val {v} ({time.time() - tic0:.0f}s)", flush=True)
    log.write(json.dumps({"step": 0, **{f"val_{k}": x for k, x in v.items()}}) + "\n"); log.flush()
    threading.Thread(target=producer, daemon=True).start()
    tic = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        opt.zero_grad(set_to_none=True)
        tl = tm = 0.0
        for _ in range(args.batch):
            i, batch_inputs = order.get()
            if i is None:
                raise RuntimeError("input preprocessing thread failed") from batch_inputs
            loss, mse, _ = trainer.sample_loss(batch_inputs, actions[i], args.repeat, args.freq_coefficient)
            (loss / args.batch).backward()
            tl += float(loss.detach()) / args.batch; tm += float(mse.detach()) / args.batch
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainer.params, args.grad_clip))
        opt.step()
        row = {"step": step, "loss": tl, "mse": tm, "grad_norm": gnorm, "lr": lr_at(step, args),
               "elapsed_s": time.time() - tic}
        if step % args.val_every == 0 or step == args.steps:
            v = validate()
            row.update({f"val_{k}": x for k, x in v.items()})
            print(f"[finetune] step {step} loss {tl:.4f} val {v} peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB "
                  f"({row['elapsed_s']:.0f}s)", flush=True)
        elif step % 25 == 0:
            print(f"[finetune] step {step} loss {tl:.4f} mse {tm:.4f} gnorm {gnorm:.2f} ({row['elapsed_s']:.0f}s)", flush=True)
        log.write(json.dumps(row) + "\n"); log.flush()
        if step % args.checkpoint_every == 0 or step == args.steps:
            state = {k: v.detach().cpu() for k, v in adapter.model.state_dict().items()
                     if k.split(".")[0] in adapter.ACTION_EXPERT}
            torch.save(state, out / "checkpoints" / f"action_expert_step{step:06d}.pt")
    torch.save(state, out / "action_expert.pt")
    summary = dict(task=args.task, demos=args.demos, samples=n, steps=args.steps, batch=args.batch, repeat=args.repeat,
                   trainable_params=n_params, finetune_time_s=time.time() - tic, total_time_s=time.time() - tic0,
                   gpu=torch.cuda.get_device_name(), peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                   final_val=v)
    (out / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[finetune] done: {summary}", flush=True)


if __name__ == "__main__":
    main()
