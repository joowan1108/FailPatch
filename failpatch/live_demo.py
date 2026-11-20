"""Fast live demonstration branch for XR-1 (paper S1.3: "the demonstration path performs a live
forward pass through the frozen backbone and retains the default image augmentation").

Per training step the paper draws a batch of 32 demonstration samples and runs the frozen policy on
them. Done one sample at a time this costs ~0.17 s per sample on an RTX 4090 (5.5 s per step).
This module makes the step fast in two ways, without changing what is computed:

  * CPU: a pool of worker processes builds the model inputs of each sample exactly as the official
    evaluator does (entry.EvalClient: centre crop, chat template, processor), after XR-1's training
    colour augmentation, and keeps the next batches ready while the GPU works on the current one.
  * GPU: the B samples go through the VLM and the action expert in ONE batched forward, the way the
    released model's own forward() handles a batch (per-sample position ids from the VLM, attention
    mask over the padded prompt), with an independent (noise, time) draw per sample.

`check_equivalence` compares the batched hidden states with the per-sample path of the adapter.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp  # registers the shared-memory reductions for tensors

_W: dict = {}  # worker globals


def _worker_init(model_path: str, repo: str, robot_type: str, crop_ratio: float, shared: dict, n_frames: int):
    torch.set_num_threads(1)
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo / "deploy"))
    sys.path.insert(0, str(repo / "eval_robocasa365"))
    import entry
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    client = entry.EvalClient.__new__(entry.EvalClient)
    client.processor, client.robot_type, client.crop_ratio = processor, robot_type, crop_ratio
    _W.update(entry=entry, client=client, shared=shared, n_frames=n_frames)


def _augment_frames(frames: list[dict], aug) -> list[dict]:
    """XR-1 training colour augmentation (mibot JsonDataset._augment) with pre-drawn parameters."""
    if aug is None:
        return frames
    import torchvision.transforms.functional as TF
    ops = [(TF.adjust_brightness, aug[0]), (TF.adjust_contrast, aug[1]), (TF.adjust_saturation, aug[2])]
    out = []
    for f in frames:
        g = dict(f)
        for k, v in f.items():
            if k.startswith("video."):
                img = torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1)
                for value, (op, _) in zip(aug, ops):
                    if value is not None:
                        img = op(img, float(value))
                g[k] = img.permute(1, 2, 0).numpy()
        out.append(g)
    return out


def _prepare(job):
    """(task, sample index, augmentation) -> processed model inputs of one sample (CPU tensors)."""
    task, i, aug = job
    arrays, keys, instructions = _W["shared"][task]
    frames = [{k: arrays[k][i, j].numpy() for k in keys} for j in range(_W["n_frames"])]
    frames = _augment_frames(frames, aug)
    entry, client = _W["entry"], _W["client"]
    states = np.stack([entry.observation_to_state(f) for f in frames])
    images = {k: np.stack([np.ascontiguousarray(f[k], dtype=np.uint8) for f in frames]) for k in entry.CAMERA_KEYS}
    captured = {}

    def capture(**data):
        captured.update(data)
        return torch.zeros(1, 16, 64)

    client.client = capture
    client.infer(states, images, str(instructions[i]))
    captured.pop("task_id", None)
    return task, i, captured


def collate(items: list[dict], pad_id: int) -> dict:
    """Batch the processed inputs of several samples like a processor call on a batch would:
    prompts left-padded, video patches and grids concatenated, state / action mask stacked."""
    length = max(x["input_ids"].shape[1] for x in items)
    ids = torch.full((len(items), length), pad_id, dtype=items[0]["input_ids"].dtype)
    att = torch.zeros((len(items), length), dtype=items[0]["attention_mask"].dtype)
    for b, x in enumerate(items):
        n = x["input_ids"].shape[1]
        ids[b, length - n:] = x["input_ids"][0]
        att[b, length - n:] = x["attention_mask"][0]
    out = dict(input_ids=ids, attention_mask=att,
               pixel_values_videos=torch.cat([x["pixel_values_videos"] for x in items]),
               video_grid_thw=torch.cat([x["video_grid_thw"] for x in items]),
               state=torch.cat([x["state"] for x in items]), action_mask=torch.cat([x["action_mask"] for x in items]))
    for k in items[0]:
        if k not in out:
            raise KeyError(f"unexpected processor output {k}")
    return out


class LiveDemoLoader:
    """Background producer of processed demonstration batches (CPU workers + prefetch queue)."""

    def __init__(self, adapter, model_path: str, repo: str, live: dict, batch: int, workers: int, augment: bool,
                 seed: int, prefetch: int = 3):
        self.adapter, self.live, self.batch, self.augment = adapter, live, batch, augment
        self.rng = np.random.default_rng(seed)
        shared = {}
        for task, L in live.items():
            arrays = {k: torch.from_numpy(np.ascontiguousarray(L["arrays"][k])).share_memory_() for k in L["keys"]}
            shared[task] = (arrays, L["keys"], L["instructions"])
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_worker_init,
                             initargs=(model_path, repo, adapter.robot_type, adapter.client.crop_ratio, shared,
                                       len(adapter.frame_deltas)))
        self.pad_id = adapter.processor.tokenizer.pad_token_id
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _jobs(self):
        tasks = list(self.live)
        picks = [tasks[i % len(tasks)] for i in self.rng.permutation(self.batch)]
        jobs = []
        for task in picks:
            i = int(self.rng.choice(self.live[task]["pool"]))
            aug = None
            if self.augment:  # each op applied with probability 1/2, same draw for all images of the sample
                values = (1.0 + self.rng.uniform(-32 / 255, 32 / 255), self.rng.uniform(0.5, 1.5), self.rng.uniform(0.5, 1.5))
                use = self.rng.integers(0, 2, size=3) == 0
                aug = tuple(v if u else None for v, u in zip(values, use))
            jobs.append((task, i, aug))
        return jobs

    def _run(self):
        try:
            while not self.stop:
                results = self.pool.map(_prepare, self._jobs(), chunksize=1)
                inputs = collate([r[2] for r in results], self.pad_id)
                actions = [self.live[t]["actions"][i] for t, i, _ in results]
                self.q.put((inputs, actions))
        except Exception as e:  # surfaced in next()
            self.q.put(e)

    def next(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise RuntimeError("live demonstration worker failed") from item
        inputs, actions = item
        return demo_forward_batch(self.adapter, inputs, actions)

    def close(self):
        self.stop = True
        self.pool.terminate()


@torch.no_grad()
def demo_forward_batch(adapter, inputs: dict, actions: list, noise: torch.Tensor | None = None,
                       t: torch.Tensor | None = None):
    """Batched counterpart of XR1Adapter._demo_forward with one (noise, time) draw per sample:
    -> z (B, H, d) action-token hidden states before the action projection, target (B, H, D), mask (B, H, D)."""
    model = adapter.model
    data = adapter._to_device({k: v.pin_memory() if v.device.type == "cpu" else v for k, v in inputs.items()})
    state, action_mask = data.pop("state"), data.pop("action_mask")
    vlm_outputs = model.vlm(**data, use_cache=True)
    batch, length, _ = action_mask.shape
    query = length + state.shape[1] + 1
    position_ids = (torch.arange(0, query, device=action_mask.device).view(1, 1, -1).repeat(3, batch, 1)
                    + vlm_outputs.position_ids.max(dim=-1)[0][..., None] + 1)
    position_embeds = model.rotary_emb(action_mask, position_ids)
    dit_mask = torch.tril(torch.ones((batch, query, query), device=action_mask.device), diagonal=0)
    cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, query, -1)
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    state_embed = model.state_projector(state)

    mask = action_mask.float().cpu()
    clean = torch.stack([adapter.normalize_actions(a, mask[b]) for b, a in enumerate(actions)])  # (B, H, D)
    if noise is None:
        noise = torch.randn_like(clean).to(torch.bfloat16).float()
    if t is None:
        t = (1 - torch.distributions.Beta(1.5, 1.0).sample((batch,))) * 0.999
    noisy = ((1 - t[:, None, None]) * noise + t[:, None, None] * clean).to(action_mask)
    t_in = t.view(batch, 1, 1).to(noisy)
    with adapter.hook.running("capture"):
        model.dit_forward(noisy, t_in, action_mask, state_embed, position_embeds, vlm_outputs.past_key_values, attn_mask)
        z = adapter.hook.tokens
    return z, (clean - noise).to(z.device), mask.to(z.device)


def check_equivalence(adapter, model_path: str, repo: str, live: dict, n: int = 4, workers: int = 2):
    """Batched forward (workers + collate + one forward) vs the adapter's per-sample path, same draws."""
    task = next(iter(live))
    L = live[task]
    shared = {task: ({k: torch.from_numpy(np.ascontiguousarray(L["arrays"][k][:n])) for k in L["keys"]},
                     L["keys"], L["instructions"])}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init,
                  initargs=(model_path, repo, adapter.robot_type, adapter.client.crop_ratio, shared,
                            len(adapter.frame_deltas))) as pool:
        results = pool.map(_prepare, [(task, i, None) for i in range(n)])
    inputs = collate([r[2] for r in results], adapter.processor.tokenizer.pad_token_id)
    actions = [L["actions"][i] for i in range(n)]
    lengths = [r[2]["input_ids"].shape[1] for r in results]
    clean_shape = (n, adapter.horizon, adapter.action_width)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(clean_shape, generator=g).to(torch.bfloat16).float()
    t = torch.rand(n, generator=g) * 0.999
    z_batch, target_batch, _ = demo_forward_batch(adapter, inputs, actions, noise, t)
    diffs = []
    for b in range(n):
        single = {k: (v[b:b + 1] if k in ("state", "action_mask") else v) for k, v in results[b][2].items()}
        z1, target1, _ = demo_forward_batch(adapter, single, [actions[b]], noise[b:b + 1], t[b:b + 1])
        rel = float((z_batch[b] - z1[0]).norm() / z1[0].norm())
        diffs.append(rel)
        assert torch.allclose(target_batch[b], target1[0])
    return dict(prompt_lengths=lengths, rel_diff_batched_vs_single=diffs)
