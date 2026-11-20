"""Model adapters: everything that is specific to one VLA lives here.

Both adapters expose the same interface (run inside that model's own Python environment):

  act(obs_list, reset, session_ids) -> actions (B, H, 12), hidden (B, S, d), routes (B, S)
      Official inference pipeline. `hidden` is the action hidden representation h of the paper
      (mean over the action tokens, taken right before the frozen action projection W_v) at each
      of the S denoising steps. If a Residual Expert Bank is attached it patches the hidden
      states and `routes` tells which route was taken (0 = Null).
  demo_features(sample, draws) -> z (M, H, d), target (M, H, A), mask (M, H, A)
      For one demonstration sample and M random (noise, time) draws: the frozen action-token
      hidden states and the flow-matching target, i.e. the inputs of the demo loss.
  project(z) -> W_v z, the frozen action projection (differentiable w.r.t. z).

An observation is {"frames": [raw gym obs at each of `frame_deltas`], "instruction": str};
a demonstration sample additionally has "actions" (H, 12) in the RoboCasa env order
[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)].
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import torch

from failpatch.noise_hook import inject_initial_noise, require_single_hit

ACTION_KEYS = ["end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode"]
ACTION_DIMS = [3, 3, 1, 4, 1]


def seed_everything(seed: int) -> None:
    """Seed every RNG the policy's inference may draw from (torch CPU/CUDA, numpy, random)."""
    import random

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class HiddenHook:
    """forward_pre_hook on the action projection: records the action-token hidden states and,
    when a REB is attached, replaces them with the patched ones."""

    def __init__(self, module, horizon: int):
        self.horizon = horizon
        self.mode = "off"  # off | infer | capture
        self.reb = None
        self.pooled: list[torch.Tensor] = []
        self.routes: list[torch.Tensor] = []
        self.tokens: torch.Tensor | None = None
        module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if self.mode == "off":
            return None
        full = args[0]
        z = full[:, -self.horizon :]
        if self.mode == "capture":
            self.tokens = z.detach().float()
            return None
        self.pooled.append(z.detach().float().mean(dim=1).cpu())
        if self.reb is None:
            self.routes.append(torch.zeros(z.shape[0], dtype=torch.long))
            return None
        patched, route = self.reb.patch(z)
        self.routes.append(route.cpu())
        full = torch.cat([full[:, : -self.horizon], patched], dim=1)
        return (full, *args[1:])

    @contextlib.contextmanager
    def running(self, mode: str):
        self.mode, self.pooled, self.routes, self.tokens = mode, [], [], None
        try:
            yield self
        finally:
            self.mode = "off"

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        hidden = torch.stack(self.pooled, dim=1).numpy().astype(np.float32)  # (B, S, d)
        routes = torch.stack(self.routes, dim=1).numpy().astype(np.int64)  # (B, S)
        return hidden, routes


class XR1Adapter:
    name = "xr1"
    robot_type = "robocasa365"

    # The action expert: everything except the VLM (DiT, its projections, timestep embedding, attention sink).
    ACTION_EXPERT = ("dit", "state_projector", "action_projector", "action_output_layer", "t_embedder", "t_projector", "sink")

    def __init__(self, model_path: str, repo: str, obs_history=4, obs_interval=2, crop_ratio=0.95, exec_steps=16,
                 action_expert: str | None = None):
        repo = Path(repo).resolve()
        sys.path.insert(0, str(repo / "deploy"))
        sys.path.insert(0, str(repo / "eval_robocasa365"))
        import entry  # official RoboCasa365 evaluator (pre/post-processing helpers)
        from transformers import AutoModel, AutoProcessor

        self.entry = entry
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = (
            AutoModel.from_pretrained(
                model_path, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
            ).cuda().to(torch.bfloat16).eval()
        )
        if action_expert:  # fine-tuned action-expert weights (failpatch.finetune) replace the released ones
            self.load_action_expert(action_expert)
        # Reuse EvalClient.infer for preprocessing; its socket client is replaced by `self._call`.
        self.client = entry.EvalClient.__new__(entry.EvalClient)
        self.client.client = self._call
        self.client.processor = self.processor
        self.client.robot_type = self.robot_type
        self.client.crop_ratio = crop_ratio

        mask = self.processor.get_action_mask(self.robot_type)[0].float()  # (H, D)
        self.horizon, self.action_width = int(mask.shape[0]), int(mask.shape[1])
        cfg = self.processor.action_config[self.robot_type]
        self.act_mean, self.act_std = cfg["mean"][0].float(), cfg["std"][0].float()  # (H, D)
        self.frame_deltas = [-(obs_history - 1 - i) * obs_interval for i in range(obs_history)]
        self.exec_steps = exec_steps
        # action_output_layer is an MLPProjector (the frozen action projection W_v of the paper)
        self.hidden_dim = int(self.model.action_output_layer.input_dim)
        self.hook = HiddenHook(self.model.action_output_layer, self.horizon)
        self._demo = None  # set while extracting demo features

    def load_action_expert(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        own = {k for k in self.model.state_dict() if k.split(".")[0] in self.ACTION_EXPERT}
        if set(state) != own:
            raise RuntimeError(f"{path}: action-expert keys differ (missing {sorted(own - set(state))[:5]}, "
                               f"unexpected {sorted(set(state) - own)[:5]})")
        self.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=False)
        print(f"[xr1] action expert loaded from {path}", flush=True)

    # -- shared preprocessing ------------------------------------------------------------
    def _to_device(self, data: dict) -> dict:
        m = self.model  # same placement rule as deploy/server.py
        return {
            k: (v.to(device=m.device, dtype=m.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.to(device=m.device) if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        }

    def _call(self, **data):
        robot_type = data.get("task_id")
        data = self._to_device(data)
        if self._demo is None:
            outputs = self.model(**data)
            return self.processor.decode_action(outputs.actions.cpu(), robot_type=robot_type)
        if self._demo.get("inputs_only"):
            self._demo["out"] = data
        else:
            self._demo["out"] = self._demo_forward(data, self._demo["actions"], self._demo["draws"])
        return torch.zeros(1, self.horizon, self.action_width)

    def _infer_one(self, obs: dict) -> np.ndarray:
        entry = self.entry
        frames = obs["frames"]
        states = np.stack([entry.observation_to_state(f) for f in frames])
        images = {k: np.stack([np.ascontiguousarray(f[k], dtype=np.uint8) for f in frames]) for k in entry.CAMERA_KEYS}
        return self.client.infer(states, images, obs["instruction"])

    # -- interface -----------------------------------------------------------------------
    def act(self, obs_list, reset, session_ids, noise_seeds=None):
        actions, hidden, routes = [], [], []
        for j, obs in enumerate(obs_list):
            if noise_seeds is not None:  # one query at a time, so each one gets its own sampling seed
                seed_everything(noise_seeds[j])
            with self.hook.running("infer"):
                actions.append(self._infer_one(obs))
                h, r = self.hook.result()
            hidden.append(h[0])
            routes.append(r[0])
        return np.stack(actions).astype(np.float32), np.stack(hidden), np.stack(routes)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        layer = self.model.action_output_layer
        return layer(z.to(next(layer.parameters()).dtype)).float()

    @torch.no_grad()
    def _demo_forward(self, data: dict, actions: np.ndarray, draws: int):
        """Mirrors MiBoTForActionGeneration.forward up to the action projection, but with the
        training-time interpolation of xr1/mibot/models/VLA/XR1.py:
            t = (1 - Beta(1.5, 1)) * 0.999,  x_t = (1 - t) noise + t action,  target = action - noise."""
        model = self.model
        state, action_mask = data.pop("state"), data.pop("action_mask")
        vlm_outputs = model.vlm(**data, use_cache=True)
        batch, length, _ = action_mask.shape
        query = length + state.shape[1] + 1
        position_ids = (
            torch.arange(0, query, device=action_mask.device).view(1, 1, -1).repeat(3, batch, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None] + 1
        )
        position_embeds = model.rotary_emb(action_mask, position_ids)
        dit_mask = torch.tril(torch.ones((batch, query, query), device=action_mask.device), diagonal=0)
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, query, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        state_embed = model.state_projector(state)

        mask = action_mask[0].float().cpu()
        clean = self.normalize_actions(actions, mask)
        beta = torch.distributions.Beta(1.5, 1.0)
        zs, targets = [], []
        for _ in range(draws):
            noise = torch.randn_like(clean).to(torch.bfloat16).float()
            t = float((1 - beta.sample()) * 0.999)
            noisy = ((1 - t) * noise + t * clean)[None].to(action_mask)
            t_in = torch.full((1, 1, 1), t, device=noisy.device, dtype=noisy.dtype)
            with self.hook.running("capture"):
                model.dit_forward(noisy, t_in, action_mask, state_embed, position_embeds,
                                  vlm_outputs.past_key_values, attn_mask)
                zs.append(self.hook.tokens[0].cpu())
            targets.append(clean - noise)
        return torch.stack(zs), torch.stack(targets), mask[None].expand(draws, -1, -1).clone()

    def model_inputs(self, sample: dict) -> dict:
        """The processed model inputs of one observation (on the GPU), as the official pipeline builds them."""
        self._demo = dict(inputs_only=True)
        try:
            self._infer_one(sample)
            return self._demo["out"]
        finally:
            self._demo = None

    def normalize_actions(self, actions: np.ndarray, mask: torch.Tensor) -> torch.Tensor:
        """Env-order action chunk (H, 12) -> normalised model action (H, D), zero outside the mask."""
        clean = torch.zeros(self.horizon, self.action_width)
        clean[:, : actions.shape[1]] = torch.from_numpy(np.asarray(actions, np.float32))
        mask = mask.float().cpu()
        return torch.where(mask > 0, (clean - self.act_mean) / self.act_std.clamp_min(1e-5), torch.zeros_like(clean))

    def demo_features(self, sample: dict, draws: int):
        self._demo = dict(actions=np.asarray(sample["actions"], np.float32), draws=draws)
        try:
            self._infer_one(sample)
            return self._demo["out"]
        finally:
            self._demo = None


class RLDXAdapter:
    name = "rldx"

    def __init__(self, model_path: str, exec_steps: int = 8, device: str = "cuda"):
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

        self.policy = RLDXPolicy(embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT, model_path=model_path, device=device)
        self.sim_policy = RLDXSimPolicyWrapper(self.policy)
        self.model = self.policy.model
        cfg = self.policy.modality_configs
        self.video_keys = list(cfg["video"].modality_keys)
        self.state_keys = list(cfg["state"].modality_keys)
        self.language_key = list(cfg["language"].modality_keys)[0]
        if list(cfg["action"].modality_keys) != ACTION_KEYS:
            raise RuntimeError(f"unexpected action keys {cfg['action'].modality_keys}")
        video_deltas = [int(d) for d in cfg["video"].delta_indices]
        state_deltas = [int(d) for d in cfg["state"].delta_indices]
        self.frame_deltas = sorted(set(video_deltas) | set(state_deltas))
        self.video_idx = [self.frame_deltas.index(d) for d in video_deltas]
        self.state_idx = [self.frame_deltas.index(d) for d in state_deltas]
        am = self.model.action_model
        self.horizon, self.action_width = int(am.config.action_horizon), int(am.action_dim)
        self.exec_steps = exec_steps
        self.hook = HiddenHook(am.action_decoder, self.horizon)
        self.hidden_dim = int(am.hidden_size)  # input width of the embodiment-specific action decoder
        self.use_memory = bool(self.policy.use_memory)
        self._embodiment_id = None

    def _video(self, obs, key):
        return np.stack([np.asarray(obs["frames"][j][f"video.{key}"], np.uint8) for j in self.video_idx])

    def _state(self, obs, key):
        return np.stack([np.asarray(obs["frames"][j][f"state.{key}"], np.float32).reshape(-1) for j in self.state_idx])

    def act(self, obs_list, reset, session_ids, noise_seeds=None):
        if noise_seeds is not None:
            raise NotImplementedError("fixed noise: RLDX-1 samples the whole batch with one call")
        observation = {f"video.{k}": np.stack([self._video(o, k) for o in obs_list]) for k in self.video_keys}
        observation.update({f"state.{k}": np.stack([self._state(o, k) for o in obs_list]) for k in self.state_keys})
        observation[self.language_key] = [str(o["instruction"]) for o in obs_list]
        options = {"reset_memory": [bool(x) for x in reset], "session_ids": [str(x) for x in session_ids]}
        with self.hook.running("infer"):
            action, _ = self.sim_policy.get_action(observation, options)
            hidden, routes = self.hook.result()
        actions = np.concatenate([action[f"action.{k}"] for k in ACTION_KEYS], axis=-1)
        return actions.astype(np.float32), hidden, routes

    def project(self, z: torch.Tensor) -> torch.Tensor:
        decoder = self.model.action_model.action_decoder
        emb = self._embodiment_id[:1].expand(z.shape[0])
        dtype = next(decoder.parameters()).dtype
        return decoder(z.to(dtype), emb).float()

    @torch.no_grad()
    def demo_features(self, sample: dict, draws: int):
        """Same processor / collator path as PolicyRuntime, with the demonstrated action chunk
        attached so the processor also returns the normalised action and its mask. The action
        head is then run in its training form (x_t = (1-t) noise + t action, target = action - noise)."""
        from rldx.data.types import MessageType
        from rldx.policy.policy_runtime import _rec_to_dtype

        if self.use_memory:
            raise NotImplementedError("demo features for memory-enabled RLDX checkpoints")
        runtime = self.policy.runtime
        obs = {
            "video": {k: self._video(sample, k) for k in self.video_keys},
            "state": {k: self._state(sample, k) for k in self.state_keys},
            "language": {self.language_key: [str(sample["instruction"])]},
        }
        step = runtime._to_vla_step_data(obs)
        actions = np.asarray(sample["actions"], np.float32)
        offsets = np.cumsum([0] + ACTION_DIMS)
        step.actions = {k: actions[:, offsets[i] : offsets[i + 1]] for i, k in enumerate(ACTION_KEYS)}
        processed = self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        collated = _rec_to_dtype(self.policy.collate_fn([processed]), dtype=torch.bfloat16)
        zs, targets = [], []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # the collator wraps the batch as {"inputs": ...} (PolicyRuntime calls get_action(**collated))
            backbone_inputs, action_inputs = self.model.prepare_input(dict(collated["inputs"]))
            backbone_outputs = self.model.backbone(backbone_inputs)
            self._embodiment_id = action_inputs.embodiment_id
            clean = action_inputs.action.float().cpu()[0]
            mask = action_inputs.action_mask.float().cpu()[0]
            for _ in range(draws):
                # The action head samples its noise with one torch.randn call; supply it so the
                # flow-matching target (action - noise) is known exactly.
                noise = torch.randn(1, *clean.shape).to(torch.bfloat16).float()
                with inject_initial_noise(noise, "randn") as state, self.hook.running("capture"):
                    self.model.action_model(backbone_outputs, action_inputs)
                    zs.append(self.hook.tokens[0].cpu())
                require_single_hit(state, noise, "RLDX-1 training forward")
                targets.append((clean - noise[0]) * mask)
        return torch.stack(zs), torch.stack(targets), mask[None].expand(draws, -1, -1).clone()


def build(name: str, args) -> "XR1Adapter | RLDXAdapter":
    if name == "xr1":
        return XR1Adapter(args.model_path, args.xr1_repo, action_expert=getattr(args, "action_expert", None))
    if name == "rldx":
        return RLDXAdapter(args.model_path)
    raise ValueError(name)
