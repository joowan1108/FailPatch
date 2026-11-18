"""Null-gated Residual Expert Bank (REB) and the Preserve-Redirect-Trust objective of FailPatch
(arXiv 2609.34175, sections 3.2-3.3 and supplement S1).

The REB sits between the frozen VLA and its frozen action projection W_v:
  * router   p(h) = softmax(Linear(LayerNorm(h)))  over [Null, expert_1..K], h = mean of the
             action-token hidden states of the current chunk
  * expert k delta_k(h) = W_down( SiLU(W_gate h) * (W_up h) ), W_down zero-initialised, computed
             from the pooled h (supplement eq. 18), so c_k = h + delta_k(h) exactly
  * inference: Top-1 route; Null keeps z, expert k adds the same delta_k(h) to every action token
             ("applied to all action-token hidden states", S1.1); the frozen W_v then maps each
             token to its action.
  Until 11-17 the residual was computed token by token (delta_k(z_t)); the Redirect/Preserve
  geometry, which is measured on the pooled h, then did not describe what inference executed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualExpertBank(nn.Module):
    def __init__(self, dim: int = 1024, n_experts: int = 8, bottleneck: int = 704):
        super().__init__()
        self.dim, self.k, self.bottleneck = dim, n_experts, bottleneck
        scale = dim**-0.5
        self.w_gate = nn.Parameter(torch.randn(n_experts, dim, bottleneck) * scale)
        self.w_up = nn.Parameter(torch.randn(n_experts, dim, bottleneck) * scale)
        self.w_down = nn.Parameter(torch.zeros(n_experts, bottleneck, dim))  # experts start as identity
        self.norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, n_experts + 1)
        nn.init.zeros_(self.router.bias)

    def config(self) -> dict:
        return dict(dim=self.dim, n_experts=self.k, bottleneck=self.bottleneck)

    def deltas(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., d) -> residuals of every expert, shape (K, ..., d)."""
        flat = z.reshape(1, -1, self.dim).expand(self.k, -1, -1)
        hidden = F.silu(torch.bmm(flat, self.w_gate)) * torch.bmm(flat, self.w_up)
        return torch.bmm(hidden, self.w_down).reshape(self.k, *z.shape)

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Pooled representation (B, d) -> log p over K+1 routes, probabilities clamped at 1e-6."""
        probs = F.softmax(self.router(self.norm(h)), dim=-1).clamp_min(1e-6)
        return probs.log()

    @torch.no_grad()
    def patch(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference. z (B, T, d) action-token hidden states -> (patched z, route (B,))."""
        dtype = z.dtype
        zf = z.float()
        h = zf.mean(dim=1)
        route = self.router(self.norm(h)).argmax(dim=-1)
        out = zf.clone()
        for b in torch.nonzero(route > 0).flatten().tolist():
            k = int(route[b]) - 1
            hidden = F.silu(h[b] @ self.w_gate[k]) * (h[b] @ self.w_up[k])
            out[b] = zf[b] + (hidden @ self.w_down[k])[None, :]  # same residual on every token
        return out.to(dtype), route


def _balanced_soft_ce(target: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Class-balanced cross-entropy with soft targets: each route's contribution is divided by
    its share of the batch target mass, so rarely selected routes are not drowned out."""
    mass = target.mean(dim=0)
    weight = torch.where(mass > 0, 1.0 / (mass * (mass > 0).sum()).clamp_min(1e-8), torch.zeros_like(mass))
    return -(weight * target * log_p).sum(dim=-1).mean()


def demo_loss(reb, z, target, mask, project, temperature: float, null_enabled: bool, reduction: str = "sum",
              group: int = 1):
    """Action-grounded patch learning on successful demonstrations (eqs. 6-9).

    z (B, T, d) frozen action-token hidden states, target (B, T, A) flow-matching target,
    mask (B, T, A) valid action dims, project: frozen W_v mapping (B, T, d) -> (B, T, A).

    reduction: the paper writes l_k = ||W_v c_k - u||_2^2, a sum over the action entries ("sum").
    "mean" divides by the number of valid entries; the losses are then ~200x smaller than the
    temperature T_d (1.0 -> 0.2), every expert gets the same responsibility and the experts
    collapse onto one another (observed: pairwise cosine 0.7-0.8 after training).

    group > 1: the batch holds `group` consecutive (noise, time) draws of the same demonstration
    sample. Responsibilities w_k and the Null decision are then computed from each expert's loss
    averaged over those draws (the expected loss of the scene), so the routing target says which
    expert suits the scene rather than which one happened to fit one noise draw. The expert loss
    itself is still per row.
    """
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0) if reduction == "mean" else 1.0
    per_sample = lambda pred: (((pred.float() - target) ** 2) * mask).sum(dim=(1, 2)) / denom  # noqa: E731
    with torch.no_grad():
        loss_null = per_sample(project(z))  # (B,)
    deltas = reb.deltas(z.float().mean(dim=1))  # (K, B, d): from the pooled h (supplement eq. 18)
    loss_k = torch.stack([per_sample(project(z + deltas[k][:, None, :])) for k in range(reb.k)], dim=1)  # (B, K)
    if group > 1:
        k = loss_k.shape[1]
        ref_k = loss_k.detach().view(-1, group, k).mean(dim=1).repeat_interleave(group, dim=0)
        ref_null = loss_null.view(-1, group).mean(dim=1).repeat_interleave(group, dim=0)
    else:
        ref_k, ref_null = loss_k.detach(), loss_null
    w = F.softmax(-ref_k / temperature, dim=1)
    loss_expert = (w * loss_k).sum(dim=1).mean()

    if null_enabled:
        q0 = (ref_k.min(dim=1).values >= ref_null).float()
    else:  # Null-disabled warm-up: responsibility only among the residual experts
        q0 = torch.zeros_like(loss_null)
    q = torch.cat([q0[:, None], (1 - q0)[:, None] * w], dim=1)
    loss_router = _balanced_soft_ce(q, reb.log_probs(z.mean(dim=1)))
    stats = dict(loss_null=float(loss_null.mean()), loss_best=float(loss_k.detach().min(dim=1).values.mean()),
                 demo_null_target=float(q0.mean()))
    return loss_expert, loss_router, stats


def preserve_loss(reb, h, scale: float = 1.0):
    """Reliable states H+: choose Null and keep residual energy small (eq. 10).
    `scale` is the typical norm of h; residual energy is measured relative to it (see train.py)."""
    log_p = reb.log_probs(h)
    energy = (reb.deltas(h) / scale).pow(2).sum(dim=-1).transpose(0, 1)  # (B, K)
    return (-log_p[:, 0] + (log_p[:, 1:].exp() * energy).sum(dim=1)).mean()


def redirect_trust_loss(reb, h, proto, lambda_u, temperature, null_margin, lambda_r, margin_r, scale: float = 1.0,
                        bootstrap: bool = False, center: torch.Tensor | None = None):
    """Failure-associated states H-: route to experts that move the representation toward the
    local success prototype (eqs. 11-13) with bounded intervention (eq. 14).

    proto is the prototype in the l2-normalised space, where all distances are measured.
    null_margin / margin_r: floats, or (B,) tensors when every row carries its own (task) margin."""
    if torch.is_tensor(null_margin):
        null_margin = null_margin.to(h.device)
    if torch.is_tensor(margin_r):
        margin_r = margin_r.to(h.device)[:, None]
    if torch.is_tensor(temperature):  # per-row (task) routing temperature
        temperature = temperature.to(h.device)[:, None]
    # center: (B, d) mean representation subtracted before normalising (None = the paper's space)
    shift = center.to(h.device) if center is not None else 0.0
    dist = lambda x: (F.normalize(x - shift, dim=-1) - proto).norm(dim=-1)  # noqa: E731
    deltas = reb.deltas(h)  # (K, B, d)
    energy = (deltas / scale).pow(2).sum(dim=-1).transpose(0, 1)  # (B, K)
    progress = dist(h)[:, None] - dist(h[None] + deltas).transpose(0, 1)  # (B, K)
    with torch.no_grad():
        utility = progress - lambda_u * energy
        q0 = (utility.max(dim=1).values <= null_margin).float()
        q = torch.cat([q0[:, None], (1 - q0)[:, None] * F.softmax(utility / temperature, dim=1)], dim=1)
    log_p = reb.log_probs(h)
    route_ce = -(q * log_p).sum(dim=1)
    # Paper: the push is weighted by the utility target q_k, which is non-zero for every expert
    # (softmax of the utilities) on states where at least one expert clears the Null threshold,
    # and zero on all other states. If no expert ever clears it, the push never starts;
    # `bootstrap` uses the same utility softmax on every H- state, regardless of the Null gate.
    push = F.softmax(utility / temperature, dim=1) if bootstrap else q[:, 1:]
    hinge = (push * F.relu(margin_r - progress)).sum(dim=1)
    loss_redirect = (route_ce + lambda_r * hinge).mean()
    loss_trust = (q[:, 1:] * energy).sum(dim=1).mean()
    stats = dict(redirect_null_target=float(q0.mean()), progress_best=float(progress.detach().max(dim=1).values.mean()),
                 energy_best=float(energy.detach().gather(1, progress.detach().argmax(dim=1, keepdim=True)).mean()))
    return loss_redirect, loss_trust, stats


@torch.no_grad()
def prototype_distance_gap(minus: torch.Tensor, pool: torch.Tensor, k_nn: int = 20) -> tuple[float, float]:
    """(mean distance of failure-associated representations to their local success prototype,
    mean distance of reliable-success representations to theirs, the query itself excluded).
    The difference is how far failures sit outside the success region, i.e. the largest progress
    a patch can meaningfully make."""
    pool_n = F.normalize(pool.float(), dim=-1)
    d_minus = (F.normalize(minus.float(), dim=-1) - local_success_prototypes(minus, pool, k_nn)).norm(dim=-1).mean()
    idx = torch.cdist(pool_n, pool_n).topk(min(k_nn + 1, len(pool_n)), largest=False).indices[:, 1:]
    d_pool = (pool_n - pool_n[idx].median(dim=1).values).norm(dim=-1).mean()
    return float(d_minus), float(d_pool)


@torch.no_grad()
def local_success_prototypes(queries: torch.Tensor, pool: torch.Tensor, k_nn: int = 20) -> torch.Tensor:
    """Coordinate-wise median of the k nearest (Euclidean, l2-normalised space) reliable-success
    representations for every failure-associated representation (supplement S1.3)."""
    pool_n = F.normalize(pool.float(), dim=-1)
    out = []
    for chunk in F.normalize(queries.float(), dim=-1).split(2048):
        idx = torch.cdist(chunk, pool_n).topk(min(k_nn, len(pool_n)), largest=False).indices
        out.append(pool_n[idx].median(dim=1).values)
    return torch.cat(out)
