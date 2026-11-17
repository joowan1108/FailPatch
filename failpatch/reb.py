"""Null-gated Residual Expert Bank (REB) and the Preserve-Redirect-Trust objective of FailPatch
(arXiv 2609.34175, sections 3.2-3.3 and supplement S1).

The REB sits between the frozen VLA and its frozen action projection W_v:
  * router   p(h) = softmax(Linear(LayerNorm(h)))  over [Null, expert_1..K], h = mean of the
             action-token hidden states of the current chunk
  * expert k delta_k(z) = W_down( SiLU(W_gate z) * (W_up z) ), W_down zero-initialised
  * inference: Top-1 route; Null keeps z, expert k adds delta_k(z) to every action token.
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
        route = self.router(self.norm(zf.mean(dim=1))).argmax(dim=-1)
        out = zf.clone()
        for b in torch.nonzero(route > 0).flatten().tolist():
            k = int(route[b]) - 1
            hidden = F.silu(zf[b] @ self.w_gate[k]) * (zf[b] @ self.w_up[k])
            out[b] = zf[b] + hidden @ self.w_down[k]
        return out.to(dtype), route


def _balanced_soft_ce(target: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Class-balanced cross-entropy with soft targets: each route's contribution is divided by
    its share of the batch target mass, so rarely selected routes are not drowned out."""
    mass = target.mean(dim=0)
    weight = torch.where(mass > 0, 1.0 / (mass * (mass > 0).sum()).clamp_min(1e-8), torch.zeros_like(mass))
    return -(weight * target * log_p).sum(dim=-1).mean()


def demo_loss(reb, z, target, mask, project, temperature: float, null_enabled: bool, reduction: str = "sum"):
    """Action-grounded patch learning on successful demonstrations (eqs. 6-9).

    z (B, T, d) frozen action-token hidden states, target (B, T, A) flow-matching target,
    mask (B, T, A) valid action dims, project: frozen W_v mapping (B, T, d) -> (B, T, A).

    reduction: the paper writes l_k = ||W_v c_k - u||_2^2, a sum over the action entries ("sum").
    "mean" divides by the number of valid entries; the losses are then ~200x smaller than the
    temperature T_d (1.0 -> 0.2), every expert gets the same responsibility and the experts
    collapse onto one another (observed: pairwise cosine 0.7-0.8 after training).
    """
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0) if reduction == "mean" else 1.0
    per_sample = lambda pred: (((pred.float() - target) ** 2) * mask).sum(dim=(1, 2)) / denom  # noqa: E731
    with torch.no_grad():
        loss_null = per_sample(project(z))  # (B,)
    deltas = reb.deltas(z)  # (K, B, T, d)
    loss_k = torch.stack([per_sample(project(z + deltas[k])) for k in range(reb.k)], dim=1)  # (B, K)
    w = F.softmax(-loss_k.detach() / temperature, dim=1)
    loss_expert = (w * loss_k).sum(dim=1).mean()

    if null_enabled:
        q0 = (loss_k.detach().min(dim=1).values >= loss_null).float()
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
                        bootstrap: bool = False):
    """Failure-associated states H-: route to experts that move the representation toward the
    local success prototype (eqs. 11-13) with bounded intervention (eq. 14).

    proto is the prototype in the l2-normalised space, where all distances are measured."""
    dist = lambda x: (F.normalize(x, dim=-1) - proto).norm(dim=-1)  # noqa: E731
    deltas = reb.deltas(h)  # (K, B, d)
    energy = (deltas / scale).pow(2).sum(dim=-1).transpose(0, 1)  # (B, K)
    progress = dist(h)[:, None] - dist(h[None] + deltas).transpose(0, 1)  # (B, K)
    with torch.no_grad():
        utility = progress - lambda_u * energy
        q0 = (utility.max(dim=1).values <= null_margin).float()
        q = torch.cat([q0[:, None], (1 - q0)[:, None] * F.softmax(utility / temperature, dim=1)], dim=1)
    log_p = reb.log_probs(h)
    route_ce = -(q * log_p).sum(dim=1)
    # Paper: only experts already selected by the utility target (q_k > 0) are pushed towards the
    # prototype. If no expert ever clears the Null threshold, that push never starts; `bootstrap`
    # weights the push by the experts' relative utilities regardless of the Null gate.
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
