"""
KUT v13.3 — Minimal Unified Kähler Transformer (Occam Branch)
================================================================

Single source:

    K(z, zbar) = (alpha / 2) |z|^4 + beta |z|^2

v13.3-minimal keeps only the non-redundant Kähler channels:

  Geometry:    K -> g -> Gamma -> kappa
  Physics:     K-derived SSB + Calabi field evolution
  Information: Psi(h,z,K) + causal flow(Psi) -> logits / hbar
  Probability: hbar = pi / omega_bar sets readout and attention temperature

Occam reductions relative to v13.2:
  * no retarded field propagator
  * no wave / causal_laplacian force
  * no explicit Boltzmann exp/log prior
  * one kahler_bundle call per layer
  * attention is the sole token-mixing mechanism
  * loss = CE + Kähler-class anchor only

This is the main verification branch for speed, causal cleanliness, and flow-readout
activity. v13.2 remains the theory-complete comparison branch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover
    from dataclasses import dataclass as _dc
    @_dc
    class CausalLMOutputWithPast:
        loss: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None
        past_key_values: Optional[Tuple] = None
        hidden_states: Optional[Tuple] = None
        attentions: Optional[Tuple] = None


ARCH_TAG = "KUT_V13_3_MINIMAL_UNIFIED"
OMEGA_K_STAR = math.pi / math.e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KUTv13Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    complex_rank: int = 128
    rms_norm_eps: float = 1e-6
    metric_floor: float = 1e-8
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Shared Kähler geometry: these parameters define the single global K.
    init_alpha_raw: float = -3.0
    init_beta_raw: float = 0.0
    init_vacuum_sq: float = 1.0

    # Readout
    phi_init_scale: float = 0.0

    # Dynamics
    cognition_dt_scale: float = 1.0
    field_drive_scale: float = 1.0

    def __post_init__(self):
        if self.hidden_size != self.n_heads * self.head_dim:
            raise ValueError("hidden_size must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for complex split")
        if self.phi_init_scale <= 0.0:
            self.phi_init_scale = 1.0 / math.sqrt(max(self.complex_rank, 1))


def inverse_softplus(x: float) -> float:
    return math.log(math.exp(x) - 1.0) if x < 20.0 else x


# ---------------------------------------------------------------------------
# Causal operator used by flow readout
# ---------------------------------------------------------------------------

def causal_grad(x: torch.Tensor) -> torch.Tensor:
    """First-order causal gradient: x[t] - x[t-1]."""
    if x.size(1) <= 1:
        return torch.zeros_like(x)
    x_pad = F.pad(x, (0, 0, 1, 0), mode="replicate")
    return x - x_pad[:, :-1]


def causal_grad_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-step causal gradient: x[t] - x[t-k]. Multi-timescale prediction.
    Corresponds to brain's hierarchical prediction across cortical areas.
    k=1: temporal cortex (next word), k=2: parietal (short phrase), k=4: frontal (long range).
    """
    if x.size(1) <= 1 or k < 1:
        return torch.zeros_like(x)
    x_pad = F.pad(x, (0, 0, k, 0), mode="replicate")
    return x - x_pad[:, :-k]


# ---------------------------------------------------------------------------
# Kähler potential and derived quantities
# ---------------------------------------------------------------------------

def shell_chart_ratio(s: torch.Tensor) -> torch.Tensor:
    """Bounded shell ratio exp(tanh(s)) in [1/e, e]."""
    return torch.exp(torch.tanh(s))


def chart_to_z(s: torch.Tensor, theta: torch.Tensor, v_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = torch.sqrt(v_sq.clamp_min(1e-8)) * shell_chart_ratio(s)
    return rho * torch.cos(theta), rho * torch.sin(theta), rho


def force_to_chart(f_a: torch.Tensor, f_b: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_a = z_a / rho.clamp_min(1e-8)
    n_b = z_b / rho.clamp_min(1e-8)
    radial = n_a * f_a + n_b * f_b
    tangential = -n_b * f_a + n_a * f_b
    return radial / rho.clamp_min(1e-8), tangential / rho.clamp_min(1e-8)


def kahler_bundle(z_a: torch.Tensor, z_b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                  v_sq: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """All geometry derived from K=(alpha/2)r^2 + beta*r.

    No explicit Boltzmann exponential is computed in v13.3-minimal. The
    probability side enters through hbar=pi/omega_bar, which sets attention and
    readout temperature.
    """
    r = z_a.pow(2) + z_b.pow(2)
    K = 0.5 * alpha * r.pow(2) + beta * r
    K_vac = 0.5 * alpha * v_sq.pow(2) + beta * v_sq
    delta_K = K - K_vac

    g = 2.0 * alpha * r + beta + eps
    g_vac = 2.0 * alpha * v_sq + beta + eps
    g_ratio = (g / g_vac).clamp_min(eps)
    conn = 2.0 * alpha / g
    curv = (2.0 * alpha * beta) / g.pow(2)
    omega_k = torch.sqrt(g_vac / v_sq.clamp_min(eps))
    kappa_vac = (2.0 * alpha * beta) / g_vac.pow(2)

    # Exposed for diagnostics / future variants, not injected as a separate force.
    grad_pref = alpha * r + beta
    grad_a = grad_pref * z_a
    grad_b = grad_pref * z_b

    metric_amp = torch.sqrt(g_ratio.clamp_min(eps))

    return {
        "r": r, "K": K, "K_vac": K_vac, "delta_K": delta_K,
        "g": g, "g_vac": g_vac, "g_ratio": g_ratio,
        "conn": conn, "curv": curv, "omega_k": omega_k,
        "kappa_vac": kappa_vac,
        "gradK_a": grad_a, "gradK_b": grad_b,
        "metric_amp": metric_amp,
        "section_amp": metric_amp,
    }


def hbar_from_bath(alpha: torch.Tensor, beta: torch.Tensor, v_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_vac = 2.0 * alpha * v_sq + beta
    omega_bar = torch.sqrt(g_vac / v_sq.clamp_min(1e-8)).mean().clamp_min(1e-6)
    hbar = omega_bar.new_tensor(math.pi) / omega_bar
    return hbar, omega_bar


def state_section(h_re: torch.Tensor, h_im: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor,
                  section_amp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rho = torch.sqrt(z_a.pow(2) + z_b.pow(2) + 1e-8)
    cos_t = z_a / rho
    sin_t = z_b / rho
    w = section_amp.to(dtype=h_re.dtype, device=h_re.device)
    psi_a = w * (cos_t.to(h_re.dtype) * h_re - sin_t.to(h_re.dtype) * h_im)
    psi_b = w * (sin_t.to(h_re.dtype) * h_re + cos_t.to(h_re.dtype) * h_im)
    return psi_a, psi_b


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1).to(dtype=x.dtype)
    sin = sin.unsqueeze(1).to(dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Geometry-controlled computation
# ---------------------------------------------------------------------------

class RiemannianNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, g_hidden: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x * x * g_hidden.to(x.dtype)).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class CovariantGatedMLP(nn.Module):
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.gate = nn.Linear(dim, inter, bias=False)
        self.up = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor, g_hidden: torch.Tensor) -> torch.Tensor:
        dt = self.gate.weight.dtype
        gs = torch.sqrt(g_hidden.to(dt).clamp_min(1e-8))
        xg = x.to(dt) * gs
        y = self.down(torch.tanh(self.gate(xg)) * self.up(xg))
        return (y / gs).to(x.dtype)


def parallel_transport_hidden(h: torch.Tensor, conn_hidden: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    factor = torch.exp((-conn_hidden.to(h.dtype) * dt.to(h.dtype)).clamp(-2.0, 2.0))
    return h * factor


class KahlerAttention(nn.Module):
    """Kähler attention as parallel transport followed by one inner product.

    The expanded kernel is
        H + Gamma*A
      = q_a*k_a + q_b*k_b + Gamma*(q_a*k_b - q_b*k_a).

    Since Gamma is local to the query position in this model, the exact transported
    query is
        q'_a = q_a - Gamma*q_b,
        q'_b = q_b + Gamma*q_a,
    and therefore score = <q', k>.  This implements the first-principles operation:
    transport to a common tangent frame, then take one inner product.
    """
    def __init__(self, cfg: KUTv13Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_kv_groups = cfg.n_heads // cfg.n_kv_heads
        self.m_head = cfg.head_dim // 2

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)

    def forward(self, h: torch.Tensor, geo: Dict[str, torch.Tensor], hbar: torch.Tensor,
                position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t, _ = h.shape
        dtype = h.dtype

        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(position_ids)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.n_kv_groups > 1:
            k = k[:, :, None].expand(-1, -1, self.n_kv_groups, -1, -1).reshape(b, self.n_heads, t, self.head_dim)
            v = v[:, :, None].expand(-1, -1, self.n_kv_groups, -1, -1).reshape(b, self.n_heads, t, self.head_dim)

        # Parallel transport in the complex head chart.
        q_a, q_b = q[..., :self.m_head], q[..., self.m_head:]
        gamma_q = geo["conn"].mean(-1, keepdim=True).unsqueeze(1).to(dtype)  # [B,1,T,1]
        q_rot_a = q_a - gamma_q * q_b
        q_rot_b = q_b + gamma_q * q_a
        q_rot = torch.cat([q_rot_a, q_rot_b], dim=-1)

        # Temperature from Kähler frequency.  Multiplying Q keeps the operation in
        # standard scaled-dot-product form for FlashAttention-compatible paths.
        q_rot = q_rot * (math.pi / hbar.to(dtype).detach().clamp_min(1e-6))

        # Fast path: no padding and no explicit attention weights requested.
        all_tokens_valid = attention_mask is None or bool(torch.all(attention_mask != 0).item())
        if not output_attentions and all_tokens_valid:
            out = F.scaled_dot_product_attention(
                q_rot, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).reshape(b, t, -1)
            return self.o_proj(out), None

        # Fallback: still one QK^T matmul, but returns attention weights and handles padding.
        score = torch.matmul(q_rot, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(t, t, device=h.device, dtype=torch.bool).triu(diagonal=1)
        score = score.masked_fill(causal.unsqueeze(0).unsqueeze(0), torch.finfo(score.dtype).min)
        if attention_mask is not None:
            mask_2d = attention_mask.unsqueeze(1).unsqueeze(2)
            score = score.masked_fill(mask_2d == 0, torch.finfo(score.dtype).min)

        att = F.softmax(score.float(), dim=-1).to(dtype)
        out = torch.matmul(att, v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), att


class MinimalKahlerLayer(nn.Module):
    """One bundle per layer; attention is the only token mixer."""
    def __init__(self, cfg: KUTv13Config):
        super().__init__()
        self.cfg = cfg
        R = cfg.complex_rank
        self.attn_norm = RiemannianNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = KahlerAttention(cfg)
        self.mlp_norm = RiemannianNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = CovariantGatedMLP(cfg.hidden_size, cfg.intermediate_size)
        self.to_chart = nn.Linear(cfg.hidden_size, R * 2, bias=False)
        self.from_chart = nn.Linear(R * 2, cfg.hidden_size, bias=False)
        nn.init.normal_(self.from_chart.weight, mean=0.0, std=1e-4)

    def forward(self, h: torch.Tensor, s: torch.Tensor, theta: torch.Tensor,
                alpha: torch.Tensor, beta: torch.Tensor, v_sq: torch.Tensor,
                hbar: torch.Tensor, omega_bar: torch.Tensor,
                position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        R = self.cfg.complex_rank
        mdtype = h.dtype

        # 1. Single Kähler bundle at layer start.
        z_a, z_b, rho = chart_to_z(s, theta, v_sq)
        geo = kahler_bundle(z_a, z_b, alpha, beta, v_sq)
        dt = (1.0 / (self.cfg.num_hidden_layers * omega_bar.detach().clamp_min(1e-4))) * self.cfg.cognition_dt_scale

        g_hidden = geo["g_ratio"].mean(-1, keepdim=True).to(mdtype)
        conn_hidden = geo["conn"].mean(-1, keepdim=True).to(mdtype)

        # 2. Kähler attention: sole token-mixing channel.
        h_attn = self.attn_norm(h, g_hidden)
        attn_out, att = self.self_attn(h_attn, geo, hbar, position_ids, attention_mask, output_attentions=output_attentions)
        h = parallel_transport_hidden(h, conn_hidden, dt) + attn_out

        # 3. Field evolution: SSB + Calabi only.
        drive = self.to_chart(h_attn)
        g_vac = geo["g_vac"].clamp_min(1e-8)
        drive_s = self.cfg.field_drive_scale * torch.tanh(drive[..., :R].float()) / g_vac
        drive_theta = self.cfg.field_drive_scale * torch.tanh(drive[..., R:].float()) / g_vac

        kappa_vac = geo["kappa_vac"]
        ssb_a = -kappa_vac * (geo["r"] - v_sq) * z_a
        ssb_b = -kappa_vac * (geo["r"] - v_sq) * z_b
        curv_bar = kappa_vac.mean(-1, keepdim=True).detach()
        curv_centered = geo["curv"] - curv_bar
        calabi_a = -geo["omega_k"] * curv_centered * z_a
        calabi_b = -geo["omega_k"] * curv_centered * z_b

        d_s, d_theta = force_to_chart(ssb_a + calabi_a, ssb_b + calabi_b, z_a, z_b, rho)
        s = s + dt * (d_s + drive_s)
        theta = theta + dt * (d_theta + drive_theta)

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            s = s * m
            theta = theta * m

        # 4. Field -> hidden coupling. No second bundle.
        z_a_new, z_b_new, _ = chart_to_z(s, theta, v_sq)
        chart_feat = torch.cat([z_a_new, z_b_new], -1).to(mdtype)
        h = parallel_transport_hidden(h, conn_hidden, dt) + dt.to(mdtype) * self.from_chart(chart_feat)

        # 5. Covariant MLP using the same layer-start metric.
        h_mlp = self.mlp_norm(h, g_hidden)
        h = parallel_transport_hidden(h, conn_hidden, dt) + self.mlp(h_mlp, g_hidden)

        anchor = torch.log(geo["omega_k"].mean() / geo["omega_k"].new_tensor(OMEGA_K_STAR)).pow(2)
        curv_flat = geo["curv"].reshape(-1)
        ke = curv_flat.var(unbiased=False) / (curv_flat.mean().pow(2) + 1e-8)
        flow_vitality = (d_s.pow(2) + d_theta.pow(2)).mean() / (geo["r"].mean() + 1e-8)

        return {
            "h": h, "s": s, "theta": theta,
            "anchor_loss": anchor,
            "omega_bar": geo["omega_k"].mean().detach(),
            "ke_loss": ke.detach(),
            "flow_vitality": flow_vitality.detach(),
            "attentions": att if output_attentions else None,
        }


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

class KahlerFlowReadout(nn.Module):
    """Information readout: state section + multi-timescale causal flow section.

    Multi-scale flow corresponds to brain's hierarchical prediction:
      k=1: Ψ[t] - Ψ[t-1]  (temporal cortex, next-token prediction)
      k=2: Ψ[t] - Ψ[t-2]  (parietal cortex, short-phrase prediction)
      k=4: Ψ[t] - Ψ[t-4]  (frontal cortex, long-range prediction)

    All scales share the same Ξ basis — they are different velocities
    of the same section, combined before measurement. Zero extra parameters.
    """
    FLOW_SCALES = (1, 2, 4)

    def __init__(self, cfg: KUTv13Config):
        super().__init__()
        R = cfg.complex_rank
        self.proj_re = nn.Linear(cfg.hidden_size, R, bias=False)
        self.proj_im = nn.Linear(cfg.hidden_size, R, bias=False)
        std = cfg.phi_init_scale
        self.phi_re = nn.Parameter(torch.empty(cfg.vocab_size, R).normal_(0, std))
        self.phi_im = nn.Parameter(torch.empty(cfg.vocab_size, R).normal_(0, std))
        self.xi_re = nn.Parameter(torch.empty(cfg.vocab_size, R).normal_(0, std))
        self.xi_im = nn.Parameter(torch.empty(cfg.vocab_size, R).normal_(0, std))

    def forward(self, h: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor,
                section_amp: torch.Tensor, hbar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h_re = self.proj_re(h)
        h_im = self.proj_im(h)
        psi_a, psi_b = state_section(h_re, h_im, z_a, z_b, section_amp)
        state_logits = F.linear(psi_a, self.phi_re) + F.linear(psi_b, self.phi_im)

        # Multi-timescale flow: sum normalized flows from k=1,2,4
        flow_a = torch.zeros_like(psi_a)
        flow_b = torch.zeros_like(psi_b)
        for k in self.FLOW_SCALES:
            fk_a = causal_grad_k(psi_a, k)
            fk_b = causal_grad_k(psi_b, k)
            fk_norm = torch.sqrt(fk_a.pow(2) + fk_b.pow(2) + 1e-8).mean(-1, keepdim=True)
            flow_a = flow_a + fk_a / (1.0 + fk_norm)
            flow_b = flow_b + fk_b / (1.0 + fk_norm)

        flow_logits = F.linear(flow_a, self.xi_re) + F.linear(flow_b, self.xi_im)

        logits = (state_logits + flow_logits) / hbar.to(h.dtype).clamp_min(1e-6)
        state_rms = state_logits.detach().float().pow(2).mean().sqrt()
        flow_rms = flow_logits.detach().float().pow(2).mean().sqrt()
        diag = {
            "state_logit_rms": state_rms,
            "flow_logit_rms": flow_rms,
            "flow_logit_share": flow_rms / (state_rms + flow_rms + 1e-8),
            "flow_section_speed": torch.sqrt(flow_a.detach().float().pow(2) + flow_b.detach().float().pow(2) + 1e-8).mean(),
        }
        return logits, psi_a, psi_b, diag


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class KUTv13System(nn.Module):
    def __init__(self, cfg: KUTv13Config):
        super().__init__()
        cfg.__post_init__()
        self.config = cfg
        R = cfg.complex_rank
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.alpha_raw = nn.Parameter(torch.full((R,), cfg.init_alpha_raw))
        self.beta_raw = nn.Parameter(torch.full((R,), cfg.init_beta_raw))
        self.vacuum_sq_raw = nn.Parameter(torch.full((R,), inverse_softplus(cfg.init_vacuum_sq)))
        self.init_chart = nn.Linear(cfg.hidden_size, R * 2, bias=False)
        self.layers = nn.ModuleList([MinimalKahlerLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.final_norm = RiemannianNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.readout = KahlerFlowReadout(cfg)
        self._last_diagnostics: Dict[str, float] = {}
        self._last_ce_loss = None
        self._last_geo = None
        self._last_hbar_eff_oper = None
        self._last_information_state_norm = None
        self._last_information_alignment = None

    def _abv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = F.softplus(self.alpha_raw).unsqueeze(0).unsqueeze(0)
        b = (F.softplus(self.beta_raw) + self.config.metric_floor).unsqueeze(0).unsqueeze(0)
        v = F.softplus(self.vacuum_sq_raw).unsqueeze(0).unsqueeze(0)
        return a, b, v

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values=None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: bool = False, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: bool = True):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        bsz, t = h.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(bsz, t, device=h.device, dtype=torch.long)
        if position_ids is None:
            position_ids = torch.arange(t, device=h.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        alpha, beta, v_sq = self._abv()
        hbar, omega_bar = hbar_from_bath(alpha, beta, v_sq)
        hbar = hbar.to(device=h.device)
        self._last_hbar_eff_oper = hbar.detach()

        R = self.config.complex_rank
        chart_init = self.init_chart(h)
        s = torch.tanh(chart_init[..., :R].float())
        theta = math.pi * torch.tanh(chart_init[..., R:].float())

        all_hidden = [] if output_hidden_states else None
        all_att = [] if output_attentions else None
        diag_acc = {"omega_bar": [], "anchor_loss": [], "ke_loss": [], "flow_vitality": []}

        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(h)
            out = layer(h, s, theta, alpha, beta, v_sq, hbar, omega_bar, position_ids,
                        attention_mask=attention_mask, output_attentions=output_attentions)
            h, s, theta = out["h"], out["s"], out["theta"]
            for k in diag_acc:
                diag_acc[k].append(out[k] if torch.is_tensor(out[k]) else h.new_tensor(float(out[k])))
            if output_attentions:
                all_att.append(out["attentions"])

        z_a, z_b, _ = chart_to_z(s, theta, v_sq)
        geo_final = kahler_bundle(z_a, z_b, alpha, beta, v_sq)
        g_final = geo_final["g_ratio"].mean(-1, keepdim=True).to(h.dtype)
        h = self.final_norm(h, g_final)
        if output_hidden_states:
            all_hidden.append(h)

        logits, psi_a, psi_b, readout_diag = self.readout(h, z_a, z_b, geo_final["section_amp"], hbar)
        self._last_information_state_norm = torch.sqrt((psi_a.pow(2) + psi_b.pow(2)).sum(-1) + 1e-8).mean().detach()

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            self._last_ce_loss = ce.detach()
            anchor = torch.stack(diag_acc["anchor_loss"]).mean()
            loss = ce + anchor

            valid = shift_labels.ne(-100)
            safe = shift_labels.masked_fill(~valid, 0)
            sp_a, sp_b = psi_a[:, :-1], psi_b[:, :-1]
            ph_re = self.readout.phi_re[safe].to(h.dtype)
            ph_im = self.readout.phi_im[safe].to(h.dtype)
            pn = torch.sqrt((sp_a.pow(2) + sp_b.pow(2)).sum(-1, keepdim=True) + 1e-8)
            fn = torch.sqrt((ph_re.pow(2) + ph_im.pow(2)).sum(-1, keepdim=True) + 1e-8)
            cos_i = ((sp_a / pn) * (ph_re / fn) + (sp_b / pn) * (ph_im / fn)).sum(-1)
            geo_diag = (1.0 - cos_i).masked_select(valid).mean() if valid.any() else ce.new_zeros(())
            self._last_geo = geo_diag.detach()
            self._last_information_alignment = cos_i.masked_select(valid).mean().detach() if valid.any() else h.new_zeros(())

        # Stasis diagnostics only; not in loss.
        with torch.no_grad():
            if logits.size(1) > 2:
                sl = logits[:, :-1]
                top1 = F.softmax(sl.float(), dim=-1).amax(dim=-1).mean()
                logit_cos = F.cosine_similarity(sl[:, 1:].float(), sl[:, :-1].float(), dim=-1).clamp(-1, 1)
                logit_stasis = (0.5 * (1.0 + logit_cos)).mean()
                psi = torch.cat([psi_a[:, :-1], psi_b[:, :-1]], dim=-1).float()
                dpsi = psi[:, 1:] - psi[:, :-1]
                denom = (psi[:, 1:].norm(dim=-1) + psi[:, :-1].norm(dim=-1)).clamp_min(1e-8)
                tangent_speed = (dpsi.norm(dim=-1) / denom).clamp(0, 1).mean()
            else:
                top1 = logits.new_tensor(0.0)
                logit_stasis = logits.new_tensor(0.0)
                tangent_speed = logits.new_tensor(0.0)

        diag = {k: float(torch.stack(v).mean().detach()) for k, v in diag_acc.items()}
        diag.update({k: float(v.detach()) for k, v in readout_diag.items()})
        diag.update({
            "hbar_eff_oper": float(hbar.detach()),
            "top1_prob_mean": float(top1.detach()),
            "logit_stasis_mean": float(logit_stasis.detach()),
            "tangent_speed_mean": float(tangent_speed.detach()),
            "tangent_stasis_mean": float((1.0 - tangent_speed).detach()),
        })
        self._last_diagnostics = diag

        if not return_dict:
            return (loss, logits, None, tuple(all_hidden) if output_hidden_states else None, tuple(all_att) if output_attentions else None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
            attentions=tuple(all_att) if output_attentions else None,
        )


def create_model(cfg: Optional[KUTv13Config] = None, device: Optional[torch.device] = None) -> KUTv13System:
    if cfg is None:
        cfg = KUTv13Config()
    cfg.__post_init__()
    model = KUTv13System(cfg)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    cfg = KUTv13Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       n_heads=4, n_kv_heads=2, head_dim=16, complex_rank=16, max_position_embeddings=128)
    model = create_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids=x, labels=x)
    print("logits", tuple(out.logits.shape), "loss", float(out.loss))
    print(model._last_diagnostics)
