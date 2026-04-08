"""
OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning.

This module contains the core OmniField architecture:
  - GaussianFourierFeatures (GFF) for spatial/temporal encoding
  - Sinusoidal initialization for learnable queries
  - CascadedBlock (per-modality encoder with cross-attention)
  - CascadedPerceiverIO (full model with MCT + ICMR + per-modality decoders)

Reference: Valencia et al., "OmniField: Conditioned Neural Fields for Robust
Multimodal Spatiotemporal Learning", ICLR 2026.
"""

import numpy as np
from math import log
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.latest_attn = None

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        self.latest_attn = attn.detach()
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Gaussian Fourier Features (GFF)  –  Section 4.1
# ---------------------------------------------------------------------------

class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier Features for spatial/temporal coordinate encoding.

    Replaces fixed sinusoidal Fourier features with randomly sampled
    frequencies from N(0, scale^2), yielding a richer spectral
    representation that captures high-frequency detail (Tancik et al., 2020).
    """

    def __init__(self, in_features: int, mapping_size: int, scale: float = 15.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

    def forward(self, coords):
        projections = coords @ self.B
        return torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)


# ---------------------------------------------------------------------------
# Sinusoidal Initialization  –  Section 4.1 / Appendix B.2
# ---------------------------------------------------------------------------

def get_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
    """Multi-scale sinusoidal initialization for learnable query tokens.

    Args:
        n: Number of latent tokens (num_latents).
        d: Embedding dimension (latent_dim). Must be even.

    Returns:
        Tensor of shape (n, d) with unit-norm rows.
    """
    assert d % 2 == 0, "latent_dim must be even for sinusoidal embeddings"
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# CascadedBlock  –  per-modality encoder stage
# ---------------------------------------------------------------------------

class CascadedBlock(nn.Module):
    """Single encoder stage: cross-attend from input tokens into learnable
    latents (sinusoidally initialized), optionally fusing a residual from
    the previous stage's global latent."""

    def __init__(self, dim, n_latents, input_dim, cross_heads, cross_dim_head,
                 self_heads, self_dim_head, residual_dim=None):
        super().__init__()
        self.latents = nn.Parameter(
            get_sinusoidal_embeddings(n_latents, dim), requires_grad=True
        )
        self.cross_attn = PreNorm(
            dim, Attention(dim, input_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=input_dim
        )
        self.self_attn = PreNorm(
            dim, Attention(dim, heads=self_heads, dim_head=self_dim_head)
        )
        self.residual_proj = (
            nn.Linear(residual_dim, dim) if residual_dim and residual_dim != dim else None
        )
        self.ff = PreNorm(dim, FeedForward(dim))

    def forward(self, x, context, mask=None, residual=None):
        b = context.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        latents = self.cross_attn(latents, context=context, mask=mask) + latents
        if residual is not None:
            if self.residual_proj:
                residual = self.residual_proj(residual)
            latents = latents + residual
        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


# ---------------------------------------------------------------------------
# CascadedPerceiverIO  –  Full OmniField Architecture  (Section 4)
# ---------------------------------------------------------------------------

class CascadedPerceiverIO(nn.Module):
    """OmniField: encoder–processor–decoder with MCT blocks and ICMR.

    Architecture overview (Fig. 2d in paper):
      - Per-modality input projections (value → embedding)
      - L stages of CascadedBlock per modality (encoder E_m)
      - At each stage, modality latents are concatenated (MCT) and
        processed through shared self-attention blocks (processor P)
      - Global feature z is mean-pooled and fed back as residual (ICMR)
      - Per-modality decoder heads decode from the final fused latent

    Args:
        input_dim:      Dimension of per-modality input tokens (after projection + pos enc).
        queries_dim:    Dimension of decoder query tokens (pos enc + time enc).
        logits_dim:     If set, project decoder output to this dim. None = identity.
        latent_dims:    Tuple of latent dimensions per ICMR stage.
        num_latents:    Tuple of number of latent tokens per stage.
        cross_heads:    Number of cross-attention heads.
        cross_dim_head: Dimension per cross-attention head.
        self_heads:     Number of self-attention heads.
        self_dim_head:  Dimension per self-attention head.
        decoder_ff:     Whether to include feedforward in decoder.
    """

    def __init__(
        self,
        *,
        input_dim,
        queries_dim,
        logits_dim=None,
        latent_dims=(128, 128, 128),
        num_latents=(128, 128, 128),
        cross_heads=4,
        cross_dim_head=128,
        self_heads=8,
        self_dim_head=128,
        decoder_ff=True,
    ):
        super().__init__()

        assert len(latent_dims) == len(num_latents)
        self.latent_dims = list(latent_dims)
        self.num_latents = list(num_latents)
        num_stages = len(latent_dims)
        final_latent_dim = latent_dims[-1]

        # --- Per-modality input projections ---
        self.input_proj_T = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.input_proj_Q = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.input_proj_V = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))

        # --- Per-modality encoder blocks (L stages each) ---
        def make_encoder_blocks():
            blocks = nn.ModuleList()
            prev_dim = None
            for dim, n_lat in zip(latent_dims, num_latents):
                blocks.append(CascadedBlock(
                    dim=dim, n_latents=n_lat, input_dim=input_dim,
                    cross_heads=cross_heads, cross_dim_head=cross_dim_head,
                    self_heads=self_heads, self_dim_head=self_dim_head,
                    residual_dim=prev_dim,
                ))
                prev_dim = dim
            return blocks

        self.encoder_blocks_T = make_encoder_blocks()
        self.encoder_blocks_Q = make_encoder_blocks()
        self.encoder_blocks_V = make_encoder_blocks()

        # --- ICMR: global-to-latent residual projections ---
        self.global2latent_proj_T = nn.ModuleList([
            nn.Linear(final_latent_dim, num_latents[i] * latent_dims[i])
            for i in range(num_stages)
        ])
        self.global2latent_proj_Q = nn.ModuleList([
            nn.Linear(final_latent_dim, num_latents[i] * latent_dims[i])
            for i in range(num_stages)
        ])
        self.global2latent_proj_V = nn.ModuleList([
            nn.Linear(final_latent_dim, num_latents[i] * latent_dims[i])
            for i in range(num_stages)
        ])

        # --- Shared self-attention trunk (processor P) ---
        self.self_attn_blocks = nn.Sequential(*[
            nn.Sequential(
                PreNorm(final_latent_dim, Attention(final_latent_dim, heads=self_heads, dim_head=self_dim_head)),
                PreNorm(final_latent_dim, FeedForward(final_latent_dim)),
            )
            for _ in range(3)
        ])

        # --- Cross-modal attention (MCT block components) ---
        def _cross_attn():
            return PreNorm(
                final_latent_dim,
                Attention(final_latent_dim, context_dim=final_latent_dim,
                          heads=cross_heads, dim_head=cross_dim_head),
            )

        self.cross_T_from_Q = _cross_attn()
        self.cross_T_from_V = _cross_attn()
        self.cross_Q_from_T = _cross_attn()
        self.cross_Q_from_V = _cross_attn()
        self.cross_V_from_T = _cross_attn()
        self.cross_V_from_Q = _cross_attn()

        # --- Per-modality decoder heads ---
        self.decoder_cross_attn_T = PreNorm(
            queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=final_latent_dim,
        )
        self.decoder_ff_T = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits_T = nn.Linear(queries_dim, 1)

        self.decoder_cross_attn_Q = PreNorm(
            queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=final_latent_dim,
        )
        self.decoder_ff_Q = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits_Q = nn.Linear(queries_dim, 1)

        self.decoder_cross_attn_V = PreNorm(
            queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=final_latent_dim,
        )
        self.decoder_ff_V = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits_V = nn.Linear(queries_dim, 1)

        # --- Query self-attention ---
        self.sa_queries_T = PreNorm(queries_dim, Attention(queries_dim, heads=4, dim_head=64))
        self.sa_queries_Q = PreNorm(queries_dim, Attention(queries_dim, heads=4, dim_head=64))
        self.sa_queries_V = PreNorm(queries_dim, Attention(queries_dim, heads=4, dim_head=64))

        # --- Global projection (for ICMR feedback) ---
        self.global_proj_T = nn.Linear(final_latent_dim, input_dim)
        self.global_proj_Q = nn.Linear(final_latent_dim, input_dim)
        self.global_proj_V = nn.Linear(final_latent_dim, input_dim)

        # Legacy encoder blocks (kept for checkpoint compat)
        self.encoder_blocks = nn.ModuleList()
        prev_dim = None
        for dim, n_lat in zip(latent_dims, num_latents):
            self.encoder_blocks.append(CascadedBlock(
                dim=dim, n_latents=n_lat, input_dim=input_dim,
                cross_heads=cross_heads, cross_dim_head=cross_dim_head,
                self_heads=self_heads, self_dim_head=self_dim_head,
                residual_dim=prev_dim,
            ))
            prev_dim = dim

        # Legacy shared decoder (kept for checkpoint compat)
        self.decoder_cross_attn = PreNorm(
            queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=final_latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        self.input_proj = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.projection_matrix = nn.Parameter(torch.randn(4, 128) / np.sqrt(4))

    def forward(self, x_T, x_Q, x_V, queries, used_modalities):
        """Forward pass with fleximodal fusion.

        Args:
            x_T: [B, S_T, input_dim] or None — encoded T modality tokens.
            x_Q: [B, S_Q, input_dim] or None — encoded Q modality tokens.
            x_V: [B, S_V, input_dim] or None — encoded V modality tokens.
            queries: [B, N, queries_dim] — decoder query tokens (pos + time).
            used_modalities: Tuple of 3 bools indicating which modalities are present.

        Returns:
            (T_out, Q_out, V_out): each [B, N, 1].
        """

        def residual_from_global(global_latent, proj_layer, n_latents_k, dim_k):
            if global_latent is None:
                return None
            G_pool = global_latent.mean(dim=1)                          # [B, Dg]
            R = proj_layer(G_pool).view(G_pool.size(0), n_latents_k, dim_k)
            return R

        global_latent = None
        num_stages = len(self.latent_dims)

        # === ICMR: iterative cross-modal refinement (Eq. 1) ===
        for stage_idx in range(num_stages):
            stage_latents = []
            nL_k = self.num_latents[stage_idx]
            d_k = self.latent_dims[stage_idx]

            # Per-modality encoding with global residual feedback
            if used_modalities[0] and x_T is not None:
                R_T = residual_from_global(global_latent, self.global2latent_proj_T[stage_idx], nL_k, d_k)
                latent_T = self.encoder_blocks_T[stage_idx](x=None, context=x_T, residual=R_T)
                stage_latents.append(latent_T)

            if used_modalities[1] and x_Q is not None:
                R_Q = residual_from_global(global_latent, self.global2latent_proj_Q[stage_idx], nL_k, d_k)
                latent_Q = self.encoder_blocks_Q[stage_idx](x=None, context=x_Q, residual=R_Q)
                stage_latents.append(latent_Q)

            if used_modalities[2] and x_V is not None:
                R_V = residual_from_global(global_latent, self.global2latent_proj_V[stage_idx], nL_k, d_k)
                latent_V = self.encoder_blocks_V[stage_idx](x=None, context=x_V, residual=R_V)
                stage_latents.append(latent_V)

            if not stage_latents:
                raise ValueError("No modalities present — cannot run forward pass.")

            # MCT: concatenate per-modality latents + shared self-attention
            fused_latent = torch.cat(stage_latents, dim=1)
            for sa_block in self.self_attn_blocks:
                fused_latent = sa_block[0](fused_latent) + fused_latent
                fused_latent = sa_block[1](fused_latent) + fused_latent

            # ICMR: mean-pool → global feature z^(k+1)
            global_latent = fused_latent

        # === Decoder ===
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=global_latent.size(0))

        def decode_branch(cross_attn, ff, head):
            q = queries
            x = cross_attn(q, context=global_latent)
            x = x + q
            if ff:
                x = x + ff(x)
            return head(x)

        T_out = decode_branch(self.decoder_cross_attn_T, self.decoder_ff_T, self.to_logits_T)
        Q_out = decode_branch(self.decoder_cross_attn_Q, self.decoder_ff_Q, self.to_logits_Q)
        V_out = decode_branch(self.decoder_cross_attn_V, self.decoder_ff_V, self.to_logits_V)

        return T_out, Q_out, V_out


# ---------------------------------------------------------------------------
# Convenience: build default OmniField for ClimSim-THW
# ---------------------------------------------------------------------------

def build_omnifield_climsim(device="cuda"):
    """Instantiate OmniField with default ClimSim-THW hyperparameters (Table A1)."""
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)
    time_enc = GaussianFourierFeatures(1, 16, scale=10.0).to(device)

    model = CascadedPerceiverIO(
        input_dim=192,       # 128 (value proj) + 64 (pos enc: 2*32)
        queries_dim=96,      # 64 (pos enc) + 32 (time enc: 2*16)
        logits_dim=None,
        latent_dims=(128, 128, 128),
        num_latents=(128, 128, 128),
        cross_heads=4,
        cross_dim_head=128,
        self_heads=8,
        self_dim_head=128,
        decoder_ff=True,
    ).to(device)

    return model, pos_enc, time_enc
