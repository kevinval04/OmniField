"""
OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning.

Core architecture for ClimSim-THW (3 modalities: Temperature, Humidity, Wind speed).

Components:
  - GaussianFourierFeatures (GFF)          — Section 4.1
  - Sinusoidal initialization              — Section 4.1 / Appendix B.2
  - CascadedBlock (per-modality encoder)   — Section 4, E_m
  - CascadedPerceiverIO (full OmniField)   — Section 4.2 (MCT + ICMR)

Reference: Valencia et al., "OmniField: Conditioned Neural Fields for Robust
Multimodal Spatiotemporal Learning", ICLR 2026.
"""

import numpy as np
from math import log

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
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Gaussian Fourier Features (GFF)  —  Section 4.1
# ---------------------------------------------------------------------------

class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier Features for spatial/temporal coordinate encoding.

    Replaces fixed sinusoidal Fourier features with randomly sampled
    frequencies from N(0, scale^2), yielding a richer spectral
    representation (Tancik et al., 2020).

    Output dim = 2 * mapping_size (sin + cos).
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
# Sinusoidal Initialization  —  Section 4.1 / Appendix B.2
# ---------------------------------------------------------------------------

def get_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
    """Multi-scale sinusoidal initialization for learnable query tokens.

    Each row has unit norm (s = d^{-1/2}), keeping initial attention
    logits well-scaled. Uses log-spaced frequency bands.
    """
    assert d % 2 == 0, "latent_dim must be even for sinusoidal embeddings"
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# CascadedBlock  —  per-modality encoder stage (E_m in paper)
# ---------------------------------------------------------------------------

class CascadedBlock(nn.Module):
    """Single encoder stage: cross-attend from input tokens into
    sinusoidally-initialized learnable latents, with optional residual
    from global feature z (ICMR feedback, Eq. 1)."""

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

        # Cross-attend: learnable latents query the per-modality input tokens
        latents = self.cross_attn(latents, context=context, mask=mask) + latents

        # ICMR residual: inject global feature z from previous stage (⊕ in Eq. 1)
        if residual is not None:
            if self.residual_proj:
                residual = self.residual_proj(residual)
            latents = latents + residual

        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


# ---------------------------------------------------------------------------
# OmniField base  —  shared ICMR + decode logic
# ---------------------------------------------------------------------------

class OmniFieldBase(nn.Module):
    """Base class implementing the ICMR loop and decoder pattern.

    Subclasses define modality-specific encoders, projections, and decoders.
    This keeps the core MCT+ICMR logic (Eq. 1) in one place.
    """

    def _icmr_forward(self, xs, used, enc_blocks_list, g2l_projs_list):
        """Run the ICMR loop over L stages.

        Args:
            xs: list of per-modality token tensors (or None if absent).
            used: list of bools — which modalities are present.
            enc_blocks_list: list of nn.ModuleList, one per modality.
            g2l_projs_list: list of nn.ModuleList (global→latent projections),
                            one per modality. If a single nn.ModuleList is passed,
                            it is shared across all modalities.

        Returns:
            global_latent: [B, n_fused, D] — the final fused latent g = h^(L-1).
        """
        shared_proj = not isinstance(g2l_projs_list, list)

        def residual_from_global(global_latent, proj_layer, n_lat, d_lat):
            if global_latent is None:
                return None
            G_pool = global_latent.mean(dim=1)  # z^(k+1) = mean(h^(k))
            return proj_layer(G_pool).view(G_pool.size(0), n_lat, d_lat)

        global_latent = None  # z^(0) = 0

        for stage_idx in range(len(self.latent_dims)):
            stage_latents = []
            nL = self.num_latents[stage_idx]
            dL = self.latent_dims[stage_idx]

            for mi in range(len(xs)):
                if not (used[mi] and xs[mi] is not None):
                    continue
                if shared_proj:
                    proj = g2l_projs_list[stage_idx]
                else:
                    proj = g2l_projs_list[mi][stage_idx]
                R = residual_from_global(global_latent, proj, nL, dL)
                latent_m = enc_blocks_list[mi][stage_idx](x=None, context=xs[mi], residual=R)
                stage_latents.append(latent_m)

            if not stage_latents:
                if global_latent is None:
                    raise ValueError("No modalities present — cannot run forward pass.")
                continue  # carry previous global forward

            # MCT: ⊙ concatenate + P self-attn trunk
            fused = torch.cat(stage_latents, dim=1)
            for sa_block in self.self_attn_blocks:
                fused = sa_block[0](fused) + fused
                fused = sa_block[1](fused) + fused
            global_latent = fused

        return global_latent

    def _decode(self, queries, global_latent, decoder_specs):
        """Decode per-modality outputs from global latent.

        Args:
            queries: [B, Nq, queries_dim]
            global_latent: [B, n_fused, D]
            decoder_specs: list of (cross_attn, ff_or_None, linear_head)

        Returns:
            list of [B, Nq, 1] tensors.
        """
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(global_latent.size(0), -1, -1)

        outputs = []
        for cross_attn, ff, head in decoder_specs:
            x = cross_attn(queries, context=global_latent) + queries
            if ff is not None:
                x = x + ff(x)
            outputs.append(head(x))
        return outputs


# ---------------------------------------------------------------------------
# CascadedPerceiverIO  —  OmniField for ClimSim-THW (3 modalities)
# ---------------------------------------------------------------------------

class CascadedPerceiverIO(OmniFieldBase):
    """OmniField for ClimSim-THW: Temperature, Humidity, Wind speed.

    Hyperparameters from Appendix A:
        input_dim=192, queries_dim=96, latent_dims=(128,128,128),
        num_latents=(128,128,128), cross_heads=4, cross_dim_head=128,
        self_heads=8, self_dim_head=128, decoder_ff=True, trunk_layers=3.
    """

    def __init__(
        self, *, input_dim, queries_dim, logits_dim=None,
        latent_dims=(128, 128, 128), num_latents=(128, 128, 128),
        cross_heads=4, cross_dim_head=128, self_heads=8, self_dim_head=128,
        decoder_ff=True, num_trunk_layers=3,
    ):
        super().__init__()
        assert len(latent_dims) == len(num_latents)
        self.latent_dims = list(latent_dims)
        self.num_latents = list(num_latents)
        final_dim = latent_dims[-1]

        # Per-modality input projections: [lon, lat, value] → R^128
        self.input_proj_T = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.input_proj_Q = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.input_proj_V = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))

        # Per-modality encoder blocks E_m
        def _enc():
            blocks = nn.ModuleList()
            prev = None
            for d, n in zip(latent_dims, num_latents):
                blocks.append(CascadedBlock(d, n, input_dim, cross_heads, cross_dim_head,
                                            self_heads, self_dim_head, prev))
                prev = d
            return blocks

        self.encoder_blocks_T = _enc()
        self.encoder_blocks_Q = _enc()
        self.encoder_blocks_V = _enc()

        # ICMR: per-modality global→latent residual projections
        def _g2l():
            return nn.ModuleList([
                nn.Linear(final_dim, num_latents[i] * latent_dims[i])
                for i in range(len(latent_dims))
            ])

        self.global2latent_proj_T = _g2l()
        self.global2latent_proj_Q = _g2l()
        self.global2latent_proj_V = _g2l()

        # Processor P: shared self-attention trunk
        self.self_attn_blocks = nn.Sequential(*[
            nn.Sequential(
                PreNorm(final_dim, Attention(final_dim, heads=self_heads, dim_head=self_dim_head)),
                PreNorm(final_dim, FeedForward(final_dim)),
            ) for _ in range(num_trunk_layers)
        ])

        # Per-modality decoder heads D_{ω,m}
        def _dec():
            return (
                PreNorm(queries_dim, Attention(queries_dim, final_dim, heads=cross_heads,
                        dim_head=cross_dim_head), context_dim=final_dim),
                PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None,
                nn.Linear(queries_dim, 1),
            )

        self.decoder_cross_attn_T, self.decoder_ff_T, self.to_logits_T = _dec()
        self.decoder_cross_attn_Q, self.decoder_ff_Q, self.to_logits_Q = _dec()
        self.decoder_cross_attn_V, self.decoder_ff_V, self.to_logits_V = _dec()

    def forward(self, x_T, x_Q, x_V, queries, used_modalities):
        """
        Args:
            x_T/Q/V: [B, S_m, input_dim] or None.
            queries: [B, N, queries_dim].
            used_modalities: (bool, bool, bool).
        Returns:
            (T_out, Q_out, V_out): each [B, N, 1].
        """
        global_latent = self._icmr_forward(
            xs=[x_T, x_Q, x_V],
            used=used_modalities,
            enc_blocks_list=[self.encoder_blocks_T, self.encoder_blocks_Q, self.encoder_blocks_V],
            g2l_projs_list=[self.global2latent_proj_T, self.global2latent_proj_Q, self.global2latent_proj_V],
        )
        T_out, Q_out, V_out = self._decode(queries, global_latent, [
            (self.decoder_cross_attn_T, self.decoder_ff_T, self.to_logits_T),
            (self.decoder_cross_attn_Q, self.decoder_ff_Q, self.to_logits_Q),
            (self.decoder_cross_attn_V, self.decoder_ff_V, self.to_logits_V),
        ])
        return T_out, Q_out, V_out


# ---------------------------------------------------------------------------
# OmniFieldAQS  —  OmniField for EPA-AQS (6 modalities)
# ---------------------------------------------------------------------------

class OmniFieldAQS(OmniFieldBase):
    """OmniField for EPA-AQS: O3, PM2.5, PM10, NO2, CO, SO2.

    Key difference from ClimSim variant: uses a **shared** global→latent
    projection across all modalities (single set of projection layers),
    matching the paper's Eq. 1 where z is a single global feature.

    Hyperparameters from Appendix A:
        input_dim=128, queries_dim=64, latent_dims=(64,64,64),
        num_latents=(64,64,64), cross_heads=2, cross_dim_head=32,
        self_heads=2, self_dim_head=32, decoder_ff=False, trunk_layers=3.
    """

    MODALITIES = ["Ozone", "PM2.5", "PM10", "NO2", "CO", "SO2"]

    def __init__(
        self, *, input_dim=128, queries_dim=64, logits_dim=None,
        latent_dims=(64, 64, 64), num_latents=(64, 64, 64),
        cross_heads=2, cross_dim_head=32, self_heads=2, self_dim_head=32,
        decoder_ff=False, num_trunk_layers=3,
    ):
        super().__init__()
        assert len(latent_dims) == len(num_latents)
        self.latent_dims = list(latent_dims)
        self.num_latents = list(num_latents)
        final_dim = latent_dims[-1]

        # Per-modality input projections: [AQI, lat, lon] → R^input_dim
        def _proj():
            return nn.Sequential(nn.Linear(3, input_dim), nn.GELU(), nn.Linear(input_dim, input_dim))

        self.input_proj_O3   = _proj()
        self.input_proj_PM25 = _proj()
        self.input_proj_PM10 = _proj()
        self.input_proj_NO2  = _proj()
        self.input_proj_CO   = _proj()
        self.input_proj_SO2  = _proj()

        # Context dim = input_dim + spatial GFF dim (64 for 2→32 bands)
        context_dim = input_dim + 64

        # Per-modality encoder blocks
        def _enc():
            blocks = nn.ModuleList()
            prev = None
            for d, n in zip(latent_dims, num_latents):
                blocks.append(CascadedBlock(d, n, context_dim, cross_heads, cross_dim_head,
                                            self_heads, self_dim_head, prev))
                prev = d
            return blocks

        self.encoder_blocks_O3   = _enc()
        self.encoder_blocks_PM25 = _enc()
        self.encoder_blocks_PM10 = _enc()
        self.encoder_blocks_NO2  = _enc()
        self.encoder_blocks_CO   = _enc()
        self.encoder_blocks_SO2  = _enc()

        # ICMR: SHARED global→latent projections (one set for all modalities)
        self.global2latent_proj = nn.ModuleList([
            nn.Linear(final_dim, num_latents[i] * latent_dims[i])
            for i in range(len(latent_dims))
        ])

        # Processor P
        self.self_attn_blocks = nn.Sequential(*[
            nn.Sequential(
                PreNorm(final_dim, Attention(final_dim, heads=self_heads, dim_head=self_dim_head)),
                PreNorm(final_dim, FeedForward(final_dim)),
            ) for _ in range(num_trunk_layers)
        ])

        # Per-modality decoder heads
        def _dec():
            return (
                PreNorm(queries_dim, Attention(queries_dim, final_dim, heads=cross_heads,
                        dim_head=cross_dim_head), context_dim=final_dim),
                PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None,
                nn.Linear(queries_dim, 1),
            )

        self.decoder_cross_attn_O3,   self.decoder_ff_O3,   self.to_logits_O3   = _dec()
        self.decoder_cross_attn_PM25, self.decoder_ff_PM25, self.to_logits_PM25 = _dec()
        self.decoder_cross_attn_PM10, self.decoder_ff_PM10, self.to_logits_PM10 = _dec()
        self.decoder_cross_attn_NO2,  self.decoder_ff_NO2,  self.to_logits_NO2  = _dec()
        self.decoder_cross_attn_CO,   self.decoder_ff_CO,   self.to_logits_CO   = _dec()
        self.decoder_cross_attn_SO2,  self.decoder_ff_SO2,  self.to_logits_SO2  = _dec()

    def forward(self, x_O3, x_PM25, x_PM10, x_NO2, x_CO, x_SO2,
                queries, used_modalities):
        """
        Args:
            x_*: [B, S_m, context_dim] or None — per-modality tokens.
            queries: [B, Nq, queries_dim].
            used_modalities: list/tuple of 6 bools.
        Returns:
            (O3, PM25, PM10, NO2, CO, SO2): each [B, Nq, 1].
        """
        xs = [x_O3, x_PM25, x_PM10, x_NO2, x_CO, x_SO2]
        enc_list = [
            self.encoder_blocks_O3, self.encoder_blocks_PM25, self.encoder_blocks_PM10,
            self.encoder_blocks_NO2, self.encoder_blocks_CO, self.encoder_blocks_SO2,
        ]

        global_latent = self._icmr_forward(
            xs=xs, used=used_modalities,
            enc_blocks_list=enc_list,
            g2l_projs_list=self.global2latent_proj,  # shared across modalities
        )

        outputs = self._decode(queries, global_latent, [
            (self.decoder_cross_attn_O3,   self.decoder_ff_O3,   self.to_logits_O3),
            (self.decoder_cross_attn_PM25, self.decoder_ff_PM25, self.to_logits_PM25),
            (self.decoder_cross_attn_PM10, self.decoder_ff_PM10, self.to_logits_PM10),
            (self.decoder_cross_attn_NO2,  self.decoder_ff_NO2,  self.to_logits_NO2),
            (self.decoder_cross_attn_CO,   self.decoder_ff_CO,   self.to_logits_CO),
            (self.decoder_cross_attn_SO2,  self.decoder_ff_SO2,  self.to_logits_SO2),
        ])
        return tuple(outputs)


# ---------------------------------------------------------------------------
# Convenience builders (Appendix A hyperparameters)
# ---------------------------------------------------------------------------

def build_omnifield_climsim(device="cuda"):
    """Instantiate OmniField with ClimSim-THW hyperparameters from Appendix A."""
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)   # spatial: 2→64
    time_enc = GaussianFourierFeatures(1, 16, scale=10.0).to(device)  # temporal: 1→32

    model = CascadedPerceiverIO(
        input_dim=192, queries_dim=96, latent_dims=(128, 128, 128),
        num_latents=(128, 128, 128), cross_heads=4, cross_dim_head=128,
        self_heads=8, self_dim_head=128, decoder_ff=True, num_trunk_layers=3,
    ).to(device)
    return model, pos_enc, time_enc


def build_omnifield_aqs(device="cuda"):
    """Instantiate OmniField with EPA-AQS hyperparameters from Appendix A."""
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)   # spatial: 2→64
    time_enc = GaussianFourierFeatures(1, 32, scale=15.0).to(device)  # temporal: 1→64

    model = OmniFieldAQS(
        input_dim=128, queries_dim=64, latent_dims=(64, 64, 64),
        num_latents=(64, 64, 64), cross_heads=2, cross_dim_head=32,
        self_heads=2, self_dim_head=32, decoder_ff=False, num_trunk_layers=3,
    ).to(device)
    return model, pos_enc, time_enc
