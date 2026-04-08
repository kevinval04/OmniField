import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import repeat
from tqdm import tqdm
from einops import rearrange, repeat
# ============================================================
# --- Fourier Encoders (Positional + Temporal) ---
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, n_bands=16):
        super().__init__()
        self.in_features = in_features
        fourier_dim = in_features * 2 * n_bands
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )
        self.register_buffer('freqs', 2**torch.arange(n_bands) * torch.pi)

    def forward(self, coords):
        # coords: (B, N, D)
        b, n, d = coords.shape
        projections = coords.unsqueeze(-1) * self.freqs   # (B,N,D,bands)
        fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        fourier_feats = fourier_feats.view(b, n, -1)      # flatten bands
        return self.mlp(fourier_feats)

# ============================================================
# --- PerceiverIO Core ---
# ============================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context:
            context = kwargs['context']
            kwargs.update(context=self.norm_context(context))
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),  # -> (…, 2*m*dim)
            GEGLU(),                          # gates to (…, m*dim)
            nn.Linear(dim * mult, dim)        # -> (…, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # SAFE reshape into heads (handles non-contiguous)
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            # mask: (b, n_ctx) → expand over heads
            mask = rearrange(mask, 'b j -> (b h) 1 j', h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)

        # merge heads back (b*h, n, d) → (b, n, h*d)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class PerceiverIO(nn.Module):
    def __init__(self, *, depth, dim, queries_dim, num_latents=256, latent_dim=512,
                 cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64,
                 decoder_ff=True):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = PreNorm(latent_dim,
            Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=dim
        )
        self.cross_ff = PreNorm(latent_dim, FeedForward(latent_dim))

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head)),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]) for _ in range(depth)
        ])

        self.decoder_cross_attn = PreNorm(queries_dim,
            Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=latent_dim
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

    def forward(self, data, queries, mask=None):
        b = data.shape[0]
        x = self.latents.unsqueeze(0).repeat(b,1,1)
        x = self.cross_attn(x, context=data, mask=mask) + x
        x = self.cross_ff(x) + x
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        latents = self.decoder_cross_attn(queries, context=x)
        if self.decoder_ff:
            latents = latents + self.decoder_ff(latents)
        return latents

# ============================================================
# --- SCENT Baseline Wrapper ---
# ============================================================

class SCENTBaseline(nn.Module):
    def __init__(self, pos_dim=192, time_dim=32, val_dim=16,
                 perceiver_dim=256, perceiver_depth=6, perceiver_latent=512):
        super().__init__()
        # encoders
        self.pos_enc = FourierFeatures(2, pos_dim)
        self.time_enc = FourierFeatures(1, time_dim)
        self.enc_T = nn.Linear(1, val_dim)
        self.enc_Q = nn.Linear(1, val_dim)
        self.enc_V = nn.Linear(1, val_dim)

        # perceiver core
        input_dim = pos_dim + time_dim + val_dim
        self.perceiver = PerceiverIO(
            depth=perceiver_depth,
            dim=input_dim,
            queries_dim=pos_dim+time_dim,
            num_latents=256,
            latent_dim=perceiver_latent,
            decoder_ff=True
        )

        # modality decoders
        qdim = pos_dim+time_dim
        self.dec_T = nn.Linear(qdim, 1)
        self.dec_Q = nn.Linear(qdim, 1)
        self.dec_V = nn.Linear(qdim, 1)

    def forward(self, data_T, data_Q, data_V,
                mesh_T, mesh_Q, mesh_V,
                mesh_y, tau):
        B = data_T.shape[0]
        tau = tau.view(B,1,1).expand(-1, mesh_y.shape[1], -1)  # (B,N,1)

        # --- encode inputs ---
        seqs = []
        def encode(data, mesh, enc_val):
            if data.shape[1] == 0: return None
            val = enc_val(data[...,[-1]])
            pos = self.pos_enc(mesh)
            t   = self.time_enc(tau[:,:mesh.shape[1]])
            return torch.cat([pos,t,val],dim=-1)

        if data_T.shape[1]>0: seqs.append(encode(data_T, mesh_T, self.enc_T))
        if data_Q.shape[1]>0: seqs.append(encode(data_Q, mesh_Q, self.enc_Q))
        if data_V.shape[1]>0: seqs.append(encode(data_V, mesh_V, self.enc_V))
        seq = torch.cat(seqs, dim=1) if seqs else None

        # --- queries: full field mesh + tau ---
        pos_y = self.pos_enc(mesh_y)
        time_y = self.time_enc(tau)
        queries = torch.cat([pos_y,time_y],dim=-1)

        latents = self.perceiver(seq, queries)

        # --- decoders ---
        pred_T = self.dec_T(latents).squeeze(-1)
        pred_Q = self.dec_Q(latents).squeeze(-1)
        pred_V = self.dec_V(latents).squeeze(-1)
        return pred_T, pred_Q, pred_V

# ============================================================
# --- Training / Validation ---
# ============================================================

def _fmt(x, nd=6):
    return f"{x:.{nd}f}"   # fixed-point decimal (no scientific notation)

def _avg(d, n):
    return {k: (v / max(n, 1)) for k, v in d.items()}

def train_epoch(model, loader, optimizer, device):
    model.train()
    sums = {"T": 0.0, "Q": 0.0, "V": 0.0, "total": 0.0}
    tbar = tqdm(loader, desc="Train", leave=True)

    for batch in tbar:
        (dT, dQ, dV, mT, mQ, mV, y, mesh_y, _, _, _, tau) = [b.to(device) for b in batch]
        y_T, y_Q, y_V = y[..., 0], y[..., 1], y[..., 2]

        pred_T, pred_Q, pred_V = model(dT, dQ, dV, mT, mQ, mV, mesh_y, tau)

        loss_T = F.mse_loss(pred_T, y_T)
        loss_Q = F.mse_loss(pred_Q, y_Q)
        loss_V = F.mse_loss(pred_V, y_V)
        loss   = loss_T + loss_Q + loss_V

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running sums
        sums["T"]     += loss_T.item()
        sums["Q"]     += loss_Q.item()
        sums["V"]     += loss_V.item()
        sums["total"] += loss.item()

        # live running average over processed batches so far
        cnt = max(1, tbar.n)  # number of batches processed so far
        avgs_live = {k: sums[k] / cnt for k in sums}

        tbar.set_postfix(
            MSE_T=_fmt(avgs_live['T']),
            MSE_Q=_fmt(avgs_live['Q']),
            MSE_V=_fmt(avgs_live['V']),
            TOTAL=_fmt(avgs_live['total'])
        )

    # final epoch averages
    return _avg(sums, len(loader))

@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    sums = {"T": 0.0, "Q": 0.0, "V": 0.0, "total": 0.0}
    vbar = tqdm(loader, desc="Val", leave=True)

    for batch in vbar:
        (dT, dQ, dV, mT, mQ, mV, y, mesh_y, _, _, _, tau) = [b.to(device) for b in batch]
        y_T, y_Q, y_V = y[..., 0], y[..., 1], y[..., 2]

        pred_T, pred_Q, pred_V = model(dT, dQ, dV, mT, mQ, mV, mesh_y, tau)

        loss_T = F.mse_loss(pred_T, y_T)
        loss_Q = F.mse_loss(pred_Q, y_Q)
        loss_V = F.mse_loss(pred_V, y_V)
        loss   = loss_T + loss_Q + loss_V

        sums["T"]     += loss_T.item()
        sums["Q"]     += loss_Q.item()
        sums["V"]     += loss_V.item()
        sums["total"] += loss.item()

        # live running average over processed batches so far
        cnt = max(1, vbar.n)
        avgs_live = {k: sums[k] / cnt for k in sums}

        vbar.set_postfix(
            MSE_T=_fmt(avgs_live['T']),
            MSE_Q=_fmt(avgs_live['Q']),
            MSE_V=_fmt(avgs_live['V']),
            TOTAL=_fmt(avgs_live['total'])
        )

    return _avg(sums, len(loader))

def run_training(train_loader, val_loader, device="cuda", save_dir="checkpoints",
                 epochs=20, lr=1e-4,
                 model_ctor=lambda: SCENTBaseline()):
    model = model_ctor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * max(1, len(train_loader)))

    os.makedirs(save_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device)
        val = validate_epoch(model, val_loader, device)
        scheduler.step()

        print(
            "Epoch {:03d} | Train: total {}  (T {} | Q {} | V {})  | Val: total {} (T {} | Q {} | V {})".format(
                epoch,
                _fmt(tr['total']), _fmt(tr['T']), _fmt(tr['Q']), _fmt(tr['V']),
                _fmt(val['total']), _fmt(val['T']), _fmt(val['Q']), _fmt(val['V'])
            )
        )

        if val["total"] < best_val:
            best_val = val["total"]
            path = os.path.join(save_dir, "best_interp_model.pt")
            torch.save(model.state_dict(), path)
            print(f"  ✅ Saved best model: {path}  (val_total={_fmt(best_val)})")

    return model



# import glob, numpy as np
# from torch.utils.data import DataLoader, Subset

# norm_stats    = dict(np.load("norm_TQV_full.npz", allow_pickle=True))
# file_list     = sorted(glob.glob("processed/**/*.npz", recursive=True))
# grid_meta_path = "ClimSim_high-res_grid-info.nc"


# sparsity = 0.02  # 2% -> 432 T points
# dataset = ClimSimTQVForecastVennFixed(
#     file_list=file_list,
#     grid_meta_path="ClimSim_high-res_grid-info.nc",
#     sparsity=0.02,
#     triple_fraction=0.25,
#     norm_stats=norm_stats,
#     input_modalities=(1,1,1),   # T-only inputs
#     seed=123
# )

# dataset = ClimSimTQVForecastVennFixed(
#     file_list=file_list,
#     grid_meta_path="ClimSim_high-res_grid-info.nc",
#     sparsity=0.02,
#     triple_fraction=0.25,
#     norm_stats=norm_stats,
#     input_modalities=(1,1,1),   # only T values
#     input_region="intersection",      # <- restrict locations to T∩Q∩V (≈108 pts)
#     seed=123,
# )
# dataset = ClimSimTQVForecastVennFixed(
#     file_list=file_list,
#     grid_meta_path="ClimSim_high-res_grid-info.nc",
#     sparsity=0.02, triple_fraction=0.25,
#     norm_stats=norm_stats,
#     input_modalities=(1,1,1),
#     input_region="union",       # union points are the "sensor locations"
#     rbf_fill=True,              # <— turn on interpolation
#     rbf_kernel="thin_plate_spline",
#     rbf_smoothing=0.0,
#     rbf_neighbors=None,
#     seed=123,
# )

# train_len = 9000
# assert len(dataset) > train_len
# train_T = Subset(dataset, range(0, train_len))
# val_T   = Subset(dataset, range(train_len, len(dataset)))


# train_loader_T = DataLoader(
#     train_T, batch_size=8, shuffle=True,
# )
# val_loader_T = DataLoader(
#     val_T, batch_size=1, shuffle=False,
# )



# device = "cuda" if torch.cuda.is_available() else "cpu"

# # --- model hyperparams (tweak as you like) ---
# POS_EMBED_DIM  = 192
# TIME_EMBED_DIM = 32
# VAL_EMBED_DIM  = 16
# PERCEIVER_DEPTH = 6
# PERCEIVER_LATENT = 512

# def make_scent_model():
#     return SCENTBaseline(
#         pos_dim=POS_EMBED_DIM,
#         time_dim=TIME_EMBED_DIM,
#         val_dim=VAL_EMBED_DIM,
#         perceiver_depth=PERCEIVER_DEPTH,
#         perceiver_latent=PERCEIVER_LATENT,
#     )

# # --- naming / checkpoint dir ---
# region_tag = dataset.input_region  # "union" or "triple"
# mods_tag   = "".join(["T" if dataset.input_modalities[0] else "",
#                       "Q" if dataset.input_modalities[1] else "",
#                       "V" if dataset.input_modalities[2] else ""]) or "none"
# run_name = f"scent_{region_tag}_{mods_tag}_s{int(100*0.02)}"  # adjust if sparsity changes
# save_dir = os.path.join("ClimSim_checkpoints/SCENT", run_name)

# # --- train ---
# model = run_training(
#     train_loader=train_loader_T,
#     val_loader=val_loader_T,
#     device=device,
#     save_dir=save_dir,
#     epochs=20,           # change as needed
#     lr=1e-4,             # change as needed
#     model_ctor=make_scent_model,
# )
