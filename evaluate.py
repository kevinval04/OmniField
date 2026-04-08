"""
Evaluation script for OmniField on ClimSim-THW.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --data_dir ./processed --grid_meta ClimSim_high-res_grid-info.nc
"""

import os
import glob
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from omnifield.model import CascadedPerceiverIO, GaussianFourierFeatures
from omnifield.data.climsim import ClimSimTQVForecastVennFixed


TQV_KEYS = ("T", "Q", "V")


def load_tqv_norm_stats(npz_path):
    """Load per-modality (mean, std) normalization statistics."""
    z = dict(np.load(npz_path, allow_pickle=True))
    stats = {}
    for k in TQV_KEYS:
        v = z[k]
        if isinstance(v, np.ndarray) and v.dtype == object:
            v = v.reshape(()).item()
        if isinstance(v, dict):
            mu = float(v.get("mean", v.get("mu", 0.0)))
            sd = max(float(v.get("std", v.get("sigma", 1.0))), 1e-6)
        else:
            arr = np.asarray(v)
            mu, sd = float(arr[0]), max(float(arr[1]), 1e-6)
        stats[k] = (mu, sd)
    return stats


def denorm(val, mu, sd):
    return val * sd + mu


def pos_enc_batched(pos_enc_fn, coords):
    leading = coords.shape[:-1]
    flat = coords.reshape(-1, coords.shape[-1])
    enc = pos_enc_fn(flat)
    return enc.view(*leading, enc.shape[-1])


@torch.no_grad()
def evaluate(args):
    device = args.device

    # --- Data ---
    norm_stats = dict(np.load(args.norm_stats, allow_pickle=True))
    norm_tuples = load_tqv_norm_stats(args.norm_stats)
    file_list = sorted(glob.glob(os.path.join(args.data_dir, "**/*.npz"), recursive=True))

    dataset = ClimSimTQVForecastVennFixed(
        file_list=file_list,
        grid_meta_path=args.grid_meta,
        sparsity=0.02, triple_fraction=0.25,
        norm_stats=norm_stats,
        input_modalities=(1, 1, 1),
        input_region="union", seed=123,
    )

    train_len = 9000
    val_ds = Subset(dataset, range(train_len, len(dataset)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- Model ---
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)
    time_enc = GaussianFourierFeatures(1, 16, scale=10.0).to(device)

    model = CascadedPerceiverIO(
        input_dim=192, queries_dim=96, logits_dim=None,
        latent_dims=(128, 128, 128), num_latents=(128, 128, 128),
        decoder_ff=True,
    ).to(device)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    pos_enc.load_state_dict(ckpt["pos_enc_state"])
    time_enc.load_state_dict(ckpt["time_enc_state"])
    print(f"Loaded checkpoint from {args.checkpoint} (iter={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', 'N/A')})")

    model.eval()

    # --- Evaluate ---
    mse_sums = {"T": 0.0, "Q": 0.0, "V": 0.0}
    rmse_phys_sums = {"T": 0.0, "Q": 0.0, "V": 0.0}
    count = 0

    for batch in val_loader:
        (data_T, data_Q, data_V,
         mesh_T, mesh_Q, mesh_V,
         data_y, mesh_y,
         supervised_idx, used_modalities, supervision_mask, tau) = batch

        data_T, mesh_T = data_T.to(device), mesh_T.to(device)
        data_Q, mesh_Q = data_Q.to(device), mesh_Q.to(device)
        data_V, mesh_V = data_V.to(device), mesh_V.to(device)
        data_y, mesh_y = data_y.to(device), mesh_y.to(device)
        tau = tau.to(device).view(-1)

        x_T = torch.cat([model.input_proj_T(data_T), pos_enc_batched(pos_enc, mesh_T)], dim=-1) if data_T.numel() > 0 else None
        x_Q = torch.cat([model.input_proj_Q(data_Q), pos_enc_batched(pos_enc, mesh_Q)], dim=-1) if data_Q.numel() > 0 else None
        x_V = torch.cat([model.input_proj_V(data_V), pos_enc_batched(pos_enc, mesh_V)], dim=-1) if data_V.numel() > 0 else None

        q_spatial = pos_enc_batched(pos_enc, mesh_y)
        tfeat = time_enc(tau[:, None])[:, None, :].expand(-1, mesh_y.shape[1], -1)
        queries_full = torch.cat([q_spatial, tfeat], dim=-1)

        pred_T, pred_Q, pred_V = model(x_T=x_T, x_Q=x_Q, x_V=x_V,
                                        queries=queries_full, used_modalities=(True, True, True))

        preds = {"T": pred_T.squeeze(-1), "Q": pred_Q.squeeze(-1), "V": pred_V.squeeze(-1)}
        tgts = {"T": data_y[..., 0], "Q": data_y[..., 1], "V": data_y[..., 2]}

        for k in TQV_KEYS:
            mse_norm = F.mse_loss(preds[k], tgts[k]).item()
            mse_sums[k] += mse_norm

            # Physical-units RMSE
            mu, sd = norm_tuples[k]
            pred_phys = denorm(preds[k], mu, sd)
            tgt_phys = denorm(tgts[k], mu, sd)
            rmse_phys = torch.sqrt(F.mse_loss(pred_phys, tgt_phys)).item()
            rmse_phys_sums[k] += rmse_phys

        count += 1

    print(f"\n{'='*60}")
    print(f"Evaluation over {count} validation samples")
    print(f"{'='*60}")
    for k in TQV_KEYS:
        units = {"T": "K", "Q": "10⁻³ kg/kg", "V": "m/s"}[k]
        mse_avg = mse_sums[k] / count
        rmse_avg = rmse_phys_sums[k] / count
        print(f"  {k}: MSE(norm)={mse_avg:.6f}  RMSE(phys)={rmse_avg:.4f} {units}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OmniField on ClimSim-THW")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./processed")
    parser.add_argument("--grid_meta", type=str, default="ClimSim_high-res_grid-info.nc")
    parser.add_argument("--norm_stats", type=str, default="norm_TQV_full.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate(parser.parse_args())
