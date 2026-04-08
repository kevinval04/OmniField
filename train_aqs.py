"""
Training script for OmniField on EPA-AQS.

Usage:
    python train_aqs.py --csv_path daily_aqi_by_county_2017.csv --device cuda:0

Data: download from https://www.epa.gov/aqs or via Kaggle.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from omnifield.model import OmniFieldAQS, GaussianFourierFeatures
from omnifield.data.epa_aqs import (
    AQSPointCloudLeadForecast, aqs_collate_lead, POLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_features(values, coords, mu=None, sd=None, norm_inputs=True):
    """Make [N, 3] features = [AQI, lat, lon], optionally normalizing AQI."""
    val = values.view(-1, 1)
    if norm_inputs and mu is not None and sd is not None:
        sd = max(sd, 1e-6)
        val = (val - mu) / sd
    lat = coords[:, 0:1]
    lon = coords[:, 1:2]
    return torch.cat([val, lat, lon], dim=-1)


def _posenc_lonlat(pos_enc, coords):
    """coords: [N, 3] (lat, lon, _) → [1, N, D_pos]"""
    lat = coords[:, 0]
    lon = coords[:, 1]
    xy = torch.stack([lon, lat], dim=-1) * (math.pi / 180.0)
    return pos_enc(xy).unsqueeze(0)


def pe_time(pos_enc_time, lead, Nq, device):
    lead = torch.as_tensor(lead, dtype=torch.float32, device=device).view(1, 1, 1)
    lead = lead.expand(1, Nq, 1)
    return pos_enc_time(lead)


def build_mod_tokens(model, pos_enc_space, ex_inputs, device, stats,
                     norm_inputs=True, enabled_mods=None):
    """Build per-modality tokens: input_proj(features) ∥ pos_enc(coords)."""
    enabled_mods = set(enabled_mods) if enabled_mods else set(POLS)
    proj_map = {
        "Ozone": model.input_proj_O3, "PM2.5": model.input_proj_PM25,
        "PM10": model.input_proj_PM10, "NO2": model.input_proj_NO2,
        "CO": model.input_proj_CO, "SO2": model.input_proj_SO2,
    }
    x = {}
    for m in POLS:
        if m not in enabled_mods:
            x[m] = None
            continue
        vals = ex_inputs[m]["values"]
        coords = ex_inputs[m]["coords"]
        if vals.numel() == 0:
            x[m] = None
            continue
        mu, sd = stats[m]["mean"], stats[m]["std"]
        feats = _stack_features(vals.to(device), coords.to(device), mu, sd, norm_inputs)
        proj = proj_map[m](feats)
        pe_sp = _posenc_lonlat(pos_enc_space, coords.to(device)).squeeze(0)
        tok = torch.cat([proj, pe_sp], dim=-1)  # [N, input_dim + D_SPACE]
        x[m] = tok.unsqueeze(0)
    return x


def loss_for_one_sample(model, pos_enc_space, pos_enc_time, ex, device, stats,
                        w_mod=None, input_mods=None, target_mods=None):
    """Compute weighted MSE loss for one sample over target modalities."""
    input_mods = set(POLS) if input_mods is None else set(input_mods)
    target_mods = set(POLS) if target_mods is None else set(target_mods)
    w_mod = w_mod or {m: 1.0 for m in POLS}

    x_dict = build_mod_tokens(model, pos_enc_space, ex["inputs"], device, stats,
                              enabled_mods=input_mods)
    used_bits = [
        bool((m in input_mods) and x_dict[m] is not None and x_dict[m].size(1) > 0)
        for m in POLS
    ]

    lead_scalar = float(ex.get("lead", 1.0))
    q_chunks, tgt_chunks, meta = [], [], []
    total = 0

    for m in POLS:
        if m not in target_mods:
            continue
        t_vals = ex["targets"][m]["values"]
        t_crds = ex["targets"][m]["coords"]
        if t_vals.numel() == 0:
            continue

        Nq = t_crds.shape[0]
        q_sp = _posenc_lonlat(pos_enc_space, t_crds.to(device))
        q_tm = pe_time(pos_enc_time, lead_scalar, Nq, device)
        q = torch.cat([q_sp, q_tm], dim=-1)
        q_chunks.append(q)

        mu, sd = stats[m]["mean"], max(stats[m]["std"], 1e-6)
        tgt_chunks.append((t_vals.to(device).view(1, -1, 1) - mu) / sd)
        meta.append((m, total, total + Nq))
        total += Nq

    if total == 0:
        return None, {m: None for m in POLS}

    queries_cat = torch.cat(q_chunks, dim=1)
    tgt_cat = torch.cat(tgt_chunks, dim=1)

    O3, PM25, PM10, NO2, CO, SO2 = model(
        x_O3=x_dict["Ozone"], x_PM25=x_dict["PM2.5"], x_PM10=x_dict["PM10"],
        x_NO2=x_dict["NO2"], x_CO=x_dict["CO"], x_SO2=x_dict["SO2"],
        queries=queries_cat, used_modalities=used_bits,
    )
    head_map = {"Ozone": O3, "PM2.5": PM25, "PM10": PM10, "NO2": NO2, "CO": CO, "SO2": SO2}

    total_loss = torch.tensor(0.0, device=device)
    total_w = 0.0
    per_mod_mse = {m: None for m in POLS}

    for m, s, e in meta:
        mse = F.mse_loss(head_map[m][:, s:e, :], tgt_cat[:, s:e, :])
        per_mod_mse[m] = mse.item()
        w = float(w_mod[m])
        total_loss = total_loss + w * mse
        total_w += w

    return total_loss / max(1.0, total_w), per_mod_mse


def compute_target_norms(ds_subset, max_samples=None):
    """Compute per-modality (mean, std) from target values in the dataset."""
    accum = {m: [] for m in POLS}
    for i in range(min(len(ds_subset), max_samples or len(ds_subset))):
        ex = ds_subset[i]
        for m in POLS:
            v = ex["targets"][m]["values"]
            if v.numel() > 0:
                accum[m].append(v.numpy())

    stats = {}
    for m in POLS:
        if accum[m]:
            arr = np.concatenate(accum[m])
            stats[m] = {"mean": float(arr.mean()), "std": float(arr.std())}
        else:
            stats[m] = {"mean": 0.0, "std": 1.0}
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = args.device

    # --- Data ---
    print(f"Loading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    dataset = AQSPointCloudLeadForecast(
        df, t_in=1, t_out=(1, 5), sample_lead=True,
        start_date=args.start_date, end_date=args.end_date,
    )
    print(f"Dataset: {len(dataset)} anchor days")

    split = int(0.8 * len(dataset))
    train_ds = Subset(dataset, range(0, split))
    val_ds = Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=aqs_collate_lead)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=aqs_collate_lead)

    # Compute normalization stats from training targets
    print("Computing normalization stats...")
    stats = compute_target_norms(train_ds, max_samples=500)
    for m in POLS:
        print(f"  {m}: mean={stats[m]['mean']:.2f}, std={stats[m]['std']:.2f}")

    # --- Model ---
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)
    time_enc = GaussianFourierFeatures(1, 32, scale=15.0).to(device)

    model = OmniFieldAQS(
        input_dim=128, queries_dim=64,
        latent_dims=(64, 64, 64), num_latents=(64, 64, 64),
        cross_heads=2, cross_dim_head=32,
        self_heads=2, self_dim_head=32,
        decoder_ff=False, num_trunk_layers=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"OmniFieldAQS parameters: {n_params / 1e6:.1f}M")

    # --- Optimizer ---
    opt = AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = CosineAnnealingWarmupRestarts(
        opt, first_cycle_steps=total_steps,
        max_lr=args.max_lr, min_lr=args.min_lr,
        warmup_steps=int(0.1 * total_steps),
    )

    # --- Training ---
    best_val = float("inf")
    w_mod = {m: 1.0 for m in POLS}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # AQS uses batch_size=1 with per-sample loss (variable point counts)
            ex = {
                "inputs": batch["inputs"][0],
                "targets": batch["targets"][0],
                "lead": batch["lead"][0].item(),
            }

            loss, per_mod = loss_for_one_sample(
                model, pos_enc, time_enc, ex, device, stats, w_mod=w_mod,
            )
            if loss is None:
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            count += 1

        avg_loss = epoch_loss / max(count, 1)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.6f}")

        # --- Validation ---
        if epoch % args.val_every == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    ex = {
                        "inputs": batch["inputs"][0],
                        "targets": batch["targets"][0],
                        "lead": batch["lead"][0].item(),
                    }
                    loss, _ = loss_for_one_sample(
                        model, pos_enc, time_enc, ex, device, stats, w_mod=w_mod,
                    )
                    if loss is not None:
                        val_loss += loss.item()
                        val_count += 1

            avg_val = val_loss / max(val_count, 1)
            print(f"[VAL] loss={avg_val:.6f}")

            if avg_val < best_val:
                best_val = avg_val
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "pos_enc_state": pos_enc.state_dict(),
                    "time_enc_state": time_enc.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_loss": avg_val,
                    "stats": stats,
                }, args.save_path)
                print(f"  Saved best model (val_loss={avg_val:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OmniField on EPA-AQS")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to AQS CSV (e.g., daily_aqi_by_county_*.csv)")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=8e-6)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_aqs_model.pt")
    main(parser.parse_args())
