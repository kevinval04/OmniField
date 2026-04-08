"""
Training script for OmniField on ClimSim-THW.

Usage:
    python train.py --data_dir ./processed --grid_meta ClimSim_high-res_grid-info.nc \
                    --norm_stats norm_TQV_full.npz --device cuda:0
"""

import os
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from itertools import cycle
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from omnifield.model import CascadedPerceiverIO, GaussianFourierFeatures
from omnifield.data.climsim import ClimSimTQVForecastVennFixed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pos_enc_batched(pos_enc_fn, coords):
    """Apply positional encoding to batched coordinates [..., 2] → [..., D]."""
    leading = coords.shape[:-1]
    flat = coords.reshape(-1, coords.shape[-1])
    enc = pos_enc_fn(flat)
    return enc.view(*leading, enc.shape[-1])


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(model, pos_enc, time_enc, optimizer, iteration, val_loss, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": iteration,
        "model_state": model.state_dict(),
        "pos_enc_state": pos_enc.state_dict(),
        "time_enc_state": time_enc.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"Saved checkpoint to {path} (val_loss={val_loss:.6f})")


@torch.no_grad()
def run_validation(model, val_loader, pos_enc, time_enc, device, max_batches=None):
    """Compute mean per-modality MSE over the validation set."""
    model.eval()
    sums = {"T": 0.0, "Q": 0.0, "V": 0.0}
    count = 0

    for b_idx, batch in enumerate(val_loader):
        if max_batches and b_idx >= max_batches:
            break

        (data_T, data_Q, data_V,
         mesh_T, mesh_Q, mesh_V,
         data_y, mesh_y,
         supervised_idx, used_modalities, supervision_mask, tau) = batch

        data_T, mesh_T = data_T.to(device), mesh_T.to(device)
        data_Q, mesh_Q = data_Q.to(device), mesh_Q.to(device)
        data_V, mesh_V = data_V.to(device), mesh_V.to(device)
        data_y = data_y.to(device)
        mesh_y = mesh_y.to(device)
        tau = tau.to(device).view(-1)
        used_bits = tuple(bool(x) for x in used_modalities[0].tolist()) if used_modalities.ndim == 2 else (True, True, True)

        # Encode inputs
        x_T = torch.cat([model.input_proj_T(data_T), pos_enc_batched(pos_enc, mesh_T)], dim=-1) if data_T.numel() > 0 else None
        x_Q = torch.cat([model.input_proj_Q(data_Q), pos_enc_batched(pos_enc, mesh_Q)], dim=-1) if data_Q.numel() > 0 else None
        x_V = torch.cat([model.input_proj_V(data_V), pos_enc_batched(pos_enc, mesh_V)], dim=-1) if data_V.numel() > 0 else None

        # Build queries
        q_spatial = pos_enc_batched(pos_enc, mesh_y)
        tfeat = time_enc(tau[:, None])[:, None, :].expand(-1, mesh_y.shape[1], -1)
        queries_full = torch.cat([q_spatial, tfeat], dim=-1)

        pred_T, pred_Q, pred_V = model(x_T=x_T, x_Q=x_Q, x_V=x_V,
                                        queries=queries_full, used_modalities=used_bits)

        sums["T"] += F.mse_loss(pred_T.squeeze(-1), data_y[..., 0]).item()
        sums["Q"] += F.mse_loss(pred_Q.squeeze(-1), data_y[..., 1]).item()
        sums["V"] += F.mse_loss(pred_V.squeeze(-1), data_y[..., 2]).item()
        count += 1

    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = args.device

    # --- Data ---
    norm_stats = dict(np.load(args.norm_stats, allow_pickle=True))
    file_list = sorted(glob.glob(os.path.join(args.data_dir, "**/*.npz"), recursive=True))
    assert len(file_list) > 0, f"No .npz files found in {args.data_dir}"
    print(f"Found {len(file_list)} snapshot files.")

    dataset = ClimSimTQVForecastVennFixed(
        file_list=file_list,
        grid_meta_path=args.grid_meta,
        sparsity=0.02,
        triple_fraction=0.25,
        norm_stats=norm_stats,
        input_modalities=(1, 1, 1),
        input_region="union",
        seed=123,
    )

    train_len = 9000
    train_ds = Subset(dataset, range(0, train_len))
    val_ds = Subset(dataset, range(train_len, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- Model ---
    pos_enc = GaussianFourierFeatures(2, 32, scale=15.0).to(device)
    time_enc = GaussianFourierFeatures(1, 16, scale=10.0).to(device)

    model = CascadedPerceiverIO(
        input_dim=192,
        queries_dim=96,
        logits_dim=None,
        latent_dims=(128, 128, 128),
        num_latents=(128, 128, 128),
        decoder_ff=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"OmniField parameters: {n_params / 1e6:.1f}M")

    # --- Optimizer ---
    opt = AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(
        opt, first_cycle_steps=args.total_iters,
        max_lr=args.max_lr, min_lr=args.min_lr,
        warmup_steps=int(0.1 * args.total_iters),
    )

    # --- Training loop ---
    train_iter = cycle(train_loader)
    best_val = float("inf")

    running_loss = 0.0
    for it in tqdm(range(1, args.total_iters + 1)):
        model.train()

        (data_T, data_Q, data_V,
         mesh_T, mesh_Q, mesh_V,
         data_y, mesh_y,
         supervised_idx, used_modalities, supervision_mask, tau) = next(train_iter)

        data_T, mesh_T = data_T.to(device), mesh_T.to(device)
        data_Q, mesh_Q = data_Q.to(device), mesh_Q.to(device)
        data_V, mesh_V = data_V.to(device), mesh_V.to(device)
        data_y, mesh_y = data_y.to(device), mesh_y.to(device)
        tau = tau.to(device).view(-1)

        # Encode inputs
        x_T = torch.cat([model.input_proj_T(data_T), pos_enc_batched(pos_enc, mesh_T)], dim=-1) if data_T.numel() > 0 else None
        x_Q = torch.cat([model.input_proj_Q(data_Q), pos_enc_batched(pos_enc, mesh_Q)], dim=-1) if data_Q.numel() > 0 else None
        x_V = torch.cat([model.input_proj_V(data_V), pos_enc_batched(pos_enc, mesh_V)], dim=-1) if data_V.numel() > 0 else None

        # Build queries
        tfeat = time_enc(tau[:, None])[:, None, :].expand(-1, mesh_y.shape[1], -1)
        queries_spatial = pos_enc_batched(pos_enc, mesh_y)
        queries_full = torch.cat([queries_spatial, tfeat], dim=-1)

        # Forward
        pred_T, pred_Q, pred_V = model(
            x_T=x_T, x_Q=x_Q, x_V=x_V,
            queries=queries_full, used_modalities=(True, True, True),
        )

        # Loss: full-field MSE over all modalities
        loss_T = F.mse_loss(pred_T.squeeze(-1), data_y[..., 0])
        loss_Q = F.mse_loss(pred_Q.squeeze(-1), data_y[..., 1])
        loss_V = F.mse_loss(pred_V.squeeze(-1), data_y[..., 2])
        loss = loss_T + loss_Q + loss_V

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        running_loss += loss.item()

        if it % args.print_every == 0:
            avg = running_loss / args.print_every
            print(f"[Iter {it}] loss={avg:.6f}  lr={get_lr(opt):.2e}")
            running_loss = 0.0

        if it % args.val_every == 0:
            val_metrics = run_validation(model, val_loader, pos_enc, time_enc, device, max_batches=50)
            val_total = sum(val_metrics.values())
            print(f"[VAL] T={val_metrics['T']:.6f}  Q={val_metrics['Q']:.6f}  V={val_metrics['V']:.6f}  total={val_total:.6f}")

            if val_total < best_val:
                best_val = val_total
                save_checkpoint(model, pos_enc, time_enc, opt, it, val_total, args.save_path)

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OmniField on ClimSim-THW")
    parser.add_argument("--data_dir", type=str, default="./processed", help="Directory with .npz snapshot files")
    parser.add_argument("--grid_meta", type=str, default="ClimSim_high-res_grid-info.nc")
    parser.add_argument("--norm_stats", type=str, default="norm_TQV_full.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=100000)
    parser.add_argument("--max_lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=8e-6)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    main(parser.parse_args())
