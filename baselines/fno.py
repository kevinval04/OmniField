import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer


torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

    
class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=15.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

    def forward(self, coords):
        projections = coords @ self.B
        fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return fourier_feats
    
################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1


        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)  # shape: (B, C, N//2 + 1)

        # Convert complex weights from (in, out, modes, 2) to complex tensor
        weight = torch.view_as_complex(self.weights1)  # shape: (in, out, modes)

        # Perform multiplication
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], weight)

        # Inverse FFT back to real
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)  # shape: (B, out, N)
        return x


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels):
        super().__init__()

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, N, C]
        x = self.fc0(x)         # now valid
        x = x.permute(0, 2, 1)  # [B, C, N] for convs

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels):
        super().__init__()
        self.conv1 = SimpleBlock1d(modes, width, in_channels, out_channels)

    def forward(self, x):
        return self.conv1(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def rel_error(self, x, y, p=2, reduction=True):
        """Compute relative Lp error"""
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p, dim=1)
        rel_err = diff_norms / y_norms.clamp(min=1e-8)

        if reduction:
            return rel_err.mean()
        return rel_err



class FNOWithEncoders(nn.Module):
    def __init__(self, modes, width, d_in_coord, d_in_time, d_in_val=1, d_in_mask=1, out_channels=3):
        super().__init__()
        self.modes = modes
        self.width = width

        # === Per-modality projections (value + mask)
        d_in_mod = d_in_val + d_in_mask   # here = 2
        self.proj_T = nn.Linear(d_in_mod, width // 4)
        self.proj_Q = nn.Linear(d_in_mod, width // 4)
        self.proj_V = nn.Linear(d_in_mod, width // 4)

        # === Coordinate and time encoders
        self.proj_coord = nn.Linear(d_in_coord, width // 4)   # lon, lat
        self.proj_time  = nn.Linear(d_in_time,  width // 4)

        # === Final concat projection
        self.input_proj = nn.Linear(width, width)

        # === FNO backbone (reuse your Net1d / SimpleBlock1d)
        self.backbone = Net1d(modes, width, in_channels=width, out_channels=out_channels)

    def forward(self, X_dict):
        """
        X_dict = {
          'coord': [B, N, 2],
          'time':  [B, N, D_t],
          'T': {'val': [B, N, 1], 'mask': [B, N, 1]},
          'Q': {...}, 'V': {...}
        }
        """

        # Per-modality embeddings
        emb_T = self.proj_T(torch.cat([X_dict['T']['val'], X_dict['T']['mask']], dim=-1))
        emb_Q = self.proj_Q(torch.cat([X_dict['Q']['val'], X_dict['Q']['mask']], dim=-1))
        emb_V = self.proj_V(torch.cat([X_dict['V']['val'], X_dict['V']['mask']], dim=-1))

        # Coord + time embeddings
        emb_coord = self.proj_coord(X_dict['coord'])
        emb_time  = self.proj_time(X_dict['time'])

        # Combine all: [B, N, width]
        emb = torch.cat([emb_T, emb_Q, emb_V, emb_coord, emb_time], dim=-1)
        emb = self.input_proj(emb)

        # Rearrange for FNO: [B, C, N]
        emb = emb.permute(0, 2, 1)

        # FNO backbone → [B, N, 3]
        return self.backbone(emb)

    
class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, width, 1)

    def forward(self, x):
        return self.proj(x)   # [B, width, N] 
    
class MultiFNO(nn.Module):
    def __init__(self, modes, width, in_channels_dict, out_channels=3, fuse="sum"):
        """
        in_channels_dict: dict with #channels for each modality, e.g.
                          {"T": 2+2+Dt, "Q": 2+2+Dt, "V": 2+2+Dt}
        fuse: "sum" or "concat"
        """
        super().__init__()
        self.enc_T = ModalityEncoder(in_channels_dict["T"], width)
        self.enc_Q = ModalityEncoder(in_channels_dict["Q"], width)
        self.enc_V = ModalityEncoder(in_channels_dict["V"], width)

        self.fuse = fuse
        if fuse == "concat":
            fused_in = 3 * width
        else:  # sum
            fused_in = width

        self.backbone = SimpleBlock1d(modes, width, fused_in, out_channels)

    def forward(self, X_dict):
        out_T = self.enc_T(X_dict["T"])
        out_Q = self.enc_Q(X_dict["Q"])
        out_V = self.enc_V(X_dict["V"])

        if self.fuse == "concat":
            fused = torch.cat([out_T, out_Q, out_V], dim=1)  # [B, 3*width, N]
        else:
            fused = out_T + out_Q + out_V  # [B, width, N]

        return self.backbone(fused)  # → [B, N, out_channels]


def build_fno_batch_dict(batch, pos_enc=None):
    (data_T, data_Q, data_V,
     mesh_T, mesh_Q, mesh_V,
     data_y, mesh_y,
     supervised_idx, used_modalities, supervision_mask, tau) = batch

    device = data_y.device
    B, N = data_y.shape[0], data_y.shape[1]   # N = 21600

    # --- coords ---
    lon = _minmax_norm(mesh_y[..., 0])
    lat = _minmax_norm(mesh_y[..., 1])
    coords = torch.stack([lon, lat], dim=-1)   # [B, N, 2]

    # # --- time ---
    # if tau.ndim == 1: tau_in = tau[:, None]
    # else: tau_in = tau
    # tfeat = time_enc_fn(tau_in)                # [B, Dt]
    # time_feats = tfeat.unsqueeze(1).expand(-1, N, -1)  # [B, N, Dt]

    pos_feats = pos_enc(coords)

    def scatter_mod(data_mod, sm_mask, idx):
        val = torch.zeros(B, N, 1, device=device)
        msk = torch.zeros(B, N, 1, device=device)
        for b in range(B):
            p = 0
            for j in range(idx.shape[1]):  # loop over union points
                if sm_mask[b, j]:
                    gi = idx[b, j].item()
                    val[b, gi, 0] = data_mod[b, p, 2]
                    msk[b, gi, 0] = 1.0
                    p += 1
        return val, msk

    T_val, T_mask = scatter_mod(data_T, supervision_mask[..., 0], supervised_idx)
    Q_val, Q_mask = scatter_mod(data_Q, supervision_mask[..., 1], supervised_idx)
    V_val, V_mask = scatter_mod(data_V, supervision_mask[..., 2], supervised_idx)

    # --- stack per modality ---
    def build_mod(val, mask):
        x = torch.cat([coords, val, mask, pos_feats], dim=-1)  # [B, N, C_mod]
        return x.permute(0, 2, 1)  # [B, C_mod, N]

    X_dict = {
        "T": build_mod(T_val, T_mask),
        "Q": build_mod(Q_val, Q_mask),
        "V": build_mod(V_val, V_mask)
    }

    return X_dict, data_y
    
def _minmax_norm(x, eps=1e-6):
    # normalize to [-1, 1] per-batch
    xmin, xmax = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0