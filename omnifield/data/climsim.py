"""
ClimSim-THW dataset with Venn-diagram sensor partitioning.

Implements the multimodal sparse sensing setup described in Section 5.1 and
Appendix D of the paper: each modality (T, H, W) observes 432 of 21,600
grid locations, with controlled overlap (108 triple, 81 pairwise, 162 exclusive).
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import netCDF4 as nc


def load_idx2latlon(grid_meta_path: str):
    """Load (lat, lon) pairs from ClimSim grid metadata."""
    with nc.Dataset(grid_meta_path) as ds:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
    return [(float(lat[i]), float(lon[i])) for i in range(len(lat))]


# ---------------------------------------------------------------------------
# Venn partition utilities
# ---------------------------------------------------------------------------

REGIONS = ['T', 'Q', 'V', 'TQ', 'TV', 'QV', 'TQV']
REG2BITS = {
    'T':   (1, 0, 0),
    'Q':   (0, 1, 0),
    'V':   (0, 0, 1),
    'TQ':  (1, 1, 0),
    'TV':  (1, 0, 1),
    'QV':  (0, 1, 1),
    'TQV': (1, 1, 1),
}


def _assign_venn_indices(n_points, sparsity, triple_fraction, seed, triple_fixed_count=None):
    """Create fixed Venn-diagram sensor masks for T/Q/V modalities.

    Returns:
        fixed_idx: Sorted array of global indices in the union of all sensors.
        region_of_local: List mapping each union index to its Venn region string.
        mod_masks: Dict with boolean arrays {'T': ..., 'Q': ..., 'V': ...}.
    """
    rng = np.random.RandomState(seed)
    K_mod = int(round(sparsity * n_points))
    assert K_mod > 0, "sparsity too small"
    a = int(triple_fixed_count if triple_fixed_count is not None else round(triple_fraction * K_mod))
    assert 0 <= a <= K_mod, "invalid triple size"

    # Distribute counts per region with t > p
    p = (K_mod - a) // 4
    t = K_mod - a - 2 * p
    if t <= p:
        p = max(0, p - 1)
        t = K_mod - a - 2 * p
    assert t >= 0 and p >= 0
    assert a + 2 * p + t == K_mod

    cnt = {'TQV': a, 'TQ': p, 'TV': p, 'QV': p, 'T': t, 'Q': t, 'V': t}

    perm = rng.permutation(n_points)
    cursor = 0
    region_indices = {}
    for r in ['TQV', 'TQ', 'TV', 'QV', 'T', 'Q', 'V']:
        k = cnt[r]
        if k > 0:
            region_indices[r] = perm[cursor:cursor + k]
            cursor += k
        else:
            region_indices[r] = np.empty((0,), dtype=int)

    mask_T = np.zeros(n_points, dtype=bool)
    mask_Q = np.zeros(n_points, dtype=bool)
    mask_V = np.zeros(n_points, dtype=bool)
    for r, idxs in region_indices.items():
        bT, bQ, bV = REG2BITS[r]
        if bT: mask_T[idxs] = True
        if bQ: mask_Q[idxs] = True
        if bV: mask_V[idxs] = True

    assert mask_T.sum() == K_mod and mask_Q.sum() == K_mod and mask_V.sum() == K_mod
    assert (mask_T & mask_Q & mask_V).sum() == a

    union_mask = mask_T | mask_Q | mask_V
    fixed_idx = np.sort(np.where(union_mask)[0])

    inv = {}
    for r, idxs in region_indices.items():
        for gi in idxs:
            inv[gi] = r
    region_of_local = [inv[gi] for gi in fixed_idx]

    return fixed_idx, region_of_local, {'T': mask_T, 'Q': mask_Q, 'V': mask_V}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClimSimTQVForecastVennFixed(Dataset):
    """ClimSim-THW forecasting dataset with fixed Venn sensor partition.

    Each sample provides:
      - Per-modality sparse inputs at t_in (lon, lat, value).
      - Full-field targets [T, Q, V] at t_out = t_in + tau.

    Args:
        file_list: Sorted list of .npz file paths (one per timestep).
        grid_meta_path: Path to ClimSim_high-res_grid-info.nc.
        sparsity: Fraction of grid points per modality (default 0.02 = 432).
        triple_fraction: Fraction of per-modality sensors in the triple overlap.
        norm_stats: Dict with keys 'T','Q','V', each a (mean, std) tuple.
        input_modalities: Tuple of 3 bools — which modalities to include as input.
        input_region: "union" (all sensor locations) or "triple" (only T∩Q∩V).
        seed: Random seed for sensor placement.
        train: If True, use fixed tau=3; else sample randomly from horizons.
    """

    def __init__(self, file_list, grid_meta_path, sparsity=0.02, triple_fraction=0.25,
                 norm_stats=None, input_modalities=(1, 1, 1), input_region="union",
                 seed=123, train=False):
        self.file_list = file_list
        self.idx2latlon = load_idx2latlon(grid_meta_path)
        self.norm_stats = norm_stats
        self.horizons = [3, 6, 9, 12, 15, 18]
        self.seq_len = 19
        self.N = len(self.idx2latlon)
        self.train = train

        self.fixed_idx, self.region_of_local, self.mod_masks = _assign_venn_indices(
            n_points=self.N, sparsity=sparsity, triple_fraction=triple_fraction, seed=seed
        )
        self.union_idx = self.fixed_idx
        self.triple_idx = np.sort(
            np.where(self.mod_masks['T'] & self.mod_masks['Q'] & self.mod_masks['V'])[0]
        )

        self.input_region = input_region
        self.input_idx = self.triple_idx if input_region == "triple" else self.union_idx

        self._rng = random.Random(seed)
        assert len(input_modalities) == 3
        self.input_modalities = tuple(bool(int(x)) for x in input_modalities)

    def __len__(self):
        return len(self.file_list) - self.seq_len

    def _norm(self, arr, key):
        if self.norm_stats and key in self.norm_stats:
            mu, sigma = self.norm_stats[key]
            arr = (arr - mu) / sigma
        return arr

    def __getitem__(self, idx):
        seq = [np.load(self.file_list[idx + i]) for i in range(self.seq_len)]
        t_in = 0
        t_out = 3 if self.train else self._rng.choice(self.horizons)

        T_t = self._norm(seq[t_in]["state_t"], "T")
        Q_t = self._norm(seq[t_in]["state_q"], "Q")
        V_t = self._norm(seq[t_in]["state_v"], "V")
        T_tp = self._norm(seq[t_out]["state_t"], "T")
        Q_tp = self._norm(seq[t_out]["state_q"], "Q")
        V_tp = self._norm(seq[t_out]["state_v"], "V")

        data_T, mesh_T = [], []
        data_Q, mesh_Q = [], []
        data_V, mesh_V = [], []
        supervision_mask = []

        tau = float(t_out) / 18.0  # normalize to [0, 1]

        # --- Inputs from chosen region ---
        for gi in self.input_idx:
            lat, lon = self.idx2latlon[gi]
            bT = bool(self.mod_masks['T'][gi])
            bQ = bool(self.mod_masks['Q'][gi])
            bV = bool(self.mod_masks['V'][gi])
            supervision_mask.append([bT, bQ, bV])

            if self.input_modalities[0] and bT:
                data_T.append([lon, lat, float(T_t[gi])]); mesh_T.append([lon, lat])
            if self.input_modalities[1] and bQ:
                data_Q.append([lon, lat, float(Q_t[gi])]); mesh_Q.append([lon, lat])
            if self.input_modalities[2] and bV:
                data_V.append([lon, lat, float(V_t[gi])]); mesh_V.append([lon, lat])

        # --- Full-field targets ---
        data_y, mesh_y = [], []
        for gi in range(self.N):
            lat, lon = self.idx2latlon[gi]
            data_y.append([float(T_tp[gi]), float(Q_tp[gi]), float(V_tp[gi])])
            mesh_y.append([lon, lat])

        used_modalities = [int(self.input_modalities[0]),
                           int(self.input_modalities[1]),
                           int(self.input_modalities[2])]

        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        return (
            to_tensor(data_T), to_tensor(data_Q), to_tensor(data_V),
            to_tensor(mesh_T), to_tensor(mesh_Q), to_tensor(mesh_V),
            to_tensor(data_y), to_tensor(mesh_y),
            torch.tensor(self.input_idx, dtype=torch.long),
            torch.tensor(used_modalities, dtype=torch.bool),
            torch.tensor(supervision_mask, dtype=torch.bool),
            torch.tensor(tau, dtype=torch.float32),
        )

    def venn_counts(self):
        counts = {r: 0 for r in REGIONS}
        for r in self.region_of_local:
            counts[r] += 1
        return counts
