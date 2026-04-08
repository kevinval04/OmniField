"""
EPA-AQS dataset for multimodal air quality forecasting.

Implements the point-cloud dataset described in Section 5.1:
  - 6 modalities: O3, PM2.5, PM10, NO2, CO, SO2
  - Native day-specific masks (no imputation)
  - Lead-time forecasting with variable sensor availability

Data source: U.S. EPA Air Quality System (https://www.epa.gov/aqs)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, Union
from collections import defaultdict

POLS = ["Ozone", "PM2.5", "PM10", "NO2", "CO", "SO2"]
POL2IDX = {p: i for i, p in enumerate(POLS)}
DEF_MIN_SITES = {"Ozone": 20, "PM2.5": 20, "PM10": 15, "NO2": 10, "CO": 10, "SO2": 10}


class AQSPointCloudLeadForecast(Dataset):
    """AQS point-cloud dataset with lead-time forecasting.

    Each sample provides:
      - inputs: per-modality point clouds {values, coords} over a t_in-day window
      - targets: per-modality point clouds on the target day (t_in + lead)
      - presence masks indicating which modalities have sufficient data

    Args:
        df: DataFrame with columns [Date, Defining Parameter/Pollutant, AQI, Latitude, Longitude].
        t_in: Number of input days in the window.
        t_out: Lead time(s) — int, tuple (min, max), or list of specific leads.
        sample_lead: If True, randomly sample lead per instance; else use first valid.
        min_sites_by_modality: Minimum stations required per modality per day.
        start_date, end_date: Optional date filters.
        require_any_input: Require at least one modality in input window.
        require_any_target: Require at least one modality on target day.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        t_in: int = 1,
        t_out: Union[int, Sequence[int]] = 1,
        sample_lead: bool = False,
        min_sites_by_modality: Optional[Dict[str, int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        require_any_input: bool = True,
        require_any_target: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.t_in = int(t_in)
        self.require_any_input = require_any_input
        self.require_any_target = require_any_target
        self.min_sites_by_modality = (min_sites_by_modality or DEF_MIN_SITES).copy()
        self.rng = rng or np.random.default_rng()

        # Parse leads
        if isinstance(t_out, int):
            leads = [int(t_out)]
        elif isinstance(t_out, tuple) and len(t_out) == 2:
            a, b = int(t_out[0]), int(t_out[1])
            assert a <= b and a >= 1
            leads = list(range(a, b + 1))
        else:
            leads = [int(x) for x in t_out]
            assert all(x >= 1 for x in leads)
        self.leads = sorted(set(leads))
        self.sample_lead = bool(sample_lead and len(self.leads) > 1)

        # Clean & filter dataframe
        d = df.rename(columns={"Defining Parameter": "Pollutant"}).copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.loc[d["Pollutant"].isin(POLS), ["Date", "Pollutant", "AQI", "Latitude", "Longitude"]].dropna()
        if start_date:
            d = d[d["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            d = d[d["Date"] <= pd.to_datetime(end_date)]

        # Pre-index: (day, mod) → arrays
        self.by_day_mod: Dict[tuple, Dict[str, np.ndarray]] = {}
        for (day, mod), g in d.groupby(["Date", "Pollutant"]):
            self.by_day_mod[(pd.Timestamp(day), mod)] = {
                "aqi": g["AQI"].to_numpy(np.float32),
                "lat": g["Latitude"].to_numpy(np.float32),
                "lon": g["Longitude"].to_numpy(np.float32),
            }

        # Build valid anchor days
        all_days = sorted({day for (day, _) in self.by_day_mod.keys()})
        self.anchor_days = []
        self.valid_leads_by_day: Dict[pd.Timestamp, list] = {}

        if all_days:
            last_day = all_days[-1]
            max_lead = max(self.leads)
            for d0 in all_days:
                out_day = d0 + pd.Timedelta(days=(self.t_in - 1 + max_lead))
                if out_day > last_day:
                    continue

                p_in = self._presence_window(d0, self.t_in)
                valid_leads = []
                for L in self.leads:
                    tgt_day = d0 + pd.Timedelta(days=(self.t_in - 1 + L))
                    if self._any_modality_present_on_day(tgt_day):
                        valid_leads.append(L)

                if (not self.require_any_input or p_in.any()) and \
                   (not self.require_any_target or len(valid_leads) > 0):
                    self.anchor_days.append(d0)
                    self.valid_leads_by_day[d0] = valid_leads if self.require_any_target else list(self.leads)

    def __len__(self):
        return len(self.anchor_days)

    def _any_modality_present_on_day(self, day):
        for m in POLS:
            key = (day, m)
            if key in self.by_day_mod and len(self.by_day_mod[key]["aqi"]) >= self.min_sites_by_modality.get(m, 0):
                return True
        return False

    def _presence_window(self, start_day, window_len):
        present = np.zeros(len(POLS), dtype=bool)
        for j, mod in enumerate(POLS):
            for t in range(window_len):
                day = start_day + pd.Timedelta(days=t)
                key = (day, mod)
                if key in self.by_day_mod and len(self.by_day_mod[key]["aqi"]) >= self.min_sites_by_modality.get(mod, 0):
                    present[j] = True
                    break
        return present

    def _collect_window(self, start_day, window_len):
        out = {}
        present = torch.zeros(len(POLS), dtype=torch.bool)
        for mod in POLS:
            vals_list, lat_list, lon_list = [], [], []
            for t in range(window_len):
                day = start_day + pd.Timedelta(days=t)
                key = (day, mod)
                if key not in self.by_day_mod:
                    continue
                arr = self.by_day_mod[key]
                if len(arr["aqi"]) < self.min_sites_by_modality.get(mod, 0):
                    continue
                vals_list.append(arr["aqi"])
                lat_list.append(arr["lat"])
                lon_list.append(arr["lon"])

            if not vals_list:
                out[mod] = {"values": torch.empty(0), "coords": torch.empty(0, 3)}
            else:
                values = torch.from_numpy(np.concatenate(vals_list).astype(np.float32))
                lat = np.concatenate(lat_list).astype(np.float32)
                lon = np.concatenate(lon_list).astype(np.float32)
                toff = np.zeros_like(lat, dtype=np.float32)
                coords = torch.from_numpy(np.stack([lat, lon, toff], axis=1))
                out[mod] = {"values": values, "coords": coords}
                present[POL2IDX[mod]] = True
        return out, present

    def __getitem__(self, idx):
        d0 = self.anchor_days[idx]
        inputs, present_in = self._collect_window(d0, self.t_in)

        leads_here = self.valid_leads_by_day.get(d0, self.leads)
        lead = int(self.rng.choice(leads_here)) if self.sample_lead and leads_here else int(leads_here[0])

        tgt_day = d0 + pd.Timedelta(days=(self.t_in - 1 + lead))
        targets, present_out = self._collect_window(tgt_day, 1)

        return {
            "date": d0,
            "lead": lead,
            "inputs": inputs,
            "targets": targets,
            "present_in": present_in,
            "present_out": present_out,
        }


def aqs_collate_lead(batch):
    """Custom collate for AQS dataset — keeps dicts intact."""
    return {
        "date": [b["date"] for b in batch],
        "lead": torch.tensor([b["lead"] for b in batch], dtype=torch.float32),
        "inputs": [b["inputs"] for b in batch],
        "targets": [b["targets"] for b in batch],
        "present_in": torch.stack([b["present_in"] for b in batch]),
        "present_out": torch.stack([b["present_out"] for b in batch]),
    }
