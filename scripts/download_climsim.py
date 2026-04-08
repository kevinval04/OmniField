import os
import requests
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

# === Constants ===
BASE_URL = "https://huggingface.co/datasets/LEAP/ClimSim_high-res/resolve/main"
MONTHS = ["0001-02", "0001-03", "0001-04","0001-05","0001-06","0001-07","0001-08","0001-09","0001-09"]
OUT_ROOT = "processed"
LEVEL = 59
VARS = {
    "state_t": "state_t",
    "state_q": "state_q0001",
    "state_v": "state_v"
}
TIME_STEPS = list(range(0, 86400, 1200))  # 72 steps per day (0 to 85200)
TOTAL_SAMPLES = 10018


# === Helpers ===
def build_url_and_paths(month, date_str, step):
    folder = f"train/{month}"
    filename = f"E3SM-MMF.mli.{date_str}-{step:05}.nc"
    url = f"{BASE_URL}/{folder}/{filename}"
    out_dir = os.path.join(OUT_ROOT, month)
    out_path = os.path.join(out_dir, filename.replace(".nc", ".npz"))
    nc_path = os.path.join(out_dir, filename)
    return url, out_dir, nc_path, out_path

def download_file(url, local_path):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def process_nc_file(nc_path, out_path):
    ds = xr.open_dataset(nc_path)
    np.savez_compressed(
        out_path,
        state_t=ds[VARS["state_t"]].values[LEVEL],
        state_q=ds[VARS["state_q"]].values[LEVEL],
        state_v=ds[VARS["state_v"]].values[LEVEL],
    )
    ds.close()

# === Main ===
def main():
    count = 0
    current_date = datetime.strptime("0001-02-01", "%Y-%m-%d")
    with open("success.log", "a") as slog, open("error.log", "a") as elog:
        while count < TOTAL_SAMPLES:
            year = f"{current_date.year:04}"
            month = f"{current_date.month:02}"
            day = f"{current_date.day:02}"
            month_folder = f"{year}-{month}"
            if month_folder not in MONTHS:
                current_date += timedelta(days=1)
                continue

            date_str = f"{year}-{month}-{day}"

            for step in TIME_STEPS:
                if count >= TOTAL_SAMPLES:
                    break

                url, out_dir, nc_path, out_path = build_url_and_paths(month_folder, date_str, step)

                if os.path.exists(out_path):
                    print(f"[{count:05}] ✓ Skipping (already processed): {out_path}")
                    count += 1
                    continue

                try:
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"[{count:05}] ↓ Downloading: {url}")
                    download_file(url, nc_path)

                    print(f"[{count:05}] ✂ Extracting → {out_path}")
                    process_nc_file(nc_path, out_path)

                    os.remove(nc_path)
                    slog.write(f"{url}\n")
                    slog.flush()
                    count += 1
                except Exception as e:
                    print(f"[{count:05}] ⚠ Error: {e}")
                    elog.write(f"{url}\t{e}\n")
                    elog.flush()
                    if os.path.exists(nc_path):
                        os.remove(nc_path)

            current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
