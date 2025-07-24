import dask.array as da
import zarr
from pathlib import Path

import zarr
from pathlib import Path

z = zarr.open("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr", mode="r")
print(z.tree())

# --- Config ---
# Source Zarr store and dataset path
source_zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/users/mwahlin/zarr/supertrab.zarr")

source_group = "1955_L"
source_dataset = "image"

# Target Zarr store and destination group
target_zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")

target_group = "1955_L"
target_dataset = "image" 

# Optional: chunking for the output
output_chunks =(128, 512, 512)

# --- Load data ---
source_path = source_zarr_path / source_group / source_dataset
print(f"[INFO] Loading dataset from: {source_path}")
data = da.from_zarr(source_path)

# Optionally rechunk before saving
if output_chunks is not None:
    print(f"[INFO] Rechunking to {output_chunks}")
    data = data.rechunk(output_chunks)

# --- Save to target ---
target_dataset_path = target_zarr_path / target_group / target_dataset
print(f"[INFO] Saving to: {target_dataset_path}")
data.to_zarr(str(target_dataset_path), overwrite=False)

print("[✅] Done.")
