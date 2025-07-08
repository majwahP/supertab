import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import dask.array as da
from scipy.ndimage import zoom
import zarr
import numpy as np

# --- Config ---
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
group_name = "2019_L"
original_dataset = "registered_LR"
new_dataset = "registered_LR_upscaled"
rechunked_dataset = "registered_LR_upscaled_rechunked"
factor = 8
in_chunks = (64, 64, 64)
order = 3  # interpolation: 3=cubic, 0=nearest for masks
final_chunks = (128, 512, 512)

# --- Load original image lazily ---
root = zarr.open(str(zarr_path), mode="a")
image_dask = da.from_zarr(f"{zarr_path}/{group_name}/{original_dataset}")
image_dask = image_dask.rechunk(in_chunks)
print(f"Original shape: {image_dask.shape}, chunks: {image_dask.chunks}")

# --- Block-wise zoom ---
def zoom_block(block, zoom_factor=factor, order=order):
    return zoom(block, zoom=zoom_factor, order=order)

image_upscaled = image_dask.map_blocks(zoom_block, dtype=image_dask.dtype)
print(f"Upscaled shape (lazy): {image_upscaled.shape}, chunks: {image_upscaled.chunks}")

# --- Save first (with whatever chunks came from zoom) ---
if new_dataset in root[group_name]:
    print(f"Overwriting existing dataset: {new_dataset}")
    del root[group_name][new_dataset]

image_upscaled.to_zarr(f"{zarr_path}/{group_name}/{new_dataset}", overwrite=True)
print(f"Saved initial upscaled image to '{new_dataset}'.")

# --- Re-load and rechunk to regular shape ---
image_loaded = da.from_zarr(f"{zarr_path}/{group_name}/{new_dataset}")
image_rechunked = image_loaded.rechunk(final_chunks)

# --- Save rechunked version ---
if rechunked_dataset in root[group_name]:
    print(f"Overwriting existing dataset: {rechunked_dataset}")
    del root[group_name][rechunked_dataset]

image_rechunked.to_zarr(f"{zarr_path}/{group_name}/{rechunked_dataset}", overwrite=True)
print(f"Saved rechunked image to '{rechunked_dataset}'.")
