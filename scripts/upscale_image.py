import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import dask.array as da
from scipy.ndimage import zoom
import zarr
import numpy as np

# --- Config ---
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
group_name = "1996_R"
original_dataset = "registered_LR"
new_dataset = "registered_LR_upscaled"
rechunked_dataset = "registered_LR_upscaled_rechunked"
factor = 8
input_chunks = (25, 96, 96)
order = 3  # 3=cubic, 0=nearest (for masks)
final_chunks = (128, 512, 512)

# --- Open the root Zarr group ---
root = zarr.open(str(zarr_path), mode="a")
image_dask = da.from_zarr(f"{zarr_path}/{group_name}/{original_dataset}")
image_dask = image_dask.rechunk(input_chunks)
print(f"[INFO] Original shape: {image_dask.shape}, chunks: {image_dask.chunks}")

# --- Define zoom block ---
def zoom_block(block, zoom_factor=factor, order=order):
    return zoom(block, zoom=zoom_factor, order=order)

# --- Compute new shape ---
orig_shape = image_dask.shape
upscaled_shape = tuple(s * factor for s in orig_shape)
print(f"[INFO] Target upscaled shape: {upscaled_shape}")

# --- Compute zoomed array ---
image_upscaled = image_dask.map_blocks(
    zoom_block,
    dtype=image_dask.dtype,
    chunks=tuple(s * factor for s in input_chunks)
)
print(f"[INFO] Upscaled shape (lazy): {image_upscaled.shape}, chunks: {image_upscaled.chunks}")

# --- Sanity check ---
assert image_upscaled.shape == upscaled_shape, "[ERROR] Shape mismatch after zoom!"

# --- Save un-rechunked upscaled data ---
if new_dataset in root[group_name]:
    print(f"[WARNING] Overwriting existing dataset: {new_dataset}")
    del root[group_name][new_dataset]

image_upscaled.to_zarr(f"{zarr_path}/{group_name}/{new_dataset}", overwrite=True)
print(f"[INFO] Saved raw upscaled image to '{new_dataset}'.")

# --- Reload and rechunk ---
image_loaded = da.from_zarr(f"{zarr_path}/{group_name}/{new_dataset}")
image_rechunked = image_loaded.rechunk(final_chunks)
print(f"[INFO] Rechunked to: {image_rechunked.shape}, chunks: {image_rechunked.chunks}")

# --- Save rechunked version ---
if rechunked_dataset in root[group_name]:
    print(f"[WARNING] Overwriting existing dataset: {rechunked_dataset}")
    del root[group_name][rechunked_dataset]

image_rechunked.to_zarr(f"{zarr_path}/{group_name}/{rechunked_dataset}", overwrite=True)
print(f"[✅] Done. Saved rechunked image to '{rechunked_dataset}' with chunks {final_chunks}")
