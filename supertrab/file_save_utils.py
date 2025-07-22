import sys
from pathlib import Path
from scipy.ndimage import zoom
import dask.array as da
import zarr
import numpy as np

# --- Config ---
zarr_root_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
group_name = "1996_R"
source_dataset = "registered_LR_upscaled"  # created by your `save_mhd_image_to_zarr_group` function
target_dataset = "registered_LR_upscaled_zoomed"
factor = 8
in_chunks = (128, 64, 64)
out_chunks = (128, 512, 512)
order = 3  # cubic interpolation

# --- Load original dataset lazily ---
print(f"\n🔹 Opening Zarr group: {zarr_root_path}/{group_name}/{source_dataset}")
try:
    image_dask = da.from_zarr(f"{zarr_root_path}/{group_name}/{source_dataset}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Zarr dataset: {e}")

print(f"🔹 Original shape: {image_dask.shape}, dtype: {image_dask.dtype}")

# --- Rechunk before zoom ---
image_dask = image_dask.rechunk(in_chunks)
print(f"🔹 Rechunked to: {image_dask.chunks}")

# --- Define block-wise zoom ---
def zoom_block(block, zoom_factor=factor, order=order):
    return zoom(block, zoom=zoom_factor, order=order)

# --- Predict output shape and chunking ---
expected_shape = tuple(s * factor for s in image_dask.shape)
expected_chunks = tuple(c * factor for c in in_chunks)
print(f"🔹 Expected output shape: {expected_shape}")
print(f"🔹 Expected output chunks: {expected_chunks}")

# --- Apply block-wise zoom with correct output chunking ---
try:
    image_upscaled = image_dask.map_blocks(
        zoom_block,
        dtype=image_dask.dtype,
        chunks=expected_chunks
    )
except Exception as e:
    raise RuntimeError(f"Zooming failed: {e}")

print(f"Zoom applied lazily. Shape: {image_upscaled.shape}, Chunks: {image_upscaled.chunks}")

# --- Save to new Zarr dataset ---
root = zarr.open(str(zarr_root_path), mode="a")
group = root.require_group(group_name)

# Delete target dataset if exists
if target_dataset in group:
    print(f"⚠️ Overwriting existing dataset: {target_dataset}")
    del group[target_dataset]

print(f"\nSaving to Zarr at: {zarr_root_path}/{group_name}/{target_dataset}")
try:
    image_upscaled.to_zarr(store=group.store, component=f"{group_name}/{target_dataset}", overwrite=True)
except Exception as e:
    raise RuntimeError(f"Failed to save upscaled data to Zarr: {e}")

# --- Verify ---
loaded = da.from_zarr(f"{zarr_root_path}/{group_name}/{target_dataset}")
print(f"\nReloaded saved data. Shape: {loaded.shape}, Chunks: {loaded.chunks}")

if loaded.shape == expected_shape:
    print("SUCCESS: Upscaled data saved with correct shape.")
else:
    print("WARNING: Saved shape doesn't match expected upscaled shape!")

