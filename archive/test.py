import dask.array as da
import zarr
from pathlib import Path
import shutil

# Define the path to your Zarr dataset
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
group_path = zarr_path / "2019_L" / "registered_LR_upscaled"  # Modify this to the dataset you want to trim

# Load the original data
image = da.from_zarr(group_path)
print(f"Original shape: {image.shape}")

# Remove the first 33 slices along the Z axis (axis 0)
trimmed = image[2:]
print(f"Trimmed shape: {trimmed.shape}")

new_group = zarr_path / "2019_L" / "registered_LR_upscaled_trimmed"

# Save the trimmed version 
trimmed.to_zarr(new_group, overwrite=True)
print(f"Trimmed data saved to: {new_group}")
