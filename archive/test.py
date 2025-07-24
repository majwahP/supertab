import dask.array as da
import zarr
from pathlib import Path
import shutil
import dask

dask.config.set(scheduler="threads")

group = "2007_L"
# Define the path to your Zarr dataset
# zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")
group_path = zarr_path / group / "registered_LR_upscaled_rechunked"  # Modify this to the dataset you want to trim

# Load the original data
image = da.from_zarr(group_path)
print(f"Original shape: {image.shape}")

#add slices
# empty_slices = da.zeros((1, image.shape[1], image.shape[2]), dtype=image.dtype)
# trimmed = da.concatenate([empty_slices, image], axis=0)
# print(f"Padded shape: {trimmed.shape}")

#remove slices
trimmed = image[2:]
print(f"Trimmed shape: {trimmed.shape}")

new_group = zarr_path / group / "registered_LR_upscaled_trimmed"

# Save the trimmed version 
trimmed.to_zarr(new_group, overwrite=False)
print(f"Trimmed data saved to: {new_group}")
