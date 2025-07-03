import dask.array as da
import zarr
from pathlib import Path
import shutil

# Define the path to your Zarr dataset
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
group_path = zarr_path / "1955_L" / "image"  # Modify this to the dataset you want to trim

# Load the original data
image = da.from_zarr(group_path)
print(f"Original shape: {image.shape}")

# Remove the first 33 slices along the Z axis (axis 0)
trimmed = image[33:]
print(f"Trimmed shape: {trimmed.shape}")

# Delete the existing dataset
shutil.rmtree(group_path)

# Save the trimmed version in the same location
trimmed.to_zarr(group_path, overwrite=True)
print("Done. Original dataset overwritten with trimmed version.")
