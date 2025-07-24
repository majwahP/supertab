import dask.array as da
import zarr
from pathlib import Path

split_num = 16

# Define paths
# zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")

# image_path = zarr_path / "1955_L" / "image"  
# save_root = zarr_path / "1955_L" / "image_split" 

image_path = zarr_path / "1955_L" / "image_split" / "part_2" 
save_root = zarr_path / "1955_L" / "image_split" / "part_2_split"

# image_path = zarr_path / "2019_L" / "registered_LR_upscaled_trimmed"  
# save_root = zarr_path / "2019_L" / "registered_LR_upscaled_trimmed_split" 

# image_path = zarr_path / "2019_L" / "registered_LR_upscaled_trimmed_split" / "part_2" 
# save_root = zarr_path / "2019_L" / "registered_LR_upscaled_trimmed_split" / "part_2_split"

# image_path = zarr_path / "1955_L" / "image_trabecular_mask"  
# save_root = zarr_path / "1955_L" / "image_trabecular_mask_split" 

# image_path = zarr_path / "2007_L" / "image_trabecular_mask_split" / "part_2"
# save_root = zarr_path / "2007_L" / "image_trabecular_mask_split" / "part_2_split"

image = da.from_zarr(image_path)

# Get total number of slices and calculate chunk size
num_slices = image.shape[0]
parts = num_slices // split_num

save_root.mkdir(parents=True, exist_ok=True)

# Save each quarter
for i in range(split_num):
    start = i * parts
    end = (i + 1) * parts if i < (split_num-1) else num_slices  
    part = image[start:end]
    
    save_path = save_root / f"part_{i+1}"
    print(f"Saving slices {start}:{end} to {save_path}")
    if save_path.exists():
        print(f"[WARNING] {save_path} already exists! Skipping to prevent overwrite.")
        continue
    else:
        part.to_zarr(save_path)



print("Done splitting and saving.")
