import numpy as np
from pathlib import Path
import zarr
import zarr.hierarchy
import tqdm
import SimpleITK as sitk

patch_size = (1, 32, 32) 

## Open .zarr file
file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/00_resampled_data/1955_L" # define group

root: zarr.hierarchy.Group = zarr.open(str(file_path))

group_name = "1955_L" # define group
scan_group: zarr.hierarchy.Group = root[group_name]


for dataset_name in scan_group:
    #create group of not already exist
    if (
        dataset_name.endswith("_bone_mask")
        or dataset_name.endswith("_trabecular_mask")
        or dataset_name.endswith("_variance")
        or dataset_name.endswith("_mean")
    ):
        continue
    ## Load mask ##
    mhd_mask_path = Path(f"{data_dir}/QCT/QCTFEMUR_1955L/masks/QCTFEMUR_1955L_R_HR_trab.mhd") # define group

    #Check if mask exist
    if not mhd_mask_path.exists():
        print(f"Skipping {group_name}, no mask file found.")
        continue
        
    mhd_mask = sitk.ReadImage(mhd_mask_path)
    import_mask = sitk.GetArrayFromImage(mhd_mask) 
    import_mask = import_mask.astype(np.uint8)

    ## Save mask to group ## 
    scan_group.create_dataset(
        f"{dataset_name}_trabecular_mask",
        data=import_mask,
        dtype="uint8",
        overwrite=True,
        chunks=patch_size,  
    )

print("done")
print(root.tree())