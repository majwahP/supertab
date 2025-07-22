"""
This script prepares patch-based trabecular bone masks from external MHD mask 
files and optionally filters patches based on local intensity variance. The generated masks,
the one loaded directly from .mhd and variance filtered are saved in zarr format as a new 
group togehter with its sample.
A notebook for generation of .mhd mask for trabecular bone can be found here:
https://github.com/gianthk/pyfabric/blob/master/notebooks/supertrab_trabecular_core_mask.ipynb

"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import supertrab.file_save_utils as prepare
from supertrab.file_save_utils import save_mhd_image_to_zarr_group



file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/QCT/QCT1996R/1996_R_R_HR_elastix_03/result.0.mhd"
#choose group

# group_name = "1955_L"
# group_name = "1956_L"
group_name = "1996_R"
# group_name = "2005_L"
# group_name = "2007_L"
# group_name = "2019_L"
#patch_size = (2, 4, 4) #must be bigger for the variance mask
#variance_threshold = 1

#root, scan_group, trabecular_mask = prepare.create_and_save_trabecular_mask(file_path, data_dir, group_name, patch_size)

# variance filtering --------------------------------------------------

# for dataset_name in scan_group:
#     if not dataset_name.endswith("_trabecular_mask"):
#         continue
#     original_name = dataset_name.replace("_trabecular_mask", "")
#     prepare.filter_mask_by_variance(root, scan_group, trabecular_mask, original_name, patch_size, variance_threshold)

print("Start")

save_mhd_image_to_zarr_group(
    mhd_path=data_dir,
    zarr_root_path=file_path,
    group_name=group_name,
    dataset_name="registered_LR_upscaled"
)


print("done")
