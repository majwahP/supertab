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

import supertrab.trabecular_mask_utils as prepare


file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/00_resampled_data/2019_L"
#choose group

# group_name = "1955_L"
# group_name = "1956_L"
# group_name = "1996_R"
# group_name = "2005_L"
# group_name = "2007_L"
group_name = "2019_L"
patch_size = (2, 2, 2) #must be bigger for the variance mask
variance_threshold = 1

root, scan_group, trabecular_mask = prepare.create_and_save_trabecular_mask(file_path, data_dir, group_name, patch_size)

# variance filtering --------------------------------------------------

# for dataset_name in scan_group:
#     if not dataset_name.endswith("_trabecular_mask"):
#         continue
#     original_name = dataset_name.replace("_trabecular_mask", "")
#     prepare.filter_mask_by_variance(root, scan_group, trabecular_mask, original_name, patch_size, variance_threshold)

print("done")
print(root.tree())
