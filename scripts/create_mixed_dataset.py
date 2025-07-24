import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader, Dataset
import zarr
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.inferance_utils import scale
# --- CONFIG ---
root_zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/paired_patch_dataset.zarr")
group_name = "2019_L"
patch_size = (1, 256, 256)
sigma = 1.3
downsample_factor = 4
batch_size = 1

# --- Step 1: Create Output Group ---
root_zarr_path.mkdir(parents=True, exist_ok=True) 

zarr_root = zarr.open(str(root_zarr_path), mode="a")  
zarr_group = zarr_root.require_group(group_name)    

z_qct     = zarr_group.require_dataset("qct",     shape=(0,) + patch_size, chunks=(1,) + patch_size, dtype="float32", maxshape=(None,) + patch_size)
z_hrpqct  = zarr_group.require_dataset("hrpqct",  shape=(0,) + patch_size, chunks=(1,) + patch_size, dtype="float32", maxshape=(None,) + patch_size)
z_lr      = zarr_group.require_dataset("lr",      shape=(0,) + patch_size, chunks=(1,) + patch_size, dtype="float32", maxshape=(None,) + patch_size)

# --- Step 2: Create Dataloaders ---
gt_train_dataloader = create_dataloader(
    zarr_path=str(Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")), 
    downsample_factor=downsample_factor,
    patch_size=patch_size,
    groups_to_use=[group_name],
    data_dim="2d",
    with_blur=True,
    draw_same_chunk=True,
    shuffle=False,
    image_group="image",
    batch_size=batch_size
)

train_dataloader = create_dataloader(
    zarr_path=str(Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")),
    downsample_factor=downsample_factor,
    patch_size=patch_size,
    groups_to_use=[group_name],
    data_dim="2d",
    with_blur=False,
    draw_same_chunk=True,
    shuffle=False,
    image_group="registered_LR_upscaled_trimmed",
    batch_size=batch_size
)

threshold = 1000  
air_fraction_limit = 0.05
idx = z_qct.shape[0]  # continue from previous index if appending

for batch_lr, batch_hr in tqdm(zip(train_dataloader, gt_train_dataloader), desc=f"Saving to group {group_name}"):
    
    pos_lr = batch_lr["position"]
    pos_hr = batch_hr["position"]

    if not torch.equal(pos_lr, pos_hr):
        raise ValueError(f"Position mismatch! LR: {pos_lr} vs HR: {pos_hr}")


    qct_patch    = batch_lr["hr_image"].squeeze(0)
    hrpqct_patch = batch_hr["hr_image"].squeeze(0)
    lr_patch     = batch_hr["lr_image"].squeeze(0)

    # Filter out air
    qct_scaled = scale(qct_patch * 32768.0)
    air_mask = qct_scaled < threshold
    air_fraction = air_mask.float().mean().item()
    if air_fraction > air_fraction_limit:
        continue
    
    qct_patch = qct_scaled/32768.0

    z_qct.resize((idx + 1,) + z_qct.shape[1:])
    z_hrpqct.resize((idx + 1,) + z_hrpqct.shape[1:])
    z_lr.resize((idx + 1,) + z_lr.shape[1:])


    z_qct[idx]     = qct_patch.numpy()
    z_hrpqct[idx]  = hrpqct_patch.numpy()
    z_lr[idx]      = lr_patch.numpy()
    
    idx += 1

print(f" Done: Saved to group '{group_name}' at {root_zarr_path}")