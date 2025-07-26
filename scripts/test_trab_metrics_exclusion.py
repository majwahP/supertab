import torch
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import ensure_3d_volume
from supertrab.metrics_utils import get_mask_ormir, get_mask_otsu

# --- CONFIG ---
PATCH_SIZE = (256, 256, 256)
DS_FACTOR = 10
VOXEL_SIZE = 0.0303
BVF_THRESHOLD = 0.04
BVF_THRESHOLD_HIGH = 0.96
BATCH_SIZE = 1
GROUPS = ["2019_L"]
ZARR_PATH = "/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"
SAVE_DIR = "low_bvf_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

def compute_bvf(mask):
    mask_np = mask.cpu().numpy().astype(bool)
    bvf = np.sum(mask_np) / mask_np.size
    print(f"[DEBUG] BVF: {bvf}")
    return bvf

def has_empty_slice(volume: torch.Tensor) -> bool:
    if volume.ndim == 2:
        # For 2D: check if the whole patch is zero
        return torch.all(volume == 0)
    elif volume.ndim == 3:
        # For 3D: check if any slice along depth is completely zero
        return torch.any(torch.all(volume == 0, dim=(1, 2)))
    else:
        raise ValueError(f"Unsupported volume shape {volume.shape}")

def visualize_patch_and_mask(patch, mask, label, position, save_dir):
    patch = patch.cpu().numpy()
    mask = mask.cpu().numpy()
    d, h, w = patch.shape
    mid_slices = {
        "XY": (patch[d//2], mask[d//2]),
        "XZ": (patch[:, h//2], mask[:, h//2]),
        "YZ": (patch[:, :, w//2], mask[:, :, w//2])
    }

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for i, (plane, (img, msk)) in enumerate(mid_slices.items()):
        axs[0, i].imshow(img, cmap="gray")
        axs[0, i].set_title(f"{label} - {plane}")
        axs[0, i].axis("off")
        axs[1, i].imshow(msk, cmap="gray")
        axs[1, i].set_title(f"Mask - {plane}")
        axs[1, i].axis("off")

    fig.suptitle(f"{label} @ {position}")
    plt.tight_layout()
    save_path = Path(save_dir) / f"{label}_{position}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# --- LOAD DATALOADERS ---
dataloader_HR_LR = create_dataloader(
    zarr_path=ZARR_PATH,
    patch_size=PATCH_SIZE,
    downsample_factor=DS_FACTOR,
    groups_to_use=GROUPS,
    batch_size=BATCH_SIZE,
    draw_same_chunk=True,
    shuffle=False,
    enable_sr_dataset=True,
    data_dim="3d",
    image_group="image_split/reassembled_HR",
    mask_base_path="image_trabecular_mask_split/reassembled",
    mask_group=""
)

dataloader_SR = create_dataloader(
    zarr_path=ZARR_PATH,
    patch_size=PATCH_SIZE,
    downsample_factor=DS_FACTOR,
    groups_to_use=GROUPS,
    batch_size=BATCH_SIZE,
    draw_same_chunk=True,
    shuffle=False,
    enable_sr_dataset=True,
    data_dim="3d",
    image_group=f"sr_volume_256_{DS_FACTOR}/reassembled",
    mask_base_path="image_trabecular_mask_split/reassembled",
    mask_group=""
)

# --- PROCESS PATCHES ---
no_slice_counter = 0
bvf_counter = 0
total = 0
for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Scanning patches"):

    total += 1

    hr = ensure_3d_volume(batch_HR_LR["hr_image"][0])
    lr = ensure_3d_volume(batch_HR_LR["lr_image"][0])
    sr = ensure_3d_volume(batch_SR["hr_image"][0]) * 32768.0 

    pos = tuple(batch_HR_LR["position"][0].tolist())

    sr = sr.cpu()
    if sr.sum() == 0 or has_empty_slice(sr):
        no_slice_counter  += 1
        continue

    try:
        mask_hr = get_mask_ormir(hr)
        mask_lr = get_mask_otsu(lr)
        mask_sr = get_mask_ormir(sr)

        bvf_hr = compute_bvf(mask_hr)
        bvf_lr = compute_bvf(mask_lr)
        bvf_sr = compute_bvf(mask_sr)

        if min(bvf_hr, bvf_lr, bvf_sr) <= BVF_THRESHOLD or max(bvf_hr, bvf_lr, bvf_sr) >= BVF_THRESHOLD_HIGH:
            # visualize_patch_and_mask(hr, mask_hr, "HR", pos, SAVE_DIR)
            # visualize_patch_and_mask(lr, mask_lr, "LR", pos, SAVE_DIR)
            # visualize_patch_and_mask(sr, mask_sr, "SR", pos, SAVE_DIR)
            bvf_counter += 1
        


    except Exception as e:
        print(f"Error at {pos}: {e}")
        continue

print("Total: ", total)
print("excluded due to missing slice: ", no_slice_counter)
print("Excluded due to high or low BVF: ", bvf_counter)