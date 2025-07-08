import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import torch
import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm
import json
import os
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np

from supertrab.sr_dataset_utils import create_dataloader
from sklearn.metrics import jaccard_score
from supertrab.metrics_utils import get_mask_ormir, get_mask_otsu
from supertrab.analysis_utils import has_empty_slice, visualize_3d_masks, visualize_masks

PATCH_SIZE = 256
DS_FACTOR = 6

def compute_dice(mask1, mask2):
    intersection = torch.sum((mask1 & mask2).float())
    return 2. * intersection / (mask1.sum() + mask2.sum())

def compute_jaccard(mask1, mask2):
    mask1_np = mask1.cpu().numpy().astype(bool).ravel()
    mask2_np = mask2.cpu().numpy().astype(bool).ravel()
    return jaccard_score(mask1_np, mask2_np)



def main(
    zarr_path,
    dim = "2d",
    patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    downsample_factor=DS_FACTOR,
    batch_size=4,
    output_dir="metric_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    nr_samples = 10,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Mask evaluation for downsapling factor {DS_FACTOR} and patch size {PATCH_SIZE}")

    dataloader_HR_LR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=dim, 
        num_workers=0, 
        prefetch=None,
        image_group="image_split/reassembled_HR", 
        #mask_base_path="image_trabecular_mask",
        mask_base_path="image_trabecular_mask_split/reassembled",
        mask_group=""
    )

    dataloader_SR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=dim, 
        num_workers=0, 
        prefetch=None,
        image_group=f"sr_volume_256_{DS_FACTOR}/reassembled",        
        #mask_base_path="image_trabecular_mask",
        mask_base_path="image_trabecular_mask_split/reassembled",
        mask_group=""
    )


    total_patches = 0
    sample_idx = 0
    dice_scores = []
    jaccard_scores = []
    saved_masks_hr = []
    saved_masks_lr = []
    saved_masks_sr = []

    for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Processing patches"):
        lr_images = batch_HR_LR["lr_image"].to(device)  
        hr_images = batch_HR_LR["hr_image"].to(device)
        sr_images = batch_SR["hr_image"].to(device)* 32768.0 #otherwise double devision! devide when fetching hr_image but SR data alsready in range -1,1

        for hr_patch, lr_patch, sr_patch in zip(hr_images, lr_images, sr_images):
            if sample_idx % 10 == 0:
                sr = sr_patch[0].cpu()
                hr = hr_patch[0].cpu()
                lr = lr_patch[0].cpu()

                if sr.sum() == 0 or has_empty_slice(sr):
                    continue
                
                mask_hr = get_mask_ormir(hr)
                # mask_lr = get_mask_ormir(lr)
                mask_lr = get_mask_otsu(lr)
                mask_sr = get_mask_ormir(sr)

                # dice_lr_hr = compute_dice(mask_lr, mask_hr).item()
                # dice_sr_hr = compute_dice(mask_sr, mask_hr).item()
                # jaccard_lr_hr = compute_jaccard(mask_lr, mask_hr)
                # jaccard_sr_hr = compute_jaccard(mask_sr, mask_hr)

                # dice_scores.append((dice_lr_hr, dice_sr_hr))
                # jaccard_scores.append((jaccard_lr_hr, jaccard_sr_hr))

                saved_masks_hr.append(mask_hr)
                saved_masks_lr.append(mask_lr)
                saved_masks_sr.append(mask_sr)

            sample_idx += 1

            total_patches += 1
        if len(saved_masks_hr) >= nr_samples: 
            break
    
    print(f"Total patches included: {total_patches}")

    dice_scores = np.array(dice_scores)
    jaccard_scores = np.array(jaccard_scores)

    print("\n--- Summary ---")
    print(dim)
    print(f"Downsapling factor {DS_FACTOR} and patch size {PATCH_SIZE}")
    # print(f"Dice LR-HR: {dice_scores[:,0].mean():.4f} ± {dice_scores[:,0].std():.4f}")
    # print(f"Dice SR-HR: {dice_scores[:,1].mean():.4f} ± {dice_scores[:,1].std():.4f}")
    # print(f"Jaccard LR-HR: {jaccard_scores[:,0].mean():.4f} ± {jaccard_scores[:,0].std():.4f}")
    # print(f"Jaccard SR-HR: {jaccard_scores[:,1].mean():.4f} ± {jaccard_scores[:,1].std():.4f}")

    if dim == "2d":
        visualize_masks(saved_masks_hr, saved_masks_lr, saved_masks_sr,
                    save_path=os.path.join(output_dir, f"mask_grid_2d_ds{DS_FACTOR}.png"))
    elif dim == "3d":
        visualize_3d_masks(saved_masks_hr, saved_masks_lr, saved_masks_sr,
                           save_path=os.path.join(output_dir, f"mask_grid_3d_ds{DS_FACTOR}_otsu.png"))

    

if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        dim = "3d",
        patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        output_dir="stat_outputs",
        groups_to_use=["2019_L"],
        nr_samples = 10
    )