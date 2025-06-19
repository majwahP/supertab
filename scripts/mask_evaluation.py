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

from supertrab.sr_dataset_utils import create_dataloader
from sklearn.metrics import jaccard_score
from supertrab.metrics_utils import get_mask_ormir
from supertrab.inferance_utils import generate_sr_images, load_model, generate_dps_sr_images

PATCH_SIZE = 256
DS_FACTOR = 8

def compute_dice(mask1, mask2):
    intersection = torch.sum((mask1 & mask2).float())
    return 2. * intersection / (mask1.sum() + mask2.sum())

def compute_jaccard(mask1, mask2):
    mask1_np = mask1.cpu().numpy().astype(bool).ravel()
    mask2_np = mask2.cpu().numpy().astype(bool).ravel()
    return jaccard_score(mask1_np, mask2_np)

def ensure_3d_volume(t: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input tensor is in (D, H, W) format for metric computation.
    Handles 2D patches with channel dim or real 3D volumes.
    """
    if t.dim() == 3 and t.shape[0] == 1:
        # (1, H, W) → (H, W) → (1, H, W)
        return t.squeeze(0).unsqueeze(0)
    elif t.dim() == 3:
        return t  # already (D, H, W)
    elif t.dim() == 4 and t.shape[0] == 1:
        return t.squeeze(0)  # (1, D, H, W) → (D, H, W)
    else:
        raise ValueError(f"Unsupported tensor shape: {t.shape}")


def main(
    zarr_path,
    weights_path,
    patch_size=(1, PATCH_SIZE, PATCH_SIZE),
    downsample_factor=DS_FACTOR,
    batch_size=8,
    voxel_size_mm=0.0303,
    output_dir="metric_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    nr_samples = 100,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"ds factor = {downsample_factor}, patch sixe = {patch_size[-1]}")
    #print("Method: DDPS with DPS")
    print("Method: DDPS")

    # Load SR model
    image_size = patch_size[-1]
    model = load_model(weights_path, image_size=image_size, device=device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True
    )

    total_patches = 0
    dice_scores = []
    jaccard_scores = []

    for batch in tqdm(dataloader, desc="Processing patches"):
        lr_images = batch["lr_image"].to(device)  
        hr_images = batch["hr_image"].to(device)

        # Generate SR images
        sr_images = generate_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)
        #sr_images = generate_dps_sr_images(model, scheduler, lr_images, target_size=image_size, downsample_factor=downsample_factor, device=device)

        for hr_patch, lr_patch, sr_patch in zip(hr_images, lr_images, sr_images):
            
            mask_hr = get_mask_ormir(hr_patch.cpu())
            mask_lr = get_mask_ormir(lr_patch.cpu())
            mask_sr = get_mask_ormir(sr_patch.cpu())

            dice_lr_hr = compute_dice(mask_lr, mask_hr).item()
            dice_sr_hr = compute_dice(mask_sr, mask_hr).item()
            jaccard_lr_hr = compute_jaccard(mask_lr, mask_hr)
            jaccard_sr_hr = compute_jaccard(mask_sr, mask_hr)

            print(f"Dice LR-HR: {dice_lr_hr:.4f}, SR-HR: {dice_sr_hr:.4f}")
            print(f"Jaccard LR-HR: {jaccard_lr_hr:.4f}, SR-HR: {jaccard_sr_hr:.4f}")

            dice_scores.append((dice_lr_hr, dice_sr_hr))
            jaccard_scores.append((jaccard_lr_hr, jaccard_sr_hr))

            total_patches += 1
        if total_patches > nr_samples: 
            break
    
    print(f"Total patches included: {total_patches}")

    dice_scores = np.array(dice_scores)
    jaccard_scores = np.array(jaccard_scores)

    print("\n--- Summary ---")
    print(f"Dice LR-HR: {dice_scores[:,0].mean():.4f} ± {dice_scores[:,0].std():.4f}")
    print(f"Dice SR-HR: {dice_scores[:,1].mean():.4f} ± {dice_scores[:,1].std():.4f}")
    print(f"Jaccard LR-HR: {jaccard_scores[:,0].mean():.4f} ± {jaccard_scores[:,0].std():.4f}")
    print(f"Jaccard SR-HR: {jaccard_scores[:,1].mean():.4f} ± {jaccard_scores[:,1].std():.4f}")

    

if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        weights_path=f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth",
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        output_dir="stat_outputs",
        groups_to_use=["2019_L"],
        nr_samples = 20
    )