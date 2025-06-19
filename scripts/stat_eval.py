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
from supertrab.metrics_utils import compute_trab_metrics
from supertrab.inferance_utils import generate_sr_images, load_model, generate_dps_sr_images

PATCH_SIZE = 256
DS_FACTOR = 4

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
    batch_size=16,
    voxel_size_mm=0.0303,
    output_dir="metric_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    nr_samples = 100,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"ds factor = {downsample_factor}, patch sixe = {patch_size[-1]}")
    print("Method: DDPS with DPS")
    #print("Method: DDPS")

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

    hr_metrics_list, lr_metrics_list, sr_metrics_list = [], [], []
    total_patches = 0

    for batch in tqdm(dataloader, desc="Processing patches"):
        lr_images = batch["lr_image"].to(device)  
        hr_images = batch["hr_image"].to(device)

        # Generate SR images
        #sr_images = generate_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)
        sr_images = generate_dps_sr_images(model, scheduler, lr_images, target_size=image_size, downsample_factor=downsample_factor, device=device)

        for hr_patch, lr_patch, sr_patch in zip(hr_images, lr_images, sr_images):
            hr_vol = ensure_3d_volume(hr_patch)
            lr_vol = ensure_3d_volume(lr_patch)
            sr_vol = ensure_3d_volume(sr_patch)

            hr_metrics = compute_trab_metrics(hr_vol, voxel_size_mm)
            lr_metrics = compute_trab_metrics(lr_vol, voxel_size_mm * downsample_factor)
            sr_metrics = compute_trab_metrics(sr_vol, voxel_size_mm)

            hr_metrics_list.append(hr_metrics)
            lr_metrics_list.append(lr_metrics)
            sr_metrics_list.append(sr_metrics)

            total_patches += 1
        if total_patches > nr_samples: 
            break
    
    print(f"Total patches included: {total_patches}")

    # Save raw metrics
    with open(os.path.join(output_dir, "hr_metrics.json"), "w") as f:
        json.dump(hr_metrics_list, f, indent=2)
    with open(os.path.join(output_dir, "lr_metrics.json"), "w") as f:
        json.dump(lr_metrics_list, f, indent=2)
    with open(os.path.join(output_dir, "sr_metrics.json"), "w") as f:
        json.dump(sr_metrics_list, f, indent=2)

    
    keys = hr_metrics_list[0].keys()
    with open(os.path.join(output_dir, "ttest_summary.txt"), "w") as f:
        f.write(f"Total patches processed: {total_patches}\n\n")

        for key in keys:
            hr_vals = np.array([m[key] for m in hr_metrics_list])
            lr_vals = np.array([m[key] for m in lr_metrics_list])
            sr_vals = np.array([m[key] for m in sr_metrics_list])

            diff_lr = lr_vals - hr_vals
            diff_sr = sr_vals - hr_vals

            stat_lr, pval_lr = ttest_rel(hr_vals, lr_vals)
            stat_sr, pval_sr = ttest_rel(hr_vals, sr_vals)

            mean_diff_lr = np.mean(diff_lr)
            std_diff_lr = np.std(diff_lr, ddof=1)
            mean_diff_sr = np.mean(diff_sr)
            std_diff_sr = np.std(diff_sr, ddof=1)

            mae_lr = np.mean(np.abs(lr_vals - hr_vals))
            mae_sr = np.mean(np.abs(sr_vals - hr_vals))
            
            # Save to file
            f.write(f"Metric: {key}\n")
            f.write(f"LR vs HR: t={stat_lr:.4f}, p={pval_lr:.4e}, mean_diff={mean_diff_lr:.4f}, std_diff={std_diff_lr:.4f}\n")
            f.write(f"SR vs HR: t={stat_sr:.4f}, p={pval_sr:.4e}, mean_diff={mean_diff_sr:.4f}, std_diff={std_diff_sr:.4f}\n\n")
            f.write(f"MAE - LR: {mae_lr:.4f}, SR: {mae_sr:.4f}\n\n")

            # Print to console
            print(f"[{key}]")
            print(f"  LR vs HR: t={stat_lr:.4f}, p={pval_lr:.4e}, mean_diff={mean_diff_lr:.4f}, std_diff={std_diff_lr:.4f}")
            print(f"  SR vs HR: t={stat_sr:.4f}, p={pval_sr:.4e}, mean_diff={mean_diff_sr:.4f}, std_diff={std_diff_sr:.4f}")
            print(f"MAE - LR: {mae_lr:.4f}, SR: {mae_sr:.4f}")

if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        weights_path=f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth",
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        output_dir="stat_outputs",
        groups_to_use=["2019_L"],
        nr_samples = 100
    )