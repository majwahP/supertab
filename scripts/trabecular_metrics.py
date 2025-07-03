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
from supertrab.metrics_utils import compute_trab_metrics, ensure_3d_volume
from supertrab.inferance_utils import generate_sr_images, load_model, generate_dps_sr_images

PATCH_SIZE = 256
DS_FACTOR = 4


def main(
    zarr_path,
    patch_size,
    downsample_factor=DS_FACTOR,
    batch_size=8,
    voxel_size_mm=0.0303,
    output_dir="metric_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    nr_samples = 7,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"ds factor = {downsample_factor}, patch size = {patch_size[-1]}")

    dataloader_HR_LR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim="3d", 
        num_workers=0, 
        prefetch=None,
        image_group="image_split/reassembled_HR", 
        mask_base_path="image_trabecular_mask", 
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
        data_dim="3d", 
        num_workers=0, 
        prefetch=None,
        image_group=f"sr_volume_256_{DS_FACTOR}/reassembled",        
        mask_base_path="image_trabecular_mask",
        mask_group=""
    )

    hr_metrics_list, lr_metrics_list, sr_metrics_list = [], [], []
    total_patches = 0
    print("Starting loop")

    for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Processing patches"):
        print("Get batch")
        lr_images = batch_HR_LR["lr_image"].to(device)  
        hr_images = batch_HR_LR["hr_image"].to(device)
        sr_images = batch_SR["hr_image"].to(device) * 32768.0 

        pos_HR_LR = batch_HR_LR["position"]
        pos_SR = batch_SR["position"]

        if not torch.equal(pos_HR_LR, pos_SR):
            print("WARNING: Mismatched positions detected!")
            print(f"HR/LR positions: {pos_HR_LR}")
            print(f"SR positions: {pos_SR}")

        print("Have patches")
        for hr_patch, lr_patch, sr_patch in zip(hr_images, lr_images, sr_images):
            if sr_patch.sum() == 0:
                continue
            print("3D volume")
            hr_vol = ensure_3d_volume(hr_patch)
            lr_vol = ensure_3d_volume(lr_patch)
            sr_vol = ensure_3d_volume(sr_patch)

            print("get metrics")
            hr_metrics = compute_trab_metrics(hr_vol, voxel_size_mm)
            lr_metrics = compute_trab_metrics(lr_vol, voxel_size_mm * downsample_factor)
            sr_metrics = compute_trab_metrics(sr_vol, voxel_size_mm)

            print("add metrics")
            hr_metrics_list.append(hr_metrics)
            lr_metrics_list.append(lr_metrics)
            sr_metrics_list.append(sr_metrics)

            total_patches += 1
        print(f"patches evaluated {total_patches}")
        if total_patches > nr_samples-1: 
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
    with open(os.path.join(output_dir, "metric_summary.txt"), "w") as f:
        f.write(f"Total patches processed: {total_patches}\n\n")

        for key in keys:
            hr_vals = np.array([m[key] for m in hr_metrics_list])
            lr_vals = np.array([m[key] for m in lr_metrics_list])
            sr_vals = np.array([m[key] for m in sr_metrics_list])

            # Differences
            diff_lr = lr_vals - hr_vals
            diff_sr = sr_vals - hr_vals

            # Stats
            mean_diff_lr = np.mean(diff_lr)
            std_diff_lr = np.std(diff_lr, ddof=1)
            mean_diff_sr = np.mean(diff_sr)
            std_diff_sr = np.std(diff_sr, ddof=1)

            # MAE
            mae_lr = np.mean(np.abs(diff_lr))
            mae_sr = np.mean(np.abs(diff_sr))

            # Save to file
            f.write(f"Metric: {key}\n")
            f.write(f"LR - HR: mean_diff = {mean_diff_lr:.4f}, std_diff = {std_diff_lr:.4f}, MAE = {mae_lr:.4f}\n")
            f.write(f"SR - HR: mean_diff = {mean_diff_sr:.4f}, std_diff = {std_diff_sr:.4f}, MAE = {mae_sr:.4f}\n\n")

            # Print to console
            print(f"[{key}]")
            print(f"  LR - HR: mean_diff = {mean_diff_lr:.4f}, std_diff = {std_diff_lr:.4f}, MAE = {mae_lr:.4f}")
            print(f"  SR - HR: mean_diff = {mean_diff_sr:.4f}, std_diff = {std_diff_sr:.4f}, MAE = {mae_sr:.4f}")

if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        batch_size=1,
        downsample_factor=DS_FACTOR,
        output_dir="stat_outputs",
        groups_to_use=["2019_L"],
        nr_samples = 4
    )