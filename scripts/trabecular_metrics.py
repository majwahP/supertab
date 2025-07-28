import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import compute_trab_metrics, ensure_3d_volume
from supertrab.analysis_utils import has_empty_slice

PATCH_SIZE = 256
DS_FACTOR = 10


def main(
    zarr_path,
    patch_size,
    batch_size=1,
    dim = "3d",
    voxel_size_mm=0.0303,
    output_dir="metric_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    # nr_samples = 2,
):
    os.makedirs(output_dir, exist_ok=True)

    # print(f"ds factor = {DS_FACTOR}, patch size = {patch_size[-1]}")

    dataloader_HR_LR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=DS_FACTOR,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=dim, 
        num_workers=0, 
        prefetch=None,
        image_group="image_split/reassembled_HR", 
        mask_base_path="image_trabecular_mask_split/reassembled",
        mask_group=""
    )

    dataloader_SR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=DS_FACTOR,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=dim, 
        num_workers=0, 
        prefetch=None,
        image_group=f"sr_volume_256_{DS_FACTOR}_200ep_given_QCT/reassembled", 
        # image_group="sr_volume_256_QCT_ds10_blur_model_with_scaling/reassembled",     
        mask_base_path="image_trabecular_mask_split/reassembled",
        mask_group=""
    )

    # volume = np.ones([256, 256, 256])

    hr_metrics_list, lr_metrics_list, sr_metrics_list = [], [], []
    total_patches = 0
    total_attempted = 0
    excluded_empty_or_zero = 0
    excluded_metric_fail = 0

    # DBG - for batch_HR_LR, batch_SR in tqdm(zip([1], [2]), desc="Processing patches"):
    for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Processing patches"):
        
        lr_images = batch_HR_LR["lr_image"].to(device)  
        # DBG - lr_images = hr_images = sr_images = torch.ones([1, 1, 256, 256, 256])
        hr_images = batch_HR_LR["hr_image"].to(device)
        sr_images = batch_SR["hr_image"].to(device) * 32768.0 

        pos_HR_LR = batch_HR_LR["position"]
        pos_SR = batch_SR["position"]

        if not torch.equal(pos_HR_LR, pos_SR):
            print("WARNING: Mismatched positions detected!")
            print(f"HR/LR positions: {pos_HR_LR}")
            print(f"SR positions: {pos_SR}")

        for hr_patch, lr_patch, sr_patch, pos in zip(hr_images, lr_images, sr_images, pos_HR_LR):
            total_attempted += 1

            sr = sr_patch[0].cpu()
            if sr.sum() == 0 or has_empty_slice(sr):
                excluded_empty_or_zero += 1
                continue
            
            hr_vol = ensure_3d_volume(hr_patch)
            lr_vol = ensure_3d_volume(lr_patch)
            sr_vol = ensure_3d_volume(sr_patch)

            try:
        
                hr_metrics = compute_trab_metrics(hr_vol, voxel_size_mm, masktype="ormir")
                lr_metrics = compute_trab_metrics(lr_vol, voxel_size_mm, masktype="otsu")
                sr_metrics = compute_trab_metrics(sr_vol, voxel_size_mm, masktype="ormir")
            except Exception as e:
                excluded_metric_fail += 1
                print(f"Skipping patch at {tuple(pos.tolist())} due to metric error: {e}")
                continue

            position = tuple(pos.tolist())
            hr_metrics["position"] = position
            lr_metrics["position"] = position
            sr_metrics["position"] = position

            hr_metrics["patch_idx"] = total_patches
            lr_metrics["patch_idx"] = total_patches
            sr_metrics["patch_idx"] = total_patches

            # print("add metrics")
            hr_metrics_list.append(hr_metrics)
            lr_metrics_list.append(lr_metrics)
            sr_metrics_list.append(sr_metrics)

            total_patches += 1   
            # if total_patches >= nr_samples:
            #     stop_flag = True
            #     break
            # print(f"patches evaluated {total_patches}")
        # if stop_flag: 
        #     break
    
    print(f"Total patches attempted: {total_attempted}")
    print(f"Total patches included: {total_patches}")
    print(f"Excluded (empty or mostly air, empty slices): {excluded_empty_or_zero}")
    print(f"Excluded (metric computation failed): {excluded_metric_fail}")

    # Add source label to each metric set
    hr_df = pd.DataFrame(hr_metrics_list)
    hr_df["source"] = "HR"

    lr_df = pd.DataFrame(lr_metrics_list)
    lr_df["source"] = "LR"

    sr_df = pd.DataFrame(sr_metrics_list)
    sr_df["source"] = "SR"

    # Add patch index
    for i, df in enumerate([hr_df, lr_df, sr_df]):
        df["patch_idx"] = list(range(len(df)))

    # Combine into one dataframe
    all_metrics_df = pd.concat([hr_df, lr_df, sr_df], ignore_index=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"trabecular_metrics_ds{DS_FACTOR}_200ep.csv")
    all_metrics_df.to_csv(csv_path, index=False, na_rep="NaN")
    print(f"Saved per-patch metrics to: {csv_path}")


    
    keys = hr_metrics_list[0].keys()
    with open(os.path.join(output_dir, f"trabecular_metrics_ds{DS_FACTOR}_200ep.txt"), "w") as f:
        f.write(f"Total patches processed: {total_patches}\n\n")

        for key in keys:
            hr_vals = np.array([m[key] for m in hr_metrics_list])
            lr_vals = np.array([m[key] for m in lr_metrics_list])
            sr_vals = np.array([m[key] for m in sr_metrics_list])

            print(f"\nAll values for metric: {key}")
            print(f"HR: {hr_vals}")
            print(f"LR: {lr_vals}")
            print(f"SR: {sr_vals}")

            
if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"),
        patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        batch_size=1,
        dim = "3d",
        output_dir="stat_outputs",
        groups_to_use=["2019_L"],
        # nr_samples = 2
    )