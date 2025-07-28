import torch
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import sys

# Append project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import ensure_3d_volume, get_mask_ormir

# --- CONFIG ---
PATCH_SIZE = (256, 256, 256)
DS_FACTOR = 4
BATCH_SIZE = 1
GROUPS = ["2019_L"]
ZARR_PATH = "/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"
RESULT_CSV = "patch_similarity_results.csv"

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
    image_group=f"sr_volume_256_{DS_FACTOR}_200ep/reassembled",
    mask_base_path="image_trabecular_mask_split/reassembled",
    mask_group=""
)

# --- METRIC FUNCTIONS ---

def pearson_corr(x, y):
    x_flat, y_flat = x.flatten(), y.flatten()
    x_mean, y_mean = torch.mean(x_flat), torch.mean(y_flat)
    x_diff, y_diff = x_flat - x_mean, y_flat - y_mean
    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff**2)) * torch.sqrt(torch.sum(y_diff**2))
    return (numerator / denominator).item() if denominator != 0 else float("nan")

def jaccard_index(mask1, mask2):
    mask1 = mask1.bool()
    mask2 = mask2.bool()
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    return intersection / union if union != 0 else float("nan")

# --- MAIN LOOP ---
results = []
for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Processing patches"):
    try:
        hr = ensure_3d_volume(batch_HR_LR["hr_image"][0])
        sr = ensure_3d_volume(batch_SR["hr_image"][0]) * 32768.0
        pos = tuple(batch_SR["position"][0].tolist())

        # Compute metrics
        corr = pearson_corr(hr, sr)
        mask_hr = get_mask_ormir(hr)
        mask_sr = get_mask_ormir(sr)
        jaccard = jaccard_index(mask_hr, mask_sr)

        results.append({
            "position": pos,
            "pearson_corr": corr,
            "jaccard_index": jaccard
        })

    except Exception as e:
        print(f"[ERROR] at {pos}: {e}")
        continue

# --- SAVE RESULTS ---
df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\nSaved results to {RESULT_CSV}")

# --- PRINT STATS ---
mean_corr = df["pearson_corr"].mean()
std_corr = df["pearson_corr"].std()
mean_jaccard = df["jaccard_index"].mean()
std_jaccard = df["jaccard_index"].std()

print(f"\n--- METRIC SUMMARY ---")
print(f"Pearson Correlation: mean = {mean_corr:.4f}, std = {std_corr:.4f}")
print(f"Jaccard Index:       mean = {mean_jaccard:.4f}, std = {std_jaccard:.4f}")

# Save summary to file
summary_txt = RESULT_CSV.replace(".csv", "_summary.txt")
with open(summary_txt, "w") as f:
    f.write(f"Pearson Correlation: mean = {mean_corr:.4f}, std = {std_corr:.4f}\n")
    f.write(f"Jaccard Index:       mean = {mean_jaccard:.4f}, std = {std_jaccard:.4f}\n")
print(f"Saved summary to {summary_txt}")
