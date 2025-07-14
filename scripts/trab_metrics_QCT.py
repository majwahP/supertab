import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
import torch
import os
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import compute_trab_metrics
from supertrab.inferance_utils import scale

# --- Settings ---
PATCH_SIZE = (256, 256, 256)
DOWNSAMPLE_FACTOR = 10
BATCH_SIZE = 1
SAMPLING_STEP = 2
NUM_TO_COLLECT = 20
VOXEL_SIZE_MM = 0.0303
SIGMA_BLUR = 9
AIR_THRESHOLD = 1000
MAX_AIR_PERCENTAGE = 5.0

# Output
OUTPUT_CSV = "trabecular_metrics_LR_HR_HRblurred.csv"
os.makedirs("metric_outputs", exist_ok=True)
OUTPUT_PATH = os.path.join("metric_outputs", OUTPUT_CSV)


# --- Dataloaders ---
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")

dataloader_HR_pQCT = create_dataloader(
    zarr_path=zarr_path,
    patch_size=PATCH_SIZE,
    downsample_factor=DOWNSAMPLE_FACTOR,
    batch_size=BATCH_SIZE,
    draw_same_chunk=True,
    shuffle=False,
    enable_sr_dataset=True,
    num_workers=0,
    prefetch=None,
    data_dim="3d",
    groups_to_use=["2019_L"],
    image_group="image",
    mask_base_path="image_trabecular_mask",
    mask_group=""
)

dataloader_QCT = create_dataloader(
    zarr_path=zarr_path,
    patch_size=PATCH_SIZE,
    downsample_factor=1,
    batch_size=BATCH_SIZE,
    draw_same_chunk=True,
    shuffle=False,
    enable_sr_dataset=True,
    num_workers=0,
    prefetch=None,
    data_dim="3d",
    groups_to_use=["2019_L"],
    image_group="registered_LR_upscaled_trimmed",
    mask_base_path="image_trabecular_mask",
    mask_group=""
)

# --- Loop and collect metrics ---
lr_metrics_list = []
hr_metrics_list = []
hr_blur_metrics_list = []

index = 0
collected = 0

for batch_hr, batch_lr in tqdm(zip(dataloader_HR_pQCT, dataloader_QCT), desc="Processing patches"):
    if index % SAMPLING_STEP == 0:
        hr_patch = batch_hr["lr_image"].squeeze().cpu().numpy() * 32768.0
        lr_patch = batch_lr["hr_image"].squeeze().cpu().numpy() * 32768.0

        lr_patch = scale(lr_patch)

        air_mask = lr_patch < AIR_THRESHOLD
        air_percentage = np.sum(air_mask) / lr_patch.size * 100

        if air_percentage > MAX_AIR_PERCENTAGE:
            print(f"Skipped patch {index} due to air ({air_percentage:.2f}%)")
            index += 1
            continue

        # Convert to tensors
        lr = torch.tensor(lr_patch)
        hr = torch.tensor(hr_patch)
        hr_blur = torch.tensor(gaussian_filter(hr_patch, sigma=SIGMA_BLUR))

        # Compute metrics
        lr_metrics = compute_trab_metrics(lr, VOXEL_SIZE_MM, masktype="otsu")
        hr_metrics = compute_trab_metrics(hr, VOXEL_SIZE_MM, masktype="otsu")
        hr_blur_metrics = compute_trab_metrics(hr_blur, VOXEL_SIZE_MM, masktype="otsu")

        for m in [lr_metrics, hr_metrics, hr_blur_metrics]:
            m["patch_idx"] = collected

        lr_metrics["source"] = "QCT"
        hr_metrics["source"] = "HR-pQCT_LR"
        hr_blur_metrics["source"] = "HR_blurred"

        lr_metrics_list.append(lr_metrics)
        hr_metrics_list.append(hr_metrics)
        hr_blur_metrics_list.append(hr_blur_metrics)

        collected += 1
        if collected >= NUM_TO_COLLECT:
            break

    index += 1

# --- Save to CSV ---
df_all = pd.DataFrame(lr_metrics_list + hr_metrics_list + hr_blur_metrics_list)
df_all.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved metrics to: {OUTPUT_PATH}")
