import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import get_mask_ormir
from supertrab.inferance_utils import load_model, generate_sr_images
from diffusers import DDPMScheduler

# --- Settings ---
PATCH_SIZE = 256
DS_FACTOR = 4
NUM_SAMPLES_TO_KEEP = 5
AREA_DIFF_THRESHOLD = 1000
NUM_SR_REPEATS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_hr_lr_srs(hr_list, lr_list, sr_repeat_list, save_path):
    """
    For each sample, show HR (row 0), LR (row 1), and 10 SRs (rows 2-11).
    Each column = one sample.
    """
    num_samples = len(hr_list)
    num_rows = 2 + NUM_SR_REPEATS

    fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 2.2, num_rows * 2.2))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(num_samples):
        axes[0][col].imshow(hr_list[col], cmap="gray")
        axes[0][col].set_title("HR")
        axes[0][col].axis("off")

        axes[1][col].imshow(lr_list[col], cmap="gray")
        axes[1][col].set_title("LR")
        axes[1][col].axis("off")

        for i in range(NUM_SR_REPEATS):
            axes[2 + i][col].imshow(sr_repeat_list[col][i], cmap="gray")
            axes[2 + i][col].set_title(f"SR {i+1}")
            axes[2 + i][col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved HR–LR–10xSR grid to: {save_path}")


def visualize_hr_lr_sr_rows(hr_list, lr_list, sr_list, save_path):
    """
    Visualize HR, LR, SR in one row per sample.
    Each row = 1 sample. Each column = HR, LR, SR
    """
    num_samples = len(hr_list)
    fig, axes = plt.subplots(num_samples, 3, figsize=(3 * 3, num_samples * 3))  # 3 columns: HR, LR, SR

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure 2D indexing

    for i in range(num_samples):
        axes[i][0].imshow(hr_list[i], cmap="gray")
        axes[i][0].set_title("HR")
        axes[i][0].axis("off")

        axes[i][1].imshow(lr_list[i], cmap="gray")
        axes[i][1].set_title("LR")
        axes[i][1].axis("off")

        axes[i][2].imshow(sr_list[i], cmap="gray")
        axes[i][2].set_title("SR")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved HR–LR–SR row-wise grid to: {save_path}")


def main():
    output_dir = "outlier_sr_samples_repeat"
    os.makedirs(output_dir, exist_ok=True)

    dataloader_hr = create_dataloader(
        zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"),
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        groups_to_use=["2019_L"],
        batch_size=16,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True,
        data_dim="2d",
        num_workers=0,
        prefetch=None,
        image_group="image_split/part_2_split/part_3",
        mask_base_path="image_trabecular_mask_split/part_2_split/part_3",
        mask_group=""
    )

    dataloader_sr = create_dataloader(
        zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"),
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        groups_to_use=["2019_L"],
        batch_size=16,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True,
        data_dim="2d",
        num_workers=0,
        prefetch=None,
        image_group=f"sr_volume_{PATCH_SIZE}_{DS_FACTOR}_200ep/part_2_split/part_3",
        # image_group=f"sr_volume_{PATCH_SIZE}_{DS_FACTOR}/part_2_split/part_7",
        mask_base_path="image_trabecular_mask_split/part_2_split/part_3",
        mask_group=""
    )

    hr_images, lr_images, sr_images = [], [], []
    sample_idx = 0

    for batch_hr, batch_sr in tqdm(zip(dataloader_hr, dataloader_sr), desc="Scanning patches"):
        hr_imgs = batch_hr["hr_image"].to(device)
        lr_imgs = batch_hr["lr_image"].to(device)
        sr_imgs = batch_sr["hr_image"].to(device) * 32768.0
        position = batch_sr["position"]

        for hr, lr, sr in zip(hr_imgs, lr_imgs, sr_imgs):
            hr_img = hr[0].cpu()
            sr_img = sr[0].cpu()
            lr_img = lr[0].cpu()

            hr_mask = get_mask_ormir(hr_img)
            sr_mask = get_mask_ormir(sr_img)

            hr_area = hr_mask.sum().item()
            sr_area = sr_mask.sum().item()
            area_diff = sr_area - hr_area

            if area_diff > AREA_DIFF_THRESHOLD:
                hr_images.append(hr_img)
                lr_images.append(lr_img)
                sr_images.append(sr_img)
                sample_idx += 1
                print(position)
                print(f"Collected: {sample_idx}")
                if sample_idx >= NUM_SAMPLES_TO_KEEP:
                    break
        if sample_idx >= NUM_SAMPLES_TO_KEEP:
            break

    # # Load model and scheduler
    # weights_path = f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth"
    # model = load_model(weights_path, image_size=PATCH_SIZE, device=device)
    # model.eval()
    # scheduler = DDPMScheduler(num_train_timesteps=1000)

    # # Run inference multiple times for each LR
    # print(f"Generating {NUM_SR_REPEATS} SR samples per LR...")
    # all_sr_repeats = []
    # with torch.no_grad():
    #     for i, lr_img in enumerate(tqdm(lr_images, desc="Inferring SRs")):
    #         lr_input = lr_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    #         repeats = []
    #         for _ in range(NUM_SR_REPEATS):
    #             sr = generate_sr_images(model, scheduler, lr_input, target_size=PATCH_SIZE, device=device)
    #             repeats.append(sr[0][0].cpu())  # remove batch and channel
    #         all_sr_repeats.append(repeats)

    # Plot and save
    save_path = os.path.join(output_dir, f"hr_lr_sr__ds{DS_FACTOR}_200ep.png")
    # visualize_hr_lr_srs(hr_images, lr_images, all_sr_repeats, save_path)
    visualize_hr_lr_sr_rows(hr_images, lr_images, sr_images, save_path)



if __name__ == "__main__":
    main()
