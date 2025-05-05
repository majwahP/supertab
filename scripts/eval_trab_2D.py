import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import os
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import compute_trab_metrics


def load_model(weights_path, image_size, device="cpu"):
    """Reconstructs the model architecture and loads trained weights."""
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_sr_images(model, scheduler, lr_images, target_size, device="cpu"):
    """Generates super-resolved images from low-resolution inputs."""
    noisy_images = torch.randn((lr_images.size(0), 1, target_size, target_size), device=device)

    for t in reversed(range(scheduler.config.num_train_timesteps)):
        timesteps = torch.full((lr_images.size(0),), t, device=device, dtype=torch.long)
        model_input = torch.cat([noisy_images, lr_images], dim=1)
        noise_pred = model(model_input, timesteps).sample
        noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample

    return noisy_images


def main():
    # Settings
    weights_path = "samples/supertrab-diffusion-sr-2d-v3-checkpoints/models/final_model_weights.pth"
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_dir = "inference_outputs"
    image_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Load model + scheduler
    model = load_model(weights_path, image_size=image_size, device=device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Load data
    dataloader = create_dataloader(
        zarr_path,
        downsample_factor=4,
        patch_size=(1, image_size, image_size),
        groups_to_use=["2019_L"],  # test group
        batch_size=4
    )
    batch = next(iter(dataloader))
    lr_images = batch["lr_image"].to(device)

    # Run inference
    sr_images = generate_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)
    sr_images = sr_images.clamp(0.0, 1.0).cpu()

    hr_images = batch["hr_image"].to(device)

    # use images for metrics
    print("Trabecular metrics per image:\n")
    for i in range(sr_images.size(0)):
        lr_img = lr_images[i].cpu()    
        sr_img = sr_images[i]          # already on CPU
        hr_img = hr_images[i].cpu()  

        # Compute metrics
        lr_metrics = compute_trab_metrics(lr_img)
        sr_metrics = compute_trab_metrics(sr_img)
        hr_metrics = compute_trab_metrics(hr_img)

        print(f"Image {i+1}:")
        #3D
        # print(f"  LR  - BV/TV: {lr_metrics['bone_volume_fraction']:.4f},  Thickness: {lr_metrics['trabecular_thickness_mean']:.3f} ± {lr_metrics['trabecular_thickness_std']:.3f} mm")
        # print(f"  SR  - BV/TV: {sr_metrics['bone_volume_fraction']:.4f},  Thickness: {sr_metrics['trabecular_thickness_mean']:.3f} ± {sr_metrics['trabecular_thickness_std']:.3f} mm")
        # print(f"  HR  - BV/TV: {hr_metrics['bone_volume_fraction']:.4f},  Thickness: {hr_metrics['trabecular_thickness_mean']:.3f} ± {hr_metrics['trabecular_thickness_std']:.3f} mm\n")

        bv_lr = lr_metrics["bone_volume_fraction"]
        bv_sr = sr_metrics["bone_volume_fraction"]
        bv_hr = hr_metrics["bone_volume_fraction"]
        bv_diff = bv_sr - bv_hr
        
        print(f"  LR  - BV/TV: {bv_lr:.4f}")
        print(f"  SR  - BV/TV: {bv_sr:.4f}")
        print(f"  HR  - BV/TV: {bv_hr:.4f}")
        print(f"  Δ(SR−HR): {bv_diff:+.4f}")



if __name__ == "__main__":
    main()
