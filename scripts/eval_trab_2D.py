import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))



import os
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from supertrab.sr_dataset_utils import create_dataloader
from supertrab.metrics_utils import compute_trab_metrics
from supertrab.metrics_utils import get_mask
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from supertrab.training_utils import normalize_tensor
from supertrab.inferance_utils import generate_sr_images, load_model



PATCH_SIZE = 128
DS_FACTOR = 8




def main():
    # Settings
    weights_path = f"samples/supertrab-diffusion-sr-2d-v4/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth"
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_dir = "inference_outputs"
    image_size = PATCH_SIZE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Load model + scheduler
    model = load_model(weights_path, image_size=image_size, device=device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Load data
    dataloader = create_dataloader(
        zarr_path,
        downsample_factor=DS_FACTOR,
        patch_size=(1, image_size, image_size),
        groups_to_use=["2019_L"],  # test group
        batch_size=8
    )
    print("Dataloader created")
    batch = next(iter(dataloader))
    lr_images = batch["lr_image"].to(device)


    # Run inference
    print("run inferance")
    sr_images = generate_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)
    sr_images = sr_images.clamp(0.0, 1.0).cpu()

    hr_images = batch["hr_image"].to(device)

    # use images for metrics
    print("Trabecular metrics per image:\n")
    for i in range(sr_images.size(0)):
        lr_img = lr_images[i].cpu()    
        sr_img = sr_images[i]          # already on CPU
        hr_img = hr_images[i].cpu()  

        lr_img = normalize_tensor(lr_img)
        sr_img = normalize_tensor(sr_img)
        hr_img = normalize_tensor(hr_img)

        # Compute metrics
        lr_metrics = compute_trab_metrics(lr_img)
        sr_metrics = compute_trab_metrics(sr_img)
        hr_metrics = compute_trab_metrics(hr_img)

        # Save SR image and mask side-by-side
        sr_mask = get_mask(sr_img)
        #sr_img_norm = normalize_tensor(sr_img)

        # sr_stack = torch.stack([sr_img_norm, sr_mask.unsqueeze(0)])
        sr_stack = torch.stack([sr_img, sr_mask.unsqueeze(0)])
        sr_grid = make_grid(sr_stack, nrow=2)  # add channel dim: [2,1,H,W]
        sr_pil = to_pil_image(sr_grid)
        sr_pil.save(os.path.join(output_dir, f"sr_image_and_mask_{i+1}.png"))

        # Save HR image and mask side-by-side
        hr_mask = get_mask(hr_img)
        # hr_img_norm = normalize_tensor(hr_img)

        #hr_stack = torch.stack([hr_img_norm, hr_mask.unsqueeze(0)])
        hr_stack = torch.stack([hr_img, hr_mask.unsqueeze(0)])
        hr_grid = make_grid(hr_stack, nrow=2)
        hr_pil = to_pil_image(hr_grid)
        hr_pil.save(os.path.join(output_dir, f"hr_image_and_mask_{i+1}.png"))

        image_triplet = torch.stack([
        hr_img,   
        lr_img,   
        sr_img    
        ])

        image_grid = make_grid(image_triplet, nrow=3)

        triplet_pil = to_pil_image(image_grid)
        triplet_pil.save(os.path.join(output_dir, f"lr_sr_hr_triplet_{i+1}.png"))

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
