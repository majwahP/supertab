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
from supertrab.metrics_utils import get_mask, get_mask_ormir
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from supertrab.training_utils import normalize_tensor
from supertrab.inferance_utils import generate_sr_images, load_model, generate_dps_sr_images



PATCH_SIZE = 256
DS_FACTOR = 8




def main():
    # Settings
    weights_path = f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth"
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_dir = "inference_outputs"
    image_size = PATCH_SIZE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Load model + scheduler
    model = load_model(weights_path, image_size=image_size, device=device)
    #model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Load data
    dataloader = create_dataloader(
        zarr_path,
        downsample_factor=DS_FACTOR,
        patch_size=(1, image_size, image_size),
        groups_to_use=["2019_L"],  # test group
        batch_size=8,
    )
    print("Dataloader created")
    batch = next(iter(dataloader))

    lr_images = batch["lr_image"].to(device)
    hr_images = batch["hr_image"].to(device)


    # Run inference
    print("run inferance")
    sr_images = generate_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)
    #sr_images = generate_dps_sr_images(model, scheduler, lr_images, target_size=image_size, device=device)

    # use images for metrics
    print("Trabecular metrics per image:\n")
    for i in range(sr_images.size(0)):
        lr_img = lr_images[i].cpu().detach()    
        sr_img = sr_images[i].cpu().detach()     
        hr_img = hr_images[i].cpu().detach()

        # print(f"Image {i}:")
        # print(f"  LR  - dtype: {lr_img.dtype}, shape: {lr_img.shape}, range: ({lr_img.min():.3f}, {lr_img.max():.3f})")
        # print(f"  SR  - dtype: {sr_img.dtype}, shape: {sr_img.shape}, range: ({sr_img.min():.3f}, {sr_img.max():.3f})")
        # print(f"  HR  - dtype: {hr_img.dtype}, shape: {hr_img.shape}, range: ({hr_img.min():.3f}, {hr_img.max():.3f})")

        # hr_pil = to_pil_image(normalize_tensor(hr_img))
        # hr_pil.save(os.path.join(output_dir, f"hr_{i+1}.png"))

        # hr_pil = to_pil_image(hr_img.squeeze(0), mode="F")
        # hr_pil.save(os.path.join(output_dir, f"hr_{i+1}.tiff"))

        # lr_img = normalize_tensor(lr_img)
        # sr_img = normalize_tensor(sr_img)
        # hr_img = normalize_tensor(hr_img)

        # Compute metrics
        # lr_metrics = compute_trab_metrics(lr_img)
        # sr_metrics = compute_trab_metrics(sr_img)
        # hr_metrics = compute_trab_metrics(hr_img)


        #print(i+1)
        # sr_mask = get_mask(sr_img)
        # hr_mask = get_mask(hr_img)

        lr_mask = get_mask_ormir(lr_img)
        sr_mask = get_mask_ormir(sr_img)
        hr_mask = get_mask_ormir(hr_img)

        # image_stack = torch.stack([
        # normalize_tensor(hr_img),   
        # normalize_tensor(lr_img),   
        # normalize_tensor(sr_img), 
        # hr_mask.unsqueeze(0).cpu(),
        # sr_mask.unsqueeze(0).cpu()   
        # ])

        image_stack = torch.stack([   
        normalize_tensor(lr_img),   
        normalize_tensor(sr_img),
        normalize_tensor(hr_img), 
        lr_mask.unsqueeze(0).cpu(),
        sr_mask.unsqueeze(0).cpu(),
        hr_mask.unsqueeze(0).cpu()   
        ])

        image_grid = make_grid(image_stack, nrow=6)

        triplet_pil = to_pil_image(image_grid)
        triplet_pil.save(os.path.join(output_dir, f"lr_sr_hr_stack_{i+1}.png"))
        
        # hr_metrics = compute_trab_metrics(hr_img)
        # lr_metrics = compute_trab_metrics(lr_img)
        # sr_metrics = compute_trab_metrics(sr_img)

        # print("Trabecular Bone Metrics:")
        # for metric_name in hr_metrics.keys():
        #     hr_val = hr_metrics[metric_name]
        #     lr_val = lr_metrics[metric_name]
        #     sr_val = sr_metrics[metric_name]
            
        #     diff_lr_hr = lr_val - hr_val
        #     diff_sr_hr = sr_val - hr_val

        #     print(f"{metric_name}: HR = {hr_val:.4f}, LR = {lr_val:.4f}, SR = {sr_val:.4f} | LR-HR = {diff_lr_hr:.4f}, SR-HR = {diff_sr_hr:.4f}")

        #show mask evolution
        # steps = get_mask(hr_img, return_steps=True)
        # # stack = [
        # #     normalize_tensor(step).unsqueeze(0) if i < 2 else step.float().unsqueeze(0)
        # #     for i, step in enumerate(steps.values())
        # # ]
        # stack = [
        #     ((step+1)/2).unsqueeze(0) if i < 2 else step.float().unsqueeze(0)
        #     for i, step in enumerate(steps.values())
        # ]
        # grid = make_grid(torch.stack(stack), nrow=len(stack))
        # img = to_pil_image(grid)
        # img.save(os.path.join(output_dir, f"mask_debug_stack_hr_{i+1}.png"))

        # print(f"Image {i+1}:")
        #3D
        # print(f"  LR  - BV/TV: {lr_metrics['bone_volume_fraction']:.4f},  Thickness: {lr_metrics['trabecular_thickness_mean']:.3f} ± {lr_metrics['trabecular_thickness_std']:.3f} mm")
        # print(f"  SR  - BV/TV: {sr_metrics['bone_volume_fraction']:.4f},  Thickness: {sr_metrics['trabecular_thickness_mean']:.3f} ± {sr_metrics['trabecular_thickness_std']:.3f} mm")
        # print(f"  HR  - BV/TV: {hr_metrics['bone_volume_fraction']:.4f},  Thickness: {hr_metrics['trabecular_thickness_mean']:.3f} ± {hr_metrics['trabecular_thickness_std']:.3f} mm\n")

        # bv_lr = lr_metrics["bone_volume_fraction"]
        # bv_sr = sr_metrics["bone_volume_fraction"]
        # bv_hr = hr_metrics["bone_volume_fraction"]
        # bv_diff = bv_sr - bv_hr
        
        # print(f"  LR  - BV/TV: {bv_lr:.4f}")
        # print(f"  SR  - BV/TV: {bv_sr:.4f}")
        # print(f"  HR  - BV/TV: {bv_hr:.4f}")
        # print(f"  Δ(SR−HR): {bv_diff:+.4f}")



if __name__ == "__main__":
    main()
