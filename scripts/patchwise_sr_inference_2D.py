import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from diffusers import DDPMScheduler
from supertrab.inferance_utils import load_model, generate_sr_images, reassemble_patches, split_into_patches, load_zarr_slice

PATCH_SIZE = 128
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    image_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    weights_path = "samples/supertrab-diffusion-sr-2d-v3-checkpoints/models/final_model_weights.pth"
    output_dir = "samples/inference"

    os.makedirs(output_dir, exist_ok=True)

    # Load model and scheduler
    model = load_model(weights_path, image_size=PATCH_SIZE, device=DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Load image
    slice_index = 2960
    image_tensor = load_zarr_slice(
        zarr_path=image_path,
        group_name="2019_L",        
        dataset_name="image",        
        slice_index=slice_index,             
        normalize=True
    ).to(DEVICE)

    # image_tensor shape: [1, H, W] -> add batch dim
    hr_image = image_tensor.unsqueeze(0)  # shape [1, 1, H, W]

    # Apply Gaussian smoothing
    sigma = 1.3  # or any float value you want
    gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
    smoothed = gaussian_blur(hr_image)

    # Downsample by factor 4
    lr_downsampled = F.interpolate(
        smoothed,
        scale_factor=1/4,
        mode="bilinear",
        align_corners=False
    )

    # Upsample back to original HR resolution (same shape as HR)
    lr_resized = F.interpolate(
        lr_downsampled,
        size=hr_image.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # Remove batch/channel dims for patching
    lr_image = lr_resized.squeeze(0)  # shape: [1, H, W]


    # Extract patches
    print("extract patches")
    patches, padded_shape, original_shape = split_into_patches(lr_image, PATCH_SIZE)

    # SR in batches
    sr_patches = []
    print(len(patches))
    for i in tqdm(range(0, len(patches), BATCH_SIZE), desc="Super-resolving patches"):
        batch = patches[i:i+BATCH_SIZE]
        batch = batch.to(DEVICE)
        sr_batch = generate_sr_images(model, scheduler, batch, target_size=PATCH_SIZE, device=DEVICE)
        sr_batch = sr_batch.clamp(0, 1).cpu()
        sr_patches.append(sr_batch)

    sr_patches = torch.cat(sr_patches, dim=0)  

    # Reassemble full image
    full_sr_image = reassemble_patches(sr_patches, padded_shape, original_shape, PATCH_SIZE)

    # Save result
    to_pil_image(image_tensor).save(os.path.join(output_dir, "hr_image.png"))
    to_pil_image(lr_image).save(os.path.join(output_dir, "lr_image.png"))
    to_pil_image(full_sr_image).save(os.path.join(output_dir, "sr_image.png"))
    print(f"Saved images")


if __name__ == "__main__":
    main()