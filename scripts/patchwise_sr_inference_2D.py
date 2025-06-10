import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import to_pil_image, to_tensor

from diffusers import DDPMScheduler
from supertrab.inferance_utils import (
    load_model,
    generate_sr_images,
    load_zarr_slice,
    split_into_patches,
    reassemble_patches,
    normalize_tensor,
    manual_crop
)

# Parameters
PATCH_SIZE = 128
DS_FACTOR = 8
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLICE_INDEX = 4000
GROUP_NAME = "2019_L"

# Paths
image_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
weights_path = f"samples/supertrab-diffusion-sr-2d-v4/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth"
output_dir = f"samples/inference/{PATCH_SIZE}_ds{DS_FACTOR}"
os.makedirs(output_dir, exist_ok=True)

def main():
    # Load model + scheduler
    model = load_model(weights_path, image_size=PATCH_SIZE, device=DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    blur_transform = GaussianBlur(kernel_size=5, sigma=1.3)

    # Load single Z slice
    image_tensor = load_zarr_slice(
        slice_index=SLICE_INDEX,
        zarr_path=image_path,
        group_name=GROUP_NAME,
        dataset_name="image",
        normalize=False
    )  # shape: [1, H, W]

    H, W = image_tensor.shape
    crop_h = 512
    crop_w = 512

    top = int(0.8 * H) - crop_h // 2
    top = max(0, min(top, H - crop_h))
    left = (W - crop_w) // 2
    left = max(0, min(left, W - crop_w))

    # Crop
    image_tensor = manual_crop(image_tensor.unsqueeze(0), top, left, crop_h, crop_w)

    to_pil_image(image_tensor).save(os.path.join(output_dir, "hr_full.png"))

    lr_image = blur_transform(image_tensor.unsqueeze(0))
    lr_image = F.interpolate(lr_image, scale_factor=1 / DS_FACTOR, mode="bilinear", align_corners=False)
    lr_image = F.interpolate(lr_image, size=(PATCH_SIZE*4, PATCH_SIZE*4), mode="bilinear", align_corners=False)

    to_pil_image(normalize_tensor(lr_image.squeeze(0))).save(os.path.join(output_dir, "lr_image.png"))

    # Split into patches
    hr_patches, padded_shape, image_shape = split_into_patches(image_tensor, PATCH_SIZE)

    # Generate LR patches (Gaussian blur + down/upsample)
    lr_patches = []

    for i, patch in enumerate(hr_patches):
        patch = patch.unsqueeze(0)
        blurred = blur_transform(patch)
        down = F.interpolate(blurred, scale_factor=1 / DS_FACTOR, mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
        lr_patches.append(up.squeeze(0))  # [H, W]

    lr_patches = torch.stack(lr_patches) 

    # Wrap in dataset
    dataset = TensorDataset(lr_patches) 

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inference
    sr_patches = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Super-resolving patches"):
            lr_batch = batch[0].to(DEVICE)
            sr_batch = generate_sr_images(model, scheduler, lr_batch, target_size=PATCH_SIZE, device=DEVICE)
            sr_patches.append(sr_batch.cpu())

    sr_patches = torch.cat(sr_patches, dim=0).squeeze(1)  # [N, H, W]

    # Reassemble full images
    full_sr_image = reassemble_patches(sr_patches, padded_shape, image_shape, PATCH_SIZE)
    full_lr_image = reassemble_patches(lr_patches, padded_shape, image_shape, PATCH_SIZE)
    full_hr_image = reassemble_patches(hr_patches, padded_shape, image_shape, PATCH_SIZE)

    # Save images
    to_pil_image(normalize_tensor(full_hr_image)).save(os.path.join(output_dir, "hr_image.png"))
    to_pil_image(normalize_tensor(full_sr_image)).save(os.path.join(output_dir, "sr_image.png"))
    print("Saved SR, LR, and HR images.")

if __name__ == "__main__":
    main()
