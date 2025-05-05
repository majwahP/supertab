import torch
from diffusers import UNet2DModel
import torch.nn.functional as F
import zarr
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from supertrab.training_utils import normalize_tensor


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

def load_zarr_slice(slice_index, zarr_path, group_name, dataset_name="image", normalize=True):
    """
    Efficiently loads a single 2D slice from a 3D Zarr volume.

    Args:
        zarr_path (str or Path): Path to the Zarr store.
        dataset_name (str): Name of the array within the Zarr store.
        slice_index (int): Index of the slice to load (along first axis).
        normalize (bool): Whether to normalize values to [0, 1].

    Returns:
        torch.Tensor: A tensor of shape [1, H, W], ready for SR inference.
    """
    # Open Zarr file lazily
    z = zarr.open(zarr_path, mode="r")
    volume = z[f"{group_name}/{dataset_name}"]  

    # Load only one slice (no full volume loading!)
    slice_2d = volume[slice_index, :, :]  
    
    # Convert to float tensor
    slice_tensor = to_tensor(slice_2d.astype(np.float32))  

    if normalize:
        normalize_tensor(slice_tensor)

    return slice_tensor

def split_into_patches(image_tensor, patch_size):
    """Splits a 2D image into non-overlapping 128x128 patches."""
    c, h, w = image_tensor.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    _, new_h, new_w = image_tensor.shape

    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(c, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # [N_patches, C, H, W]
    return patches, (new_h, new_w), (h, w)

def reassemble_patches(patches, image_shape, original_shape, patch_size):
    """Reassembles patches into a full image."""
    new_h, new_w = image_shape
    num_patches_w = new_w // patch_size
    num_patches_h = new_h // patch_size

    patches = patches.permute(1, 0, 2, 3)  # [C, N, H, W]
    patches = patches.contiguous().view(1, patches.size(0), num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches = patches.view(1, patches.size(1), new_h, new_w)
    full_image = patches[0]

    return full_image[:, :original_shape[0], :original_shape[1]]