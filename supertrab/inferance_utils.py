import torch
from diffusers import UNet2DModel
import torch.nn.functional as F
import zarr
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from supertrab.training_utils import normalize_tensor
import torchvision.transforms as T


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


def generate_dps_sr_images(model, scheduler, lr_images, target_size, downsample_factor, device="cpu", lambda_reg=1.0):
    """
    Generates super-resolved images using DPS (Diffusion Posterior Sampling).
    
    Args:
        model: Trained UNet diffusion model (e.g. from HuggingFace diffusers).
        scheduler: DDPM scheduler providing noise schedule.
        lr_images: Tensor of shape (B, 1, H, W), low-resolution input patches.
        target_size: Size of the HR image to generate.
        device: Torch device.
        lambda_reg: Strength of the likelihood gradient correction.
    
    Returns:
        sr_images: Tensor of shape (B, 1, target_size, target_size)
    """
    model.eval()
    B, _, H_lr, W_lr = lr_images.shape
    x_t = torch.randn((B, 1, target_size, target_size), device=device)

    sigma = 1.3            
    blur = T.GaussianBlur(kernel_size=5, sigma=sigma)

    for t in reversed(range(scheduler.config.num_train_timesteps)):
        timesteps = torch.full((B,), t, device=device, dtype=torch.long)

        # Concatenate noisy HR + LR conditioning
        model_input = torch.cat([x_t, lr_images], dim=1)
        eps_theta = model(model_input, timesteps).sample

        # Compute Tweedie estimate of x_0
        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar = alpha_bar_t.view(1, 1, 1, 1).to(device=device, dtype=x_t.dtype)
        x0_hat = (x_t - (1 - alpha_bar).sqrt() * eps_theta) / alpha_bar.sqrt()

        x0_hat.requires_grad_(True)
        blurred = blur(x0_hat)

        lr_down = F.interpolate(blurred, scale_factor=1/downsample_factor, mode="bilinear", align_corners=False)
        simulated_lr = F.interpolate(lr_down, size=(H_lr, W_lr), mode="bilinear", align_corners=False)

        # Likelihood gradient
        loss = F.mse_loss(simulated_lr, lr_images, reduction="sum")
        grad = torch.autograd.grad(loss, x0_hat)[0].detach()
        grad_log_likelihood = -lambda_reg * grad

        # DPS update
        beta_t = scheduler.betas[t]
        x_t = x_t - 0.5 * beta_t * (x_t + eps_theta + grad_log_likelihood)

        if t > 0:
            x_t += beta_t.sqrt() * torch.randn_like(x_t)

    return x_t.clamp(0.0, 1.0).cpu()


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
    print("Shape of volume:", volume.shape)

    # Load only one slice (no full volume loading!)
    slice_2d = volume[slice_index, :, :]  
    
    # Convert to float tensor
    slice_tensor = torch.from_numpy(slice_2d).float()
    slice_tensor= slice_tensor / 65535.0

    if normalize:
        slice_tensor = normalize_tensor(slice_tensor)

    return slice_tensor

def manual_crop(tensor, top, left, crop_h, crop_w):
    """
    Crop a tensor [C, H, W] or [1, H, W] starting at (top, left)
    """
    return tensor[:, top:top + crop_h, left:left + crop_w]

def center_crop(tensor, crop_size):
    _, h, w = tensor.shape
    crop_h, crop_w = crop_size
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return tensor[:, top:top + crop_h, left:left + crop_w]


def split_into_patches(image_tensor, patch_size):
    """
    Splits the image into non-overlapping patches of `patch_size`.
    Pads the image if needed.

    Args:
        image_tensor (Tensor): Shape [C, H, W]
        patch_size (int): Size of the patches (assumed square)

    Returns:
        patches: Tensor of shape [N_patches, C, patch_size, patch_size]
        padded_shape: shape after padding
        image_shape: original shape before padding
    """
    c, h, w = image_tensor.shape

    # Pad if needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    new_h, new_w = image_tensor.shape[1:]

    # Extract patches
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(c, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # [N_patches, C, patch_size, patch_size]

    return patches, (new_h, new_w), (h, w)  # where h, w are after crop

def reassemble_patches(patches, padded_shape, image_shape, patch_size):
    """Reassembles [N, H, W] patches into a full image assuming regular grid order."""
    new_h, new_w = padded_shape
    num_patches_w = new_w // patch_size
    num_patches_h = new_h // patch_size

    # Reshape patches into grid
    patches = patches.view(num_patches_h, num_patches_w, patch_size, patch_size)  # [H_tiles, W_tiles, ph, pw]
    full_image = patches.permute(0, 2, 1, 3).contiguous()  # [H_tiles, ph, W_tiles, pw]
    full_image = full_image.view(new_h, new_w)

    # Crop to original image size
    return full_image[:image_shape[0], :image_shape[1]]
