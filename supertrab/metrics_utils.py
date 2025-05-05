import torch
import torch.nn.functional as F
from skimage.filters import gaussian, threshold_otsu
import numpy as np
from ormir_xct.util.hildebrand_thickness import calc_structure_thickness_statistics


def compute_image_metrics(sr_images: torch.Tensor, hr_images: torch.Tensor):
    """
    Computes evaluation metrics between super-resolved (SR) and high-resolution (HR) images.

    Args:
        sr_images (Tensor): Super-resolved images of shape (B, 1, H, W).
        hr_images (Tensor): Ground truth high-resolution images of shape (B, 1, H, W).

    Returns:
        List[dict]: A list of dictionaries, one per image, each containing:
            - 'mse': Mean Squared Error
            - 'psnr': Peak Signal-to-Noise Ratio
    """
    # Compute per-pixel squared error without reducing, then flatten each image and average 
    # over all pixels to get one MSE value per image in the batch
    mse_batch = F.mse_loss(sr_images, hr_images, reduction='none')
    mse_per_image = mse_batch.view(mse_batch.size(0), -1).mean(dim=1)

    psnr_per_image = 10 * torch.log10(1.0 / (mse_per_image + 1e-8)) 

    return [{"mse": mse.item(), "psnr": psnr.item()} for mse, psnr in zip(mse_per_image, psnr_per_image)]


def compute_trab_metrics(volume: torch.Tensor, voxel_size_mm: float = 0.0303) -> dict:
    """
    Computes bone structural metrics from a 3D volume:
    - Bone Volume Fraction (BV/TV)
    - Mean and standard deviation of trabecular thickness

    Args:
        volume (torch.Tensor): A 3D volume tensor of shape (D, H, W) or (1, D, H, W), values in [0, 1].
        voxel_size_mm (float): Voxel size in mm (default is 30,3 μm = 0.0303 mm).

    Returns:
        dict: {
            "bone_volume_fraction": float,
            "trabecular_thickness_mean": float,
            "trabecular_thickness_std": float
        }
    """

     # Remove channel dimension if present
    if volume.dim() == 4:
        volume = volume.squeeze(0)

    # Generate binary mask using some thresholding/masking logic
    bone_mask = get_mask(volume)  # shape: (D, H, W), dtype float32, values 0.0 or 1.0

    mask_np = bone_mask.cpu().numpy().astype(bool)
    spacing_mm = [voxel_size_mm] * 3

    # Compute bone volume fraction (BV/TV)
    bone_volume_fraction = np.sum(mask_np) / mask_np.size

    # Compute trabecular thickness statistics 
    # - only valid for 3D
    #th_mean, th_std = calc_structure_thickness_statistics(mask_np, spacing_mm, 0)

    return {
        "bone_volume_fraction": bone_volume_fraction,
        #"trabecular_thickness_mean": th_mean,
        #"trabecular_thickness_std": th_std,
    }
    


def get_mask(image: torch.Tensor, sigma: float = 1.3) -> torch.Tensor:
    """
    Applies Gaussian blur and returns a binary mask using Otsu thresholding.
    
    Args:
        image (torch.Tensor): A single-channel image tensor (C=1 or shape HxW).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        torch.Tensor: A binary mask (same height and width as input).
    """
    # Ensure image is CPU and numpy format (HxW)
    if image.dim() == 3:
        image_np = image.squeeze().cpu().numpy()
    elif image.dim() == 2:
        image_np = image.cpu().numpy()
    else:
        raise ValueError("Input must be 2D or 3D (C=1)")

    # Apply Gaussian blur
    blurred = gaussian(image_np, sigma=sigma)

    # Compute Otsu threshold and create binary mask
    thresh = threshold_otsu(blurred)
    mask = (blurred >= thresh).astype(np.float32)

    #TODO remove small islands

    return torch.from_numpy(mask)
