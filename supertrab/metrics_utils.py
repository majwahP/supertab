import torch
import torch.nn.functional as F
import math

def compute_image_metrics(sr_images: torch.Tensor, hr_images: torch.Tensor):
    """
    Computes evaluation metrics between super-resolved (SR) and high-resolution (HR) images.

    Args:
        sr_images (Tensor): Super-resolved images of shape (B, 1, H, W).
        hr_images (Tensor): Ground truth high-resolution images of shape (B, 1, H, W).

    Returns:
        dict: Dictionary containing:
            - 'mse': Mean Squared Error (averaged over batch)
            - 'psnr': Peak Signal-to-Noise Ratio (averaged over batch)
    """
    # Compute per-pixel squared error without reducing, then flatten each image and average 
    # over all pixels to get one MSE value per image in the batch
    mse_batch = F.mse_loss(sr_images, hr_images, reduction='none')
    mse_per_image = mse_batch.view(mse_batch.size(0), -1).mean(dim=1)

    psnr_per_image = 10 * torch.log10(1.0 / (mse_per_image + 1e-8)) 

    return {
        "mse": mse_per_image.mean().item(),
        "psnr": psnr_per_image.mean().item()
    }
