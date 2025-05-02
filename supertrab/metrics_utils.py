import torch
import torch.nn.functional as F


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
