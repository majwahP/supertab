import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from pytorch_msssim import ssim
import torchvision
import lpips
from skimage.filters import gaussian, threshold_otsu
import numpy as np
import SimpleITK as sitk
from ormir_xct.segmentation.ipl_seg import ipl_seg, threshold_dict
from ormir_xct.util.hildebrand_thickness import calc_structure_thickness_statistics
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects, remove_small_holes, disk


import torch
import torch.nn.functional as F
import torchvision
import lpips

def compute_image_metrics(sr_images: torch.Tensor, hr_images: torch.Tensor):
    """
    Computes evaluation metrics between super-resolved (SR) and high-resolution (HR) images.

    Args:
        sr_images (Tensor): Super-resolved images of shape (B, 1, H, W), values in [-1, 1].
        hr_images (Tensor): Ground truth high-resolution images of shape (B, 1, H, W), values in [-1, 1].

    Returns:
        List[dict]: A list of dictionaries per image with:
            - 'mse': Mean Squared Error
            - 'psnr': Peak Signal-to-Noise Ratio
            - 'ssim': Structural Similarity Index
            - 'lpips': Perceptual similarity distance
    """

    device = sr_images.device
    #create and prepare the LPIPS metric model
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    results = []

    #MSE and PSNR
    mse_batch = F.mse_loss(sr_images, hr_images, reduction='none')
    mse_per_image = mse_batch.view(mse_batch.size(0), -1).mean(dim=1)
    psnr_per_image = 10 * torch.log10(1.0 / (mse_per_image + 1e-8))

    for i in range(sr_images.size(0)):
        sr_img = sr_images[i].unsqueeze(0) 
        hr_img = hr_images[i].unsqueeze(0)

        # print(sr_img.shape)
        # print(hr_img.shape)

        # SSIM: rescale to [0, 1] (requirement)
        sr_ssim = (sr_img + 1) / 2
        hr_ssim = (hr_img + 1) / 2
        ssim_val = ssim(sr_ssim, hr_ssim, data_range=1.0)

        # LPIPS: replicate to 3 channels
        sr_lpips = sr_img.expand(-1, 3, -1, -1)
        hr_lpips = hr_img.expand(-1, 3, -1, -1)
        lpips_val = lpips_fn(sr_lpips, hr_lpips).item()

        results.append({
            "mse": mse_per_image[i].item(),
            "psnr": psnr_per_image[i].item(),
            "ssim": ssim_val.item(),
            "lpips": lpips_val
        })

    return results


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
    bone_mask = get_mask_ormir(volume)  # shape: (D, H, W), dtype float32, values 0.0 or 1.0

    mask_np = bone_mask.cpu().numpy().astype(bool)
    spacing_mm = [voxel_size_mm] * 3

    # Compute bone volume fraction (BV/TV)
    bone_volume_fraction = np.sum(mask_np) / mask_np.size

    # Compute trabecular thickness statistics 
    # - only valid for 3D
    #th_mean, th_std = calc_structure_thickness_statistics(mask_np, spacing_mm, structure = 0)

    #trabecular spacing
    # marrow_mask_np = ~mask_np
    # sp_mean, sp_std = calc_structure_thickness_statistics(marrow_mask_np, spacing_mm, structure=0)

    # #trabecular number
    # trabecular_number = bone_volume_fraction / th_mean if th_mean > 0 else 0.0

    return {
        "bone_volume_fraction": bone_volume_fraction,
        #"trabecular_thickness_mean": th_mean,
        #"trabecular_thickness_std": th_std,
        # "trabecular_spacing_mean": sp_mean,
        # "trabecular_spacing_std": sp_std,
        # "trabecular_number": trabecular_number,
    }
    


def get_mask(image: torch.Tensor, sigma: float = 0.8, erosion_radius=1.5, dilation_radius=1, return_steps=False) -> torch.Tensor:
    """
    Applies Gaussian blur and returns a binary mask using Otsu thresholding.
    
    Args:
        image (torch.Tensor): A single-channel image tensor (C=1 or shape HxW).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        torch.Tensor: A binary mask (same height and width as input).
    """
    from supertrab.training_utils import normalize_tensor

    # Ensure image is CPU and numpy format (HxW)
    if image.dim() == 3:
        image_np = image.squeeze().cpu().numpy()
    elif image.dim() == 2:
        image_np = image.cpu().numpy()
    else:
        raise ValueError("Input must be 2D or 3D (C=1)")
    
    steps = {}
    steps["original"] = image_np.copy()

    # Apply Gaussian blur
    blurred = gaussian(image_np, sigma=sigma)
    steps["blurred"] = blurred

    # Compute Otsu threshold and create binary mask
    thresh = threshold_otsu(blurred)
    print(f"Threshold: {thresh}")
    

    if thresh < 0.06 or thresh > 0.19:
        mask = np.zeros_like(blurred, dtype=np.float32)
        if return_steps:
            return {k: torch.from_numpy(v) for k, v in steps.items()}
        else:
            return torch.from_numpy(mask)


    mask = (blurred >= thresh).astype(np.float32)
    steps["thresholded"] = mask

    mask = binary_erosion(mask, disk(erosion_radius))
    steps["eroded"] = mask
    mask = binary_dilation(mask, disk(dilation_radius))
    steps["dilated"] = mask
    mask = remove_small_objects(mask, min_size=40)
    steps["no_small_objects"] = mask
    mask = remove_small_holes(mask, area_threshold=10)
    steps["final_mask"] = mask

    if return_steps:
        return {k: torch.from_numpy(v) for k, v in steps.items()}
    else:
        return torch.from_numpy(mask)



def get_mask_ormir(image: torch.Tensor, sigma: float = 1.3, voxel_size_mm: float = 0.0303, erosion_radius=1, dilation_radius=1) -> torch.Tensor:
    """
    Performs trabecular bone segmentation using the IPL-based method from ORMiR_XCT.
    Kuczynski, M.T., et al. "ORMIR_XCT: A Python package for high resolution peripheral quantitative computed tomography image processing." arXiv preprint arXiv:2309.04602 (2023).
    Args:
        image (torch.Tensor): A single-channel image tensor of shape (H, W) or (1, H, W).
        sigma (float): Gaussian smoothing parameter (used as sigma * voxel size).
        voxel_size_mm (float): Size of one voxel in millimeters.

    Returns:
        torch.Tensor: A binary mask (same height and width as input).
    """

    if image.dim() == 3:
        image = image.squeeze()
    np_img = image.cpu().numpy().astype('float32')
    sitk_img = sitk.GetImageFromArray(np_img)

    lower = threshold_dict["BMD_Lower"]/ 32768.0 # same normalization as done for images
    upper = threshold_dict["BMD_Upper"]/ 32768.0

    # print(f"Min: {image.min().item()}, Max: {image.max().item()}, Mean: {image.mean().item()}")
    # print(f"BMD_Lower: {lower}, BMD_Upper: {upper}")



    seg_sitk = ipl_seg(
    input_image=sitk_img,
    lower_threshold=lower,
    upper_threshold=upper,       
    voxel_size= voxel_size_mm,
    sigma=sigma
    )

    seg_np = sitk.GetArrayFromImage(seg_sitk)
    mask = seg_np.astype(np.uint8)

    mask = binary_erosion(mask, disk(erosion_radius))
    mask = binary_dilation(mask, disk(dilation_radius))
    mask = remove_small_objects(mask, min_size=50)
    mask = remove_small_holes(mask, area_threshold=50)

    mask = 1 - mask
    
    return torch.from_numpy(mask).to(torch.uint8)
