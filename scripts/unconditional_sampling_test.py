import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import torchvision
import torchvision.transforms as T
from PIL import Image
from supertrab.training_utils import normalize_tensor


@torch.no_grad()
def unconditional_sample(model, scheduler, num_samples, image_size, device="cuda"):
    noisy_images = torch.randn((num_samples, 1, image_size, image_size), device=device)
    dummy_condition = torch.zeros_like(noisy_images)  # zeroed conditioning input

    for t in reversed(range(scheduler.config.num_train_timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        model_input = torch.cat([noisy_images, dummy_condition], dim=1)  # shape [N, 2, H, W]
        noise_pred = model(model_input, t_tensor).sample
        noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample

    return noisy_images.clamp(0.0, 1.0)

def save_grid(images, output_path):
    """Normalize each image, scale to [0,255], and save grid as PNG."""
    # Normalize each image individually
    images = torch.stack([normalize_tensor(img) for img in images])

    # Scale and convert to uint8
    images = (images * 255).clamp(0, 255).to(torch.uint8)

    # Make grid and save
    grid = torchvision.utils.make_grid(images, nrow=4)
    Image.fromarray(grid.permute(1, 2, 0).cpu().numpy()).save(output_path)

def load_model(weights_path, image_size, device="cpu"):
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=2,       
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    weights_path = "samples/supertrab-diffusion-sr-2d-v3-checkpoints/models/final_model_weights.pth"
    output_dir = "unconditional_samples"
    image_size = 128
    num_samples = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(weights_path, image_size, device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    samples = unconditional_sample(model, scheduler, num_samples, image_size, device)
    samples_cpu = samples.cpu()

    save_grid(samples_cpu, os.path.join(output_dir, "sample_grid.png"))
    print(f"Saved unconditional samples to: {output_dir}/sample_grid.png")

if __name__ == "__main__":
    main()
