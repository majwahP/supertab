"""
Implementation of a simple Denoising diffusion probebilistic Model based on: 
https://huggingface.co/docs/diffusers/en/tutorials/basic_training
The code is adjusted to use the train_dataloader created for trabeclular bone in 
this library in supertrab_create_dataloader.py
and to implement super-resolution instead of generating data. 
"""

from dataclasses import dataclass
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from PIL import Image 
from torchvision.transforms.functional import to_pil_image
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from supertrab.sr_dataset_utils import create_dataloader
from PIL import Image, ImageDraw, ImageFont
import PIL
import wandb
from dataclasses import asdict
from pprint import pprint


#Define the configuration and parameters, adjust to the training train_dataloader
@dataclass
class TrainingConfig:
    image_size: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8 
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    ds_factor: int = 4
    mixed_precision: str = "fp16"
    output_dir: str = "samples/ddpm-supertrab-256-2D-v2"
    seed: int = 0
    cfg_dropout_prob: float = 0.1 # 10% of the time, drop the LR image during training

# helper functions------------------------------------------------------

def normalize_tensor(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def create_sample_image(lr_imgs, sr_imgs, hr_imgs, padding=10, header_height=60, footer_height=40):
    assert len(lr_imgs) == len(sr_imgs) == len(hr_imgs), "Image lists must be same length"
    n = len(lr_imgs)

    # Convert to PIL
    lr_pil = [to_pil_image(normalize_tensor(img.squeeze(0))) for img in lr_imgs]
    sr_pil = [to_pil_image(normalize_tensor(img.squeeze(0))) for img in sr_imgs]
    hr_pil = [to_pil_image(normalize_tensor(img.squeeze(0))) for img in hr_imgs]
    diff_pil = [to_pil_image(normalize_tensor(torch.abs(sr - hr).squeeze(0))) for sr, hr in zip(sr_imgs, hr_imgs)]
    mses = [torch.mean((sr - hr) ** 2).item() for sr, hr in zip(sr_imgs, hr_imgs)]

    # Assume all images have the same size
    w, h = lr_pil[0].size

    font_size = max(10, h // 10)
    font_path = PIL.__path__[0] + "/fonts/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # Full grid size
    total_width = 4 * w + 5 * padding
    total_height = n * (h + footer_height + padding) + header_height + padding

    grid_img = Image.new("L", (total_width, total_height), color=255)
    draw = ImageDraw.Draw(grid_img)

    # Draw headers
    headers = ["LR", "SR", "HR", "Diff"]
    for i, text in enumerate(headers):
        x = padding + i * (w + padding) + w // 2
        draw.text((x, header_height // 2), text, fill=0, anchor="mm", font=font)

    # Draw each sample row
    for idx in range(n):
        y_offset = header_height + padding + idx * (h + footer_height + padding)

        x_offsets = [padding + i * (w + padding) for i in range(4)]
        images = [lr_pil[idx], sr_pil[idx], hr_pil[idx], diff_pil[idx]]

        for x, img in zip(x_offsets, images):
            grid_img.paste(img, (x, y_offset))

        # Draw MSE below diff image
        mse_text = f"MSE: {mses[idx]:.4f}"
        mse_x = x_offsets[3] + w // 2
        mse_y = y_offset + h + footer_height // 4
        draw.text((mse_x, mse_y), mse_text, fill=0, anchor="mm", font=font)

    return grid_img


config = TrainingConfig()
print("Training Configuration:")
pprint(asdict(config))

os.environ["WANDB_PROJECT"] = "ddpm-supertrab"

### Create the train_dataloader ###
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")

train_groups = ["1955_L", "1956_L", "1996_R", "2005_L"]
val_groups = ["2007_L"]
test_groups = ["2019_L"]

train_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1,config.image_size,config.image_size), groups_to_use=train_groups)
val_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1,config.image_size,config.image_size), groups_to_use=val_groups)
test_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1,config.image_size,config.image_size), groups_to_use=test_groups)

#Define the model --------------------------------------------------------------------
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=2,  # [noisy HR | LR conditioning]
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


#Testing ---------------------------------------------------------------------------------
# Check that images are same shape as output
batch = next(iter(train_dataloader))
sample_image = batch["hr_image"][0].unsqueeze(0)
print("Input shape:", sample_image.shape)
sample_lr = batch["lr_image"][0].unsqueeze(0)
sample_lr_resized = F.interpolate(sample_lr, size=sample_image.shape[-2:], mode='bilinear', align_corners=False)
model_input = torch.cat([sample_image, sample_lr_resized], dim=1)
print("Output shape:", model(model_input, timestep=0).sample.shape)

# Define noise scheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape) # create noise with the same shape as the sample image
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
#-------------------------------------------------------------------------------------------

# Training -----------------------------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
#Scheduler to adjust learning rate during training
steps_per_epoch = 1000
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps = steps_per_epoch * config.num_epochs,
)


@torch.no_grad()
def evaluate(config, epoch, model, noise_scheduler, dataloader, device="cuda", global_step=None):
    model.eval()
    model.to(device)

    save_dir = os.path.join(config.output_dir, "samples_supertrab_2D_simple")
    os.makedirs(save_dir, exist_ok=True)

    batch = next(iter(dataloader))
    lr_images = batch["lr_image"].to(device)
    hr_images = batch["hr_image"].to(device)

    # Resize LR to HR size
    lr_resized = F.interpolate(
        lr_images, size=(config.image_size, config.image_size), mode="bilinear", align_corners=False
    )

    # Initialize noise
    noisy_images = torch.randn((lr_images.size(0), 1, config.image_size, config.image_size), device=device)

    for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
        timesteps = torch.full((lr_images.size(0),), t, device=device, dtype=torch.long)
        model_input = torch.cat([noisy_images, lr_resized], dim=1)
        noise_pred = model(model_input, timesteps).sample
        noisy_images = noise_scheduler.step(noise_pred, t, noisy_images).prev_sample

    # Clamp and bring to CPU
    sr_images = noisy_images.clamp(0.0, 1.0).cpu()
    lr_images_up = lr_resized.cpu().clamp(0.0, 1.0)
    hr_images = hr_images.cpu().clamp(0.0, 1.0)

    # Build grid row by row
    grid_rows = []
    for lr, sr, hr in zip(lr_images_up, sr_images, hr_images):
        row = torch.cat([lr, sr, hr], dim=-1)  # concatenate side-by-side
        grid_rows.append(row)

    # Stack rows vertically
    final_image = create_sample_image(lr_images_up, sr_images, hr_images)

    if isinstance(epoch, int):
        filename = f"{epoch:04d}_ds{config.ds_factor}_size{config.image_size}.png"
    else:
        filename = f"{epoch}_ds{config.ds_factor}_size{config.image_size}.png"

    final_image.save(os.path.join(save_dir, filename))

    if wandb.run is not None:
        wandb.log({f"sample_epoch_{epoch}": wandb.Image(final_image)}, step=global_step)





def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    # assure only one process creates directories or logs
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        run_name = f"supertrab_ddpm_{config.image_size}px_ds{config.ds_factor}_100ep"
        accelerator.init_trackers(
            project_name="supertrab", 
            config=vars(config),
            init_kwargs={"wandb": {"name": run_name}}
        )


    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        if accelerator.is_local_main_process:
            print(f"Starting Epoch {epoch}...")


        for step, batch in enumerate(train_dataloader):
            if step >= steps_per_epoch:
                break
            clean_images = batch["hr_image"]
            conditioning = batch["lr_image"]
            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.size(0),), device=clean_images.device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


            # Backpropagation
            with accelerator.accumulate(model):
                # Predict the noise residual
                conditioning_resized = F.interpolate(conditioning, size=clean_images.shape[-2:], mode='bilinear', align_corners=False)

                # Classifier-Free Guidance: randomly drop conditioning
                if torch.rand(1).item() < config.cfg_dropout_prob:
                    conditioning_resized = torch.zeros_like(conditioning_resized)


                # Concatenate LR with noisy HR as input to the model
                model_input = torch.cat([noisy_images, conditioning_resized], dim=1)

                # Predict noise
                noise_pred = model(model_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                # Avoid exploding gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            #evaluate every save_image:apochs
            #evaluate(config, epoch, model, noise_scheduler, val_dataloader, device=accelerator.device, global_step=global_step)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, model, noise_scheduler, val_dataloader, device=accelerator.device, global_step=global_step)


if __name__ == "__main__":
    
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    evaluate(config, "final_test", model, noise_scheduler, test_dataloader, device="cuda")
