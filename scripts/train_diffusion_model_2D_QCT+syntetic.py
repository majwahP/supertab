"""
Train a  Denoising Diffusion Probabilistic Model, DDPM-based super-resolution model on 
HR-pQCT trabecular bone images using Huggingface Diffusers.

This script defines the training configuration, prepares data loaders for training, validation,
and testing, initializes a U-Net model with low-resolution image conditioning, and trains 
the model using the (DDPM) framework. After training, the script runs an evaluation pass on 
the test dataset and saves qualitative results.

Main components:
- Loads patch-wise 2D image data from Zarr-based HR-pQCT volumes.
- Uses classifier-free guidance by randomly dropping conditioning during training.
- Applies a UNet2DModel with LR/HR concatenated inputs.
- Trains using Accelerate and logs to Weights & Biases.
- Includes shape checks for sanity before training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
import os
from pathlib import Path
from supertrab.sr_dataset_utils import create_dataloader
from dataclasses import asdict
from pprint import pprint
from supertrab.training_utils import train_loop_2D_QCT_diffusion, evaluate_QCT
from supertrab.training_config import TrainingConfig

def main():
    config = TrainingConfig(
        image_size=256,
        train_batch_size=4,
        eval_batch_size=4,
        num_epochs=100,  
        ds_factor=10,
        output_dir="samples/supertrab-diffusion-sr-2d-QCT+syntetic"
    )
    print("Training Configuration:")
    pprint(asdict(config)) 

    # Create dataloaders ------------------------------------------------------------
    # zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")


    train_groups = ["1955_L", "1956_L", "1996_R", "2005_L"',"1955_L_LR", "1956_L_LR", "1996_R_LR", "2005_L_LR"']
    val_groups = ["2007_L"]
    test_groups = ["2019_L"]

    gt_train_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1, config.image_size, config.image_size), groups_to_use=train_groups, data_dim="2d", with_blur=False)
    train_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1, config.image_size, config.image_size), groups_to_use=train_groups, data_dim="2d", with_blur=False, image_group="registered_LR_upscaled_trimmed")
    gt_val_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1, config.image_size, config.image_size), groups_to_use=val_groups, data_dim="2d", with_blur=False)
    val_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1, config.image_size, config.image_size), groups_to_use=val_groups, data_dim="2d", with_blur=False, image_group="registered_LR_upscaled_trimmed")
    gt_test_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1, config.image_size, config.image_size), groups_to_use=test_groups, data_dim="2d", with_blur=False)
    test_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=(1,config.image_size,config.image_size), groups_to_use=test_groups, data_dim="2d", with_blur=False, image_group="registered_LR_upscaled_trimmed")

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
    # print("Get one batch")
    # print(f"Dataset class: {type(train_dataloader.dataset)}")

    batch = next(iter(train_dataloader))
    print("Has one batch")
    sample_image = batch["hr_image"][0].unsqueeze(0)
    print("Input shape:", sample_image.shape)
    sample_lr = batch["hr_image"][0].unsqueeze(0)
    sample_lr_resized = F.interpolate(sample_lr, size=sample_image.shape[-2:], mode='bilinear', align_corners=False)
    model_input = torch.cat([sample_image, sample_lr_resized], dim=1)
    print("Output shape:", model(model_input, timestep=0).sample.shape)

    # Define noise scheduler ----------------------------------------------------------

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    
    # setup optimizer and scheduler ----------------------------------------------------
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    #Scheduler to adjust learning rate during training
    steps_per_epoch = 1000
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps = steps_per_epoch * config.num_epochs,
    )

    # training and evaluation ----------------------------------------------------------- 

    print("Starting training loop")

    train_loop_2D_QCT_diffusion(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, gt_train_dataloader,gt_val_dataloader, lr_scheduler, steps_per_epoch)
    evaluate_QCT(config, "final_test", model, noise_scheduler, test_dataloader, gt_test_dataloader, device="cuda")




if __name__ == "__main__":
    main()