"""
Train a Denoising Diffusion Probabilistic Model (DDPM)-based **3D super-resolution** model on
HR-pQCT trabecular bone volumes using Huggingface Diffusers + PyTorch.

This script defines the training configuration, prepares data loaders for training, validation,
and testing, initializes a 3D U-Net model with low-resolution volume conditioning, and trains
the model using the DDPM framework. After training, the script runs evaluation on test data
and saves qualitative results.

Main components:
- Loads patch-wise 3D image volumes from Zarr-based HR-pQCT dataset.
- Uses classifier-free guidance by randomly dropping LR conditioning during training.
- Applies a UNet3DModel with [noisy HR | LR] concatenated along channels.
- Trains using Accelerate and logs to Weights & Biases.
- Includes shape checks for sanity before training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from diffusers import UNet3DConditionModel, DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
import os
from pathlib import Path
from supertrab.sr_dataset_utils import create_dataloader
from dataclasses import asdict
from pprint import pprint
from supertrab.training_utils import train_loop_3D_diffusion, evaluate_3D
from supertrab.training_config import TrainingConfig



def main():
    print("Starting...")
    config = TrainingConfig()
    print("Training Configuration:")
    pprint(asdict(config))

    # Create dataloaders ------------------------------------------------------------
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")

    train_groups = ["1955_L", "1956_L", "1996_R", "2005_L"]
    val_groups = ["2007_L"]
    test_groups = ["2019_L"]

    patch=(config.image_size,config.image_size,config.image_size)
    print(patch)
    train_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=patch, groups_to_use=train_groups, data_dim="3d", num_workers=0, prefetch=None,image_group="image", mask_base_path="image_trabecular_mask", mask_group="")
    val_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=patch, groups_to_use=val_groups, data_dim="3d", num_workers=0, prefetch=None,image_group="image", mask_base_path="image_trabecular_mask", mask_group="")
    test_dataloader = create_dataloader(zarr_path, downsample_factor=config.ds_factor, patch_size=patch, groups_to_use=test_groups, data_dim="3d", num_workers=0, prefetch=None,image_group="image", mask_base_path="image_trabecular_mask", mask_group="")

    print("Dataloaders created")

    #Define the model --------------------------------------------------------------------
    model = UNet3DConditionModel(
        sample_size=(config.image_size, config.image_size, config.image_size),  
        in_channels=2,      
        out_channels=1,
        layers_per_block=2,
        block_out_channels = (128, 128, 256, 256, 512, 512),
        down_block_types = (
            "DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"
        ),
        up_block_types = (
            "UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"
        ),
        cross_attention_dim=None,
    )


    #Testing ---------------------------------------------------------------------------------
    # Check that images are same shape as output
    # batch = next(iter(train_dataloader))
    # sample_hr = batch["hr_image"][:1]
    # sample_lr = batch["lr_image"][:1]  
    # print("HR Input shape:", sample_hr.shape)
    # sample_lr_resized = F.interpolate(sample_lr, size=sample_hr.shape[-3:], mode='trilinear', align_corners=False)
    # noise = torch.randn_like(sample_hr)
    # noisy_hr = sample_hr + noise  

    # model_input = torch.cat([noisy_hr, sample_lr_resized], dim=1)
    # B, C, D, H, W = model_input.shape
    # dummy_condition = torch.zeros((B, 1), dtype=model_input.dtype, device=model_input.device)

    # output = model(sample=model_input, timestep=0,encoder_hidden_states=dummy_condition)
    # #output = model(sample=model_input, timestep=0)
    # print("Output shape:", output.sample.shape)

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
    print("Starting training")
    train_loop_3D_diffusion(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler, steps_per_epoch)
    evaluate_3D(config, "final_test", model, noise_scheduler, test_dataloader, device="cuda")




if __name__ == "__main__":
    main()