from pathlib import Path
from diffusers import DDPMScheduler
from pprint import pprint
from dataclasses import asdict
from supertrab.sr_dataset_utils import create_dataloader
from supertrab.training_config import TrainingConfig
from supertrab.training_utils import train_loop_2D_diffusion, load_model_and_optimizer, evaluate

def main():
    config = TrainingConfig(
        image_size=256,
        train_batch_size=4,
        eval_batch_size=4,
        num_epochs=200,  
        ds_factor=8,
        output_dir="samples/supertrab-diffusion-sr-2d-v5"
    )

    print("Training Configuration:")
    pprint(asdict(config))

    checkpoint_path = f"{config.output_dir}/{config.image_size}_ds{config.ds_factor}/models/final_training_checkpoint_{config.image_size}_ds{config.ds_factor}.pth"
    model, optimizer, start_epoch = load_model_and_optimizer(config, checkpoint_path)

    print(f"Continuing training from checkpoint, epoch {start_epoch}. New total epochs: {config.num_epochs}")

    # Prepare dataloaders
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")
    train_groups = ["1955_L", "1956_L", "1996_R", "2005_L"]
    val_groups = ["2007_L"]
    test_groups = ["2019_L"]
    train_dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=(1, config.image_size, config.image_size),
        batch_size=config.train_batch_size,
        downsample_factor=config.ds_factor,
        groups_to_use=train_groups,
        num_workers=4
    )   
    val_dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=(1, config.image_size, config.image_size),
        batch_size=config.train_batch_size,
        downsample_factor=config.ds_factor,
        groups_to_use=val_groups,
        num_workers=4
    )   
    test_dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=(1, config.image_size, config.image_size),
        batch_size=config.train_batch_size,
        downsample_factor=config.ds_factor,
        groups_to_use=test_groups,
        num_workers=4
    )   

    steps_per_epoch = 1000
    noise_scheduler = DDPMScheduler(num_train_timesteps=steps_per_epoch)
    
    from diffusers.optimization import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=steps_per_epoch * (config.num_epochs - start_epoch),
    )

    # Resume training from the last epoch
    train_loop_2D_diffusion(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        steps_per_epoch,
        starting_epoch=start_epoch 
    )
    evaluate(config, "final_test", model, noise_scheduler, test_dataloader, device="cuda")

if __name__ == "__main__":
    main()



