from pathlib import Path
from diffusers import DDPMScheduler
from supertrab.sr_dataset_utils import create_dataloader
from supertrab.training_config import TrainingConfig
from supertrab.training_utils import train_loop_2D_diffusion, load_model_and_optimizer, evaluate

def main():
    config = TrainingConfig(
        image_size=128,
        train_batch_size=4,
        eval_batch_size=4,
        num_epochs=100,  
        ds_factor=8,
        output_dir="samples/supertrab-diffusion-sr-2d-v4"
    )

    print(f"Continuing training from checkpoint. New total epochs: {config.num_epochs}")

    checkpoint_path = f"{config.output_dir}/{config.image_size}_ds{config.ds_factor}/models/final_training_checkpoint_{config.image_size}_ds{config.ds_factor}.pth"
    model, optimizer, start_epoch = load_model_and_optimizer(config, checkpoint_path)

    # Prepare dataloaders
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    train_groups = ["1955_L", "1956_L", "1996_R", "2005_L"]
    val_groups = ["2007_L"]
    test_groups = ["2019_L"]
    train_dataloader = create_dataloader(zarr_path, config.ds_factor, (1, config.image_size, config.image_size), train_groups)
    val_dataloader = create_dataloader(zarr_path, config.ds_factor, (1, config.image_size, config.image_size), val_groups)
    test_dataloader = create_dataloader(zarr_path, config.ds_factor, (1, config.image_size, config.image_size), test_groups)

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



