from dataclasses import dataclass

@dataclass
class TrainingConfig:

    """
    Defines hyperparameters and settings for training a DDPM super-resolution model.
    
    Attributes:
        image_size: Height and width of square input patches.
        train_batch_size: Batch size for training.
        eval_batch_size: Batch size used during evaluation.
        num_epochs: Number of training epochs.
        gradient_accumulation_steps: Steps to accumulate gradients before updating weights.
        learning_rate: Initial learning rate.
        lr_warmup_steps: Number of warmup steps before full learning rate.
        save_model_epochs: Frequency (in epochs) to generate and save evaluation images.
        ds_factor: Downsampling factor used to simulate LR from HR.
        mixed_precision: Type of mixed precision used during training (e.g., "fp16").
        output_dir: Path to save results and checkpoints.
        seed: Random seed for reproducibility.
        cfg_dropout_prob: Probability of dropping conditioning for classifier-free guidance.
    """

    image_size: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8 
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    ds_factor: int = 10
    mixed_precision: str = "fp16"
    output_dir: str = f"samples/supertrab-diffusion-sr-2d-v4" # Name change
    seed: int = 0
    cfg_dropout_prob: float = 0.1 # 10% of the time, drop the LR image during training