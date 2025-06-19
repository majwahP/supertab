from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import PIL.Image
from tqdm import tqdm
import os

@dataclass
class TrainingConfig:
    image_size: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"
    output_dir: str = "archive/ddpm-butterflies-128"
    push_to_hub: bool = False
    hub_model_id: str = None
    overwrite_output_dir: bool = True
    seed: int = 0

config = TrainingConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")


preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def transform(examples):
    images = [preprocess(img.convert("RGB")) for img in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)



model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")

)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)



optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=len(train_dataloader) * config.num_epochs
)

model.to(device)
for epoch in range(config.num_epochs):
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
        clean_images = batch["images"].to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (clean_images.size(0),)
        ).to(device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()



pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline.to(device)


os.makedirs(config.output_dir, exist_ok=True)

images = pipeline(batch_size=config.eval_batch_size).images
for i, img in enumerate(images):
    img.save(f"{config.output_dir}/sample_{i}.png")

model.save_pretrained(config.output_dir)
noise_scheduler.save_pretrained(config.output_dir)
