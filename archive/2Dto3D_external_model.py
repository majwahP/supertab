from dataclasses import dataclass
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet2DModel.from_pretrained("archive/ddpm-butterflies-128")
scheduler = DDPMScheduler.from_pretrained("archive/ddpm-butterflies-128")

pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
pipeline.to(device)

start = time.time()
# Generate samples
images = pipeline(batch_size=4).images

end = time.time()
total_time = end - start
time_per_image = total_time / 4

# Save images
for i, img in enumerate(images):
    img.save(f"sample_later_{i}.png")



print(f"Total time to generate 4 images: {total_time:.2f} seconds")
print(f"Average time per image: {time_per_image:.2f} seconds")