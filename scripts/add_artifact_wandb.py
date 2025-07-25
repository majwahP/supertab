import wandb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

PATCH_SIZE = 256
DS_FACTOR = 8

run = wandb.init(project="supertrab", job_type="upload-model", name=f"upload_weights_{PATCH_SIZE}_{DS_FACTOR}_200_ep")

artifact = wandb.Artifact(name=f"2D-ds{DS_FACTOR}-{PATCH_SIZE}-200ep-weights", type="model")
artifact.add_file(f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth")  

run.log_artifact(artifact)

run.finish()
