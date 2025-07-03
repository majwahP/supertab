#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1   
#SBATCH --time=36:00:00

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1


##################################################################################
#accelerate launch /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/train_diffusion_model_2D.py

accelerate launch /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/train_diffusion_model_3D.py

#accelerate launch /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/resume_training.py

