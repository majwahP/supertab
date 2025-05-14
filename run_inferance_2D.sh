#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1   
#SBATCH --time=24:00:00

export PYTHONPATH=$(pwd):$PYTHONPATH

##################################################################################
accelerate launch /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/patchwise_sr_inference_2D.py
