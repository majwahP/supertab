#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=5:00:00

# Variables section: 
#export NUMEXPR_MAX_THREADS=10
export PYTHONUNBUFFERED=1


##################################################################################
python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/save_to_zarr.py