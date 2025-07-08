#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --array=0
#SBATCH --nodelist=hardin01

# Define an array of specific PART values
#PART_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
PART_LIST=(16)
PART=${PART_LIST[$SLURM_ARRAY_TASK_ID]}

ds_factor=8

##################################################################################
echo "Running PART=$PART with ds_factor=$ds_factor"
python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/2Dto3D.py $PART $ds_factor
#python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/2Dto3D_external_model.py