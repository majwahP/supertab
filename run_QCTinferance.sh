#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-16




# Define an array of specific PART values
PART_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
#PART_LIST=(0 7 8 9 10 11 12 13)
# PART_LIST=(3 10 11)
PART=${PART_LIST[$SLURM_ARRAY_TASK_ID]}

ds_factor=10

GROUP_NAME="2005_L"

##################################################################################
echo "Running PART=$PART with ds_factor=$ds_factor"
# python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/QCT_inferance.py $PART $ds_factor
#python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/2Dto3D_external_model.py
python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/scripts/create_syntetic_LR_dataset.py $PART $ds_factor $GROUP_NAME
