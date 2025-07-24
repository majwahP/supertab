#!/bin/bash
#SBATCH --job-name=supertrab_%j
#SBATCH --output=logs/supertrab_%j.out
#SBATCH --error=logs/supertrab_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00


##################################################################################
python /usr/terminus/data-xrm-01/stamplab/users/mwahlin/2025/trab_master/supertrab/archive/move_file.py