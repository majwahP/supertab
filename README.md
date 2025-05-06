# Supertrab

## Master's Thesis - Revealing Trabecular Bone Architecture in 3D with Deep Learning based Super Resolution
This repository contains the code and scripts developed for my master's thesis conducted at ETH Zürich/Paul Scherrer Institute (PSI) dusing spring semester 2025.
The projects aim was to create a diffusion based Super Resolution model to improve resolution in CT images of trabecular bone.

Thesis can be found at: 

## Repository Structure
archive/ – Stores old or backup versions of code and data. Mostly contains trial script developed on the process of dertermining the final method.

logs/ – Contains training logs and error messages.

samples/ – Includes training outputs such as generated images. Note: Trained model files are excluded from version control due to large file sizes.

scripts/ – Scripts for model training, evaluation, and preprocessing.

supertrab/ – Contains utility functions used across the project (e.g., data loading, preprocessing, evaluation).

wandb/ – Contains experiment logs for Weights & Biases tracking.

.gitignore – Specifies files and folders to exclude from version control (e.g., logs, W&B).

README.md – This file.

.sh files - This repository includes several shell scripts. These shell scripts are used to run different parts of the project on a SLURM-based computing cluster. They make it easier to submit jobs for training, evaluation, and data preparation.
