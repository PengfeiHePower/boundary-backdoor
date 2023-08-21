#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=create_clean_set
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=logs/create_clean_set
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate gp

python create_clean_set.py