#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=adversarial_data-rn18-e1-p50000
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/adversarial_data-rn18-p50000
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u adversarial_data.py --model resnet18 --epsilon 0.002 --checkpoint ./pretrained_models/resnet18.pth --rate 1.0