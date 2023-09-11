#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=train_vinilla_cifar100_resnet18
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=logs/train_vinilla_cifar100_resnet18
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python train_vanilla.py -dataset cifar100 -epoch 200 -model resnet18