#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=vulnerable
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/vulnerable
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u vulnerable.py --epsilon 0.4 --idsaver vulnerable/cifar10/fgsm-rn18-all5000-e04/ --trainpath synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all5000 --model resnet18