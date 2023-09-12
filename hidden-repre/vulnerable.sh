#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=vulnerable-cf100-rn18-p50000-e02
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/vulnerable-cf100-rn18-p50000-e02
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u vulnerable.py --epsilon 0.2 --idsaver vulnerable/cifar100/fgsm-rn18-all50000-e02/ --trainpath synthesis/cifar100/adversarial_data/resnet18/fgsm_train_all50000 --model resnet18 --dataset cifar100 --checkpoint ./pretrained_models/rn18-cf100.pth