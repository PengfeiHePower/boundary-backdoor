#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=vulnerable-cf10-rn18-p50000-e035
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/vulnerable-cf10-rn18-p50000-e035
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u vulnerable.py --epsilon 0.35 --idsaver vulnerable/cifar10/fgsm-rn18-all50000-e035/ --trainpath synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all50000 --model resnet18 --dataset cifar10 --checkpoint ./pretrained_models/resnet18.pth