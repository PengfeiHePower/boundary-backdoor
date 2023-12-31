#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=check_vulnerable-cf100-rn18-p50000
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/check_vulnerable-cf100-rn18-p50000
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u check_vulnerable.py --trainpath synthesis/cifar100/adversarial_data/resnet18/fgsm_train_all50000 --dataset cifar100 --modelpath pretrained_models/rn18-cf100.pth