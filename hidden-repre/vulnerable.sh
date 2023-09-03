#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=vulnerable-rn18-p50000-e04
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/vulnerable-rn18-p50000-e04
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u vulnerable.py --epsilon 0.4 --idsaver vulnerable/cifar10/fgsm-rn18-all50000-e04-syn/ --trainpath synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all50000 --model resnet18 --checkpoint ./pretrained_models/fgsm_rn18_syn_all50000.pth