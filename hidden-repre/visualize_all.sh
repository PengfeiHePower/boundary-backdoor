#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=visualize_all
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/visualize_all
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb

python -u visualize_all.py --method tsne --dataset cifar10 --poisonsaver synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all50000 --modelpath pretrained_models/resnet18.pth --figuresaver figures/cifar10/fgsm_rn18_all50000/ --no_aug