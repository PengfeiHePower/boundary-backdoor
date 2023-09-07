#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=attack_rn18_random
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/attack_rn18_random
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb
cd ..
cd ..

python ./resource/label-consistent/craft_adv_dataset.py --dataset cifar10