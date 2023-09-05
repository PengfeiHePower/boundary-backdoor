#!/bin/bash
#SBATCH --exclude=lac-143
#SBATCH --job-name=attack_random_rn18
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SMATCH --mail-type=ALL
#SBATCH --mail-user=hepengf1@msu.edu
#SBATCH --output=log/attack_random_rn18
#SBATCH --gres=gpu:v100:1

module load Conda/3
module load Conda/3
source activate bb
cd ..
cd ..

python ./attack/wanet.py --yaml_path ../config/attack/wanet/default.yaml --cross_ratio 2 --random_rotation 10 --random_crop 5 --s 0.5 --k 4 --grid_rescale 1