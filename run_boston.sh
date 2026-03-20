#!/bin/bash
#SBATCH --job-name=grfg_boston       
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=8           
#SBATCH --mem=16G
#SBATCH --time=12:00:00              

# 1. Block Python from "bleeding" into your local user folders
export PYTHONNOUSERSITE=1

# 2. Force it to use your exact Conda environment Python!
/home/pjanga/.conda/envs/grfg_env/bin/python -u main.py --name housing_boston --hidden_size 64