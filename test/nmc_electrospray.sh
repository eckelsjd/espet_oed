#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=nmc_electrospray
#SBATCH --account=goroda0
#SBATCH --partition=gpu
#SBATCH --time=01-12:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem-per-gpu=16g
#SBATCH --cpus-per-gpu=4
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.9-anaconda
python 'test_mse.py'

echo "Finishing job script..."
