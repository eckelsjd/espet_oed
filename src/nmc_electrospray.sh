#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=nmc_electrospray
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

conda env create -f '../conda-espet_39.yaml'
conda activate espet_39
python 'test_mse.py'

echo "Finishing job script..."
