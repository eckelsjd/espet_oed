#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=gpu_electrospray
#SBATCH --account=goroda0
#SBATCH --partition=spgpu
#SBATCH --time=01-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem-per-gpu=48g
#SBATCH --cpus-per-gpu=1
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.9-anaconda
module load cuda/11.3.0
source $ANACONDA_ROOT/etc/profile.d/conda.sh
conda activate espet
pip install cupy-cuda113
python 'test_mse.py'

echo "Finishing job script..."
