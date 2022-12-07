#!/bin/bash
sbatch <<EOT
#!/bin/bash
# JOB HEADERS HERE
# Arg 1 is the iteration
# Arg 2 is the N2M ratio
# Run as bash script.sh [arg1] [arg2]

#SBATCH --job-name=nmc"$1"_N2M_"$2"
#SBATCH --account=goroda0
#SBATCH --partition=standard
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --output=/scratch/goroda_root/goroda0/eckelsjd/logs/%x-%j.log
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eckelsjd@umich.edu

echo "Starting job script..."

module load python3.9-anaconda
python test_mse.py "$1" "$2"

echo "Finishing job script..."

exit 0
EOT