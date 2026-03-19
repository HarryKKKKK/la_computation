#!/bin/bash
#SBATCH --job-name=highres_mpi
#SBATCH --partition=csc-mphil
#SBATCH --clusters=CSC
#SBATCH --account=hk597
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00 
#SBATCH --output=highres_mpi_%j.out

set -euo pipefail

mkdir -p logs

# module purge || true
# module load gcc || true

cd "${SLURM_SUBMIT_DIR}"

echo "Running on $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

./performance

echo "Done. Output CSV:"
ls -lh bench_laplace.csv || true