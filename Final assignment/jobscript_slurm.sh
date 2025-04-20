#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=08:00:00
#SBATCH --reservation=terv92681

srun apptainer exec --nv --env-file .env container_v2.sif /bin/bash main.sh
