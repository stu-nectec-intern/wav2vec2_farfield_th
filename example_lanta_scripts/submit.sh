#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 2 -c 16   			    # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		    # Specify number of tasks per node
#SBATCH --gpus=4		            # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200007                     # Specify project name
#SBATCH -J dist                      # Specify job name

module purge
module load Miniconda3/22.11.1-1
conda deactivate
conda activate pyroom

python /project/lt200007-tspai2/thepeach/notebook/distribution.py