#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH --ntasks-per-node=4		    # Specify number of tasks per node
#SBATCH --gpus=1		            # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200007                     # Specify project name
#SBATCH -J jupyter                      # Specify job name


module purge
module load Miniconda3/22.11.1-1
conda deactivate
conda activate pyroom

jupyter notebook --no-browser --ip 0.0.0.0 --NotebookApp.allow_origin='*'