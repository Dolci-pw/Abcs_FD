#!/bin/bash
#SBATCH --nodes=1                      #Número de Nós
#SBATCH --cpus-per-task=40             #Número de threads
#SBATCH -p intel_small                 #Fila (partition) a ser utilizada
#SBATCH -J devito_fwi_gdm         #Nome job
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --exclusive

module purge
module load gnu8/8.3.0
module load openmpi-3.1.6-gcc-8.3.0-n2h2i7h
eval "$(/home/felipe.augusto/anaconda3/bin/conda shell.bash hook)"
conda activate devito

export DEVITO_ARCH=gcc
export DEVITO_PLATFORM=intel64
export DEVITO_LOGGING=DEBUG
export DEVITO_LANGUAGE=openmp
export DEVITO_AUTOTUNING=aggressive

export OMP_NUM_THREADS=40
export EXEC_DEVITO=/home/felipe.augusto/FWIDevito/fwi.py
python $EXEC_DEVITO
