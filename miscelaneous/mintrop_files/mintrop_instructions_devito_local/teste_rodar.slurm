#!/bin/bash
#SBATCH --nodes=1                      #Número de Nós
#SBATCH --cpus-per-task=40             #Número de threads
#SBATCH -p intel_small                 #Fila (partition) a ser utilizada
#SBATCH -J devito-anaconda-example     #Nome job
#SBATCH --mail-user=user@usp.br        #Configura o destinatário
#SBATCH --mail-type=BEGIN              #Envia email ao iniciar o job
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --exclusive

export RUN_SCRATCH_DIR=/scratch/global/tmp-$SLURM_JOBID

module purge
module load gnu8/8.3.0
module load openmpi-3.1.6-gcc-8.3.0-n2h2i7h
eval "$(/home/felipe.augusto/anaconda3/bin/conda shell.bash hook)"
conda activate devito
cd $RUN_SCRATCH_DIR
export DEVITO_ARCH=gcc
export DEVITO_PLATFORM=intel64
export DEVITO_LOGGING=DEBUG
export DEVITO_LANGUAGE=openmp
export DEVITO_AUTOTUNING=aggressive
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export EXEC_DEVITO=/home/felipe.augusto/devito/benchmarks/user/benchmark.py
python $EXEC_DEVITO bench -P acoustic -d 512 512 512 -so 12 --arch gcc --tn 100 -a aggressive

which python
python -c "import segyio; print(segyio.__path__)"

