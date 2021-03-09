########################################################################################################################
#                                   Instalando o Devito/Anaconda no Mintrop                                            #
########################################################################################################################
module purge
module load gnu8/8.3.0 
module load openmpi-3.1.6-gcc-8.3.0-n2h2i7h 

wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh

# As opções selecionadas devem ser: Enter, yes, Enter, Enter

eval "$(/home/felipe.augusto/anaconda3/bin/conda shell.bash hook)" 

git clone https://github.com/devitocodes/devito.git
cd devito
conda env create -f environment-dev.yml
source activate devito
pip install -e .
