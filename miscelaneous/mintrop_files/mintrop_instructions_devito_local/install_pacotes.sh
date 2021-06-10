########################################################################################################################
#                         Instalando pacotes python no ambiente Devito no Mintrop                                      #
########################################################################################################################
module purge
module load gnu8/8.3.0 
module load openmpi-3.1.6-gcc-8.3.0-n2h2i7h 

eval "$(/home/felipe.augusto/anaconda3/bin/conda shell.bash hook)" 
conda activate devito

# Verifique se o seu python principal é o do Anaconda, se não for, arrume as variáveis do seu bashrc antes de prosseguir.
which python

# Instalando o segyio como exemplo

python -m pip install matplotlib
python -m pip install jupyter notebook
python -m pip install segyio
########################################################################################################################

