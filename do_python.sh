#this is a standard python submission script to use on Myriad.
# by TW 06h Nov 2023.

#!/bin/bash -l

#$ -cwd
#% -N do_metaD_sim
#$ -pe smp 1
#$ -l h_rt=00:00:10
#$ -l mem=1G

#$ -m eba
#$ -M ucnvtwe@ucl.ac.uk
module load gcc-libs
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate biophys_env

#nohup python langevin_sim_metad.py > metaD_sim_2.log &
nohup python test.py > test.log &

