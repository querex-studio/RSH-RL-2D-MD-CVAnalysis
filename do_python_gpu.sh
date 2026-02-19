#this is a gpu python submission script to use on Myriad.
# by TW 06h Nov 2023.

#!/bin/bash -l

#$ -N do_metaD_sim
#$ -l gpu=2
#$ -pe smp 12
#$ -l h_rt=48:00:00
#$ -l mem=1G

#$ -wd /home/ucnvtwe/Scratch/output

#$ -m eba
#$ -M ucnvtwe@ucl.ac.uk

module load gcc-libs
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate biophys_env

nohup python langevin_sim_metad.py > metaD_sim_06Nov2023.log &
#nohup python test.py &

