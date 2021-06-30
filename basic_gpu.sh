#!/bin/bash -l
#
# I use this script to start jobs from within another script. We pass the name of the job and a variable CFG directly to the qsub command
# The variable CFG points to a config file that is used py the python script
#
# Submit like this: qsub.tinygpu basic_gpu.sh
#
# allocate 1 node, 4 cores, for 24 hours
# one GPU per 4 cores (cores always have to be a multiple of 4)
# We only request nodes that support avx, because PyTorch requires it (if you get a non-avx node, the script crashes)
#PBS -l nodes=1:ppn=4:avx,walltime=24:00:00
#
# Give the job a name (no spaces!)
#PBS -N job_name 
#
# put output files into directory named output (needs to exist!)
#PBS -o output
#PBS -e output
#
# send mails
#PBS -M EMAIL-ADDRESS -m abe
# first non-empty non-comment line ends PBS options

# load Python 3.7
module load python/3.7-anaconda

# navigate to project directory
cd $WORK/jigsaw-puzzle-solver

# activate the venv environment stored in the "venv" directory
conda create -f environment.yml --prefix $WORK/envs/PuzzleSolver

# activate env
conda activate PuzzleSolver

# start the script main.py
python src/main.py
