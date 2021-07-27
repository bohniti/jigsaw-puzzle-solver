#!/bin/bash -l
#
# I use this script to start jobs from within another script.
# We pass the name of the job and a variable CFG directly to the qsub command
# The variable CFG points to a config file that is used py the python script
#
# Submit like this: qsub.tinygpu basic_gpu.sh
#
# allocate 1 node, 4 cores, for 24 hours
# one GPU per 4 cores (cores always have to be a multiple of 4)
# We only request nodes that support avx, because PyTorch requires it (if you get a non-avx node, the script crashes)
#PBS -l nodes=1:ppn=4:avx:v100,walltime=24:00:00
#
# Give the job a name (no spaces!)
#PBS -N JigsawTestRun
#
# put output files into directory named output (needs to exist!)
#PBS -o /home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver/results/hpc-logs/outputs
#PBS -e /home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver/results/hpc-logs/errors
#
# send mails
#PBS -M timo.bohnstedt@fau.de -m abe
# first non-empty non-comment line ends PBS options

# load Python 3.7
module load python/3.7-anaconda
module load cuda/10.2


# navigate to project directory
cd /home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver/

# activate the venv environment stored in the "venv" directory
source env/bin/activate

# start the script main.py
python3 ./main.py
