#!/bin/bash
#SBATCH --job-name=avida-ecology
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=ecology_%j.out
#SBATCH --error=ecology_%j.err

#SBATCH --mail-user=alinaqvi8014@gmail.com
#SBATCH --mail-type=END,FAIL

module load cmake
module load gcc
module load python

cd ~/scratch/avida/cbuild/work/ecology_test
../avida -c avida.cfg
