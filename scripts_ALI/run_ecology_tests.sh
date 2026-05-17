#!/bin/bash
#SBATCH --account=def-skelly

#SBATCH --ntasks=21               
#SBATCH --mem-per-cpu=15G      
#SBATCH --time=0-5:00  # time (DD-HH:MM)

#SBATCH --output=ecology_%j.out
#SBATCH --error=ecology_%j.err

#SBATCH --mail-user=alinaqvi8014@gmail.com
#SBATCH --mail-type=END,FAIL

module load cmake
module load gcc
module load python

cd ~/scratch/avida/cbuild/work/ecology_test
../avida -c avida.cfg
