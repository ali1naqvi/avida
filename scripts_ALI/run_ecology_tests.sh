#!/bin/bash
#SBATCH --account=def-skelly

#SBATCH --ntasks=1               
#SBATCH --mem-per-cpu=5G      
#SBATCH --time=0-5:00  # time (DD-HH:MM)

#SBATCH --output=ecology_%j.out
#SBATCH --error=ecology_%j.err

#SBATCH --mail-user=alinaqvi8014@gmail.com
#SBATCH --mail-type=END,FAIL

module load cmake
module load gcc
module load python

TEST_NAME="$1"

if [ -z "$TEST_NAME" ]; then
  echo "Usage: sbatch scripts_ALI/run_ecology_tests.sh <test_dir_name>"
  exit 1
fi

cd ~/scratch/avida/cbuild/work/$TEST_NAME
../avida -c avida.cfg