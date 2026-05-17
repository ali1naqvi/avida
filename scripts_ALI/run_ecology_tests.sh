#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-5:00
#SBATCH --array=1-20
#SBATCH --output=ecology_%x_%A_%a.out
#SBATCH --error=ecology_%x_%A_%a.err

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

BASE_DIR=~/scratch/avida/cbuild/work
RUN_DIR=$BASE_DIR/${TEST_NAME}_run_${SLURM_ARRAY_TASK_ID}

rm -rf "$RUN_DIR"
cp -a "$BASE_DIR/$TEST_NAME" "$RUN_DIR"

cd "$RUN_DIR"

SEED=$SLURM_ARRAY_TASK_ID
../avida -c avida.cfg -s "$SEED"
