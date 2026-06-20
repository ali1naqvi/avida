#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-0:00
#SBATCH --array=1-5
#SBATCH --output=ecology_%x_%A_%a.out
#SBATCH --error=ecology_%x_%A_%a.err
#SBATCH --mail-user=alinaqvi8014@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load cmake
module load gcc
module load python

TEST_NAME="$1"

if [ -z "$TEST_NAME" ]; then
  echo "Usage: sbatch scripts_ALI/run_ecology_tests.sh <test_dir_name>"
  exit 1
fi

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -x "$SLURM_SUBMIT_DIR/build_avida" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$SCRIPT_PROJECT_DIR"
fi
BASE_DIR="$PROJECT_DIR/cbuild/work"
AVIDA_BIN="$BASE_DIR/avida"
TEMPLATE_DIR="$BASE_DIR/$TEST_NAME"
RUN_DIR="$BASE_DIR/${TEST_NAME}_run_${SLURM_ARRAY_TASK_ID}"

if [ ! -x "$AVIDA_BIN" ]; then
  echo "Missing executable: $AVIDA_BIN"
  echo "Build Avida on the cluster before submitting this job:"
  echo "  cd $PROJECT_DIR"
  echo "  ./build_avida"
  exit 1
fi

if [ ! -d "$TEMPLATE_DIR" ]; then
  echo "Missing template directory: $TEMPLATE_DIR"
  exit 1
fi

rm -rf "$RUN_DIR"
cp -a "$TEMPLATE_DIR" "$RUN_DIR"

cd "$RUN_DIR"

SEED="$SLURM_ARRAY_TASK_ID"
"$AVIDA_BIN" -c avida.cfg -s "$SEED"
