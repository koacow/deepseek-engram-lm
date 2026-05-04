#!/bin/bash -l
#$ -N eval_lm
#$ -l h_rt=4:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y

# Exit on error
set -e

cd $SGE_O_WORKDIR

mkdir -p logs

module load python3/3.10.12

# Activate virtual environment
source venv/bin/activate

# Function to display usage
usage() {
    echo "Usage: qsub scc_evaluate.sh <MODEL_KIND> <CHECKPOINT_PATH> <TASK>"
    echo "  MODEL_KIND:      baseline, params, or engram"
    echo "  CHECKPOINT_PATH: Path to the .pt checkpoint file"
    echo "  TASK:            mmlu or blimp"
    echo "Example: qsub scc_evaluate.sh params checkpoints/params_final.pt blimp"
    exit 1
}

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

MODEL_KIND=$1
CHECKPOINT_PATH=$2
TASK=$3

# Extract checkpoint filename without extension for the output file
CKPT_BASENAME=$(basename "$CHECKPOINT_PATH" .pt)
OUTPUT_FILE="reports/${CKPT_BASENAME}_${TASK}_results.json"

echo "======================================"
echo "Starting Evaluation"
echo "Model Kind: $MODEL_KIND"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Task:       $TASK"
echo "Output:     $OUTPUT_FILE"
echo "======================================"

python scripts/evaluate.py \
    --model-kind "$MODEL_KIND" \
    --checkpoint "$CHECKPOINT_PATH" \
    --task "$TASK" \
    --batch-size 8 \
    --output-file "$OUTPUT_FILE"

echo "Evaluation finished."
