#!/bin/bash -l

# Example SLURM job script for BU SCC
# Submit with: qsub scripts/scc_train.sh

#$ -P cs505              # your SCC project allocation
#$ -l gpus=1             # Request 1 GPU
#$ -l gpu_type=A100      # Request A100 (can change to A40 if queue is long)
#$ -l h_rt=12:00:00      # 12 hours walltime (max for A100s usually)
#$ -N engram_train       # Job name
#$ -j y                  # Merge stdout/stderr
#$ -o logs/$JOB_NAME_$JOB_ID.log  # Output log file

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust versions if needed)
module load python3/3.10.12 pytorch/2.1.0

# Activate any virtualenv if you have one
# source /path/to/your/venv/bin/activate

# Execute from the project root
cd /projectnb/cs505/$USER/deepseek-engram-lm

# Train full Engram:
# python scripts/train_engram.py --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_engram

# Train frozen-gate Engram:
# python scripts/train_engram.py --frozen-gates --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_frozen

# Train baseline:
python scripts/train_baseline.py --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_baseline

# Train params control:
# python scripts/train_params_control.py --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_params
