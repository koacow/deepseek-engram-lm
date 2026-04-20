#!/bin/bash -l
#$ -N train_engram_frozen
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y



cd $SGE_O_WORKDIR

mkdir -p logs


module load python3/3.10.12

# 3. Activate virtual environment

source venv/bin/activate

# Execute the training script:
python train_engram.py --frozen-gates --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_frozen
#!/bin/bash -l
#$ -N train_engram_frozen
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y


# 1. Navigate to the directory where you submitted the qsub command
cd $SGE_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs

# 2. Load the Python module
module load python3/3.10.12

# 3. Activate your virtual environment
# (See instructions below on how to set this up first)
source venv/bin/activate

# Execute the training script:
python train_engram.py --frozen-gates --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_frozen


# python train_engram.py  --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_engram

# python train_baseline.py  --steps 100000 --batch-size 8 --grad-accum 4 --output-dir checkpoints_baseline