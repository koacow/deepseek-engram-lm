#!/bin/bash -l
#$ -N engram_gating_visualization
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

# Execute the jupyter notebook, saving it to the ipynb file
jupyter nbconvert --to notebook --execute engram_gating_visualization.ipynb
#!/bin/bash -l
#$ -N train_engram_frozen
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 8
#$ -j y