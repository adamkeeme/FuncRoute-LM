#!/bin/bash

#SBATCH --job-name=funcroute_train
#SBATCH --account=e32706               # <-- IMPORTANT: Replace it with your actual Quest allocation ID.
#SBATCH --partition=gengpu              # Request the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4               # Request 4 CPUs
#SBATCH --mem=32G                       # Request 32GB of memory
#SBATCH --time=04:00:00                 # Request 4 hours of walltime
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --output=slurm_output_%j.log    # Redirect stdout/stderr to a log file

# --- Your job's commands below ---

# It's good practice to purge modules to ensure a clean environment
module purge

# Load the anaconda module to make the 'conda' command available
module load python/anaconda3

# Activate your conda environment
# The prompt you showed '(nlp)' indicates your environment is named 'nlp'
source activate nlp

# Print information about the job allocation
echo "Starting training job on host $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script
# These are the same arguments we discussed before
python train.py \
    --train_data_path ./train_dataset.jsonl \
    --functions_path ./functions_with_params.json \
    --output_dir ./funcroute_lm_v1_gpu \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5

echo "Job finished with exit code $?"