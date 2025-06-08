#!/bin/bash

#SBATCH --job-name=funcroute_finetune
#SBATCH --account=e32706               # <-- IMPORTANT: Replace with your Quest allocation ID.
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_finetune_%j.log

# --- Your job's commands below ---

module purge
module load python/anaconda3
source activate nlp

echo "Starting fine-tuning job on host $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

python finetune.py \
    --model_path /home/iwm6052/adamus/FuncRoute-LM/gpt2-100m-custom \
    --data_path ./train_dataset.jsonl \
    --functions_path ./functions_with_params.json \
    --output_dir ./funcroute_lm_finetuned_from_custom_v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --eval_steps 100

echo "Job finished with exit code $?"