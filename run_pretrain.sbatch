#!/bin/bash

#SBATCH --job-name=c4_pretrain
#SBATCH --account=e32706             # replace with your quest allocation ID
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00                 # 2days request
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_c4_pretrain_%j.log

# --- Your job's commands below ---

module purge
module load python/anaconda3
source activate nlp

echo "Host: $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Starting C4 pre-training job..."

python pretrain.py \
    --model_path /home/iwm6052/adamus/FuncRoute-LM/gpt2-100m-custom \
    --output_dir /home/iwm6052/adamus/FuncRoute-LM/gpt2-100m-custom-pretrained-c4 \
    --max_steps 50000 \
    --eval_steps 1000

echo "Job finished with exit code $?"