#!/bin/bash
#SBATCH --job-name=build_gpu_gpt2    # Name of your job
#SBATCH --account=e32706             # Your project/allocation ID (CHANGE THIS)
#SBATCH --partition=gengpu             # The partition/queue for GPU jobs (THIS IS A KEY CHANGE)
#SBATCH --gpus-per-node=1            # Request 1 GPU per node
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=64G                    # Memory per node (Increased for GPU nodes)
#SBATCH --time=05:00:00                 # Request 4 hours of walltime
#SBATCH --output=build_model_gpu_%j.log # Log file for stdout/stderr


# --- Your commands start here ---

# Load any necessary modules
module purge
module load python/anaconda3
module load cuda

# Activate your conda environment (CHANGE "nlp" TO YOUR ENV NAME if different)
source activate nlp 

echo "Starting Python script on a GPU node..."
echo "Job running on node: $SLURM_NODELIST"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Run the Python script
# The --verbose flag prints more detailed output from the transformers library
python model.py --verbose

echo "Slurm job finished."