#!/bin/bash

# Colors
B_PURPLE=$'\033[1;35m'
B_CYAN=$'\033[1;36m'
NC=$'\033[0m'

echo -e "${B_PURPLE}=== StepDrop DDIM Sampling ===${NC}"

#SBATCH --job-name=ddim_sampling
#SBATCH --output=logs/ddim_%j.out
#SBATCH --error=logs/ddim_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Run DDIM sampling with different step counts
for steps in 10 25 50 100; do
    echo "Running DDIM with $steps steps..."
    python src/sample.py \
        --checkpoint checkpoints/model.pt \
        --method ddim \
        --n_samples 16 \
        --ddim_steps $steps \
        --ddim_eta 0.0 \
        --output_dir results/ddim \
        --device cuda
done