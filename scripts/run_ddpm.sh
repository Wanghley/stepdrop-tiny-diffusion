#!/bin/bash

# Colors
B_PURPLE=$'\033[1;35m'
B_CYAN=$'\033[1;36m'
NC=$'\033[0m'

echo -e "${B_PURPLE}=== StepDrop DDPM Sampling ===${NC}"

#SBATCH --job-name=ddpm_sampling
#SBATCH --output=logs/ddpm_%j.out
#SBATCH --error=logs/ddpm_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Run DDPM sampling
python ddpm_sampler.py \
    --num_samples 16 \
    --num_timesteps 1000 \
    --channels 3 \
    --output_dir results/ddpm