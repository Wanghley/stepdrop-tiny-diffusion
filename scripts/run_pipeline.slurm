#!/bin/bash
#SBATCH --job-name=stepdrop
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# SLURM Job Script for StepDrop Pipeline
# =============================================================================
# Submit jobs to HPC cluster with:
#   sbatch scripts/run_pipeline.slurm                              # Default
#   sbatch scripts/run_pipeline.slurm --train --dataset cifar10    # Custom args
#
# Monitor jobs:
#   squeue -u $USER          # View your jobs
#   scancel <job_id>         # Cancel a job
#   tail -f logs/slurm_*.out # View output
# =============================================================================

echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job Name:     ${SLURM_JOB_NAME}"
echo "Node:         ${SLURM_NODELIST}"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "GPUs:         ${SLURM_GPUS:-1}"
echo "Memory:       ${SLURM_MEM_PER_NODE:-64G}"
echo "Start Time:   $(date)"
echo "Working Dir:  ${SLURM_SUBMIT_DIR}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Load modules (adjust for your cluster)
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true
module load python/3.9 2>/dev/null || module load python 2>/dev/null || true
module load anaconda 2>/dev/null || module load conda 2>/dev/null || true

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}" || cd ~/stepdrop-tiny-diffusion || {
    echo "ERROR: Could not find project directory"
    exit 1
}

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
fi

# Create logs directory if needed
mkdir -p logs

# -----------------------------------------------------------------------------
# Environment Info
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "Environment Information"
echo "=============================================="
echo "Python:       $(which python)"
echo "Python Ver:   $(python --version 2>&1)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"

# CUDA check
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA:         Available"
    echo "GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory:   $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\")')"
else
    echo "CUDA:         NOT AVAILABLE"
    echo "WARNING: Running on CPU will be very slow!"
fi
echo "=============================================="

# -----------------------------------------------------------------------------
# Default Pipeline Arguments
# -----------------------------------------------------------------------------

# Default: Full pipeline on CIFAR-10
DEFAULT_ARGS="--all --dataset cifar10 --epochs 50 --eval-samples 1000"

# Use command line arguments if provided, otherwise use defaults
if [ $# -gt 0 ]; then
    PIPELINE_ARGS="$@"
else
    PIPELINE_ARGS="${DEFAULT_ARGS}"
fi

echo ""
echo "=============================================="
echo "Running Pipeline"
echo "=============================================="
echo "Arguments: ${PIPELINE_ARGS}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Run Pipeline
# -----------------------------------------------------------------------------

# Make sure pipeline is executable
chmod +x pipeline.sh

# Run with timing
START_TIME=$(date +%s)

./pipeline.sh ${PIPELINE_ARGS}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "Job Complete"
echo "=============================================="
echo "End Time:     $(date)"
echo "Elapsed:      $(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
echo "Exit Code:    $?"
echo "=============================================="

# List outputs
echo ""
echo "Output Files:"
echo "  Checkpoints: $(ls -1 checkpoints/*.pt 2>/dev/null | wc -l) models"
echo "  Samples:     $(ls -1d samples/*/ 2>/dev/null | wc -l) directories"
echo "  Results:     $(ls -1d results/*/ 2>/dev/null | wc -l) experiments"
