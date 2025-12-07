#!/bin/bash
set -e  # Exit on error

# =============================================================================
# STEPDROP DIFFUSION MODEL PIPELINE
# =============================================================================
# A complete pipeline for training, sampling, and evaluating diffusion models.
#
# Usage:
#   ./pipeline.sh --help                    # Show help
#   ./pipeline.sh --all                     # Run full pipeline
#   ./pipeline.sh --train --dataset mnist   # Train on MNIST
#   ./pipeline.sh --sample --checkpoint checkpoints/model.pt
#   ./pipeline.sh --evaluate                # Run benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# Get Project Root (works from any directory)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root for consistent relative paths
cd "$PROJECT_ROOT"
# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
DATASET="cifar10"
IMG_SIZE=32
CHANNELS=3
BATCH_SIZE=128
EPOCHS=50
LR="2e-4"
BASE_CHANNELS=64
N_TIMESTEPS=1000
SCHEDULE_TYPE="cosine"
SEED=42

# Paths
CHECKPOINT_DIR="checkpoints"
SAMPLE_DIR="samples"
RESULTS_DIR="results"
LOG_DIR="logs"
DATA_DIR="./data"

# Sampling
N_SAMPLES=64
SAMPLE_METHOD="ddim"
DDIM_STEPS=50

# Evaluation
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=32

# Pipeline flags
DO_TRAIN=false
DO_SAMPLE=false
DO_EVALUATE=false
DO_ALL=false
DO_CLEAN=false
RESUME=""
CHECKPOINT=""
CUSTOM_DATA_DIR=""
DEVICE="cuda"
DRY_RUN=false
VERBOSE=false

# -----------------------------------------------------------------------------
# Color Output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
}

# -----------------------------------------------------------------------------
# Help Function
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
${CYAN}STEPDROP DIFFUSION MODEL PIPELINE${NC}
===================================

A complete pipeline for training, sampling, and evaluating diffusion models.

${YELLOW}USAGE:${NC}
    ./pipeline.sh [OPTIONS]

${YELLOW}PIPELINE STAGES:${NC}
    --train             Run training stage
    --sample            Run sampling stage
    --evaluate          Run evaluation/benchmarking stage
    --all               Run all stages (train → sample → evaluate)
    --clean             Clean generated files before running

${YELLOW}DATASET OPTIONS:${NC}
    --dataset NAME      Dataset to use: mnist, cifar10, custom (default: ${DATASET})
    --custom-data DIR   Path to custom dataset directory
    --img-size SIZE     Image size (default: ${IMG_SIZE})
    --channels N        Number of channels (default: ${CHANNELS})
    --data-dir DIR      Data directory (default: ${DATA_DIR})

${YELLOW}TRAINING OPTIONS:${NC}
    --epochs N          Number of training epochs (default: ${EPOCHS})
    --batch-size N      Training batch size (default: ${BATCH_SIZE})
    --lr RATE           Learning rate (default: ${LR})
    --base-channels N   U-Net base channels (default: ${BASE_CHANNELS})
    --timesteps N       Diffusion timesteps (default: ${N_TIMESTEPS})
    --schedule TYPE     Noise schedule: linear, cosine (default: ${SCHEDULE_TYPE})
    --resume PATH       Resume training from checkpoint
    --seed N            Random seed (default: ${SEED})

${YELLOW}SAMPLING OPTIONS:${NC}
    --checkpoint PATH   Path to model checkpoint (required for --sample without --train)
    --n-samples N       Number of samples to generate (default: ${N_SAMPLES})
    --method METHOD     Sampling method: ddpm, ddim (default: ${SAMPLE_METHOD})
    --ddim-steps N      DDIM sampling steps (default: ${DDIM_STEPS})

${YELLOW}EVALUATION OPTIONS:${NC}
    --eval-samples N    Samples for FID/IS evaluation (default: ${EVAL_SAMPLES})
    --eval-batch N      Evaluation batch size (default: ${EVAL_BATCH_SIZE})

${YELLOW}GENERAL OPTIONS:${NC}
    --device DEVICE     Device: cuda, cpu (default: ${DEVICE})
    --checkpoint-dir D  Checkpoint directory (default: ${CHECKPOINT_DIR})
    --sample-dir DIR    Sample output directory (default: ${SAMPLE_DIR})
    --results-dir DIR   Results directory (default: ${RESULTS_DIR})
    --log-dir DIR       Log directory (default: ${LOG_DIR})
    --dry-run           Print commands without executing
    --verbose           Verbose output
    --help, -h          Show this help message

${YELLOW}EXAMPLES:${NC}
    # Quick test on MNIST
    ./pipeline.sh --all --dataset mnist --epochs 5 --n-samples 16

    # Full CIFAR-10 training
    ./pipeline.sh --train --dataset cifar10 --epochs 100 --base-channels 128

    # Sample from trained model
    ./pipeline.sh --sample --checkpoint checkpoints/cifar_model.pt --n-samples 64

    # Run full benchmark
    ./pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 5000

    # Custom dataset
    ./pipeline.sh --all --dataset custom --custom-data /path/to/images --img-size 64

    # Resume training
    ./pipeline.sh --train --resume checkpoints/checkpoint_epoch_20.pt --epochs 50

EOF
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # Pipeline stages
            --train)        DO_TRAIN=true; shift ;;
            --sample)       DO_SAMPLE=true; shift ;;
            --evaluate)     DO_EVALUATE=true; shift ;;
            --all)          DO_ALL=true; shift ;;
            --clean)        DO_CLEAN=true; shift ;;
            
            # Dataset options
            --dataset)      DATASET="$2"; shift 2 ;;
            --custom-data)  CUSTOM_DATA_DIR="$2"; shift 2 ;;
            --img-size)     IMG_SIZE="$2"; shift 2 ;;
            --channels)     CHANNELS="$2"; shift 2 ;;
            --data-dir)     DATA_DIR="$2"; shift 2 ;;
            
            # Training options
            --epochs)       EPOCHS="$2"; shift 2 ;;
            --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
            --lr)           LR="$2"; shift 2 ;;
            --base-channels) BASE_CHANNELS="$2"; shift 2 ;;
            --timesteps)    N_TIMESTEPS="$2"; shift 2 ;;
            --schedule)     SCHEDULE_TYPE="$2"; shift 2 ;;
            --resume)       RESUME="$2"; shift 2 ;;
            --seed)         SEED="$2"; shift 2 ;;
            
            # Sampling options
            --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
            --n-samples)    N_SAMPLES="$2"; shift 2 ;;
            --method)       SAMPLE_METHOD="$2"; shift 2 ;;
            --ddim-steps)   DDIM_STEPS="$2"; shift 2 ;;
            
            # Evaluation options
            --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
            --eval-batch)   EVAL_BATCH_SIZE="$2"; shift 2 ;;
            
            # General options
            --device)       DEVICE="$2"; shift 2 ;;
            --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
            --sample-dir)   SAMPLE_DIR="$2"; shift 2 ;;
            --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
            --log-dir)      LOG_DIR="$2"; shift 2 ;;
            --dry-run)      DRY_RUN=true; shift ;;
            --verbose)      VERBOSE=true; shift ;;
            --help|-h)      show_help; exit 0 ;;
            
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
    
    # If --all is set, enable all stages
    if [ "$DO_ALL" = true ]; then
        DO_TRAIN=true
        DO_SAMPLE=true
        DO_EVALUATE=true
    fi
    
    # Check if at least one stage is selected
    if [ "$DO_TRAIN" = false ] && [ "$DO_SAMPLE" = false ] && [ "$DO_EVALUATE" = false ] && [ "$DO_CLEAN" = false ]; then
        log_error "No pipeline stage selected. Use --train, --sample, --evaluate, --all, or --clean"
        echo "Use --help for usage information."
        exit 1
    fi
    
    # Auto-configure based on dataset
    if [ "$DATASET" = "mnist" ]; then
        IMG_SIZE=28
        CHANNELS=1
    elif [ "$DATASET" = "cifar10" ]; then
        IMG_SIZE=32
        CHANNELS=3
    fi
}

# -----------------------------------------------------------------------------
# Run Command (with dry-run support)
# -----------------------------------------------------------------------------
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${CYAN}[DRY-RUN]${NC} $*"
    else
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}[RUNNING]${NC} $*"
        fi
        eval "$@"
    fi
}

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------
setup_environment() {
    log_step "Setting Up Environment"
    
    # Create directories
    log_info "Creating directories..."
    run_cmd "mkdir -p ${CHECKPOINT_DIR}"
    run_cmd "mkdir -p ${SAMPLE_DIR}"
    run_cmd "mkdir -p ${RESULTS_DIR}"
    run_cmd "mkdir -p ${LOG_DIR}"
    run_cmd "mkdir -p ${DATA_DIR}"
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check CUDA
    if [ "$DEVICE" = "cuda" ]; then
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
            log_success "CUDA available: ${GPU_NAME}"
        else
            log_warning "CUDA not available, falling back to CPU"
            DEVICE="cpu"
        fi
    fi
    
    # Display configuration
    log_info "Configuration:"
    echo "  Dataset:        ${DATASET}"
    echo "  Image Size:     ${IMG_SIZE}x${IMG_SIZE}"
    echo "  Channels:       ${CHANNELS}"
    echo "  Device:         ${DEVICE}"
    echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
    
    log_success "Environment setup complete"
}

# -----------------------------------------------------------------------------
# Clean Stage
# -----------------------------------------------------------------------------
clean_stage() {
    log_step "Cleaning Generated Files"
    
    read -p "This will delete checkpoints, samples, and results. Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning checkpoints..."
        run_cmd "rm -rf ${CHECKPOINT_DIR}/*.pt"
        
        log_info "Cleaning samples..."
        run_cmd "rm -rf ${SAMPLE_DIR}/*"
        
        log_info "Cleaning results..."
        run_cmd "rm -rf ${RESULTS_DIR}/*"
        
        log_info "Cleaning logs..."
        run_cmd "rm -rf ${LOG_DIR}/*.json"
        
        log_success "Clean complete"
    else
        log_info "Clean cancelled"
    fi
}

# -----------------------------------------------------------------------------
# Training Stage
# -----------------------------------------------------------------------------
train_stage() {
    log_step "Training Stage"
    
    # Build model save path
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MODEL_NAME="${DATASET}_${BASE_CHANNELS}ch_${EPOCHS}ep"
    SAVE_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}.pt"
    
    log_info "Training configuration:"
    echo "  Dataset:       ${DATASET}"
    echo "  Epochs:        ${EPOCHS}"
    echo "  Batch Size:    ${BATCH_SIZE}"
    echo "  Learning Rate: ${LR}"
    echo "  Base Channels: ${BASE_CHANNELS}"
    echo "  Timesteps:     ${N_TIMESTEPS}"
    echo "  Schedule:      ${SCHEDULE_TYPE}"
    echo "  Save Path:     ${SAVE_PATH}"
    
    # Build training command
    TRAIN_CMD="python src/train.py"
    TRAIN_CMD+=" --dataset ${DATASET}"
    TRAIN_CMD+=" --img_size ${IMG_SIZE}"
    TRAIN_CMD+=" --channels ${CHANNELS}"
    TRAIN_CMD+=" --batch_size ${BATCH_SIZE}"
    TRAIN_CMD+=" --epochs ${EPOCHS}"
    TRAIN_CMD+=" --lr ${LR}"
    TRAIN_CMD+=" --base_channels ${BASE_CHANNELS}"
    TRAIN_CMD+=" --n_timesteps ${N_TIMESTEPS}"
    TRAIN_CMD+=" --schedule_type ${SCHEDULE_TYPE}"
    TRAIN_CMD+=" --save_path ${SAVE_PATH}"
    TRAIN_CMD+=" --log_dir ${LOG_DIR}"
    TRAIN_CMD+=" --data_dir ${DATA_DIR}"
    TRAIN_CMD+=" --device ${DEVICE}"
    TRAIN_CMD+=" --seed ${SEED}"
    
    if [ -n "$CUSTOM_DATA_DIR" ]; then
        TRAIN_CMD+=" --custom_data_dir ${CUSTOM_DATA_DIR}"
    fi
    
    if [ -n "$RESUME" ]; then
        TRAIN_CMD+=" --resume ${RESUME}"
        log_info "Resuming from: ${RESUME}"
    fi
    
    # Run training
    log_info "Starting training..."
    run_cmd "${TRAIN_CMD}"
    
    # Set checkpoint for subsequent stages
    CHECKPOINT="${SAVE_PATH}"
    
    log_success "Training complete! Model saved to: ${SAVE_PATH}"
}

# -----------------------------------------------------------------------------
# Sampling Stage
# -----------------------------------------------------------------------------
sample_stage() {
    log_step "Sampling Stage"
    
    # Check checkpoint
    if [ -z "$CHECKPOINT" ]; then
        # Try to find latest checkpoint
        CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pt 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            log_error "No checkpoint specified and none found in ${CHECKPOINT_DIR}"
            exit 1
        fi
        log_info "Using latest checkpoint: ${CHECKPOINT}"
    fi
    
    if [ ! -f "$CHECKPOINT" ]; then
        log_error "Checkpoint not found: ${CHECKPOINT}"
        exit 1
    fi
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${SAMPLE_DIR}/${TIMESTAMP}"
    
    log_info "Sampling configuration:"
    echo "  Checkpoint:    ${CHECKPOINT}"
    echo "  Samples:       ${N_SAMPLES}"
    echo "  Method:        ${SAMPLE_METHOD}"
    if [ "$SAMPLE_METHOD" = "ddim" ]; then
        echo "  DDIM Steps:    ${DDIM_STEPS}"
    fi
    echo "  Output Dir:    ${OUTPUT_DIR}"
    
    # Build sampling command
    SAMPLE_CMD="python src/sample.py"
    SAMPLE_CMD+=" --checkpoint ${CHECKPOINT}"
    SAMPLE_CMD+=" --n_samples ${N_SAMPLES}"
    SAMPLE_CMD+=" --method ${SAMPLE_METHOD}"
    SAMPLE_CMD+=" --output_dir ${OUTPUT_DIR}"
    SAMPLE_CMD+=" --device ${DEVICE}"
    SAMPLE_CMD+=" --seed ${SEED}"
    SAMPLE_CMD+=" --save_grid"
    SAMPLE_CMD+=" --save_individual"
    
    if [ "$SAMPLE_METHOD" = "ddim" ]; then
        SAMPLE_CMD+=" --ddim_steps ${DDIM_STEPS}"
    fi
    
    # Run sampling
    log_info "Generating samples..."
    run_cmd "${SAMPLE_CMD}"
    
    log_success "Sampling complete! Samples saved to: ${OUTPUT_DIR}"
}

# -----------------------------------------------------------------------------
# Evaluation Stage
# -----------------------------------------------------------------------------
evaluate_stage() {
    log_step "Evaluation Stage"
    
    # Check checkpoint
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pt 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            log_error "No checkpoint specified and none found in ${CHECKPOINT_DIR}"
            log_info "Running evaluation in dummy mode..."
            EVAL_MODE="--dummy"
        else
            log_info "Using latest checkpoint: ${CHECKPOINT}"
            EVAL_MODE="--checkpoint ${CHECKPOINT}"
        fi
    else
        if [ ! -f "$CHECKPOINT" ]; then
            log_error "Checkpoint not found: ${CHECKPOINT}"
            exit 1
        fi
        EVAL_MODE="--checkpoint ${CHECKPOINT}"
    fi
    
    log_info "Evaluation configuration:"
    echo "  Checkpoint:    ${CHECKPOINT:-DUMMY}"
    echo "  Eval Samples:  ${EVAL_SAMPLES}"
    echo "  Batch Size:    ${EVAL_BATCH_SIZE}"
    
    # Ensure real data cache exists
    log_info "Preparing real data cache for FID calculation..."
    run_cmd "python -c \"
import sys
sys.path.append('.')
from src.eval.metrics_utils import saveCifar10RealSub
saveCifar10RealSub(numImages=${EVAL_SAMPLES})
\""
    
    # Build evaluation command
    EVAL_CMD="python scripts/benchmark_strategies.py"
    EVAL_CMD+=" ${EVAL_MODE}"
    EVAL_CMD+=" --samples ${EVAL_SAMPLES}"
    EVAL_CMD+=" --batch_size ${EVAL_BATCH_SIZE}"
    
    # Run evaluation
    log_info "Running benchmark..."
    run_cmd "${EVAL_CMD}"
    
    log_success "Evaluation complete! Results saved to: ${RESULTS_DIR}/"
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print_summary() {
    log_step "Pipeline Summary"
    
    echo -e "${GREEN}Completed stages:${NC}"
    [ "$DO_TRAIN" = true ] && echo "  ✅ Training"
    [ "$DO_SAMPLE" = true ] && echo "  ✅ Sampling"
    [ "$DO_EVALUATE" = true ] && echo "  ✅ Evaluation"
    
    echo ""
    echo -e "${CYAN}Output locations:${NC}"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  Samples:     ${SAMPLE_DIR}/"
    echo "  Results:     ${RESULTS_DIR}/"
    echo "  Logs:        ${LOG_DIR}/"
    
    if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
        echo ""
        echo -e "${CYAN}Latest checkpoint:${NC}"
        echo "  ${CHECKPOINT}"
    fi
    
    # Show latest results if available
    LATEST_RESULT=$(ls -td ${RESULTS_DIR}/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ] && [ -f "${LATEST_RESULT}report.json" ]; then
        echo ""
        echo -e "${CYAN}Latest evaluation results:${NC}"
        python -c "
import json
with open('${LATEST_RESULT}report.json') as f:
    data = json.load(f)
for name, metrics in data.items():
    if isinstance(metrics, dict) and 'fid' in metrics:
        print(f\"  {name}:\")
        print(f\"    FID: {metrics.get('fid', 'N/A'):.2f}\")
        print(f\"    IS:  {metrics.get('is_mean', 'N/A'):.2f}\")
        print(f\"    Throughput: {metrics.get('throughput', 'N/A'):.2f} img/s\")
" 2>/dev/null || true
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         STEPDROP DIFFUSION MODEL PIPELINE                 ║"
    echo "║         Stochastic Step Skipping in Tiny Diffusion        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    parse_args "$@"
    
    # Run pipeline stages
    setup_environment
    
    if [ "$DO_CLEAN" = true ]; then
        clean_stage
    fi
    
    if [ "$DO_TRAIN" = true ]; then
        train_stage
    fi
    
    if [ "$DO_SAMPLE" = true ]; then
        sample_stage
    fi
    
    if [ "$DO_EVALUATE" = true ]; then
        evaluate_stage
    fi
    
    print_summary
    
    echo ""
    log_success "Pipeline completed successfully! 🎉"
}

# Run main
main "$@"