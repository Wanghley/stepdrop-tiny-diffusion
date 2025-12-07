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
SKIP_PROB=0.3
SKIP_STRATEGY="linear"

# Evaluation
EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=32
FULL_METRICS=false
EVAL_STRATEGIES="all"

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
COMPARE_STEPDROP=false
STEPDROP_ONLY=false

# -----------------------------------------------------------------------------
# Color Output
# -----------------------------------------------------------------------------
# ANSI Escapes (using $'' for cat compatibility)
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
PURPLE=$'\033[0;35m'
CYAN=$'\033[0;36m'
WHITE=$'\033[0;37m'
GREY=$'\033[0;90m'

# Bold/Bright
B_RED=$'\033[1;31m'
B_GREEN=$'\033[1;32m'
B_YELLOW=$'\033[1;33m'
B_BLUE=$'\033[1;34m'
B_PURPLE=$'\033[1;35m'
B_CYAN=$'\033[1;36m'
B_WHITE=$'\033[1;37m'

# Effects
BOLD=$'\033[1m'
ITALIC=$'\033[3m'
NC=$'\033[0m' # No Color

# Semantic Colors
C_HEADER=${B_PURPLE}
C_SUBHEAD=${B_CYAN}
C_FLAG=${GREEN}
C_ARG=${YELLOW}
C_CMD=${B_WHITE}
C_COMMENT=${GREY}
C_BORDER=${BLUE}

log_info() {
    echo -e "${B_BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${B_GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${B_YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${B_RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${C_BORDER}========================================${NC}"
    echo -e "${C_HEADER}  $1${NC}"
    echo -e "${C_BORDER}========================================${NC}\n"
}

# -----------------------------------------------------------------------------
# Help Function
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
${C_BORDER}╔══════════════════════════════════════════════════════════════════════════════╗${NC}
${C_BORDER}║${NC}   ${B_PURPLE}STEPDROP DIFFUSION MODEL PIPELINE${NC}                                      ${C_BORDER}║${NC}
${C_BORDER}║${NC}   ${ITALIC}Stochastic Step Skipping in Tiny Diffusion${NC}                             ${C_BORDER}║${NC}
${C_BORDER}╚══════════════════════════════════════════════════════════════════════════════╝${NC}

${C_COMMENT}A complete pipeline for training, sampling, and benchmarking diffusion models.${NC}

${C_SUBHEAD}USAGE:${NC}
    ${C_CMD}./pipeline.sh${NC} [OPTIONS]

${C_SUBHEAD}PIPELINE STAGES:${NC}
    ${C_FLAG}--train${NC}             Run training stage
    ${C_FLAG}--sample${NC}            Run sampling stage
    ${C_FLAG}--evaluate${NC}          Run evaluation/benchmarking stage
    ${C_FLAG}--all${NC}               Run all stages ${C_COMMENT}(train → sample → evaluate)${NC}
    ${C_FLAG}--clean${NC}             Clean generated files before running

${C_SUBHEAD}DATASET OPTIONS:${NC}
    ${C_FLAG}--dataset${NC} ${C_ARG}NAME${NC}      Dataset to use: [mnist, cifar10, custom] ${C_COMMENT}(default: ${DATASET})${NC}
    ${C_FLAG}--custom-data${NC} ${C_ARG}DIR${NC}   Path to custom dataset directory
    ${C_FLAG}--img-size${NC} ${C_ARG}SIZE${NC}     Image size ${C_COMMENT}(default: ${IMG_SIZE})${NC}
    ${C_FLAG}--channels${NC} ${C_ARG}N${NC}        Number of channels ${C_COMMENT}(default: ${CHANNELS})${NC}
    ${C_FLAG}--data-dir${NC} ${C_ARG}DIR${NC}      Data directory ${C_COMMENT}(default: ${DATA_DIR})${NC}

${C_SUBHEAD}TRAINING OPTIONS:${NC}
    ${C_FLAG}--epochs${NC} ${C_ARG}N${NC}          Number of training epochs ${C_COMMENT}(default: ${EPOCHS})${NC}
    ${C_FLAG}--batch-size${NC} ${C_ARG}N${NC}      Training batch size ${C_COMMENT}(default: ${BATCH_SIZE})${NC}
    ${C_FLAG}--lr${NC} ${C_ARG}RATE${NC}           Learning rate ${C_COMMENT}(default: ${LR})${NC}
    ${C_FLAG}--base-channels${NC} ${C_ARG}N${NC}   U-Net base channels ${C_COMMENT}(default: ${BASE_CHANNELS})${NC}
    ${C_FLAG}--timesteps${NC} ${C_ARG}N${NC}       Diffusion timesteps ${C_COMMENT}(default: ${N_TIMESTEPS})${NC}
    ${C_FLAG}--schedule${NC} ${C_ARG}TYPE${NC}     Noise schedule: [linear, cosine] ${C_COMMENT}(default: ${SCHEDULE_TYPE})${NC}
    ${C_FLAG}--resume${NC} ${C_ARG}PATH${NC}       Resume training from checkpoint
    ${C_FLAG}--seed${NC} ${C_ARG}N${NC}            Random seed ${C_COMMENT}(default: ${SEED})${NC}

${C_SUBHEAD}SAMPLING OPTIONS:${NC}
    ${C_FLAG}--checkpoint${NC} ${C_ARG}PATH${NC}   Path to model checkpoint ${C_COMMENT}(required for --sample without --train)${NC}
    ${C_FLAG}--n-samples${NC} ${C_ARG}N${NC}       Number of samples to generate ${C_COMMENT}(default: ${N_SAMPLES})${NC}
    ${C_FLAG}--method${NC} ${C_ARG}METHOD${NC}     Method: [ddpm, ddim, stepdrop] ${C_COMMENT}(default: ${SAMPLE_METHOD})${NC}
    ${C_FLAG}--ddim-steps${NC} ${C_ARG}N${NC}      DDIM sampling steps ${C_COMMENT}(default: ${DDIM_STEPS})${NC}
    ${C_FLAG}--skip-prob${NC} ${C_ARG}P${NC}       StepDrop probability [0.0-1.0]
    ${C_FLAG}--skip-strategy${NC} ${C_ARG}S${NC}   StepDrop strategy: [linear, quadratic, cosine]

${C_SUBHEAD}EVALUATION OPTIONS:${NC}
    ${C_FLAG}--eval-samples${NC} ${C_ARG}N${NC}    Samples for FID/IS evaluation ${C_COMMENT}(default: ${EVAL_SAMPLES})${NC}
    ${C_FLAG}--eval-batch${NC} ${C_ARG}N${NC}      Evaluation batch size ${C_COMMENT}(default: ${EVAL_BATCH_SIZE})${NC}
    ${C_FLAG}--strategies${NC} ${C_ARG}LIST${NC}   Comma-separated strategies to bench ${C_COMMENT}(default: all)${NC}

${C_SUBHEAD}GENERAL OPTIONS:${NC}
    ${C_FLAG}--device${NC} ${C_ARG}DEVICE${NC}     Device: [cuda, cpu] ${C_COMMENT}(default: ${DEVICE})${NC}
    ${C_FLAG}--checkpoint-dir${NC} ${C_ARG}D${NC}  Checkpoint directory ${C_COMMENT}(default: ${CHECKPOINT_DIR})${NC}
    ${C_FLAG}--sample-dir${NC} ${C_ARG}DIR${NC}    Sample output directory ${C_COMMENT}(default: ${SAMPLE_DIR})${NC}
    ${C_FLAG}--results-dir${NC} ${C_ARG}DIR${NC}   Results directory ${C_COMMENT}(default: ${RESULTS_DIR})${NC}
    ${C_FLAG}--log-dir${NC} ${C_ARG}DIR${NC}       Log directory ${C_COMMENT}(default: ${LOG_DIR})${NC}
    ${C_FLAG}--dry-run${NC}           Print commands without executing
    ${C_FLAG}--verbose${NC}           Verbose output
    ${C_FLAG}--help, -h${NC}          Show this help message

${C_SUBHEAD}EXAMPLES:${NC}
    ${C_COMMENT}# 1. Quick test on MNIST${NC}
    ${C_CMD}./pipeline.sh${NC} ${C_FLAG}--all --dataset${NC} ${C_ARG}mnist${NC} ${C_FLAG}--epochs${NC} ${C_ARG}5${NC}

    ${C_COMMENT}# 2. Full CIFAR-10 training (Serious Run)${NC}
    ${C_CMD}./pipeline.sh${NC} ${C_FLAG}--train --dataset${NC} ${C_ARG}cifar10${NC} ${C_FLAG}--epochs${NC} ${C_ARG}100${NC} ${C_FLAG}--base-channels${NC} ${C_ARG}128${NC}

    ${C_COMMENT}# 3. Sample with StepDrop (Experimental)${NC}
    ${C_CMD}./pipeline.sh${NC} ${C_FLAG}--sample --method${NC} ${C_ARG}stepdrop${NC} ${C_FLAG}--skip-prob${NC} ${C_ARG}0.5${NC} ${C_FLAG}--skip-strategy${NC} ${C_ARG}quadratic${NC}

    ${C_COMMENT}# 4. Run Comprehensive Benchmark${NC}
    ${C_CMD}./pipeline.sh${NC} ${C_FLAG}--evaluate --checkpoint${NC} ${C_ARG}checkpoints/best_model.pt${NC} ${C_FLAG}--eval-samples${NC} ${C_ARG}5000${NC}

    ${C_COMMENT}# 5. Resume from Checkpoint${NC}
    ${C_CMD}./pipeline.sh${NC} ${C_FLAG}--train --resume${NC} ${C_ARG}checkpoints/epoch_20.pt${NC}

${C_BORDER}──────────────────────────────────────────────────────────────────────────────${NC}
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
            --skip-prob)    SKIP_PROB="$2"; shift 2 ;;
            --skip-strategy) SKIP_STRATEGY="$2"; shift 2 ;;
            
            # Evaluation options
            --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
            --eval-batch)   EVAL_BATCH_SIZE="$2"; shift 2 ;;
            --full-metrics) FULL_METRICS=true; shift ;;
            --strategies)   EVAL_STRATEGIES="$2"; shift 2 ;;
            --compare-stepdrop) COMPARE_STEPDROP=true; shift ;;
            --stepdrop-only) STEPDROP_ONLY=true; shift ;;
            
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
    elif [ "$SAMPLE_METHOD" = "stepdrop" ] || [ "$SAMPLE_METHOD" = "adaptive_stepdrop" ]; then
        echo "  Skip Prob:     ${SKIP_PROB}"
        echo "  Skip Strategy: ${SKIP_STRATEGY}"
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
    elif [ "$SAMPLE_METHOD" = "stepdrop" ] || [ "$SAMPLE_METHOD" = "adaptive_stepdrop" ]; then
        SAMPLE_CMD+=" --skip_prob ${SKIP_PROB}"
        SAMPLE_CMD+=" --skip_strategy ${SKIP_STRATEGY}"
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
    
    # Determine which strategies to evaluate
    if [ "$COMPARE_STEPDROP" = true ]; then
        # Compare all StepDrop variants against DDIM baselines
        EVAL_STRATEGIES="DDIM_50,DDIM_25,StepDrop_Linear_0.3,StepDrop_Linear_0.5,StepDrop_CosineSq_0.3,StepDrop_Quadratic_0.3,StepDrop_Adaptive"
        log_info "Mode: Comparing StepDrop strategies against DDIM baselines"
    elif [ "$STEPDROP_ONLY" = true ]; then
        # Only StepDrop strategies
        EVAL_STRATEGIES="StepDrop_Linear_0.3,StepDrop_Linear_0.5,StepDrop_CosineSq_0.3,StepDrop_CosineSq_0.5,StepDrop_Quadratic_0.3,StepDrop_Quadratic_0.5,StepDrop_Adaptive"
        log_info "Mode: Evaluating only StepDrop strategies"
    fi
    
    log_info "Evaluation configuration:"
    echo "  Checkpoint:    ${CHECKPOINT:-DUMMY}"
    echo "  Eval Samples:  ${EVAL_SAMPLES}"
    echo "  Batch Size:    ${EVAL_BATCH_SIZE}"
    echo "  Full Metrics:  ${FULL_METRICS:-false}"
    echo "  Strategies:    ${EVAL_STRATEGIES:-all}"
    
    # Build evaluation command
    EVAL_CMD="python scripts/benchmark_strategies.py"
    EVAL_CMD+=" ${EVAL_MODE}"
    EVAL_CMD+=" --samples ${EVAL_SAMPLES}"
    EVAL_CMD+=" --batch_size ${EVAL_BATCH_SIZE}"
    EVAL_CMD+=" --output_dir ${RESULTS_DIR}"
    
    # Add full metrics flag if set
    if [ "$FULL_METRICS" = true ]; then
        EVAL_CMD+=" --full-metrics"
        log_info "Computing FULL metrics (FID, KID, IS, Precision, Recall, LPIPS, SSIM, etc.)"
    else
        log_info "Computing BASIC metrics (FID, IS, Throughput)"
    fi
    
    # Add specific strategies if set
    if [ -n "$EVAL_STRATEGIES" ] && [ "$EVAL_STRATEGIES" != "all" ]; then
        EVAL_CMD+=" --strategies \"${EVAL_STRATEGIES}\""
    fi
    
    # Run evaluation
    log_info "Running comprehensive benchmark..."
    run_cmd "${EVAL_CMD}"
    
    # Generate plots if results exist
    LATEST_RESULT=$(ls -td ${RESULTS_DIR}/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ] && [ -f "${LATEST_RESULT}report.json" ]; then
        log_info "Generating plots..."
        run_cmd "python scripts/plot_results.py --results ${RESULTS_DIR}"
    fi
    
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