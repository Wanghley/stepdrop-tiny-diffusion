#!/bin/bash
# =============================================================================
# Quick Start Menu for StepDrop Diffusion
# =============================================================================
# An interactive menu for common workflows.
#
# Usage:
#   ./scripts/quick_start.sh           # Interactive menu
#   ./scripts/quick_start.sh 1         # Run option 1 directly
# =============================================================================

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
# Colors
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
PURPLE=$'\033[0;35m'
CYAN=$'\033[0;36m'
GREY=$'\033[0;90m'
WHITE=$'\033[0;37m'
NC=$'\033[0m'

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

# Semantic Colors
C_HEADER=${B_PURPLE}
C_SUBHEAD=${B_CYAN}
C_FLAG=${GREEN}
C_ARG=${YELLOW}
C_CMD=${B_WHITE}
C_COMMENT=${GREY}
C_BORDER=${BLUE}

# -----------------------------------------------------------------------------
# Menu Display
# -----------------------------------------------------------------------------
show_menu() {
    clear
    echo -e "${C_BORDER}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${C_BORDER}║${NC}           ${C_HEADER}STEPDROP QUICK START MENU${NC}                        ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}  ${C_SUBHEAD}TRAINING${NC}                                                  ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}1)${NC} Quick Test (MNIST, 5 epochs)                         ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}2)${NC} Train MNIST (20 epochs)                              ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}3)${NC} Train CIFAR-10 (50 epochs)                           ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}4)${NC} Train CIFAR-10 Full (100 epochs, larger model)       ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}  ${C_SUBHEAD}SAMPLING${NC}                                                  ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}5)${NC} Sample DDIM (50 steps, fast)                         ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}6)${NC} Sample DDIM (25 steps, faster)                       ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}7)${NC} Sample DDPM (1000 steps, best quality)               ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}  ${C_SUBHEAD}EVALUATION${NC}                                                ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}8)${NC} Run Benchmark (1000 samples)                         ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}9)${NC} Run Full Benchmark (5000 samples)                    ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}  ${C_SUBHEAD}FULL PIPELINE${NC}                                             ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}10)${NC} Full Pipeline - MNIST                                ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}11)${NC} Full Pipeline - CIFAR-10                             ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}  ${C_SUBHEAD}UTILITIES${NC}                                                 ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}12)${NC} Check Environment                                    ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}13)${NC} Dry Run (Show Commands)                              ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}14)${NC} Clean All Outputs                                    ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}║${NC}   ${C_CMD}15)${NC} Show Help                                            ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${C_BORDER}║${NC}    ${C_CMD}0)${NC} Exit                                                 ${C_BORDER}║${NC}"
    echo -e "${C_BORDER}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -n -e "Select option [${C_CMD}0-15${NC}]: "
    read choice
    echo ""
}

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
check_environment() {
    echo -e "${CYAN}Checking environment...${NC}"
    echo ""
    
    # Python
    echo -n "Python:          "
    python --version 2>/dev/null || echo -e "${RED}NOT FOUND${NC}"
    
    # PyTorch
    echo -n "PyTorch:         "
    python -c "import torch; print(torch.__version__)" 2>/dev/null || echo -e "${RED}NOT FOUND${NC}"
    
    # CUDA
    echo -n "CUDA Available:  "
    python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "Unknown"
    
    # GPU
    echo -n "GPU:             "
    python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A"
    
    # Memory
    echo -n "GPU Memory:      "
    python -c "
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'{mem:.1f} GB')
else:
    print('N/A')
" 2>/dev/null || echo "N/A"
    
    echo ""
    echo -e "${CYAN}Checking required packages...${NC}"
    python -c "
packages = ['torch', 'torchvision', 'numpy', 'PIL', 'matplotlib', 'tqdm']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} - NOT INSTALLED')
"
    
    echo ""
    echo -e "${CYAN}Checking project files...${NC}"
    [ -f "src/train.py" ] && echo "  ✅ src/train.py" || echo "  ❌ src/train.py"
    [ -f "src/sample.py" ] && echo "  ✅ src/sample.py" || echo "  ❌ src/sample.py"
    [ -f "src/modules.py" ] && echo "  ✅ src/modules.py" || echo "  ❌ src/modules.py"
    [ -f "scripts/pipeline.sh" ] && echo "  ✅ scripts/pipeline.sh" || echo "  ❌ scripts/pipeline.sh"
    
    echo ""
    echo -e "${CYAN}Checkpoints found:${NC}"
    ls -la checkpoints/*.pt 2>/dev/null || echo "  (none)"
}

# -----------------------------------------------------------------------------
# Run Option
# -----------------------------------------------------------------------------
run_option() {
    case $1 in
        1)
            echo -e "${GREEN}Running Quick Test (MNIST, 5 epochs)...${NC}"
            scripts/pipeline.sh --all --dataset mnist --epochs 5 --n-samples 16 --eval-samples 100
            ;;
        2)
            echo -e "${GREEN}Training on MNIST (20 epochs)...${NC}"
            scripts/pipeline.sh --train --dataset mnist --epochs 20
            ;;
        3)
            echo -e "${GREEN}Training on CIFAR-10 (50 epochs)...${NC}"
            scripts/pipeline.sh --train --dataset cifar10 --epochs 50
            ;;
        4)
            echo -e "${GREEN}Training on CIFAR-10 Full (100 epochs)...${NC}"
            scripts/pipeline.sh --train --dataset cifar10 --epochs 100 --base-channels 128
            ;;
        5)
            echo -e "${GREEN}Sampling with DDIM (50 steps)...${NC}"
            scripts/pipeline.sh --sample --n-samples 64 --method ddim --ddim-steps 50
            ;;
        6)
            echo -e "${GREEN}Sampling with DDIM (25 steps)...${NC}"
            scripts/pipeline.sh --sample --n-samples 64 --method ddim --ddim-steps 25
            ;;
        7)
            echo -e "${GREEN}Sampling with DDPM (1000 steps)...${NC}"
            scripts/pipeline.sh --sample --n-samples 16 --method ddpm
            ;;
        8)
            echo -e "${GREEN}Running Benchmark (1000 samples)...${NC}"
            scripts/pipeline.sh --evaluate --eval-samples 1000
            ;;
        9)
            echo -e "${GREEN}Running Full Benchmark (5000 samples)...${NC}"
            scripts/pipeline.sh --evaluate --eval-samples 5000
            ;;
        10)
            echo -e "${GREEN}Running Full Pipeline on MNIST...${NC}"
            scripts/pipeline.sh --all --dataset mnist --epochs 20 --eval-samples 500
            ;;
        11)
            echo -e "${GREEN}Running Full Pipeline on CIFAR-10...${NC}"
            scripts/pipeline.sh --all --dataset cifar10 --epochs 50 --eval-samples 1000
            ;;
        12)
            check_environment
            ;;
        13)
            echo -e "${GREEN}Dry Run (showing commands)...${NC}"
            scripts/pipeline.sh --all --dataset cifar10 --epochs 10 --dry-run
            ;;
        14)
            echo -e "${YELLOW}Cleaning all outputs...${NC}"
            scripts/pipeline.sh --clean
            ;;
        15|--help|-h)
            scripts/pipeline.sh --help
            ;;
        0)
            echo -e "${GREEN}Goodbye! 👋${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option: $1${NC}"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    # Check if pipeline.sh exists
    if [ ! -f "scripts/pipeline.sh" ]; then
        echo -e "${RED}Error: scripts/pipeline.sh not found!${NC}"
        echo "Please run this script from the project root directory or scripts directory."
        exit 1
    fi
    
    # Make sure pipeline is executable
    chmod +x scripts/pipeline.sh 2>/dev/null || true
    
    if [ $# -gt 0 ]; then
        # Run with command line argument
        run_option "$1"
    else
        # Interactive menu loop
        while true; do
            show_menu
            run_option "$choice"
            echo ""
            echo -e "${BLUE}Press Enter to continue...${NC}"
            read
        done
    fi
}

main "$@"
