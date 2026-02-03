#!/bin/bash
#
# LangOverrule 실험 실행 스크립트
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Environment setup
setup_environment() {
    echo -e "${GREEN}=== Setting up environment ===${NC}"

    # Activate venv310
    if [ -f "/mnt/fr20tb/wbl_residency/jos/.venv310/bin/activate" ]; then
        source /mnt/fr20tb/wbl_residency/jos/.venv310/bin/activate
    fi

    # Set environment variables
    export BASE_CACHE_DIR="/mnt/tmp/cache"
    export HF_HOME="$BASE_CACHE_DIR/hf"
    export TORCH_HOME="$BASE_CACHE_DIR/torch"

    # Completely disable TensorFlow
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export TRANSFORMERS_NO_TF=1
    export USE_TF=0

    # CUDA settings - Use all 8 A100 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    mkdir -p "$HF_HOME" "$TORCH_HOME" results logs

    echo "Python: $(python --version 2>&1)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
}

# Run tests
run_tests() {
    echo -e "${GREEN}=== Running Tests ===${NC}"
    python test_mdi.py 2>&1 | tee "$LOG_DIR/test_mdi_$TIMESTAMP.log"
    python compare_with_paper.py 2>&1 | tee "$LOG_DIR/compare_paper_$TIMESTAMP.log"
}

# Run experiments
run_experiment() {
    local model=$1
    local num_samples=${2:-10}
    local use_real=${3:-false}

    echo -e "${GREEN}=== Running $model Experiment ===${NC}"
    
    local dataset_flag=""
    [ "$use_real" = "true" ] && dataset_flag="--use-real-dataset"

    python run_real_experiments.py \
        --model "$model" \
        --num-samples "$num_samples" \
        $dataset_flag \
        2>&1 | tee "$LOG_DIR/${model}_${TIMESTAMP}.log"
}

# Main
main() {
    local command=${1:-help}
    
    setup_environment

    case "$command" in
        test)
            run_tests
            ;;
        llava|qwen|both)
            run_experiment "$command" "${2:-10}" "${3:-false}"
            ;;
        *)
            echo "Usage: ./run.sh [test|llava|qwen|both] [num_samples] [real]"
            exit 1
            ;;
    esac
}

main "$@"
