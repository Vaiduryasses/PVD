#!/bin/bash
# Testing script for PVD model

# Default parameters
CONFIG="configs/default.yaml"
CHECKPOINT="results/checkpoints/best_model.pt"
OUTPUT_DIR="results/test"
BATCH_SIZE=4
GPU_IDS="0"
USE_DDIM=true
STEPS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --gpus)
      GPU_IDS="$2"
      shift
      shift
      ;;
    --use_ddim)
      USE_DDIM="$2"
      shift
      shift
      ;;
    --steps)
      STEPS="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Create output directory
mkdir -p $OUTPUT_DIR

# Run testing
DDIM_FLAG=""
if [ "$USE_DDIM" = true ]; then
  DDIM_FLAG="--use_ddim"
fi

python -m pvd.inference --config $CONFIG \
                         --checkpoint $CHECKPOINT \
                         --output_dir $OUTPUT_DIR \
                         --batch_size $BATCH_SIZE \
                         $DDIM_FLAG \
                         --steps $STEPS

echo "Testing completed! Results saved to $OUTPUT_DIR"