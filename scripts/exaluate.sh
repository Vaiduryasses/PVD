#!/bin/bash
# Evaluation script for PVD model

# Default parameters
CONFIG="configs/default.yaml"
CHECKPOINT="results/checkpoints/best_model.pt"
DATA_DIR="data/test"
OUTPUT_DIR="results/evaluation"
BATCH_SIZE=4
GPU_IDS="0"

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
    --data_dir)
      DATA_DIR="$2"
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
    *)
      shift
      ;;
  esac
done

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python -m pvd.evaluate --config $CONFIG \
                         --checkpoint $CHECKPOINT \
                         --data_dir $DATA_DIR \
                         --output_dir $OUTPUT_DIR \
                         --batch_size $BATCH_SIZE

echo "Evaluation completed! Results saved to $OUTPUT_DIR/metrics.json"