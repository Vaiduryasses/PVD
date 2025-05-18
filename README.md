# PVD: Parameter Voxel Diffusion Model

PVD is a deep learning framework for point cloud generation and processing using a voxel-based diffusion model. This model enables high-quality point cloud synthesis and reconstruction through a novel parameter-based voxel representation combined with a diffusion model.

## Features

- ✨ Fast and accurate point cloud generation
- 🚀 GPU-accelerated processing pipeline
- 📊 Voxel-based preprocessing for efficient representation
- 🔄 Diffusion-based generative modeling
- 🌐 Support for large-scale point cloud datasets
- 📈 Comprehensive evaluation metrics

## Installation

### Requirements

- Python 3.8+
- CUDA 11.6+ (for GPU acceleration)
- PyTorch 1.9+

### Setup

The easiest way to set up PVD is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/Vaiduryasses/PVD.git
cd PVD

# Run setup script
bash setup.sh
```

Alternatively, you can set up manually:

```bash
# Create virtual environment
python -m venv pvd_env
source pvd_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Training

Train a model from scratch:

```bash
python -m pvd.train --config configs/default.yaml --output_dir results/
```

Resume training from a checkpoint:

```bash
python -m pvd.train --config configs/default.yaml --checkpoint results/checkpoints/best_model.pt --output_dir results/
```

Alternatively, use the provided training script:

```bash
bash scripts/train.sh --config configs/default.yaml --output_dir results/ --batch_size 8 --lr 0.0001 --max_epochs 100 --gpus 0
```

### Testing

Basic test with new metrics:

```bash
python -m pvd.test --config configs/test_config.yaml --checkpoint checkpoints/best_model.pt --output_dir results
```

Specify failure threshold:

```bash
python -m pvd.test --config configs/test_config.yaml --checkpoint checkpoints/best_model.pt --output_dir results --failure_threshold 0.05
```

Visualize test results:

```bash
python -m pvd.test --config configs/test_config.yaml --checkpoint checkpoints/best_model.pt --output_dir results --visualize --save_vis
```

Use specific GPUs:

```bash
python -m pvd.test --config configs/test_config.yaml --checkpoint checkpoints/best_model.pt --gpu_ids 0 1 --batch_size 2
```

### Inference

Generate point clouds from input:

```bash
python -m pvd.inference --checkpoint results/checkpoints/best_model.pt --input_dir data/test/ --output_dir results/inference/
```

With DDIM sampling for faster generation:

```bash
python -m pvd.inference --checkpoint results/checkpoints/best_model.pt --input_dir data/test/ --output_dir results/inference/ --use_ddim --steps 50
```

Or use the provided testing script:

```bash
bash scripts/test.sh --checkpoint results/checkpoints/best_model.pt --output_dir results/test/ --batch_size 4 --use_ddim true --steps 50
```

### Evaluation

Evaluate a trained model:

```bash
python -m pvd.evaluate --checkpoint results/checkpoints/best_model.pt --data_dir data/test/
```

With the provided evaluation script:

```bash
bash scripts/evaluate.sh --checkpoint results/checkpoints/best_model.pt --data_dir data/test/ --output_dir results/evaluation/
```

## Configuration

PVD uses YAML configuration files. The main configuration options include:

```yaml
# Model Configuration
model:
  transformer_dim: 256
  transformer_depth: 8
  transformer_heads: 8
  transformer_window_size: 3
  diffusion_dim: 256
  diffusion_encoder_layers: 6
  diffusion_decoder_layers: 4

# Training Configuration
training:
  batch_size: 8
  learning_rate: 1e-4
  max_epochs: 100
  early_stopping_patience: 10
  save_frequency: 10
  
# Diffusion Parameters
diffusion:
  beta_start: 1e-4
  beta_end: 0.02
  diffusion_steps_train: 1000
  diffusion_steps_infer: 50
  
# Data Configuration
data:
  dataset: shapenet
  data_path: data/shapenet
  max_points: 2048
  output_points: 2048
  train_split: 0.8
  val_split: 0.1
```

## Code Examples

### Basic Point Cloud Generation

```python
import torch
from pvd.pipeline import PVD

# Initialize the pipeline
pipeline = PVD(checkpoint_path="path/to/checkpoint.pt")

# Load input point cloud
input_points = torch.randn(1, 1024, 3)

# Generate output point cloud
output_points = pipeline.generate(points=input_points, num_points=2048)
```

### Feature Extraction

```python
import torch
from pvd.pipeline import PVD

# Initialize the pipeline
pipeline = PVD(checkpoint_path="path/to/checkpoint.pt")

# Load input point cloud
input_points = torch.randn(1, 1024, 3)

# Extract features
features = pipeline.process(points=input_points, return_features=True)
```

### Model Training Loop

```python
import torch
from pvd.pipeline import PVD
from pvd.data.dataset import PointCloudDataset
from torch.utils.data import DataLoader

# Initialize the pipeline
pipeline = PVD()

# Create dataset and dataloader
dataset = PointCloudDataset("path/to/data")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Setup optimizer
optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        input_points = batch["input"]
        target_points = batch["target"]
        
        # Perform training step
        loss = pipeline.train_step(input_points, target_points, optimizer)
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    # Save checkpoint
    pipeline.save_checkpoint(f"checkpoints/model_epoch_{epoch}.pt", optimizer, epoch)
```

### Batch Processing for Multiple Point Clouds

```python
import torch
from pvd.pipeline import PVD

# Initialize the pipeline
pipeline = PVD(checkpoint_path="path/to/checkpoint.pt")

# Batch of input point clouds
batch_points = torch.randn(8, 1024, 3)  # 8 point clouds, each with 1024 points

# Process the entire batch
batch_output = pipeline.generate(points=batch_points, num_points=2048)

# batch_output.shape = (8, 2048, 3)
```

## License

MIT License

---

Last updated: 2025-05-16 by Vaiduryasses