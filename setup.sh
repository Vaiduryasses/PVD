#!/bin/bash
# Setup script for PVD (Parameter Voxel Diffusion model)

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv pvd_env
source pvd_env/bin/activate

# Install PyTorch with CUDA support
# Change cuda version as needed (options: 11.8, 11.7, 11.6, etc.)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Build CUDA extensions if available
echo "Building CUDA extensions..."
cd pvd/utils/cuda_ext
python setup.py install
cd ../../..

# Install the package in development mode
echo "Installing PVD package..."
pip install -e .

echo "Setup complete! Activate the environment with: source pvd_env/bin/activate"