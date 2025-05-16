"""Parallel inference script for PACO model."""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from .models.model import PacoModel
from .dataset import PointCloudDataset
from .config import cfg, get_config

def save_point_cloud(points, filepath):
    """Save point cloud to file."""
    # Simple PLY format
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def setup(rank, world_size):
    """Initialize distributed inference environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def inference_distributed(rank, world_size, args):
    """Run inference using trained model with distributed processing."""
    # Setup distributed environment
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Load model
    model = PacoModel(
        transformer_dim=cfg.transformer_dim,
        transformer_depth=cfg.transformer_depth,
        transformer_heads=cfg.transformer_heads,
        transformer_window_size=cfg.transformer_window_size,
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if rank == 0:
        print(f"Loaded model from {args.checkpoint}, epoch {checkpoint['epoch']}")
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = PointCloudDataset(
        data_path=cfg.data_path, 
        split='test',
        max_points=cfg.max_points
    )
    
    # Create distributed sampler
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Sync before starting timing
    dist.barrier()
    start_time = time.time()
    
    # Run inference
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=rank != 0)):
            # Get input point cloud
            input_pc = data['input'].to(device)
            
            # Generate point cloud
            if args.use_ddim:
                generated_pc = model.sample_with_ddim(
                    input_pc, 
                    num_points=cfg.output_points,
                    steps=cfg.diffusion_steps_infer,
                    eta=0.0
                )
            else:
                generated_pc = model.sample(
                    input_pc, 
                    num_points=cfg.output_points
                )
            
            #