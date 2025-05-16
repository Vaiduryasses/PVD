"""Inference script for PACO model with automatic single/multi-GPU support."""
import torch
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
            
            # Convert to numpy
            generated_pc = generated_pc.cpu().numpy()
            
            # Save output - each rank saves its own outputs
            batch_size = input_pc.shape[0]
            for b in range(batch_size):
                # Calculate global sample index
                sample_idx = i * args.batch_size * world_size + rank * args.batch_size + b
                
                # Save output
                output_path = os.path.join(args.output_dir, f"sample_{sample_idx:04d}.ply")
                save_point_cloud(generated_pc[b], output_path)
                
                # Save input for reference
                input_pc_np = input_pc[b].cpu().numpy()
                input_path = os.path.join(args.output_dir, f"input_{sample_idx:04d}.ply")
                save_point_cloud(input_pc_np, input_path)
                
                # Also save ground truth if available
                if 'target' in data:
                    target_pc = data['target'][b].cpu().numpy()
                    target_path = os.path.join(args.output_dir, f"target_{sample_idx:04d}.ply")
                    save_point_cloud(target_pc, target_path)
    
    # Synchronize before measuring time
    dist.barrier()
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Total inference time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time/len(test_dataset):.4f}s")
    
    # Clean up
    cleanup()

def inference_single_gpu(args):
    """Run inference on a single GPU."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    print(f"Loaded model from {args.checkpoint}, epoch {checkpoint['epoch']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = PointCloudDataset(
        data_path=cfg.data_path, 
        split='test',
        max_points=cfg.max_points
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Start timing
    start_time = time.time()
    
    # Run inference
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
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
            
            # Convert to numpy
            generated_pc = generated_pc.cpu().numpy()
            
            # Save output
            batch_size = input_pc.shape[0]
            for b in range(batch_size):
                sample_idx = i * args.batch_size + b
                
                # Save output
                output_path = os.path.join(args.output_dir, f"sample_{sample_idx:04d}.ply")
                save_point_cloud(generated_pc[b], output_path)
                
                # Save input for reference
                input_pc_np = input_pc[b].cpu().numpy()
                input_path = os.path.join(args.output_dir, f"input_{sample_idx:04d}.ply")
                save_point_cloud(input_pc_np, input_path)
                
                # Also save ground truth if available
                if 'target' in data:
                    target_pc = data['target'][b].cpu().numpy()
                    target_path = os.path.join(args.output_dir, f"target_{sample_idx:04d}.ply")
                    save_point_cloud(target_pc, target_path)
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time/len(test_dataset):.4f}s")

def inference(args):
    """Main inference function with automatic single/multi-GPU support."""
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1 and not args.force_single_gpu:
        print(f"Running inference with {num_gpus} GPUs")
        # Launch distributed processes
        mp.spawn(
            inference_distributed,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True
        )
    else:
        if args.force_single_gpu:
            print("Forced single GPU operation")
        else:
            print("Running inference with single GPU")
        # Fall back to single device inference
        inference_single_gpu(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with PACO model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to config file")
    parser.add_argument("--use_ddim", action="store_true", 
                       help="Use DDIM for faster sampling")
    parser.add_argument("--force_single_gpu", action="store_true",
                       help="Force single GPU operation even if multiple GPUs are available")
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        cfg = get_config().from_yaml(args.config)
    
    # Run inference
    inference(args)