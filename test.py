"""
PVD Model Testing Script
Test the performance of PVD model using 6-step DDIM denoising with
specialized metrics: chamfer distance, Hausdorff Distance, Normal Consistency, and Construction Failure Rate
"""
import os
import torch
import numpy as np
import argparse
import time
import yaml
from tqdm import tqdm
from datetime import datetime
import logging
import open3d as o3d
from torch.utils.data import DataLoader

from ..models.pipeline import PVD
from ..data.dataset import PointCloudDataset
from ..metrics.evaluation import (
    chamfer_distance, 
    hausdorff_distance,
    normal_consistency,
    construction_failure_rate
)
from ..config import cfg, load_config
from ..utils.visualization import (
    visualize_point_cloud, 
    visualize_diffusion_steps,
    save_point_cloud
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test PVD Model")
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save results')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='Specify GPU IDs to use (e.g. --gpu_ids 0 1 2)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--vis_diffusion_steps', action='store_true',
                        help='Visualize diffusion steps')
    parser.add_argument('--test_samples', type=int, default=None,
                        help='Number of test samples, None means all')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--failure_threshold', type=float, default=0.1,
                        help='Threshold for construction failure rate')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(checkpoint_path, gpu_ids):
    """Load model and checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize model with specified GPUs
    model = PVD(gpu_ids=gpu_ids)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_ids[0]}' if gpu_ids and torch.cuda.is_available() else 'cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    
    # Set to evaluation mode
    model.eval()
    
    return model

def prepare_data(batch_size, num_workers, test_samples=None):
    """Prepare test dataset and dataloader"""
    logger.info("Preparing test data...")
    
    # Create test dataset
    test_dataset = PointCloudDataset(
        data_path=cfg.data.test_path,
        split='test',
        voxel_size=cfg.voxel.resolution,
        transform=None,
        cache=True
    )
    
    if test_samples is not None:
        # If test samples specified, use only a subset
        test_indices = np.random.choice(len(test_dataset), min(test_samples, len(test_dataset)), replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def evaluate_model(model, test_loader, output_dir, visualize=False, save_vis=False, vis_diffusion_steps=False, failure_threshold=0.1):
    """Evaluate model performance"""
    logger.info("Starting model evaluation...")
    
    # Create result directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Initialize evaluation metrics
    chamfer_distances = []
    hausdorff_distances = []
    normal_consistencies = []
    failure_rates = []
    inference_times = []
    
    # Track failures for construction failure rate
    total_samples = 0
    failed_samples = 0
    
    # Evaluation loop
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Move data to device
        points = batch['points'].to(device)
        voxel_features = batch['voxel_features'].to(device)
        voxel_coords = batch['voxel_coords'].to(device)
        
        # Record inference time
        start_time = time.time()
        
        # Use torch.no_grad() for inference to reduce memory usage
        with torch.no_grad():
            if vis_diffusion_steps:
                # Get intermediate diffusion steps
                outputs = model.visualize_intermediate(points, voxel_features, voxel_coords)
                final_points = outputs['final_points']
                intermediates = outputs.get('diffusion_intermediates', [])
            else:
                # Normal inference
                final_points, proxy_outputs = model.inference(points, voxel_features, voxel_coords)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Calculate evaluation metrics
        cd = chamfer_distance(final_points, points).cpu().numpy()
        hd = hausdorff_distance(final_points, points).cpu().numpy()
        nc = normal_consistency(final_points, points).cpu().numpy()
        
        # Check for construction failures
        # A failure is defined as having a Chamfer distance above the threshold
        batch_size = points.shape[0]
        total_samples += batch_size
        batch_failures = (cd > failure_threshold).sum().item()
        failed_samples += batch_failures
        
        # Store metrics
        chamfer_distances.append(cd)
        hausdorff_distances.append(hd)
        normal_consistencies.append(nc)
        
        # Visualize and save results
        if visualize or save_vis:
            for i in range(points.shape[0]):
                input_pc = points[i].cpu().numpy()
                output_pc = final_points[i].cpu().numpy()
                
                # Generate unique ID
                sample_id = batch_idx * test_loader.batch_size + i
                
                if visualize:
                    visualize_point_cloud(input_pc, output_pc, f"Sample {sample_id}")
                
                if save_vis:
                    # Save point clouds
                    input_path = os.path.join(vis_dir, f"input_{sample_id}.ply")
                    output_path = os.path.join(vis_dir, f"output_{sample_id}.ply")
                    save_point_cloud(input_pc, input_path)
                    save_point_cloud(output_pc, output_path)
                    
                    # If visualizing diffusion steps, save intermediate results
                    if vis_diffusion_steps and intermediates:
                        steps_dir = os.path.join(vis_dir, f"diffusion_steps_{sample_id}")
                        os.makedirs(steps_dir, exist_ok=True)
                        
                        for step_idx, step_points in enumerate(intermediates):
                            step_pc = step_points[i].cpu().numpy()
                            step_path = os.path.join(steps_dir, f"step_{step_idx}.ply")
                            save_point_cloud(step_pc, step_path)
                        
                        # Create diffusion steps visualization GIF
                        visualize_diffusion_steps(
                            [step[i].cpu().numpy() for step in intermediates],
                            os.path.join(steps_dir, "diffusion_process.gif")
                        )
    
    # Calculate average metrics
    avg_cd = np.mean(chamfer_distances)
    avg_hd = np.mean(hausdorff_distances)
    avg_nc = np.mean(normal_consistencies)
    failure_rate = failed_samples / total_samples if total_samples > 0 else 0
    avg_time = np.mean(inference_times)
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"Average Chamfer Distance: {avg_cd:.6f}")
    logger.info(f"Average Hausdorff Distance: {avg_hd:.6f}")
    logger.info(f"Average Normal Consistency: {avg_nc:.6f}")
    logger.info(f"Construction Failure Rate: {failure_rate:.6f} ({failed_samples}/{total_samples})")
    logger.info(f"Average Inference Time: {avg_time:.4f} seconds")
    
    # Save evaluation results
    results = {
        'chamfer_distance': {
            'mean': float(avg_cd),
            'std': float(np.std(chamfer_distances)),
            'min': float(np.min(chamfer_distances)),
            'max': float(np.max(chamfer_distances))
        },
        'hausdorff_distance': {
            'mean': float(avg_hd),
            'std': float(np.std(hausdorff_distances)),
            'min': float(np.min(hausdorff_distances)),
            'max': float(np.max(hausdorff_distances))
        },
        'normal_consistency': {
            'mean': float(avg_nc),
            'std': float(np.std(normal_consistencies)),
            'min': float(np.min(normal_consistencies)),
            'max': float(np.max(normal_consistencies))
        },
        'construction_failure_rate': {
            'rate': float(failure_rate),
            'failed_samples': int(failed_samples),
            'total_samples': int(total_samples),
            'threshold': float(failure_threshold)
        },
        'inference_time': {
            'mean': float(avg_time),
            'std': float(np.std(inference_times)),
            'min': float(np.min(inference_times)),
            'max': float(np.max(inference_times))
        },
        'test_samples': len(test_loader.dataset),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': cfg.model.__dict__ if hasattr(cfg.model, '__dict__') else str(cfg.model),
        'diffusion_steps': 6,  # 6-step DDIM
    }
    
    # Save results to YAML file
    with open(os.path.join(output_dir, 'evaluation_results.yaml'), 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    return results

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Load model
    model = load_model(args.checkpoint, args.gpu_ids)
    
    # Prepare data
    test_loader = prepare_data(args.batch_size, args.num_workers, args.test_samples)
    
    # Evaluate model
    results = evaluate_model(
        model,
        test_loader,
        args.output_dir,
        visualize=args.visualize,
        save_vis=args.save_vis,
        vis_diffusion_steps=args.vis_diffusion_steps,
        failure_threshold=args.failure_threshold
    )
    
    logger.info(f"Testing completed! Results saved to {args.output_dir}")
    
    return results

if __name__ == "__main__":
    main()