"""
PVD: Parameter Voxel Diffusion Model
Main pipeline for point cloud processing and generation
"""
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from .models.diffusion import DiffusionModel
from .models.voxel_transformer import SparseVoxelTransformer
from .models.preprocessing import process_point_cloud, process_batch_parallel
from .config import cfg
from .utils.logger import setup_logger

logger = setup_logger(__name__)

class PVD:
    """
    Pipeline for Point Cloud Processing and Generation with 
    Parameter Voxel Diffusion model.
    """
    
    def __init__(self, 
                 model=None,
                 config_path=None,
                 checkpoint_path=None,
                 device=None):
        """
        Initialize the PVD pipeline.
        
        Args:
            model: Pre-initialized model (optional)
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to run the model on
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Load configuration if provided
        if config_path is not None:
            cfg.merge_from_file(config_path)
        
        # Create or use provided model
        if model is not None:
            self.model = model
        else:
            from .models.model import PVD as PVDModel
            self.model = PVDModel(
                transformer_dim=cfg.transformer_dim,
                transformer_depth=cfg.transformer_depth,
                transformer_heads=cfg.transformer_heads,
                transformer_window_size=cfg.transformer_window_size,
                dropout=0.0  # No dropout for inference
            ).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode by default
        self.model.eval()
        
        logger.info(f"PVD pipeline initialized on {self.device}")
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            logger.info(f"Loaded checkpoint from epoch {epoch}")
        else:
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint (format without epoch info)")
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
        return self
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def process(self, points, return_features=False):
        """
        Process point cloud through the transformer.
        
        Args:
            points: (B, N, 3) tensor of point clouds
            return_features: Whether to return transformer features
            
        Returns:
            features: Transformer features if return_features=True
        """
        # Ensure input is on the correct device
        if isinstance(points, torch.Tensor):
            points = points.to(self.device)
        else:
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Add batch dimension if missing
        if points.dim() == 2:
            points = points.unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.model.extract_features(points)
        
        if return_features:
            return features
        
    def generate(self, points=None, num_points=2048, features=None, use_ddim=True, steps=50):
        """
        Generate point cloud using the diffusion model.
        
        Args:
            points: (B, N, 3) tensor of input point clouds (optional)
            num_points: Number of points to generate
            features: Pre-computed features (optional)
            use_ddim: Whether to use DDIM sampling
            steps: Number of diffusion steps
            
        Returns:
            generated_points: (B, num_points, 3) tensor of generated point clouds
        """
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Process input points if provided
        if features is None and points is not None:
            # Ensure input is on the correct device
            if isinstance(points, torch.Tensor):
                points = points.to(self.device)
            else:
                points = torch.tensor(points, dtype=torch.float32, device=self.device)
            
            # Add batch dimension if missing
            if points.dim() == 2:
                points = points.unsqueeze(0)
            
            # Extract features
            features = self.process(points, return_features=True)
        
        # Handle the case when neither points nor features are provided
        if features is None:
            raise ValueError("Either points or features must be provided")
        
        # Generate point cloud
        with torch.no_grad():
            # Ensure all features have same shape for batching
            max_features = max([f.shape[0] for f in features])
            batch_size = len(features)
            
            # Pad features to same size
            padded_features = []
            for i, feats in enumerate(features):
                if feats.shape[0] < max_features:
                    padding = torch.zeros(
                        max_features - feats.shape[0], 
                        feats.shape[1], 
                        device=feats.device
                    )
                    padded_features.append(torch.cat([feats, padding], dim=0))
                else:
                    padded_features.append(feats)
                    
            # Stack into batch
            context = torch.stack(padded_features, dim=0)
            
            # Generate point cloud
            if use_ddim:
                generated_points = self.model.diffusion.sample_with_ddim(
                    context, num_points, steps, eta=0.0)
            else:
                generated_points = self.model.diffusion.sample(
                    context, num_points)
        
        return generated_points
    
    def reconstruct(self, points, num_points=None):
        """
        Reconstruct point cloud (identity mapping for training).
        
        Args:
            points: (B, N, 3) tensor of input point clouds
            num_points: Number of points in output (defaults to same as input)
            
        Returns:
            reconstructed_points: (B, num_points, 3) tensor of reconstructed point clouds
        """
        # Ensure input is on the correct device
        if isinstance(points, torch.Tensor):
            points = points.to(self.device)
        else:
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Add batch dimension if missing
        if points.dim() == 2:
            points = points.unsqueeze(0)
        
        # Set default num_points if not provided
        if num_points is None:
            num_points = points.shape[1]
        
        # Process through model
        with torch.no_grad():
            features = self.process(points, return_features=True)
            reconstructed = self.generate(features=features, num_points=num_points)
        
        return reconstructed
    
    def train_step(self, points, target_points, optimizer):
        """
        Perform a single training step.
        
        Args:
            points: (B, N, 3) tensor of input point clouds
            target_points: (B, M, 3) tensor of target point clouds
            optimizer: PyTorch optimizer
            
        Returns:
            loss: Training loss
        """
        # Ensure inputs are on the correct device
        if isinstance(points, torch.Tensor):
            points = points.to(self.device)
        else:
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        if isinstance(target_points, torch.Tensor):
            target_points = target_points.to(self.device)
        else:
            target_points = torch.tensor(target_points, dtype=torch.float32, device=self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = self.model(points, target_points, mode='train')
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path, optimizer=None, epoch=None, loss=None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            epoch: Optional epoch number
            loss: Optional loss value
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        if loss is not None:
            checkpoint['loss'] = loss
        
        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")