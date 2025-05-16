"""Complete Point Cloud Processing Model with GPU parallelization."""
import torch
import torch.nn as nn
from .preprocessing import process_point_cloud, process_batch_parallel
from .voxel_transformer import SparseVoxelTransformer
from .diffusion import DiffusionModel
from ..config import cfg

class PacoModel(nn.Module):
    """Complete model pipeline including preprocessing, transformer, and diffusion."""
    
    def __init__(self, 
                 transformer_dim=256, 
                 transformer_depth=8, 
                 transformer_heads=8,
                 transformer_window_size=3,
                 dropout=0.0):
        super().__init__()
        
        # Sparse Voxel Transformer for feature extraction
        self.transformer = SparseVoxelTransformer(
            in_dim=35,  # 32(from MLP expansion of 8D features) + 3(centroid)
            dim=transformer_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            window_size=transformer_window_size,
            dropout=dropout
        )
        
        # Diffusion model for point cloud generation
        self.diffusion = DiffusionModel(
            in_dim=3,  # x,y,z coordinates
            context_dim=transformer_dim,
            diffusion_dim=cfg.diffusion_dim,
            encoder_depth=cfg.diffusion_encoder_layers,
            decoder_depth=cfg.diffusion_decoder_layers,
            dropout=dropout
        )
        
    def extract_features_parallel(self, points, features=None):
        """
        Extract features from input point cloud in parallel.
        
        Args:
            points: (B, N, 3) - Batch of point clouds
            features: (B, N, F) - Optional point features
            
        Returns:
            transformed_features: List of (V_i, C) features for each example in batch
        """
        batch_size = points.shape[0]
        
        # Process batch in parallel
        all_coords, all_features = process_batch_parallel(points, features)
        
        # Track batch indices
        all_batch_indices = []
        for i in range(batch_size):
            all_batch_indices.append(torch.full((all_coords[i].shape[0],), i, device=points.device))
        
        # Concatenate results
        if batch_size > 0:
            all_coords_cat = torch.cat(all_coords, dim=0)
            all_features_cat = torch.cat(all_features, dim=0)
            all_batch_indices_cat = torch.cat(all_batch_indices, dim=0)
        else:
            # Handle empty batch case
            all_coords_cat = torch.zeros((0, 3), device=points.device)
            all_features_cat = torch.zeros((0, 35), device=points.device)  # 32 + 3
            all_batch_indices_cat = torch.zeros((0,), device=points.device)
        
        # Apply transformer
        transformed_features = self.transformer(all_features_cat, all_coords_cat)
        
        # Split results by batch index
        result = []
        for i in range(batch_size):
            mask = (all_batch_indices_cat == i)
            result.append(transformed_features[mask])
        
        return result
    
    def extract_features(self, points, features=None):
        """
        Extract features from input point cloud.
        
        Args:
            points: (B, N, 3) - Batch of point clouds
            features: (B, N, F) - Optional point features
            
        Returns:
            transformed_features: List of (V_i, C) features for each example in batch
        """
        # Use parallel processing for batches
        if points.shape[0] > 1:
            return self.extract_features_parallel(points, features)
        
        # Otherwise, process sequentially for single examples
        batch_size = points.shape[0]
        
        # Process each example in batch
        all_coords = []
        all_features = []
        all_batch_indices = []
        
        for i in range(batch_size):
            # Get points and features for this example
            example_points = points[i]
            example_features = features[i] if features is not None else None
            
            # Process input using preprocessing pipeline
            coords, processed_features = process_point_cloud(example_points, example_features)
            
            # Save results
            all_coords.append(coords)
            all_features.append(processed_features)
            all_batch_indices.append(torch.full((coords.shape[0],), i, device=points.device))
        
        # Concatenate results
        if batch_size > 0:
            all_coords = torch.cat(all_coords, dim=0)
            all_features = torch.cat(all_features, dim=0)
            all_batch_indices = torch.cat(all_batch_indices, dim=0)
        else:
            # Handle empty batch case
            all_coords = torch.zeros((0, 3), device=points.device)
            all_features = torch.zeros((0, 35), device=points.device)  # 32 + 3
            all_batch_indices = torch.zeros((0,), device=points.device)
        
        # Apply transformer
        transformed_features = self.transformer(all_features, all_coords)
        
        # Split results by batch index
        result = []
        for i in range(batch_size):
            mask = (all_batch_indices == i)
            result.append(transformed_features[mask])
        
        return result
    
    def forward(self, points, target_points=None, mode='train'):
        """
        Full model forward pass.
        
        Args:
            points: (B, N, 3) - Input point clouds
            target_points: (B, M, 3) - Target point clouds for diffusion (if training)
            mode: 'train' or 'sample'
            
        Returns:
            If mode == 'train': Diffusion loss
            If mode == 'sample': Generated point cloud
        """
        # Extract features using transformer
        transformer_features = self.extract_features(points)
        
        # Ensure all features have same shape for batching
        max_features = max([f.shape[0] for f in transformer_features])
        batch_size = len(transformer_features)
        
        # Pad features to same size
        padded_features = []
        for i, features in enumerate(transformer_features):
            if features.shape[0] < max_features:
                padding = torch.zeros(
                    max_features - features.shape[0], 
                    features.shape[1], 
                    device=features.device
                )
                padded_features.append(torch.cat([features, padding], dim=0))
            else:
                padded_features.append(features)
                
        # Stack into batch
        context = torch.stack(padded_features, dim=0)
        
        if mode == 'train':
            # Train diffusion model
            assert target_points is not None, "Target point cloud required for training"
            
            # Check for NaN or inf in features
            if torch.isnan(context).any() or torch.isinf(context).any():
                print("Warning: NaN or inf in transformer features, using feature clipping")
                context = torch.nan_to_num(context, nan=0.0, posinf=1e6, neginf=-1e6)
                context = torch.clamp(context, -1e6, 1e6)
            
            diff_loss, _, _ = self.diffusion(target_points, context)
            return diff_loss
        else:
            # Sample from diffusion model
            num_points = cfg.output_points
            generated_points = self.diffusion.sample(context, num_points)
            return generated_points
    
    @torch.no_grad()
    def sample(self, points, num_points=None):
        """
        Generate point cloud from input.
        
        Args:
            points: (B, N, 3) - Input point clouds
            num_points: Number of points to generate (default: cfg.output_points)
            
        Returns:
            generated_points: (B, num_points, 3) - Generated point clouds
        """
        transformer_features = self.extract_features(points)
        
        # Ensure all features have same shape for batching
        max_features = max([f.shape[0] for f in transformer_features])
        batch_size = len(transformer_features)
        
        # Pad features to same size
        padded_features = []
        for i, features in enumerate(transformer_features):
            if features.shape[0] < max_features:
                padding = torch.zeros(
                    max_features - features.shape[0], 
                    features.shape[1], 
                    device=features.device
                )
                padded_features.append(torch.cat([features, padding], dim=0))
            else:
                padded_features.append(features)
                
        # Stack into batch
        context = torch.stack(padded_features, dim=0)
        
        # Sample from diffusion model
        num_points = num_points or cfg.output_points
        return self.diffusion.sample(context, num_points)
    
    @torch.no_grad()
    def sample_with_ddim(self, points, num_points=None, steps=50, eta=0.0):
        """
        Generate point cloud using DDIM sampling.
        
        Args:
            points: (B, N, 3) - Input point clouds
            num_points: Number of points to generate
            steps: Number of DDIM sampling steps
            eta: Stochasticity parameter (0=deterministic, 1=stochastic)
            
        Returns:
            generated_points: (B, num_points, 3) - Generated point clouds
        """
        transformer_features = self.extract_features(points)
        
        # Ensure all features have same shape for batching
        max_features = max([f.shape[0] for f in transformer_features])
        batch_size = len(transformer_features)
        
        # Pad features to same size
        padded_features = []
        for i, features in enumerate(transformer_features):
            if features.shape[0] < max_features:
                padding = torch.zeros(
                    max_features - features.shape[0], 
                    features.shape[1], 
                    device=features.device
                )
                padded_features.append(torch.cat([features, padding], dim=0))
            else:
                padded_features.append(features)
                
        # Stack into batch
        context = torch.stack(padded_features, dim=0)
        
        # Sample from diffusion model using DDIM
        num_points = num_points or cfg.output_points
        return self.diffusion.sample_with_ddim(context, num_points, steps, eta)