"""
End-to-end PVD pipeline

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import cfg
from .voxel_transformer import SparseVoxelTransformer
from .proxy_decoder import ProxyDecoder
from .pvf import PointVoxelFusion
from .diffusion import DiffusionModel
from .refinement import RefinementModule
from .losses import ProxyDecoderLoss, DiffusionLoss, ResidualLoss

class PVD(nn.Module):
    """
    PVD model pipeline

    """
    
    def __init__(self):
        super().__init__()

        #Voxel transformer and Proxy decoder
        self.voxel_transformer = SparseVoxelTransformer(
            in_dim=35,
            dim=cfg.encoder_dim,
            depth=cfg.encoder_layers,
            num_heads=cfg.encoder_nhead,
            window_size=cfg.window_size
        )
        
        self.proxy_decoder = ProxyDecoder(
            dim=cfg.encoder_dim,
            num_queries=cfg.num_plane_queries,
            num_layers=4
        )

        #Point-Voxel Fusion
        self.pvf = PointVoxelFusion(dim=cfg.pvf_dim)
        
        #Diffusion
        self.diffusion = DiffusionModel()
        
        #Refinement
        self.refinement = RefinementModule()
        
        # Loss functions
        self.proxy_loss_fn = ProxyDecoderLoss(
            alpha_param=cfg.alpha_param, 
            beta_chamfer=cfg.beta_chamfer
        )
        self.diffusion_loss_fn = DiffusionLoss()
        self.residual_loss_fn = ResidualLoss()

    def forward(self, points, voxel_features, voxel_coords, mode='train'):
        """
        Args:
            points: (N, 3) - Input point cloud
            voxel_features: (V, 35) - Voxel features
            voxel_coords: (V, 3) - Voxel coordinates
            mode: 'train' or 'inference'
            
        Returns:
            outputs: dict - Model outputs
        """
        #voxel transformer
        voxel_global_features = self.voxel_transformer(voxel_features, voxel_coords)

        #Proxy Decoder
        proxy_outputs = self.proxy_decoder(voxel_global_features)
        
        #Select coarse points
        mask = proxy_outputs['logits'] > cfg.proxy_threshold
        coarse_points = voxel_coords[mask].float() / cfg.voxel_resolution - 0.5
        
        #Point-Voxel Fusion
        condition_features = self.pvf(points, voxel_global_features, voxel_coords)
        
        # Diffusion
        if mode == 'train':
            # In training, compute diffusion loss
            diffusion_loss, pred_noise, true_noise = self.diffusion(points, condition_features)
            dense_points = None
        else:
            # In inference, run DDIM sampling
            dense_points = self.diffusion.sample_with_ddim(
                context=condition_features,
                num_points=cfg.output_points,
                steps=cfg.diffusion_steps_infer
            )
            diffusion_loss = None
        
        #Refinement (only in inference)
        if mode == 'inference' and dense_points is not None:
            final_points = self.refinement(dense_points, proxy_outputs)
        else:
            final_points = None
        
        return {
            'proxy_outputs': proxy_outputs,
            'coarse_points': coarse_points,
            'condition_features': condition_features,
            'dense_points': dense_points,
            'final_points': final_points,
            'diffusion_loss': diffusion_loss
        }

    def train_step(self, batch):
        """
        Training step.
        
        Args:
            batch: dict - Input batch with points, voxels, and GT
            
        Returns:
            losses: dict - Training losses
        """
        # Unpack batch
        points = batch['points']
        voxel_features = batch['voxel_features']
        voxel_coords = batch['voxel_coords']
        gt_planes = batch['gt_planes']
        
        # Forward pass
        outputs = self.forward(points, voxel_features, voxel_coords, mode='train')
        
        # Compute losses
        losses = {}
        
        # Proxy decoder loss with GoCoPP supervision
        proxy_loss, proxy_loss_dict = self.proxy_loss_fn(
            outputs['proxy_outputs'], 
            gt_planes,
            points
        )
        losses.update(proxy_loss_dict)
        
        # Diffusion loss
        diffusion_loss = outputs['diffusion_loss']
        losses['diffusion_loss'] = diffusion_loss
        
        # Total loss
        losses['total_loss'] = proxy_loss + cfg.lambda_diff * diffusion_loss
        
        return losses
    
    def inference(self, points, voxel_features, voxel_coords):
        """
        Inference step.
        
        Args:
            points: (N, 3) - Input point cloud
            voxel_features: (V, 35) - Voxel features
            voxel_coords: (V, 3) - Voxel coordinates
            
        Returns:
            final_points: (M, 3) - Final refined point cloud
            planes: dict - Detected planes
        """
        outputs = self.forward(points, voxel_features, voxel_coords, mode='inference')
        
        return outputs['final_points'], outputs['proxy_outputs']

    def visualize_intermediate(self, points, voxel_features, voxel_coords):
        """
        Generate intermediate results for visualization.
        
        Args:
            points: (N, 3) - Input point cloud
            voxel_features: (V, 35) - Voxel features
            voxel_coords: (V, 3) - Voxel coordinates
            
        Returns:
            visualization_outputs: dict - All intermediate results
        """
        # Run entire pipeline
        outputs = self.forward(points, voxel_features, voxel_coords, mode='inference')
        
        # Add input points to outputs
        outputs['input_points'] = points
        
        # Generate diffusion intermediate steps (6 steps)
        if outputs['condition_features'] is not None:
            # Start from random noise
            batch_size = outputs['condition_features'].shape[0]
            device = outputs['condition_features'].device
            x_T = torch.randn(batch_size, cfg.num_output_points, 3, device=device)
            
            # Get DDIM timestep sequence 
            seq = [0, 5, 10, 15, 20, 25, 29]  
            
            # Store intermediate results
            intermediate_points = [x_T]
            
            # DDIM sampling with intermediate results
            x_t = x_T
            for i in range(len(seq) - 1, 0, -1):
                # Get current step
                t = seq[i]
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.diffusion.predict_noise(x_t, t_tensor, outputs['condition_features'])
                
                # Single DDIM step
                x_t = self.diffusion.ddim_step(
                    x_t, t, seq[i-1], noise_pred, outputs['condition_features'], eta=0.0
                )
                
                # Save intermediate
                intermediate_points.append(x_t)
            
            outputs['diffusion_intermediates'] = intermediate_points
        
        return outputs


