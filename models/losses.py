"""
Loss functions for PVD

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class ProxyDecoderLoss(nn.Module):
    """
    Loss function for proxy decoder with Hungarian matching
    
    """
    def __init__(self, alpha_param = 0.5, beta_chamfer = 20.0):
        super().__init__()
        self.alpha_param = alpha_param
        self.beta_chamfer = beta_chamfer
    
    def forward(self, pred_planes, gt_planes, points):
        """
        Args:
            pred_planes: dict with predicted plane parameters
            gt_planes: dict with ground truth plane parameters
            points: (N, 3) point cloud
            
        Returns:
            total_loss: Total loss
            loss_dict: Dict with individual loss components
        """    
        #extract prediction and ground truth
        pred_logits = pred_planes['logits']#(B,P)
        pred_normals = pred_planes['normals']#(B,P,3)
        pred_distances = pred_planes['distances']#(B,P)

        gt_normals = gt_planes['normals']  # (B, P', 3)
        gt_distances = gt_planes['distances']  # (B, P')
        gt_masks = gt_planes['masks']  # (B, P', N)

        #handle batch dimension
        if pred_logits.dim() == 1:
            pred_logits = pred_logits.unsqueeze(0)
            pred_normals = pred_normals.unsqueeze(0)
            pred_distances = pred_distances.unsqueeze(0)

        if gt_normals.dim() == 2:
            gt_normals = gt_normals.unsqueeze(0)
            gt_distances = gt_distances.unsqueeze(0)
            gt_masks = gt_masks.unsqueeze(0)
        
        batch_size = pred_logits.shape[0]
        num_pred = pred_logits.shape[1]
        num_gt = gt_normals.shape[1]

        #initialize losses
        cls_loss = 0
        param_loss = 0
        chamfer_loss =0

        #process each batch
        for b in range(batch_size):
            #compute cost matrix for matching
            cost_matrix = torch.zeros((num_pred, num_gt), device=pred_logits.device)

            #compute cost based on normal and distance similarity
            for i in range(num_pred):
                for j in range(num_gt):
                    #normal similarity (1 - |cos(angle)|)
                    normal_sim = 1 - torch.abs(torch.sum(pred_normals[b, i] * gt_normals[b, j]))

                    #distance difference
                    dist_diff = torch.abs(pred_distances[b, i] - gt_distances[b, j])

                    #Chamfer distance for points
                    if torch.any(gt_masks[b, j]):
                        #get points belonging to this plane
                        gt_plane_points = points[b][gt_masks[b, j]]
                        
                        #compute point-to-plane distance for predicted plane
                        point_to_plane = torch.abs(torch.sum(gt_plane_points * pred_normals[b, i].unsqueeze(0), dim=1) - 
                                                 pred_distances[b, i])
                        
                        # Mean distance as chamfer component
                        chamfer_component = torch.mean(point_to_plane)
                    else:
                        chamfer_component = torch.tensor(1.0, device=pred_logits.device)
                    
                    # Combined cost
                    cost_matrix[i, j] = normal_sim + 0.5 * dist_diff + 5.0 * chamfer_component

            # Hungarian matching
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            pred_indices = torch.tensor(pred_indices, device=pred_logits.device)
            gt_indices = torch.tensor(gt_indices, device=pred_logits.device)
            
            # Create GT classification target based on matching
            cls_target = torch.zeros_like(pred_logits[b])
            cls_target[pred_indices] = 1.0
            
            # Classification loss
            cls_loss += F.binary_cross_entropy_with_logits(pred_logits[b], cls_target)
            
            # Parameter loss for matched planes
            param_normal_loss = 0
            param_dist_loss = 0
            this_chamfer_loss = 0
            
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                # Normal loss (1 - cos similarity)
                normal_loss = 1 - torch.abs(torch.sum(pred_normals[b, pred_idx] * gt_normals[b, gt_idx]))
                param_normal_loss += normal_loss
                
                # Distance loss
                dist_loss = torch.abs(pred_distances[b, pred_idx] - gt_distances[b, gt_idx])
                param_dist_loss += dist_loss
                
                # Chamfer loss for points
                if torch.any(gt_masks[b, gt_idx]):
                    # Get points belonging to this plane
                    gt_plane_points = points[b][gt_masks[b, gt_idx]]
                    
                    # Compute point-to-plane distance for predicted plane
                    point_to_plane = torch.abs(torch.sum(gt_plane_points * pred_normals[b, pred_idx].unsqueeze(0), dim=1) - 
                                              pred_distances[b, pred_idx])
                    
                    # Mean distance as chamfer loss
                    this_chamfer_loss += torch.mean(point_to_plane)
            
            # Average losses by number of matched planes
            if len(pred_indices) > 0:
                param_normal_loss /= len(pred_indices)
                param_dist_loss /= len(pred_indices)
                this_chamfer_loss /= len(pred_indices)
            
            # Add to total parameter and chamfer losses
            param_loss += param_normal_loss + param_dist_loss
            chamfer_loss += this_chamfer_loss
        
        # Average losses by batch size
        cls_loss /= batch_size
        param_loss /= batch_size
        chamfer_loss /= batch_size
        
        # Total loss
        total_loss = cls_loss + self.alpha_param * param_loss + self.beta_chamfer * chamfer_loss
        
        return total_loss, {
            'cls_loss': cls_loss,
            'param_loss': param_loss,
            'chamfer_loss': chamfer_loss
        }

class DiffusionLoss(nn.Module):
    """MSE loss for diffusion model."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_noise, target_noise):
        """
        Args:
            pred_noise: (B, N, 3) - Predicted noise
            target_noise: (B, N, 3) - Target noise
            
        Returns:
            loss: MSE loss
        """
        return F.mse_loss(pred_noise, target_noise)

class ResidualLoss(nn.Module):
    """MSE loss for residual refinement."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_residual, target_residual):
        """
        Args:
            pred_residual: (B, N, 3) - Predicted residual
            target_residual: (B, N, 3) - Target residual
            
        Returns:
            loss: MSE loss
        """
        return F.mse_loss(pred_residual, target_residual)        
                               

