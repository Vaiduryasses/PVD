"""
Point-Voxel Fusion (PVF) for condition feature extraction

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from ..config import cfg
from ..data.preprocessing import farthest_point_sampling

class EdgeConv(nn.Module):
    """
    EdgeConv layer for point feature extraction

    """
    def __init__(self, in_dim, out_dim, k = 16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim,out_dim)
        )
    
    def forward(self, x, pos):
        """
        Args:
            x: (N, C) - Point features
            pos: (N, 3) - Point positions
            
        Returns:
            out: (N, out_dim) - Updated point features
        """     
        #compute KNN graph using torch_cluster  
        batch = torch.zeros(pos.shape[0], dtype = torch.long, device = pos.device)
        edge_index = torch_cluster.knn_graph(pos, k = self.k,batch = batch, loop = False)

        #get features for source and target nodes
        x_j = x[edge_index[1]]  # Target node features (N*k, C)
        x_i = x[edge_index[0]]  # Source node features (N*k, C)

        #compute edge features
        edge_features = torch.cat([x_i, x_j - x_i],dim = 1)#(N*k,2C)

        #apply MLP
        edge_features = self.conv(edge_features) #(N*k, out_dim)

        #aggregate features using scatter_max
        from torch_scatter import scatter_max
        out, _ = scatter_max(edge_features, edge_index[0], dim = 0, dim_size = pos.shape[0])

        return out

class PointBranch(nn.Module):
    """
    Point branch for multi-scale feature extraction

    """
    def __init__(self, scales, dim = 256):
        super().__init__()
        self.scales = scales

        #initial feature extraction
        self.edge_conv = EdgeConv(3,64,k=16)

        #MLP for each scale
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64,128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )for _ in range(len(scales))
        ])
    def forward(self, points):
        """
        Args:
            points: (N, 3) - Input point cloud
            
        Returns:
            multi_scale_features: list of (S_i, dim) - Features at multiple scales
        """
        #inital feature extraction
        features = self.edge_conv(points,points)#(N,64)

        #multi-scale smpling and feature processing using PointNet++ style FPS
        multi_scale_features = []

        for i,scale in enumerate(self.scales):
            #use FPS for multi-scale sampling
            if i == 0:
                #first scale: direct FPS on original point cload
                fps_idx = farthest_point_sampling(points, scale)
                sampled_points = points[fps_idx]
                sampled_features = features[fps_idx]
            else:
                #subsequent scales: FPS based on previous layer's points (hierarchical structure like PointNet++)
                prev_points = multi_scale_features[-1][0] # Points from previous layer
                fps_idx = farthest_point_sampling(prev_points, scale)
                sampled_points = prev_points[fps_idx]

                # calculate features for current level points
                # find closest points from original feature set or interpolate features
                idx = self._find_closest_points(sampled_points, points)
                sampled_features = features[idx]
            
            # apply MLP
            level_features = self.mlps[i](sampled_features)  # (S_i, dim)
            multi_scale_features.append((sampled_points, level_features))
        
        return multi_scale_features
    
    def _find_closest_points(self, query_points, source_points):
        """
         Find nearest neighbors of query points in source point set

        """
        #compute distance between all point pairs
        inner = -2 * torch.matmul(query_points, source_points.transpose(1, 0))
        xx =  torch.sum(query_points**2, dim = 1, keepdim = True)
        yy =  torch.sum(source_points**2, dim = 1, keepdim = True).transpose(1,0)
        dist = xx + inner + yy   # (query_size, source_size)

        #get nearest neighbor indices
        _, nn_idx = torch.min(dist, dim=1)
        return nn_idx
          