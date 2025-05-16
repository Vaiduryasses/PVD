"""
residual fine-tuing and geometric projection

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from ..config import cfg

class EdgeConvBlock(nn.Module):
    """
    EdgeConv block for residual network
    
    """
    
    def __init__(self, in_dim, out_dim, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, pos):
        """
        Args:
            x: (N, C) - Point features
            pos: (N, 3) - Point positions
            
        Returns:
            out: (N, out_dim) - Updated point features
        """
        #compute KNN graph 
        batch = torch.zeros(pos.shape[0], dtype = torch.long, device = pos.device)
        edge_index = torch_cluster.knn_graph(pos, k=self.k, batch = batch, loop = Flase)

        #get features for source and target nodes
        x_j = x[edge_index[1]]  # Target node features (N*k, C)
        x_i = x[edge_index[0]]  # Source node features (N*k, C)
        
        #compute edge features
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # (N*k, 2C)
        
        #apply MLPs
        edge_features = self.conv(edge_features)  # (N*k, out_dim)
        
        #aggregate features using scatter_max
        from torch_scatter import scatter_max
        out, _ = scatter_max(edge_features, edge_index[0], dim=0, dim_size=pos.shape[0])
        
        return out 

class ResidualNetwork(nn.Module):
    """
    residual network for fine-tuning dense points
    
    """
    def __init__(self, k=16):
        super().__init__()
        self.k = k

        #EdgeConv blocks
        self.edge_conv1 = EdgeConvBLock(3,64,k =k)
        self.edge_conv2 = EdgeConvBlock(64, 64, k=k)
        self.edge_conv3 = EdgeConvBlock(64, 64, k=k)
                
        #MLP for final residual prediction
        self.mlp = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self,points):
        """
        Args:
            points: (N, 3) - Input point cloud
            
        Returns:
            residual: (N, 3) - Predicted residual
        """
        #extract feature
        f1 = self.edge_conv1(points, points)
        f2 = self.edge_conv2(f1, points)
        f3 = self.edge_conv3(f2, points)

        #concatenate
        features = torch.cat([f1,f2,f3],dim=1)

        #predict residual
        residual = self.mlp(features)

        return residual

class SVDPlaneProjection(nn.Module):
    """
    SVD-based plane projection with GoCoPP refinement
    
    """
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, points, planes):
        """
        Args:
            points: (N, 3) - Input point cloud
            planes: dict - Plane parameters with normals and distances
            
        Returns:
            projected_points: (N, 3) - Points projected to planes
        """
        normals = planes['normals']#(P,3)
        distances = planes['distances']#(P,)

        #initialize projected points
        projected_points = points.clone()

        #process each plane
        for i in range(normals.shape[0]):
            normal = normals[i]
            distance = distances[i]

            #compute point-to-point distance
            point_distance = torch.abs(torch.sum(points * normal, dim=1)-distance)

            #select points close to the plane
            mask = point_distance < self.threshold
            plane_points = points[mask]
            
            #need at least 3 points for SVD
            if plane_points.shape[0] < 3:
                continue

            #compute centroid
            centroid = torch.mean(plane_points, dim=0)

            #center points
            centered_points = plane_points - centroid

            #compute convariance matrix
            cov = centered_points.T @ centered_points

            #SVD for plane fittinf
            U,S,V = torch.svd(cov)

            #extract refined normal (eigenvector with smallest eigenvalue)
            refined_normal = V[:, 2]
            
            #ensure normal points in the same general direction
            if torch.sum(refined_normal * normal) < 0:
                refined_normal = -refined_normal

            #compute refined distance
            refined_distance = torch.sum(centroid * refined_normal)

            #project inlier points to the refined plane
            for j in range(projected_points.shape[0]):
                if mask[j]:
                    #compute orthogonal projection
                    dot_product = torch.sum(projected_points[j] * refined_normal)
                    projection = refined_normal * (dot_product - refined_distance)
                    projected_points[j] = projected_points[j] - projection

        return projected_points

class RefinementModule(nn.Module):
    """
    Refinement module combining residual network and SVD plane projection
    
    """
    
    def __init__(self):
        super().__init__()
        self.residual_net = ResidualNetwork(k=cfg.residual_knn)
        self.plane_projection = SVDPlaneProjection(threshold=cfg.plane_proj_threshold)

    def forward(self, points, planes):
        """
        Args:
            points: (N, 3) - Input point cloud
            planes: dict - Plane parameters
            
        Returns:
            refined_points: (N, 3) - Refined point cloud
        """
        #compute and apply residual
        residual = self.residual_net(points)
        points_with_residual = points + residual

        # Apply SVD-based plane projection with GoCoPP refinement
        refined_points = self.plane_projection(points_with_residual, planes)
        
        return refined_points







