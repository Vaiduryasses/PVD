"""
Point Cloud Evaluation Metrics
Used to evaluate the quality of generated point clouds
"""
import torch
import numpy as np
import open3d as o3d

def chamfer_distance(x, y, reduce_mean=True):
    """
    Calculate Chamfer Distance
    
    Args:
        x: Point cloud 1, (B, N, 3)
        y: Point cloud 2, (B, M, 3)
        reduce_mean: Whether to average across batch
    
    Returns:
        distance: If reduce_mean=True, returns scalar; otherwise (B,)
    """
    # Ensure inputs are correct
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Add batch dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    # Move to same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Get batch size and point counts
    batch_size, n_x, _ = x.shape
    _, n_y, _ = y.shape
    
    # Calculate squared distance between all pairs of points
    xx = torch.sum(x ** 2, dim=2).view(batch_size, n_x, 1)
    yy = torch.sum(y ** 2, dim=2).view(batch_size, 1, n_y)
    xy = torch.bmm(x, y.transpose(1, 2))
    dist = xx + yy - 2 * xy
    
    # Get minimum distances
    dist_x_to_y = torch.min(dist, dim=2)[0]  # Minimum distance from x to y
    dist_y_to_x = torch.min(dist, dim=1)[0]  # Minimum distance from y to x
    
    # Calculate Chamfer distance
    cd = torch.mean(dist_x_to_y, dim=1) + torch.mean(dist_y_to_x, dim=1)
    
    if reduce_mean:
        cd = torch.mean(cd)
    
    return cd

def hausdorff_distance(x, y, reduce_mean=True):
    """
    Calculate Hausdorff Distance
    
    Args:
        x: Point cloud 1, (B, N, 3)
        y: Point cloud 2, (B, M, 3)
        reduce_mean: Whether to average across batch
    
    Returns:
        distance: If reduce_mean=True, returns scalar; otherwise (B,)
    """
    # Ensure inputs are correct
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Add batch dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    # Move to same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Get batch size and point counts
    batch_size, n_x, _ = x.shape
    _, n_y, _ = y.shape
    
    # Calculate squared distance between all pairs of points
    xx = torch.sum(x ** 2, dim=2).view(batch_size, n_x, 1)
    yy = torch.sum(y ** 2, dim=2).view(batch_size, 1, n_y)
    xy = torch.bmm(x, y.transpose(1, 2))
    dist = xx + yy - 2 * xy
    
    # Get minimum distances
    min_dist_x_to_y = torch.min(dist, dim=2)[0]  # Min distance from each x to y
    min_dist_y_to_x = torch.min(dist, dim=1)[0]  # Min distance from each y to x
    
    # Calculate Hausdorff distance
    # max_dist_x_to_y = maximum distance from any point in x to its closest point in y
    # max_dist_y_to_x = maximum distance from any point in y to its closest point in x
    max_dist_x_to_y = torch.max(min_dist_x_to_y, dim=1)[0]
    max_dist_y_to_x = torch.max(min_dist_y_to_x, dim=1)[0]
    
    # Hausdorff distance is the maximum of these two maximums
    hausdorff = torch.max(max_dist_x_to_y, max_dist_y_to_x)
    
    if reduce_mean:
        hausdorff = torch.mean(hausdorff)
    
    return hausdorff

def normal_consistency(x, y, reduce_mean=True, k_neighbors=30):
    """
    Calculate Normal Consistency between two point clouds
    
    Args:
        x: Point cloud 1, (B, N, 3)
        y: Point cloud 2, (B, M, 3)
        reduce_mean: Whether to average across batch
        k_neighbors: Number of neighbors for normal estimation
    
    Returns:
        normal_consistency: If reduce_mean=True, returns scalar; otherwise (B,)
    """
    batch_size = x.shape[0] if isinstance(x, torch.Tensor) else len(x)
    
    # Initialize consistency scores
    nc_scores = torch.zeros(batch_size, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
    
    for i in range(batch_size):
        # Extract point clouds for this batch item
        pc1 = x[i].cpu().numpy() if isinstance(x, torch.Tensor) else x[i]
        pc2 = y[i].cpu().numpy() if isinstance(y, torch.Tensor) else y[i]
        
        # Convert to Open3D point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1)
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        
        # Estimate normals
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
        
        # Orient normals consistently
        pcd1.orient_normals_consistent_tangent_plane(k=k_neighbors)
        pcd2.orient_normals_consistent_tangent_plane(k=k_neighbors)
        
        # Get normals as numpy arrays
        normals1 = np.asarray(pcd1.normals)
        normals2 = np.asarray(pcd2.normals)
        
        # Build KD Tree for second point cloud
        pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
        
        # For each point in pcd1, find closest point in pcd2 and calculate normal alignment
        consistency_sum = 0.0
        valid_points = 0
        
        for j, normal1 in enumerate(normals1):
            # Find closest point
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd1.points[j], 1)
            if idx:
                normal2 = normals2[idx[0]]
                # Calculate absolute dot product (1 = aligned, 0 = perpendicular)
                # Taking absolute value because normal direction can be flipped
                dot_product = abs(np.dot(normal1, normal2))
                consistency_sum += dot_product
                valid_points += 1
        
        # Average consistency for this batch item
        if valid_points > 0:
            nc_scores[i] = torch.tensor(consistency_sum / valid_points, device=nc_scores.device)
    
    if reduce_mean:
        return torch.mean(nc_scores)
    else:
        return nc_scores

def construction_failure_rate(distances, threshold=0.1):
    """
    Calculate the construction failure rate based on a distance metric
    
    Args:
        distances: Chamfer distances or other distance metric (B,)
        threshold: Threshold for considering a reconstruction as failed
    
    Returns:
        failure_rate: Percentage of samples exceeding the threshold
    """
    if isinstance(distances, torch.Tensor):
        failures = (distances > threshold).float().mean()
    else:
        failures = np.mean(np.array(distances) > threshold)
    
    return failures