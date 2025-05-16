"""
CUDA-accelerated operations for point cloud processing.
Provides GPU-accelerated implementations of common point cloud operations.
"""
import torch
import numpy as np
import warnings

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

if not CUDA_AVAILABLE:
    warnings.warn("CUDA is not available. Using CPU implementations instead.")

# Try importing custom CUDA extensions if available
try:
    import pvd_cuda
    CUSTOM_CUDA_OPS_AVAILABLE = True
    print("Custom CUDA operations loaded successfully!")
except ImportError:
    CUSTOM_CUDA_OPS_AVAILABLE = False
    print("Custom CUDA extensions not found. Using PyTorch implementations.")

def voxelize_points_gpu(points, voxel_size, point_features=None):
    """
    Voxelize point cloud on GPU
    
    Args:
        points: (N, 3) tensor of point coordinates
        voxel_size: size of voxels
        point_features: optional (N, C) tensor of point features
        
    Returns:
        voxel_coordinates: (M, 3) tensor of voxel indices
        voxel_features: (M, C+1) tensor of voxel features (features + point count)
    """
    if not CUDA_AVAILABLE:
        return voxelize_points_cpu(points, voxel_size, point_features)
        
    if CUSTOM_CUDA_OPS_AVAILABLE:
        return pvd_cuda.voxelize(points, voxel_size, point_features)
        
    # PyTorch implementation
    # Quantize point coordinates to voxel indices
    voxel_indices = (points / voxel_size).int()
    
    # Create a unique ID for each voxel
    voxel_hash = voxel_indices[:, 0] * 100000 + voxel_indices[:, 1] * 1000 + voxel_indices[:, 2]
    
    # Find unique voxels
    unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    
    # Count points per voxel
    ones = torch.ones(points.shape[0], 1, device=points.device)
    voxel_point_counts = torch.zeros(len(unique_hashes), 1, device=points.device)
    voxel_point_counts.index_add_(0, inverse_indices, ones)
    
    # Average features per voxel
    if point_features is None:
        point_features = points
    
    voxel_features = torch.zeros(len(unique_hashes), point_features.shape[1], device=points.device)
    
    # Use scatter_add_ for feature accumulation
    for i in range(point_features.shape[1]):
        feature_column = point_features[:, i:i+1]
        voxel_features[:, i:i+1].index_add_(0, inverse_indices, feature_column)
    
    # Divide by count for average
    voxel_features = voxel_features / voxel_point_counts
    
    # Get voxel coordinates for unique voxels
    voxel_coordinates = torch.zeros(len(unique_hashes), 3, dtype=torch.int, device=points.device)
    
    # Recover coordinates from hash
    # This is a naive implementation - in practice, you'd use a more efficient method
    for i, h in enumerate(unique_hashes):
        # Find any point that maps to this voxel
        point_idx = torch.where(voxel_hash == h)[0][0]
        voxel_coordinates[i] = voxel_indices[point_idx]
    
    # Include point count as a feature
    voxel_features = torch.cat([voxel_features, voxel_point_counts], dim=1)
    
    return voxel_coordinates, voxel_features

def voxelize_points_cpu(points, voxel_size, point_features=None):
    """CPU implementation of voxelization (fallback)"""
    # Move to CPU for processing
    points_cpu = points.cpu().numpy()
    point_features_cpu = point_features.cpu().numpy() if point_features is not None else points_cpu
    
    # Quantize point coordinates to voxel indices
    voxel_indices = (points_cpu / voxel_size).astype(np.int32)
    
    # Create a unique ID for each voxel
    voxel_hash = voxel_indices[:, 0] * 100000 + voxel_indices[:, 1] * 1000 + voxel_indices[:, 2]
    
    # Find unique voxels
    unique_hashes, inverse_indices, counts = np.unique(voxel_hash, return_inverse=True, return_counts=True)
    
    # Prepare output arrays
    voxel_coordinates = np.zeros((len(unique_hashes), 3), dtype=np.int32)
    voxel_features = np.zeros((len(unique_hashes), point_features_cpu.shape[1] + 1))  # +1 for count
    
    # Accumulate features and get coordinates
    for i, h in enumerate(unique_hashes):
        mask = (voxel_hash == h)
        voxel_coordinates[i] = voxel_indices[mask][0]  # Take first point's voxel index
        voxel_features[i, :-1] = point_features_cpu[mask].mean(axis=0)  # Average features
        voxel_features[i, -1] = counts[i]  # Store point count
    
    # Convert back to torch tensors on the original device
    voxel_coordinates = torch.tensor(voxel_coordinates, device=points.device)
    voxel_features = torch.tensor(voxel_features, device=points.device)
    
    return voxel_coordinates, voxel_features

def farthest_point_sampling(points, num_samples):
    """
    GPU implementation of farthest point sampling
    
    Args:
        points: (B, N, 3) tensor of point coordinates
        num_samples: number of points to sample
        
    Returns:
        indices: (B, num_samples) tensor of indices
    """
    if not CUDA_AVAILABLE:
        return farthest_point_sampling_cpu(points, num_samples)
        
    if CUSTOM_CUDA_OPS_AVAILABLE:
        return pvd_cuda.farthest_point_sample(points, num_samples)
    
    # PyTorch implementation
    device = points.device
    batch_size, num_points, _ = points.shape
    
    # Initialize with the first point
    indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    
    # Randomly select the first point
    indices[:, 0] = torch.randint(0, num_points, (batch_size,), device=device)
    
    # Iteratively select farthest points
    for i in range(1, num_samples):
        # Get the last selected point
        last_idx = indices[:, i-1]
        
        # Compute distances
        for b in range(batch_size):
            last_point = points[b, last_idx[b], :].view(1, 3)
            dist = torch.sum((points[b] - last_point) ** 2, dim=1)
            distance[b] = torch.min(distance[b], dist)
        
        # Select the farthest point
        indices[:, i] = torch.max(distance, dim=1)[1]
    
    return indices

def farthest_point_sampling_cpu(points, num_samples):
    """CPU implementation of farthest point sampling (fallback)"""
    # Move to CPU for processing
    points_cpu = points.cpu().numpy()
    batch_size, num_points, _ = points_cpu.shape
    
    # Initialize output
    indices = np.zeros((batch_size, num_samples), dtype=np.int64)
    
    for b in range(batch_size):
        # Initialize with the first point
        indices[b, 0] = np.random.randint(0, num_points)
        distances = np.full(num_points, 1e10)
        
        # Iteratively select farthest points
        for i in range(1, num_samples):
            last_idx = indices[b, i-1]
            last_point = points_cpu[b, last_idx]
            
            # Compute distances
            dist = np.sum((points_cpu[b] - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            
            # Select the farthest point
            indices[b, i] = np.argmax(distances)
    
    # Convert back to torch tensor on the original device
    indices = torch.tensor(indices, device=points.device)
    
    return indices

def batch_ball_query(points, centroids, radius, max_samples):
    """
    Find all points within radius of each centroid
    
    Args:
        points: (B, N, 3) tensor of point coordinates
        centroids: (B, M, 3) tensor of centroid coordinates
        radius: query radius
        max_samples: maximum number of samples to return
        
    Returns:
        indices: (B, M, max_samples) tensor of indices, padded with -1
    """
    if not CUDA_AVAILABLE:
        return batch_ball_query_cpu(points, centroids, radius, max_samples)
        
    if CUSTOM_CUDA_OPS_AVAILABLE:
        return pvd_cuda.ball_query(points, centroids, radius, max_samples)
    
    # PyTorch implementation
    device = points.device
    batch_size, num_points, _ = points.shape
    _, num_centroids, _ = centroids.shape
    
    # Initialize output tensor
    indices = torch.ones(batch_size, num_centroids, max_samples, 
                         dtype=torch.long, device=device) * -1
    
    # Compute pairwise distances for each batch
    for b in range(batch_size):
        for c in range(num_centroids):
            # Compute distances from all points to this centroid
            distances = torch.sum((points[b] - centroids[b, c].unsqueeze(0)) ** 2, dim=1)
            
            # Find points within radius
            mask = distances < radius ** 2
            neighbor_indices = torch.where(mask)[0]
            
            # Handle case where we have too many or too few neighbors
            if len(neighbor_indices) > max_samples:
                # Randomly select max_samples neighbors
                perm = torch.randperm(len(neighbor_indices), device=device)
                neighbor_indices = neighbor_indices[perm[:max_samples]]
            
            # Store the indices
            indices[b, c, :len(neighbor_indices)] = neighbor_indices
    
    return indices

def batch_ball_query_cpu(points, centroids, radius, max_samples):
    """CPU implementation of ball query (fallback)"""
    # Move to CPU for processing
    points_cpu = points.cpu().numpy()
    centroids_cpu = centroids.cpu().numpy()
    batch_size, num_points, _ = points_cpu.shape
    _, num_centroids, _ = centroids_cpu.shape
    
    # Initialize output
    indices = np.ones((batch_size, num_centroids, max_samples), dtype=np.int64) * -1
    
    # Compute pairwise distances for each batch
    for b in range(batch_size):
        for c in range(num_centroids):
            # Compute distances from all points to this centroid
            distances = np.sum((points_cpu[b] - centroids_cpu[b, c]) ** 2, axis=1)
            
            # Find points within radius
            neighbor_indices = np.where(distances < radius ** 2)[0]
            
            # Handle case where we have too many or too few neighbors
            if len(neighbor_indices) > max_samples:
                # Randomly select max_samples neighbors
                perm = np.random.permutation(len(neighbor_indices))
                neighbor_indices = neighbor_indices[perm[:max_samples]]
            
            # Store the indices
            indices[b, c, :len(neighbor_indices)] = neighbor_indices
    
    # Convert back to torch tensor on the original device
    indices = torch.tensor(indices, device=points.device)
    
    return indices

# Additional utility functions
def ransac_plane_fit(points, num_iterations=100, distance_threshold=0.05, min_inliers=100):
    """
    RANSAC-based plane fitting
    
    Args:
        points: (N, 3) tensor of point coordinates
        num_iterations: number of RANSAC iterations
        distance_threshold: maximum distance for inliers
        min_inliers: minimum number of inliers for a valid plane
        
    Returns:
        plane_params: (4,) tensor of plane parameters (a, b, c, d) where ax + by + cz + d = 0
        inlier_mask: (N,) boolean tensor indicating inliers
    """
    if not CUDA_AVAILABLE:
        return ransac_plane_fit_cpu(points, num_iterations, distance_threshold, min_inliers)
        
    if CUSTOM_CUDA_OPS_AVAILABLE:
        return pvd_cuda.ransac_plane_fit(points, num_iterations, distance_threshold, min_inliers)
    
    # PyTorch implementation
    device = points.device
    num_points = points.shape[0]
    
    best_inliers = 0
    best_plane = None
    best_mask = None
    
    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = torch.randperm(num_points, device=device)[:3]
        p1, p2, p3 = points[sample_indices]
        
        # Compute plane parameters
        v1 = p2 - p1
        v2 = p3 - p1
        normal = torch.cross(v1, v2)
        
        # Normalize the normal vector
        norm = torch.norm(normal)
        if norm < 1e-6:
            continue  # Skip if points are collinear
        normal = normal / norm
        
        # Compute d in ax + by + cz + d = 0
        d = -torch.dot(normal, p1)
        
        # Compute distances from all points to the plane
        distances = torch.abs(torch.matmul(points, normal) + d)
        
        # Find inliers
        inlier_mask = distances < distance_threshold
        num_inliers = torch.sum(inlier_mask)
        
        # Update best plane if we found more inliers
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_plane = torch.cat([normal, d.unsqueeze(0)])
            best_mask = inlier_mask
    
    # Check if we found a valid plane
    if best_inliers < min_inliers:
        return None, None
    
    return best_plane, best_mask

def ransac_plane_fit_cpu(points, num_iterations=100, distance_threshold=0.05, min_inliers=100):
    """CPU implementation of RANSAC plane fitting (fallback)"""
    # Move to CPU for processing
    points_cpu = points.cpu().numpy()
    num_points = points_cpu.shape[0]
    
    best_inliers = 0
    best_plane = None
    best_mask = None
    
    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(num_points, 3, replace=False)
        p1, p2, p3 = points_cpu[sample_indices]
        
        # Compute plane parameters
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue  # Skip if points are collinear
        normal = normal / norm
        
        # Compute d in ax + by + cz + d = 0
        d = -np.dot(normal, p1)
        
        # Compute distances from all points to the plane
        distances = np.abs(np.dot(points_cpu, normal) + d)
        
        # Find inliers
        inlier_mask = distances < distance_threshold
        num_inliers = np.sum(inlier_mask)
        
        # Update best plane if we found more inliers
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_plane = np.append(normal, d)
            best_mask = inlier_mask
    
    # Check if we found a valid plane
    if best_inliers < min_inliers:
        return None, None
    
    # Convert back to torch tensors on the original device
    best_plane = torch.tensor(best_plane, device=points.device)
    best_mask = torch.tensor(best_mask, device=points.device)
    
    return best_plane, best_mask