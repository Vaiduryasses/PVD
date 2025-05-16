"""
pre-processing functions 

"""
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_sparse
import numpy as np
from ..config import cfg

def batched_normalize_pointcloud(points_batch):
    """
    Efficiently normalize a batch of point clouds
    
    Args:
        points_batch: Tensor of shape (B, N, 3)
        
    Returns:
        normalized_points: Tensor of shape (B, N, 3)
        centroids: Tensor of shape (B, 3)
        scales: Tensor of shape (B)
    """
    # Compute centroids (B, 3)
    centroids = torch.mean(points_batch, dim=1)
    
    # Center points (B, N, 3)
    centered_points = points_batch - centroids.unsqueeze(1)
    
    # Compute scales
    scales = (centered_points.max(dim=1)[0] - centered_points.min(dim=1)[0]).max(dim=1)[0]
    
    # Normalize
    normalized_points = centered_points / scales.unsqueeze(1).unsqueeze(2)
    
    return normalized_points, centroids, scales

def normalize_pointcloud(points):
    """
    center and scale pointcloud to unit cube
    Args:
        points:Tensor of shape (N,3)
    Returns:
        normalized points,centorid,scale
    """
    if isinstance(points, np.ndarray):
        centroid = points.mean(0)
        points = points - centroid
        scale = points.ptp()  # Max range across all dimensions
        points = points / scale
        return points, centroid, scale
    else:  # PyTorch tensor
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        scale = torch.max(points) - torch.min(points)  # ptp
        points = points / scale
        return points, centroid, scale

def voxelize_pointcloud(points,resolution=64,features=None):
    """
    Voxelize a pointcloud to sparse tensor format
    Args:
        points: Tensor of shape (N, 3) in range [-0.5, 0.5]
        resolution: Voxel grid resolution
        features: Optional point features (N, F)
    Returns:
        voxel_features: Features per occupied voxel (V, F+8)
        voxel_coords: Coordinates of occupied voxels (V, 3)
        sparse_tensor: SparseTensor representation
    """
    scaled_points = (points + 0.5)*(resolution - 1)
    voxel_indices = torch.floor(scaled_points).long()
    voxel_indices = torch.clamp(voxel_indices,0,resolution - 1)

    #create voxel hash
    voxel_hash = voxel_indices[:,0]*resolution**2 + voxel_indices[:,1]*resolution + voxel_indices[:,2]
    #find unique voxels and counts
    unique_voxels,inverse_indices,counts = torch.unique(voxel_hash, return_inverse=True, return_counts=True)

    #prepare voxel coordinates
    num_voxels = len(unique_voxels)
    voxel_coords = torch.zeros((num_voxels,3), dtype=torch.long, device = points.device)
    
    #compute voxel coordinates and centroids
    for i, h in enumerate(unique_voxels):
        mask = (voxel_hash ==h)
        coords = voxel_indices[mask][0]
        voxel_coords[i] = coords

    # compute centorids and counts as initial features
    base_features = []

    #conpute centroids by scatter mean
    centroids = scatter_mean(points, inverse_indices, dim = 0)
    base_features.append(centorids)

    #add counts as feature
    log_counts = torch.log(counts.float() + 1) / torch.log(torch.tensor(100.0, device=points.device))
    base_features.append(log_counts.unsqueeze(1))

    # compute PCA normals as additional feature
    if points.shape[0] > 10:  # Only if we have enough points
        try:
            # For each voxel, compute covariance of points
            voxel_points = []
            for i, h in enumerate(unique_voxels):
                mask = (voxel_hash == h)
                if mask.sum() >= 3:  # Need at least 3 points for PCA
                    voxel_points.append(points[mask] - centroids[i])
                else:
                    voxel_points.append(torch.zeros((3, 3), device=points.device))
            
            # Compute covariance matrices
            covs = []
            for pts in voxel_points:
                if pts.shape[0] >= 3:
                    cov = pts.t() @ pts / (pts.shape[0] - 1)
                    covs.append(cov)
                else:
                    covs.append(torch.eye(3, device=points.device))
            
            # Compute eigenvalues and eigenvectors
            covs = torch.stack(covs)  # (V, 3, 3)
            evals, evecs = torch.linalg.eigh(covs)
            
            # Get normal (eigenvector with smallest eigenvalue)
            normals = evecs[:, :, 0]  # (V, 3)
            variances = evals[:, 0].unsqueeze(1)  # (V, 1)
            
            base_features.append(normals)
            base_features.append(variances)
        except:
            # Fallback if PCA fails
            normals = torch.zeros((num_voxels, 3), device=points.device)
            variances = torch.ones((num_voxels, 1), device=points.device)
            base_features.append(normals)
            base_features.append(variances)

    #concate
    voxel_features = torch.cat(base_features, dim = 1)

    # Add point features if provided
    if features is not None:
        point_features = scatter_mean(features, inverse_indices, dim=0)
        voxel_features = torch.cat([voxel_features, point_features], dim=1)
    
    # Create sparse tensor
    indices = voxel_coords.t().contiguous()  # (3, V) for sparse API
    shape = (resolution, resolution, resolution)
    sparse_tensor = torch_sparse.SparseTensor(indices=indices, values=voxel_features, size=shape)
    
    return voxel_features, voxel_coords, sparse_tensor

class FeatureExpansionMLP(nn.Module):
    """
    MLP network to expand 8D features to 32D
    
    """
    
    def __init__(self):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
    
    def forward(self, x):
        """Forward pass to expand features
        
        Args:
            x: Input tensor of shape (N, 8) - 8D voxel features
            
        Returns:
            Expanded features of shape (N, 32)
        """
        return self.mlp(x)

def process_point_cloud(points, features=None):
    """
    Complete point cloud processing pipeline:
    1. Voxelize to get 8D features
    2. Expand features to 32D using MLP
    3. Add 3D original centroids
    4. Output 35D features
    
    Args:
        points: (N, 3) input point cloud
        features: Optional (N, F) point features
        
    Returns:
        voxel_coords: (V, 3) voxel coordinates
        final_features: (V, 35) processed features
    """
    # Normalize point cloud
    normalized_points, center, scale = normalize_pointcloud(points)
    
    # Step 1: Voxelize point cloud to get 8D features
    voxel_features, voxel_coords, _ = voxelize_pointcloud(
        normalized_points, resolution=64, features=features)
    
    # Step 2: Expand to 32D using MLP
    feature_expansion = FeatureExpansionMLP()
    expanded_features = feature_expansion(voxel_features)
    
    # Step 3: Calculate centroids in original coordinate system (3D)
    # Convert from normalized voxel space back to original space
    original_centroids = voxel_coords.float() / (64 - 1)  # [0, 1]
    original_centroids = (original_centroids * 2 - 1) * scale + center  # Back to original space
    
    # Step 4: Concatenate expanded features with original centroids for final 35D
    final_features = torch.cat([expanded_features, original_centroids], dim=1)
    
    return voxel_coords, final_features

def extract_planes_gocpp(points, normals = None, min_points = 100,
                         distance_threshold = 0.01, angle_threshold = 0.1,
                         max_planes = 40, reg_weight = 0.1):
    """
    extract palnes from point cloud using GoCoPP algorithm.
    globally consistent plane extraction with global optimization.

    Args:
        points: (N, 3) point cloud
        normals: optional (N, 3) point normals
        min_points: minimum points for a valid plane
        distance_threshold: inlier distance threshold
        angle_threshold: normal angle threshold (radians)
        max_planes: maximum number of planes to extract
        reg_weight: regularization weight for plane complexity
    
    Returns:
        planes: dict with normals, distances, inliers masks
    """
    # Estimate normals if not provided
    if normals is None:
        normals = estimate_point_normals(points)
    
    #convert to numpy for processing 
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy() 
        normals_np = normals.cpu().numpy() if normals is not None else None
        device = points.device
    else:
        points_np = points
        normals_np = normals
        device = torch.device('cpu')
    
    # Step 1: Initial local plane detection via region growing
    # Group points with similar normals in local regions
    local_planes = region_growing_segmentation(
        points_np, normals_np, 
        angle_threshold=angle_threshold,
        distance_threshold=distance_threshold
    )
    
    # Step 2: Plane merging and global refinement
    # Merge similar planes and optimize globally
    merged_planes = global_plane_refinement(
        points_np, normals_np, local_planes, 
        reg_weight=reg_weight,
        max_planes=max_planes
    )
    
    # Step 3: Convert results to torch tensors
    plane_normals = []
    plane_distances = []
    plane_masks = []
    
    for plane in merged_planes:
        if len(plane['indices']) < min_points:
            continue
            
        normal = torch.tensor(plane['normal'], device=device)
        distance = torch.tensor(plane['distance'], device=device)
        
        # Create mask for this plane
        mask = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
        mask[torch.tensor(plane['indices'], device=device)] = True
        
        plane_normals.append(normal)
        plane_distances.append(distance)
        plane_masks.append(mask)
    
    # Combine results
    if plane_normals:
        return {
            'normals': torch.stack(plane_normals),
            'distances': torch.stack(plane_distances),
            'masks': torch.stack(plane_masks)
        }
    else:
        # Return empty result
        return {
            'normals': torch.zeros((1, 3), device=device),
            'distances': torch.zeros(1, device=device),
            'masks': torch.zeros((1, points.shape[0]), dtype=torch.bool, device=device)
        }

def estimate_point_normals(points, k=30):
    """
    Estimate point normals using PCA on local neighborhoods.
    
    Args:
        points: (N, 3) point cloud
        k: number of neighbors for normal estimation
        
    Returns:
        normals: (N, 3) estimated normals
    """
    # Convert to numpy for processing if needed
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
        device = points.device
    else:
        points_np = points
        device = torch.device('cpu')
    
    try:
        import open3d as o3d
        
        # Use Open3D for normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        
        # Estimate normals with k nearest neighbors
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k))
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k)
        
        # Convert back to tensor
        normals = torch.tensor(np.asarray(pcd.normals), device=device)
        return normals
        
    except ImportError:
        # Fallback implementation using numpy/sklearn
        from sklearn.neighbors import NearestNeighbors
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(points_np)
        distances, indices = nbrs.kneighbors(points_np)
        
        # Compute normals using PCA
        normals = np.zeros_like(points_np)
        
        for i in range(points_np.shape[0]):
            # Get neighbors
            neighbors = points_np[indices[i]]
            
            # Center points
            centered = neighbors - np.mean(neighbors, axis=0)
            
            # Compute covariance matrix
            cov = centered.T @ centered
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Normal is eigenvector corresponding to smallest eigenvalue
            normals[i] = eigenvectors[:, 0]
            
        # Convert to tensor
        normals = torch.tensor(normals, device=device)
        return normals

def region_growing_segmentation(points, normals, angle_threshold=0.1, distance_threshold=0.01):
    """
    Segment point cloud into local planes using region growing.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) point normals
        angle_threshold: maximum normal angle difference (radians)
        distance_threshold: maximum point-to-plane distance
        
    Returns:
        local_planes: list of plane segments
    """
    try:
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Region growing segmentation
        segments = o3d.segmentation.region_growing(
            pcd,
            max_point_to_plane_distance=distance_threshold,
            max_normal_angle_diff=angle_threshold
        )
        
        # Convert to plane representation
        local_planes = []
        
        for indices in segments:
            if len(indices) < 10:  # Minimum points for a reliable plane
                continue
                
            # Extract segment points and normals
            segment_points = np.asarray(pcd.points)[indices]
            segment_normals = np.asarray(pcd.normals)[indices]
            
            # Compute average normal
            avg_normal = np.mean(segment_normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            
            # Compute centroid
            centroid = np.mean(segment_points, axis=0)
            
            # Compute plane distance (d in plane equation ax+by+cz+d=0)
            distance = -np.dot(avg_normal, centroid)
            
            local_planes.append({
                'normal': avg_normal,
                'distance': distance,
                'indices': indices,
                'centroid': centroid
            })
        
        return local_planes
        
    except ImportError:
        # Fallback implementation without Open3D
        # This is a simplified version of region growing
        from sklearn.neighbors import NearestNeighbors
        
        # Build KD-tree for nearest neighbor search
        nbrs = NearestNeighbors(n_neighbors=20).fit(points)
        
        # Initialize visited flags
        visited = np.zeros(points.shape[0], dtype=bool)
        
        # Store plane segments
        local_planes = []
        
        # Process each point
        for seed_idx in range(points.shape[0]):
            if visited[seed_idx]:
                continue
                
            # Initialize new segment
            segment = [seed_idx]
            seed_queue = [seed_idx]
            visited[seed_idx] = True
            
            # Region growing
            while seed_queue:
                current_idx = seed_queue.pop(0)
                current_normal = normals[current_idx]
                current_point = points[current_idx]
                
                # Find neighbors
                distances, indices = nbrs.kneighbors([points[current_idx]])
                neighbors_idx = indices[0]
                
                for neighbor_idx in neighbors_idx:
                    if visited[neighbor_idx]:
                        continue
                        
                    # Check normal similarity
                    neighbor_normal = normals[neighbor_idx]
                    angle_diff = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1, 1))
                    
                    if angle_diff > angle_threshold:
                        continue
                        
                    # Check point-to-plane distance
                    neighbor_point = points[neighbor_idx]
                    dist = np.abs(np.dot(current_normal, neighbor_point - current_point))
                    
                    if dist > distance_threshold:
                        continue
                        
                    # Add to segment
                    segment.append(neighbor_idx)
                    seed_queue.append(neighbor_idx)
                    visited[neighbor_idx] = True
            
            # Create plane from segment
            if len(segment) < 10:
                continue
                
            segment_points = points[segment]
            segment_normals = normals[segment]
            
            # Compute average normal
            avg_normal = np.mean(segment_normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            
            # Compute centroid
            centroid = np.mean(segment_points, axis=0)
            
            # Compute plane distance
            distance = -np.dot(avg_normal, centroid)
            
            local_planes.append({
                'normal': avg_normal,
                'distance': distance,
                'indices': segment,
                'centroid': centroid
            })
        
        return local_planes

def global_plane_refinement(points, normals, local_planes, reg_weight=0.1, max_planes=40, iterations=5):
    """
    Refine planes globally with a joint optimization.
    This is the key component of GoCoPP algorithm.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) point normals
        local_planes: list of initial plane segments
        reg_weight: regularization weight
        max_planes: maximum number of planes
        iterations: number of optimization iterations
        
    Returns:
        refined_planes: list of refined plane parameters
    """
    # If too few local planes detected, return as is
    if len(local_planes) < 2:
        return local_planes
    
    # Extract plane parameters
    plane_params = []
    for plane in local_planes:
        plane_params.append({
            'normal': plane['normal'],
            'distance': plane['distance'],
            'support': len(plane['indices']),
            'indices': set(plane['indices'])
        })
    
    # Sort planes by support (number of points)
    plane_params.sort(key=lambda x: x['support'], reverse=True)
    
    # Keep only top-k planes
    if len(plane_params) > max_planes:
        plane_params = plane_params[:max_planes]
    
    # Iterative global refinement
    for _ in range(iterations):
        # Step 1: Assign points to planes
        point_assignments = np.full(points.shape[0], -1)
        
        for i in range(len(plane_params)):
            plane_normal = plane_params[i]['normal']
            plane_distance = plane_params[i]['distance']
            
            # Compute point-to-plane distances for all points
            distances = np.abs(np.dot(points, plane_normal) + plane_distance)
            
            # Find points close to this plane
            valid_points = np.where(distances < 0.02)[0]  # Slightly larger threshold for reassignment
            
            for p_idx in valid_points:
                if point_assignments[p_idx] == -1 or \
                   distances[p_idx] < np.abs(np.dot(points[p_idx], 
                                            plane_params[point_assignments[p_idx]]['normal']) + 
                                            plane_params[point_assignments[p_idx]]['distance']):
                    point_assignments[p_idx] = i
        
        # Update plane parameters
        for i in range(len(plane_params)):
            assigned_points = np.where(point_assignments == i)[0]
            
            if len(assigned_points) < 10:
                continue
                
            # Get points and normals
            plane_points = points[assigned_points]
            plane_normals = normals[assigned_points]
            
            # SVD-based plane fitting
            centroid = np.mean(plane_points, axis=0)
            centered = plane_points - centroid
            
            # Weight by normal consistency
            weights = np.abs(np.dot(plane_normals, plane_params[i]['normal']))
            weights = weights / np.sum(weights)
            
            # Compute covariance matrix with weights
            cov = (centered.T * weights) @ centered
            
            # SVD decomposition
            u, s, vh = np.linalg.svd(cov)
            
            # Update normal (smallest eigenvector)
            new_normal = vh[2]
            
            # Ensure normal points in consistent direction
            if np.dot(new_normal, plane_params[i]['normal']) < 0:
                new_normal = -new_normal
                
            # Update distance
            new_distance = -np.dot(new_normal, centroid)
            
            # Update plane parameters
            plane_params[i]['normal'] = new_normal
            plane_params[i]['distance'] = new_distance
            plane_params[i]['indices'] = set(assigned_points.tolist())
            plane_params[i]['support'] = len(assigned_points)
        
        # Remove planes with insufficient support
        plane_params = [p for p in plane_params if p['support'] >= 10]
        
        # Check for similar planes and merge them
        i = 0
        while i < len(plane_params):
            j = i + 1
            while j < len(plane_params):
                # Check if planes are similar
                normal_dot = np.abs(np.dot(plane_params[i]['normal'], plane_params[j]['normal']))
                dist_diff = np.abs(plane_params[i]['distance'] - plane_params[j]['distance'])
                
                if normal_dot > 0.95 and dist_diff < 0.05:
                    # Merge planes
                    # Use the parameters of the plane with more support
                    if plane_params[i]['support'] >= plane_params[j]['support']:
                        plane_params[i]['indices'].update(plane_params[j]['indices'])
                        plane_params[i]['support'] = len(plane_params[i]['indices'])
                        plane_params.pop(j)
                    else:
                        plane_params[j]['indices'].update(plane_params[i]['indices'])
                        plane_params[j]['support'] = len(plane_params[j]['indices'])
                        plane_params.pop(i)
                        i -= 1
                        break
                else:
                    j += 1
            i += 1
    
    # Convert set of indices to list
    for plane in plane_params:
        plane['indices'] = list(plane['indices'])
    
    return plane_params

def farthest_point_sampling(points, npoint):
    """
    Farthest point sampling algorithm.
    
    Args:
        points: (N, 3) - Input point cloud
        npoint: number of points to sample
        
    Returns:
        indices: (npoint) - Indices of sampled points
    """
    device = points.device
    N, D = points.shape
    
    # Ensure npoint doesn't exceed point cloud size
    npoint = min(npoint, N)
    
    # Store indices of selected points
    indices = torch.zeros(npoint, dtype=torch.long, device=device)
    
    # Store minimum distance from each point to the current set of selected points
    distances = torch.ones(N, device=device) * 1e10
    
    # Randomly select the first point
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    # Iteratively select the farthest points
    for i in range(npoint):
        # Save index of current farthest point
        indices[i] = farthest
        
        # Get coordinates of current point
        centroid = points[farthest, :].view(1, D)
        
        # Compute squared distances from all points to current point
        dist = torch.sum((points - centroid) ** 2, dim=1)
        
        # Update minimum distances
        mask = dist < distances
        distances[mask] = dist[mask]
        
        # Select point with maximum minimum distance as next point
        farthest = torch.max(distances, dim=0)[1]
        
    return indices

# Add multi-scale FPS sampling
def multi_scale_fps(points, scales):
    """
    Perform multi-scale farthest point sampling as in PointNet++.
    
    Args:
        points: (N, 3) - Input point cloud
        scales: list of int - Number of points to sample at each scale
        
    Returns:
        sampled_points_list: list of (S_i, 3) tensors - Points at different scales
        sampled_indices_list: list of (S_i) tensors - Indices of sampled points at each scale
    """
    sampled_points_list = []
    sampled_indices_list = []
    
    for scale in scales:
        # Perform FPS sampling
        fps_indices = farthest_point_sampling(points, scale)
        sampled_points = points[fps_indices]
        
        sampled_points_list.append(sampled_points)
        sampled_indices_list.append(fps_indices)
    
    return sampled_points_list, sampled_indices_list
