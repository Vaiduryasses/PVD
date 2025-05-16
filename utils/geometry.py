"""
Geometry utilities for point cloud processing.
Provides functions for geometric operations on point clouds.
"""
import torch
import numpy as np
import math

def compute_point_normals(points, k=20):
    """
    Compute point normals by fitting planes to local neighborhoods
    
    Args:
        points: (N, 3) tensor of point coordinates
        k: number of neighbors for normal estimation
        
    Returns:
        normals: (N, 3) tensor of unit normals
    """
    # Move to CPU for KNN computation if points is on GPU
    device = points.device
    use_cpu = device.type == 'cuda' and points.shape[0] > 10000
    points_for_nn = points.cpu() if use_cpu else points
    
    # Compute pairwise distances
    N = points_for_nn.shape[0]
    dist = torch.cdist(points_for_nn, points_for_nn)
    
    # Get k nearest neighbors for each point
    _, indices = torch.topk(dist, k=k+1, dim=1, largest=False)
    indices = indices[:, 1:]  # Exclude self
    
    # Move indices back to the original device
    indices = indices.to(device)
    
    # Initialize normals
    normals = torch.zeros_like(points)
    
    # Compute normal for each point using PCA on its neighborhood
    for i in range(points.shape[0]):
        # Get neighbors for this point
        neighbors = points[indices[i]]
        
        # Center the neighborhood
        centroid = torch.mean(neighbors, dim=0)
        centered = neighbors - centroid
        
        # Compute covariance matrix
        cov = torch.matmul(centered.t(), centered)
        
        # Perform eigen decomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # The eigenvector corresponding to the smallest eigenvalue is the normal
            normal = eigenvectors[:, 0]
            
            # Ensure normal points outward (heuristic: away from centroid of all points)
            center_to_point = points[i] - torch.mean(points, dim=0)
            if torch.dot(normal, center_to_point) < 0:
                normal = -normal
                
            normals[i] = normal
        except:
            # Fallback if eigendecomposition fails
            normals[i] = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    return normals

def estimate_point_curvature(points, normals, k=20):
    """
    Estimate point curvature from normal variations
    
    Args:
        points: (N, 3) tensor of point coordinates
        normals: (N, 3) tensor of point normals
        k: number of neighbors for curvature estimation
        
    Returns:
        curvature: (N,) tensor of curvature values
    """
    # Move to CPU for KNN computation if points is on GPU
    device = points.device
    use_cpu = device.type == 'cuda' and points.shape[0] > 10000
    points_for_nn = points.cpu() if use_cpu else points
    
    # Compute pairwise distances
    dist = torch.cdist(points_for_nn, points_for_nn)
    
    # Get k nearest neighbors for each point
    _, indices = torch.topk(dist, k=k+1, dim=1, largest=False)
    indices = indices[:, 1:]  # Exclude self
    
    # Move indices back to the original device
    indices = indices.to(device)
    
    # Initialize curvature values
    curvature = torch.zeros(points.shape[0], device=device)
    
    # Compute curvature for each point
    for i in range(points.shape[0]):
        # Get normals of neighbors
        neighbor_normals = normals[indices[i]]
        
        # Compute variations of normals (using dot product)
        normal_variations = 1.0 - torch.abs(torch.sum(normals[i].unsqueeze(0) * neighbor_normals, dim=1))
        
        # Curvature is the average variation
        curvature[i] = torch.mean(normal_variations)
    
    return curvature

def extract_plane_segments(points, normals, angle_threshold=10, distance_threshold=0.05, min_points=100):
    """
    Extract planar segments from point cloud
    
    Args:
        points: (N, 3) tensor of point coordinates
        normals: (N, 3) tensor of point normals
        angle_threshold: maximum angle difference in degrees
        distance_threshold: maximum distance for region growing
        min_points: minimum number of points in a segment
        
    Returns:
        segments: list of boolean masks for each segment
    """
    device = points.device
    N = points.shape[0]
    
    # Convert angle threshold to cosine similarity threshold
    angle_threshold_rad = math.radians(angle_threshold)
    cos_threshold = math.cos(angle_threshold_rad)
    
    # Initialize array to track unassigned points
    unassigned = torch.ones(N, dtype=torch.bool, device=device)
    segments = []
    
    # Region growing algorithm
    while torch.sum(unassigned) > min_points:
        # Find seed point (use point with lowest curvature)
        curvature = estimate_point_curvature(points[unassigned], normals[unassigned], k=20)
        seed_idx = torch.argmin(curvature)
        
        # Mapping from reduced index to original index
        original_indices = torch.where(unassigned)[0]
        seed_idx_original = original_indices[seed_idx]
        
        # Initialize current segment
        current_segment = torch.zeros(N, dtype=torch.bool, device=device)
        current_segment[seed_idx_original] = True
        
        # Initialize seed normal and point
        seed_normal = normals[seed_idx_original]
        seed_point = points[seed_idx_original]
        
        # Track points added in the last iteration
        last_added = current_segment.clone()
        
        # Region growing
        while torch.any(last_added):
            # Reset last added
            new_added = torch.zeros(N, dtype=torch.bool, device=device)
            
            # Get indices of last added points
            last_added_indices = torch.where(last_added)[0]
            
            # For each point in the current segment, check its neighbors
            for idx in last_added_indices:
                # Compute distances to all unassigned points
                distances = torch.sum((points - points[idx].unsqueeze(0)) ** 2, dim=1)
                
                # Find nearby unassigned points
                nearby = (distances < distance_threshold ** 2) & unassigned & (~current_segment)
                
                if not torch.any(nearby):
                    continue
                
                # Check normal consistency
                nearby_indices = torch.where(nearby)[0]
                for nearby_idx in nearby_indices:
                    normal_similarity = torch.dot(seed_normal, normals[nearby_idx])
                    
                    # Check if normal is consistent
                    if normal_similarity > cos_threshold:
                        # Check distance to plane
                        point_to_plane = torch.dot(points[nearby_idx] - seed_point, seed_normal)
                        if abs(point_to_plane) < distance_threshold:
                            current_segment[nearby_idx] = True
                            new_added[nearby_idx] = True
            
            last_added = new_added
        
        # Update unassigned points
        if torch.sum(current_segment) > min_points:
            segments.append(current_segment)
            unassigned = unassigned & (~current_segment)
        else:
            # Mark these points as assigned to avoid getting stuck
            unassigned[current_segment] = False
            
        # Break if too few points remain
        if torch.sum(unassigned) < min_points:
            break
    
    return segments

def compute_mesh_from_points(points, normals=None, depth=8, width=5, scale=1.0):
    """
    Compute a mesh from a point cloud using Poisson surface reconstruction
    
    Args:
        points: (N, 3) tensor of point coordinates
        normals: (N, 3) tensor of point normals (computed if None)
        depth: octree depth for reconstruction
        width: width parameter for reconstruction
        scale: scale factor for mesh
        
    Returns:
        vertices: (V, 3) tensor of mesh vertices
        faces: (F, 3) tensor of mesh faces
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for mesh reconstruction. Install it with `pip install open3d`")
    
    # Convert to numpy for Open3D
    points_np = points.cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # Compute normals if not provided
    if normals is None:
        pcd.estimate_normals()
    else:
        normals_np = normals.cpu().numpy()
        pcd.normals = o3d.utility.Vector3dVector(normals_np)
    
    # Run Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=False)
    
    # Extract vertices and faces
    vertices_np = np.asarray(mesh.vertices)
    faces_np = np.asarray(mesh.triangles)
    
    # Convert back to torch tensors
    vertices = torch.tensor(vertices_np, device=points.device)
    faces = torch.tensor(faces_np, device=points.device)
    
    return vertices, faces

def create_voxel_grid(points, voxel_size, grid_extent=None):
    """
    Create a voxel grid from point cloud
    
    Args:
        points: (N, 3) tensor of point coordinates
        voxel_size: size of voxels
        grid_extent: (min_xyz, max_xyz) bounds of grid, computed from points if None
        
    Returns:
        voxel_grid: boolean 3D tensor of occupied voxels
        voxel_indices: (M, 3) tensor of occupied voxel indices
    """
    device = points.device
    
    # Compute grid extent if not provided
    if grid_extent is None:
        min_bounds = torch.min(points, dim=0)[0] - voxel_size
        max_bounds = torch.max(points, dim=0)[0] + voxel_size
    else:
        min_bounds, max_bounds = grid_extent
    
    # Compute grid dimensions
    grid_dims = ((max_bounds - min_bounds) / voxel_size).ceil().int() + 1
    
    # Quantize point coordinates to voxel indices
    voxel_indices = ((points - min_bounds) / voxel_size).int()
    
    # Ensure indices are in bounds
    voxel_indices = torch.clamp(voxel_indices, 
                               min=torch.zeros(3, device=device).int(),
                               max=grid_dims - 1)
    
    # Create a unique hash for each voxel
    voxel_hash = (voxel_indices[:, 0] * grid_dims[1] * grid_dims[2] + 
                  voxel_indices[:, 1] * grid_dims[2] + 
                  voxel_indices[:, 2])
    
    # Find unique voxels
    unique_hashes, _ = torch.unique(voxel_hash, return_inverse=True)
    
    # Create voxel grid
    voxel_grid = torch.zeros(grid_dims, dtype=torch.bool, device=device)
    
    # Convert unique hashes back to 3D indices
    unique_indices = torch.zeros((len(unique_hashes), 3), dtype=torch.int64, device=device)
    
    for i, h in enumerate(unique_hashes):
        # Find any point that maps to this voxel
        point_idx = torch.where(voxel_hash == h)[0][0]
        unique_indices[i] = voxel_indices[point_idx]
        
        # Mark voxel as occupied
        voxel_grid[unique_indices[i, 0], unique_indices[i, 1], unique_indices[i, 2]] = True
    
    return voxel_grid, unique_indices

def signed_distance_function(points, mesh_vertices, mesh_faces):
    """
    Compute signed distance function from points to mesh
    
    Args:
        points: (N, 3) tensor of query points
        mesh_vertices: (V, 3) tensor of mesh vertices
        mesh_faces: (F, 3) tensor of mesh faces
        
    Returns:
        sdf: (N,) tensor of signed distances
    """
    try:
        import trimesh
        import igl
    except ImportError:
        raise ImportError("This function requires trimesh and libigl. Install with: pip install trimesh igl")
    
    # Convert to numpy for trimesh
    points_np = points.cpu().numpy()
    vertices_np = mesh_vertices.cpu().numpy()
    faces_np = mesh_faces.cpu().numpy()
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    
    # Compute signed distance using libigl
    sdf_np, _, _ = igl.signed_distance(points_np, vertices_np, faces_np)
    
    # Convert back to torch tensor
    sdf = torch.tensor(sdf_np, device=points.device)
    
    return sdf

def transform_points(points, rotation=None, translation=None, scale=None):
    """
    Apply rigid transformation to points
    
    Args:
        points: (N, 3) tensor of point coordinates
        rotation: (3, 3) rotation matrix or (4,) quaternion [w, x, y, z]
        translation: (3,) translation vector
        scale: scalar or (3,) per-axis scale
        
    Returns:
        transformed_points: (N, 3) tensor of transformed points
    """
    transformed_points = points.clone()
    
    # Apply rotation
    if rotation is not None:
        if rotation.numel() == 4:  # Quaternion
            # Convert quaternion to rotation matrix
            w, x, y, z = rotation
            
            # Build rotation matrix
            rot_matrix = torch.zeros(3, 3, device=points.device)
            rot_matrix[0, 0] = 1 - 2 * (y * y + z * z)
            rot_matrix[0, 1] = 2 * (x * y - w * z)
            rot_matrix[0, 2] = 2 * (x * z + w * y)
            rot_matrix[1, 0] = 2 * (x * y + w * z)
            rot_matrix[1, 1] = 1 - 2 * (x * x + z * z)
            rot_matrix[1, 2] = 2 * (y * z - w * x)
            rot_matrix[2, 0] = 2 * (x * z - w * y)
            rot_matrix[2, 1] = 2 * (y * z + w * x)
            rot_matrix[2, 2] = 1 - 2 * (x * x + y * y)
            
            rotation = rot_matrix
        
        # Apply rotation matrix
        transformed_points = torch.matmul(transformed_points, rotation.transpose(0, 1))
    
    # Apply scale
    if scale is not None:
        if isinstance(scale, (int, float)) or scale.numel() == 1:
            transformed_points = transformed_points * scale
        else:  # Per-axis scale
            transformed_points = transformed_points * scale.unsqueeze(0)
    
    # Apply translation
    if translation is not None:
        transformed_points = transformed_points + translation.unsqueeze(0)
    
    return transformed_points