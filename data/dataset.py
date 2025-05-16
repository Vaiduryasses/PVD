"""
Dataset classes 

"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from pathlib import Path
from ..config import cfg
from .preprocessing import normalize_pointcloud, voxelize_pointcloud, extract_planes_gocopp,farthest_point_sampling

class PointCloudDataset(Dataset):
    """
    Dataset for loading and preprocessing point clouds

    """
    def __init__(self, data_path, spilt = 'train', transform = None,
                 num_points = 8192, voxel_resolution = 64, cache_dir = None):
        """
        Args:
            data_path:Path to dataset
            spilt:'train','val', or'test'
            transform:Optional transform function
            num_points: Number of points to sample
            voxel_resolution: Voxel grid resolution
            cache_dir: Option direction for caching preprocessed data
        """
        self.data_path = data_path
        self.spilt = spilt
        self.transform = transform
        self.num_points = num_points
        self.voxel_resolution = voxel_resolution
        self.cache_dir = cache_dir
        
        #find point cloud files
        self.files = []
        #for val
        if spilt == 'val':
            extensions = ['.npy', '.obj']
        else: #for train and test
            extensions = ['.npy']
        for ext in extensions:
            self.files.extend(glob.glob(os.path.join(data_path, spilt, f'**/*{ext}'), recursive=True ))
        
        #sort for reproducibility
        self.files.sort()

        print(f"Found{len(self.files)} point clouds in {spilt} spilt")

        #create cache if needed
        if cache_dir is not None:
            os.makedirs(os.path.join(cache_dir, spilt), exist_ok=True)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = Path(file_path).stem

        #check cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, self.spilt, f"{file_name}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                points = torch.from_numpy(data['points']).float()
                voxel_features = torch.from_numpy(data['voxel_features']).float()
                voxel_coords = torch.from_numpy(data['voxel_coords']).long()

                if 'planes_normals' in data:
                    gt_planes = {
                        'normals': torch.from_numpy(data['planes_normals']).float(),
                        'distances': torch.from_numpy(data['planes_distances']).float(),
                        'masks': torch.from_numpy(data['planes_masks']).bool()
                    }
                else:
                    gt_planes = None
                
                return {
                    'points': points,
                    'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'gt_planes': gt_planes,
                    'file_name': file_name
                }

        #load point cloud
        points = self._load_point_cloud(points)
        
        #normalize
        points, centroid, scale = normalize_pointcloud(points)

        #sample points if needed
        if points.shape[0] > self.num_points:
            points_tensor = torch.from_numpy(points).float()
            indices = farthest_point_sampling(points_tensor, self.num_points)
            points = points_tensor[indices].numpy()

        #apply transform if provided
        if self.transform is not None:
            points = self.transform(points)
        
        #convert to tensor
        points = torch.from_numpy(points).float() if isinstance(points, np.ndarray) else points

        #voxelize
        voxel_features, voxel_coords, _ = voxelize_pointcloud(points, self.voxel_resolution)

        #extract planes for GT supervision using GoCoPP
        gt_planes = extract_planes_gocopp(
            points,
            min_points = 100,
            distancce_threshold = 0.01,
            angle_threshold = 0.1,
            max_planes = cfg.num_plane_queries
        )

        # Cache preprocessing results
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, self.split, f"{file_name}.npz")
            np.savez(
                cache_path,
                points=points.cpu().numpy(),
                voxel_features=voxel_features.cpu().numpy(),
                voxel_coords=voxel_coords.cpu().numpy(),
                planes_normals=gt_planes['normals'].cpu().numpy(),
                planes_distances=gt_planes['distances'].cpu().numpy(),
                planes_masks=gt_planes['masks'].cpu().numpy()
            )
        
        return {
            'points': points,
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'gt_planes': gt_planes,
            'file_name': file_name
        }

    def _load_point_cloud(self, file_path):
        """
        Load point cloud from file(optimized primarily for NPY, with OBJ support for val)

        """
        extension = os.path.splitext(file_path)[1].lower()

        if extension == '.npy':
            try:
                #load npy file directly with an array of points contained
                points = np.load(file_path)

                #make sure its the right shape(N,3)
                if points.ndim == 3 and points.shape[1] == 3:
                    #this is probably (B,N,3) format - take first batch
                    points = points[0]
                elif points.ndim > 2:
                    #try to reshape to (N,3)
                    points = points.reshape(-1,3)
                
                #ensure we have float32 type
                points = points.astype(np.float32)

                #check if the point cloud has valid points
                if points.shape[0] == 0:
                    raise ValueError(f"Empty point cloud in {file_path}")

                return points
            except Exception as e:
                raise RuntimeError(f"Failed to load NPY file: {e}")
        elif extension == '.obj':
            #obj is only used for val
            try:
                import trimesh
                mesh = trimesh.load(file_path)
                points = np.asarray(mesh.vertices, dtype = np.float32)

                #if mesh has faces, sample points on the surfaces for more uniform distribution
                if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                    points = trimesh.sample.sample_surface(mesh, self.num_points * 2)[0]
                    #take a subset if needed
                    if points.shape[0] > self.num_points:
                        points_tensor = torch.from_numpy(points).float()
                        indices = farthest_point_sampling(points_tensor, self.num_points)
                        points = points_tensor[indices].numpy()
                    
                return points
            except ImportError:
                raise ImportError("Trimesh is required for loading OBJ files in validation")
                
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Only NPY and OBJ (validation) are supported.")
