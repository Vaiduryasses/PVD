"""
Point Cloud Visualization Tools
Simplified version with only the core visualization functions
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import imageio
import os
import torch

def visualize_point_cloud(input_points, output_points, title="Point Cloud Comparison"):
    """
    Visualize comparison between input and output point clouds
    
    Args:
        input_points: Input point cloud as numpy array (N, 3)
        output_points: Output point cloud as numpy array (M, 3)
        title: Visualization title
    """
    # Create Open3D point cloud objects
    input_pc = o3d.geometry.PointCloud()
    input_pc.points = o3d.utility.Vector3dVector(input_points)
    input_pc.paint_uniform_color([1, 0, 0])  # Red for input point cloud
    
    output_pc = o3d.geometry.PointCloud()
    output_pc.points = o3d.utility.Vector3dVector(output_points)
    output_pc.paint_uniform_color([0, 0, 1])  # Blue for output point cloud
    
    # Visualize
    o3d.visualization.draw_geometries([input_pc, output_pc], window_name=title)

def save_point_cloud(points, path):
    """
    Save point cloud to PLY file
    
    Args:
        points: Point cloud as numpy array (N, 3)
        path: Save path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

def _visualize_point_cloud_matplotlib(points, color='b', title="Point Cloud", size=1):
    """
    Internal helper function to visualize a single point cloud using matplotlib
    
    Args:
        points: Point cloud as numpy array (N, 3)
        color: Point color
        title: Visualization title
        size: Point size
    
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    ax.scatter(xs, ys, zs, c=color, s=size, alpha=0.5)
    
    # Set axes scale to maintain point cloud shape
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def visualize_diffusion_steps(step_points_list, output_path):
    """
    Create visualization GIF of diffusion steps
    
    Args:
        step_points_list: List of diffusion step point clouds, each (N, 3) numpy array
        output_path: Output GIF path
    """
    # Convert tensor to numpy if needed
    step_points_list_np = []
    for points in step_points_list:
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        step_points_list_np.append(points)
    
    # Create temporary directory for frames
    temp_dir = os.path.dirname(output_path) + "/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create image for each step
    filenames = []
    for i, points in enumerate(step_points_list_np):
        # Color final step blue, intermediate steps red to yellow gradient
        if i == len(step_points_list_np)-1:
            color = 'b'  # Final output is blue
        else:
            # Gradient from red (noise) to yellow (near-final)
            progress = i / (len(step_points_list_np)-1)
            color = (1.0, progress, 0)
        
        fig = _visualize_point_cloud_matplotlib(
            points, 
            color=color, 
            title=f"Diffusion Step {i}/{len(step_points_list_np)-1}"
        )
        filename = f"{temp_dir}/step_{i:03d}.png"
        fig.savefig(filename)
        plt.close(fig)
        filenames.append(filename)
    
    # Create GIF
    with imageio.get_writer(output_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # Clean up temporary files
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)