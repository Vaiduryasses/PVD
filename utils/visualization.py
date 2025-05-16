"""
Visualization utilities

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_pointcloud(input_points, output_points, save_path=None):
    """
    visualize input and output point clouds.
    
    Args:
        input_points: (N, 3) - Input point cloud
        output_points: (M, 3) - Output point cloud
        save_path: Optional path to save visualization
    """
    fig = plt.figure(figsize=(12, 6))
    
    #Plot input point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], 
               c='r', s=1, alpha=0.5)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(False)
    
    #Plot output point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(output_points[:, 0], output_points[:, 1], output_points[:, 2], 
               c='b', s=1, alpha=0.5)
    ax2.set_title('Output Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def visualize_planes(points, normals, distances, save_path=None):
    """
    visualize detected planes.
    
    Args:
        points: (N, 3) - Point cloud
        normals: (P, 3) - Plane normals
        distances: (P,) - Plane distances
        save_path: Optional path to save visualization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    #Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='gray', s=1, alpha=0.3, label='Points')
    
    #Plot planes as wireframe
    colors = plt.cm.tab10(np.linspace(0, 1, len(normals)))
    
    for i, (normal, distance) in enumerate(zip(normals, distances)):
        #Create a basis for the plane
        basis_1 = np.array([1, 0, 0])
        if np.abs(np.dot(basis_1, normal)) > 0.9:
            basis_1 = np.array([0, 1, 0])
        basis_1 = basis_1 - np.dot(basis_1, normal) * normal
        basis_1 = basis_1 / np.linalg.norm(basis_1)
        
        basis_2 = np.cross(normal, basis_1)
        basis_2 = basis_2 / np.linalg.norm(basis_2)
        
        #Create grid for plane
        xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
        grid = np.zeros((5, 5, 3))
        
        for i_x in range(5):
            for i_y in range(5):
                point = normal * distance + basis_1 * xx[i_x, i_y] + basis_2 * yy[i_x, i_y]
                grid[i_x, i_y] = point
        
        #Plot plane
        ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], 
                       color=colors[i], alpha=0.3)
        
        #Plot normal vector at plane center
        center = normal * distance
        ax.quiver(center[0], center[1], center[2], 
                 normal[0], normal[1], normal[2], 
                 color=colors[i], length=0.2, normalize=True)
    
    #Set axis limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_title(f'Detected Planes ({len(normals)})')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()