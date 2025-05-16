"""
PVD: Parameter Voxel Diffusion Model
A deep learning framework for point cloud generation and processing
"""

__version__ = "0.1.0"

from .pipeline import PVD
from .models import DiffusionModel, SparseVoxelTransformer
from .utils import setup_logger, set_random_seed

# Configure default logger
from .utils.logger import setup_logger
logger = setup_logger()

def get_version():
    """Return the current version of PVD"""
    return __version__