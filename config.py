"""
Configuration settings
"""
import os
import torch
import yaml
from datetime import datetime


class Config:
    """
    Configuration class that manages all settings for model architecture,
    training, data processing, and system paths.
    """
    
    def __init__(self):
        # ===== General system settings =====
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = 'output'
        self.exp_name = f"pvd{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.num_workers = 4
        self.debug = False
        
        # ===== Voxelization settings =====
        self.voxel_resolution = 64    # voxel grid R
        self.normalize = True         # whether to normalize point cloud
        self.normalize_pointcloud = True
        
        # ===== Feature processing settings =====
        self.feature_mlp_hidden = 16  # Hidden dimension in feature expansion MLP
        self.base_feature_dim = 8     # Dimension of base features (3D centroid + 1D count + 3D normal + 1D variance)
        self.expanded_feature_dim = 32 # Dimension after MLP expansion
        self.final_feature_dim = 35    # Final dimension with original centroid added (32 + 3)
        
        # ===== Voxel Transformer settings =====
        self.transformer_dim = 256
        self.transformer_depth = 8
        self.transformer_heads = 8
        self.transformer_window_size = 3
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.path_dropout = 0.1
        
        # Aliased names for compatibility
        self.encoder_layers = 8
        self.encode_nhead = 8
        self.encoder_dim = 256
        self.window_size = 3
        self.num_plane_queries = 40
        self.proxy_threshold = 0.4    # logit threshold
        
        # ===== PVF settings =====
        self.fps_scales = [1024, 512, 256, 128]
        self.pvf_dim = 256
        
        # ===== Diffusion settings =====
        self.diffusion_steps_train = 30
        self.diffusion_steps_infer = 6
        self.diffusion_dim = 256
        self.diffusion_encoder_layers = 6
        self.diffusion_decoder_layers = 4
        self.bata_start = 1e-4
        self.beta_end = 0.02
        
        # ===== Refinement settings =====
        self.residual_knn = 16
        self.plane_proj_threshold = 0.01  # τ for plane projection
        
        # ===== Loss weights =====
        self.alpha_param = 0.5    # proxy parameter loss weight
        self.beta_chamfer = 20.0
        self.lambda_diff = 1.0
        self.lambda_res = 0.1
        
        # ===== Training settings =====
        self.batch_size = 8
        self.accumulate_grad_batches = 1
        self.learning_rate = 1e-4
        self.lr = 1e-4              # alias for learning_rate
        self.weight_decay = 1e-5
        self.warmup_epochs = 10
        self.max_epochs = 300
        self.epochs = 100           # alias for compatibility
        self.early_stopping_patience = 20
        self.gradient_clip_val = 1.0
        
        # ===== Optimizer settings =====
        self.optimizer_type = 'adamw'  # 'adam', 'adamw', 'sgd'
        self.scheduler_type = 'cosine' # 'step', 'cosine', 'plateau'
        self.scheduler_step_size = 30
        self.scheduler_gamma = 0.1
        
        # ===== Data settings =====
        self.data_path = "./data"
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        self.use_augmentation = True
        self.max_points = 100000      # Maximum number of points to use from each point cloud
        self.min_points = 100         # Minimum number of points required in a point cloud
        
        # ===== Augmentation settings =====
        self.aug_rotate = True
        self.aug_flip = True
        self.aug_scale_min = 0.8
        self.aug_scale_max = 1.2
        self.aug_translate = 0.1
        self.aug_jitter = 0.01
        self.aug_shuffle = True
        
        # ===== Output settings =====
        self.output_points = 16384  # Number of output points
        self.Output_points = 16384  # Alias with capital letter for compatibility
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.sample_dir = os.path.join(self.output_dir, "samples")
        
        # ===== Logging settings =====
        self.log_every_n_steps = 50
        self.val_check_interval = 1.0  # Fraction of epoch or integer steps
        
        # ===== Experiment tracking =====
        self.exp_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for outputs."""
        dirs = [self.output_dir, self.checkpoint_dir, self.log_dir, 
                self.results_dir, self.sample_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def update(self, config_dict):
        """Update config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}'")
    
    def save(self, filepath):
        """Save configuration to YAML file."""
        # Create a dictionary of all attributes
        config_dict = {key: value for key, value in self.__dict__.items() 
                       if not key.startswith('_') and not callable(value)}
        
        # Remove non-serializable objects
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def load(self, filepath):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.update(config_dict)
        
        # Restore non-serializable objects
        if 'device' in config_dict and isinstance(config_dict['device'], str):
            if config_dict['device'].startswith('cuda'):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(config_dict['device'])
    
    @classmethod
    def from_yaml(cls, yaml_file):
        """Load config from YAML file."""
        cfg = cls()
        if not os.path.exists(yaml_file):
            print(f"Config file {yaml_file} not found. Using default config.")
            return cfg
            
        with open(yaml_file, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            
        for k, v in yaml_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                
        return cfg
    
    def to_yaml(self, yaml_file):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        # Convert torch.device to string
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
            
        with open(yaml_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self):
        """String representation of configuration."""
        config_str = "Configuration:\n"
        for key, value in sorted(self.__dict__.items()):
            if not key.startswith('_') and not callable(value):
                config_str += f"  {key}: {value}\n"
        return config_str


cfg = Config()

def get_config():
    """Helper function to get config instance."""
    return cfg