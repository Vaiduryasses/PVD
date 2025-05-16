"""
sparse voxel-transformer encoder 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..config import cfg

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal poisitional encoding for voxel coordinates

    """
    def __init__(self,channels):
        super().__init__()
        self.channels = channels    

    def forward(self, coords, resolution):
        """
        Args:
            coords: (V, 3) - Integer voxel coordinates
            resolution: Voxel grid resolution
            
        Returns:
            pos_encoding: (V, channels) - Positional encoding

        """
        #normalize to [-0.5,0.5]
        coords = coords.float() / resolution - 0.5

        #compute encoding dimension
        half_channels = self.channels // 6 # 2 functions *3 dimensions

        #create frequency bands
        freq_bands = torch.arange(half_channels, device=coords.device).float()
        freq_bands = 2.0 ** freq_bands

        #create empty tensor for encoding
        pos_encoding = torch.zeros((coords.shape[0], self.channels), device=coords.device)

        #compute encoding for each dimension
        for dim in range(3):
            pos = coords[:,dim].unsqueeze(1)#(V,1)
            
            #compute sin and cos encodings
            sin_encoding = torch.sin(pos * freq_bands * math.pi)#(V,42)
            cos_encoding = torch.cos(pos * freq_bands * math.pi)

            #fill encoding tensor 
            start_sin = dim * 2 * half_channels
            start_cos = start_sin + half_channels
            pos_encoding[:, start_sin:start_sin + half_channels] = sin_encoding
            pos_encoding[:, start_cos:start_cos + half_channels] = cos_encoding

        return pos_encoding

class SparseWindowAttention(nn.Module):
    """
    Sparse window-based self-attention 

    """

    def __init__(self, dim, window_size, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # linear projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        #relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros((2*window_size-1)**3, num_heads))

        #precompute relative position indices
        self.register_buffer("rel_pos_indices", self._get_rel_pos_indices(window_size))

    def _get_rel_pos_indices(self, window_size):
        """
        precompute relative position indices for windowed attention

        """
        coords = torch.arange(window_size)
        coords_grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'))
        coords_grid = coords_grid.permute(1,2,3,0).reshape(-1,3)#(window_size^3,3)

        rel_pos = coords_grid.unsqueeze(1) - coords_grid.unsqueeze(0)  # (window_size^3, window_size^3, 3)
        rel_pos += window_size - 1  # Shift to [0, 2*window_size-1)
        
        rel_pos_idx = rel_pos[:, :, 0] * (2*window_size-1)**2 + \
                      rel_pos[:, :, 1] * (2*window_size-1) + \
                      rel_pos[:, :, 2]
                      
        return rel_pos_idx
    def forward(self, x, coords):
        """
        Args:
            x: (N,C) - features for n tokens
            coords: (N,3) - integer coordinates of tokens

        Returns:
            x: (N,C) - updated features
        """
        N, C = x.shape

        #create hash table for efficient lookup
        coords_hash = coords[:, 0] * cfg.voxel_resolution**2 + \
                      coords[:, 1] * cfg.voxel_resolution + \
                      coords[:, 2]

        hash_table = {h.item(): i for i,h in enumerate(coords_hash)}

        #compute QKV
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).permute(1,0,2,3)
        q, k, v = qkv[0], qkv[1], qkv[2] #each (N, num_heads, head_dim)

        #process points by windows
        output = torch.zeros_like(x)

        #group points by windows
        windows = {}
        window_size = self.window_size

        for i, coord in enumerate(coords):
            #compute window index
            window_idx = (coord // window_size).tolist()
            window_idx = tuple(window_idx)

            #add points to window
            if window_idx not in windows:
                windows[window_idx] = []
            windows[window_idx].append(i)

        #process each window
        for window_idx, point_indices in windows.items():
            if len(point_indices) == 0:
                continue

            #get window points
            window_points = torch.tensor(point_indices, device=x.device)

            #get window coordinates
            window_coords = coords[window_points]#(len(point_indices),3)

            #get qkv for window points
            window_q = torch.index_select(q, 0, window_points)
            window_k = torch.index_select(k, 0, window_points)
            window_v = torch.index_select(v, 0, window_points)

            #compute attention
            attn = (window_q @ window_k.transpose(-2,-1)) * self.scale # (num_heads,len(point_indices),len(point_indices))

            # Apply relative position bias for all windows
            if len(window_points) > 0:
                # Dynamically compute relative position indices for current window points
                window_rel_coords = window_coords.unsqueeze(1) - window_coords.unsqueeze(0)  # (len(point_indices),len(point_indices), 3)
                window_rel_coords += window_size - 1  # Shift to [0, 2*window_size-1) range
                
                # Compute linear indices
                window_rel_idx = window_rel_coords[:, :, 0] * (2*window_size-1)**2 + \
                                window_rel_coords[:, :, 1] * (2*window_size-1) + \
                                window_rel_coords[:, :, 2]  # (len(point_indices),len(point_indices))
                
                # Get relative position bias - initially shape is (len(point_indices),len(point_indices), num_heads)
                rel_pos_bias = self.rel_pos_bias[window_rel_idx]
                
                # Permute to (num_heads,len(point_indices),len(point_indices)) to match attention scores shape
                rel_pos_bias = rel_pos_bias.permute(2, 0, 1)
                
                # Add to attention scores
                attn = attn + rel_pos_bias
            #apply softmax
            attn = F.softmax(attn, dim = -1)

            # Apply attention
            out = (attn @ window_v).permute(1,0,2)
            out = out.reshape(len(window_points), C)  # (len(point_indices), C)
            
            # Project the output
            processed_output = self.proj(out)

            # Use scatter_ for gradient-safe assignment
            indices = window_points.unsqueeze(1).expand(-1, C)
            output.scatter_(0, indices, processed_output)
        
        return output 
class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection
    
    """
    
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block with sparse window attention
    
    """
    
    def __init__(self, dim, window_size, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * mlp_ratio, dropout)
        
    def forward(self, x, coords):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.ffn(self.norm2(x))
        return x

class SparseVoxelTransformer(nn.Module):
    """
    Sparse Voxel Transformer Encoder for global feature extraction

    """
    
    def __init__(self, 
                 in_dim=35,  # 32(from MLP expansion of 8D features) + 3(centroid)
                 dim=256, 
                 depth=8, 
                 num_heads=8, 
                 window_size=3,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # Initial feature projection
        self.feature_projection = nn.Linear(in_dim, dim)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        
    def forward(self, feats, coords):
        """
        Args:
            feats: (N, F) - Input features for N voxels
            coords: (N, 3) - Integer coordinates of voxels
            
        Returns:
            x: (N, dim) - Output features
        """
        # Initial feature projection
        x = self.feature_projection(feats)
        
        # Add positional encoding
        pos = self.pos_encoding(coords, cfg.voxel_resolution)
        x = x + pos
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, coords)
        
        # Final normalization
        x = self.output_norm(x)
        
        return x
        

