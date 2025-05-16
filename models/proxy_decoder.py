"""
Proxy Decoder for plane prediction

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import cfg

class CrossAttention(nn.Module):
    """Cross attention layer."""
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        Args:
            x: (B, N, C) - Query embeddings
            context: (B, M, C) - Context embeddings
            
        Returns:
            out: (B, N, C) - Updated embeddings
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        #compute q,k,v
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3) ## (B, num_heads, N, head_dim)
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4) ## (2, B, num_heads, M, head_dim)
        k,v = kv[0],kv[1]  # Each (B, num_heads, M, head_dim)

        #compute attention
        attn = (q @ k.transpose(-2.-1))*self.scale # (B, num_heads, N, M)
        attn = F.softmax(attn,dim = -1)
        attn = self.dropout(attn)

        #apply attention
        out = (attn @ v).permute(0,2,1,3).reshape(B, N, C)
        return self.proj(out)

class SelfAttention(nn.Module):
    """
    Self attention layer
    
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) - Input embeddings
            
        Returns:
            out: (B, N, C) - Updated embeddings
        """
        B, N, C = x.shape
        
        # Compute query, key, value
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each (B, num_heads, N, head_dim)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        return self.proj(out)

class FeedForward(nn.Module):
    """
    feed-ford network

    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)  
        )
    
    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    """
    transformer decoder block with cross attention

    """
    def __init__(self, dim, num_heads = 8, dropout = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4, dropout)

    def forward(self, x, context):
        """
        Args:
            x: (B, N, C) - Query embeddings
            context: (B, M, C) - Context embeddings
            
        Returns:
            out: (B, N, C) - Updated embeddings
        """
        # Cross attention
        x = x + self.cross_attn(self.norm1(x), context)
        
        # Self attention
        x = x + self.self_attn(self.norm2(x))
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))

class ProxyDecoder(nn.Module):
    """
    Proxy Decoder for predicting palne parameters

    """
    def __init__(self, dim = 256, num_queries = 40, num_layers = 4, dropout = 0.0 ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        #learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries,dim))

        #decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(dim, dropout=dropout) for _ in range(num_layers)
        ])

        #final layer nrom
        self.norm = nn.LayerNorm(dim)

        #output heads
        self.class_head = nn.Linear(dim, 1)  # plane vs. no-plane
        self.param_head = nn.Linear(dim, 3)  # (r, θ, φ) parameters
        self.inlier_head = nn.Linear(dim, 1)  # inlier distance
        self.conf_head = nn.Linear(dim, 1)   # confidence score

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.query_embed, std=0.02)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, context):
        """
        Args:
            context: (B, M, C) - Encoder features
            
        Returns:
            dict with plane predictions
        """
        #add batch dim for context 
        if context.dim() == 2:
            context = context.unqueeze(0)

        B, _, _ = context.shape

        #initalize queries
        queries = self.query_embed.expand(B, -1, -1)
        
        #apply decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries,context)

        #normalize final queries
        queries = self.norm(queries)

        #apply output heads
        logits = self.class_head(queries).squeeze(-1)  # (B, num_queries)
        params = self.param_head(queries)  # (B, num_queries, 3)
        inlier_dist = F.softplus(self.inlier_head(queries)).squeeze(-1)  # (B, num_queries)
        conf = torch.sigmoid(self.conf_head(queries)).squeeze(-1)  # (B, num_queries)

        # convert (r, θ, φ) to normal vector (nx, ny, nz) and distance d
        r = params[..., 0]  # (B, num_queries)
        theta = params[..., 1]  # (B, num_queries)
        phi = params[..., 2]  # (B, num_queries)
        
        # compute normal vector
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)
        
        # ensure unit normals
        normals = torch.stack([nx, ny, nz], dim=-1)  # (B, num_queries, 3)
        normals = F.normalize(normals, dim=-1)
        
        # positive distance
        distances = torch.abs(r)  # (B, num_queries)
        
        # for single batch, squeeze dimensions
        if B == 1:
            logits = logits.squeeze(0)
            normals = normals.squeeze(0)
            distances = distances.squeeze(0)
            inlier_dist = inlier_dist.squeeze(0)
            conf = conf.squeeze(0)
        
        return {
            'logits': logits,
            'normals': normals,
            'distances': distances,
            'inlier_dist': inlier_dist,
            'conf': conf
        }    