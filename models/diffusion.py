"""
Conditional Diffusion Model 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..config import cfg

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion model
    
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (B,) - Time steps
            
        Returns:
            embeddings: (B, dim) - Time embeddings
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings

class TimeMLPEmbedding(nn.Module):
    """
    MLP time embedding
    
    """
    
    def __init__(self, dim):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, t):
        return self.mlp(self.time_embed(t))

class SelfAttention(nn.Module):
    """
    Self-attention layer
    
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) - Input features
            
        Returns:
            out: (B, N, C) - Updated features
        """
        B, N, C = x.shape
        
        # Project to q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each (B, num_heads, N, head_dim)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        
        return out
    
class CrossAttention(nn.Module):
    """
    Cross-attention layer
    
    """
    
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
            x: (B, N, C) - Query features
            context: (B, M, C) - Context features
            
        Returns:
            out: (B, N, C) - Updated features
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # Project to q, k, v
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (2, B, num_heads, M, head_dim)
        k, v = kv[0], kv[1]  # Each (B, num_heads, M, head_dim)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        
        return out

class FeedForward(nn.Module):
    """
    Feed-forward network
    
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

class EncoderBlock(nn.Module):
    """
    Encoder block with self-attention
    
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)
        
    def forward(self, x, time_emb=None):
        """
        Args:
            x: (B, N, C) - Input features
            time_emb: (B, C) - Optional time embedding
            
        Returns:
            out: (B, N, C) - Updated features
        """
        # Self-attention
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        
        # Add time embedding if provided
        if time_emb is not None:
            # Reshape time_emb to add to all tokens
            time_emb = time_emb.unsqueeze(1)  # (B, 1, C)
            x = x + time_emb
        
        # Feed-forward
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        
        return x

class DecoderBlock(nn.Module):
    """
    Decoder block with self-attention and cross-attention
    
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)
        
    def forward(self, x, context, time_emb=None):
        """
        Args:
            x: (B, N, C) - Input features
            context: (B, M, C) - Context features
            time_emb: (B, C) - Optional time embedding
            
        Returns:
            out: (B, N, C) - Updated features
        """
        # Self-attention
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        
        # Add time embedding if provided
        if time_emb is not None:
            # Reshape time_emb to add to all tokens
            time_emb = time_emb.unsqueeze(1)  # (B, 1, C)
            x = x + time_emb
        
        # Cross-attention
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        
        # Feed-forward
        h = self.norm3(x)
        h = self.ffn(h)
        x = x + h
        
        return x

class Encoder(nn.Module):
    """
    Transformer encoder for context encoding
    
    """
    
    def __init__(self, dim, depth, num_heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(dim, num_heads, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, time_emb=None):
        """
        Args:
            x: (B, N, C) - Input features
            time_emb: (B, C) - Optional time embedding
            
        Returns:
            x: (B, N, C) - Encoded features
        """
        for layer in self.layers:
            x = layer(x, time_emb)
        return self.norm(x)

class Decoder(nn.Module):
    """
    Transformer decoder for noise prediction
    
    """
    
    def __init__(self, dim, depth, num_heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, context, time_emb=None):
        """
        Args:
            x: (B, N, C) - Input features (noisy points)
            context: (B, M, C) - Context features
            time_emb: (B, C) - Time embedding
            
        Returns:
            x: (B, N, C) - Denoised features
        """
        for layer in self.layers:
            x = layer(x, context, time_emb)
        return self.norm(x)

class DiffusionModel(nn.Module):
    """
    Point cloud diffusion model
    
    """
    
    def __init__(self, 
                 in_dim=3,
                 context_dim=256,
                 diffusion_dim=256,
                 encoder_depth=6,
                 decoder_depth=4,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        # Model dimensions
        self.in_dim = in_dim
        self.diffusion_dim = diffusion_dim
        
        # Diffusion parameters
        self.beta_start = cfg.bata_start  # Beta schedule start value
        self.beta_end = cfg.beta_end      # Beta schedule end value
        self.steps = cfg.diffusion_steps_train  # Number of diffusion steps
        
        # Input and output projections
        self.point_proj = nn.Linear(in_dim, diffusion_dim)
        self.output_proj = nn.Linear(diffusion_dim, in_dim)
        
        # Time embedding
        self.time_embed = TimeMLPEmbedding(diffusion_dim)
        
        # Context projection (if dimensions don't match)
        self.context_proj = nn.Linear(context_dim, diffusion_dim) if context_dim != diffusion_dim else nn.Identity()
        
        # Encoder for context processing
        self.encoder = Encoder(diffusion_dim, encoder_depth, num_heads, dropout)
        
        # Decoder for noise prediction
        self.decoder = Decoder(diffusion_dim, decoder_depth, num_heads, dropout)
        
        # Initialize parameters
        self._init_weights()
        
        # Pre-compute diffusion parameters
        self._setup_diffusion_parameters()
        
    def _init_weights(self):
        """
        Initialize model weights
        
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _setup_diffusion_parameters(self):
        """
        Set up diffusion schedule and parameters
        
        """
        # Define beta schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.steps)
        
        # Pre-compute diffusion parameters
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Posterior variance
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # Register buffers for efficient forward passes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to input according to diffusion schedule.
        
        Args:
            x_0: (B, N, D) - Clean data
            t: (B,) - Timesteps
            noise: Optional pre-defined noise
            
        Returns:
            x_t: (B, N, D) - Noisy data at timestep t
            noise: (B, N, D) - Applied noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract diffusion parameters for timesteps
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Add noise according to diffusion process
        x_t = sqrt_alphas * x_0 + sqrt_one_minus_alphas * noise
        
        return x_t, noise
    
    def predict_noise(self, x_t, t, context):
        """
        Predict noise in x_t using the model.
        
        Args:
            x_t: (B, N, D) - Noisy points at timestep t
            t: (B,) - Timesteps
            context: (B, M, C) - Context features
            
        Returns:
            pred_noise: (B, N, D) - Predicted noise
        """
        # Get time embeddings
        time_emb = self.time_embed(t)
        
        # Project points to feature space
        x = self.point_proj(x_t)
        
        # Process context
        context = self.context_proj(context)
        context = self.encoder(context, time_emb)
        
        # Predict noise via decoder
        x = self.decoder(x, context, time_emb)
        
        # Project back to point space
        pred_noise = self.output_proj(x)
        
        return pred_noise
    
    def forward(self, x_0, context, t=None):
        """
        Forward pass for training.
        
        Args:
            x_0: (B, N, D) - Clean point cloud
            context: (B, M, C) - Context features
            t: (B,) - Optional specific timesteps, otherwise random
            
        Returns:
            loss: Diffusion loss (MSE between predicted and actual noise)
            pred_noise: (B, N, D) - Predicted noise
            noise: (B, N, D) - True noise
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.steps, (batch_size,), device=x_0.device)
        
        # Add noise to input
        x_t, noise = self.add_noise(x_0, t)
        
        # Predict noise
        pred_noise = self.predict_noise(x_t, t, context)
        
        # Calculate loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss, pred_noise, noise

    def reverse_diffusion_step(self, x_t, t, pred_noise=None, context=None):
        """
        Single step of reverse diffusion process (denoising)
        
        Args:
            x_t: (B, N, D) - Point cloud with noise at time t
            t: (B,) - Current timestep
            pred_noise: Optional predicted noise, if not provided will be predicted by model
            context: (B, M, C) - Conditional context features
            
        Returns:
            x_t_prev: Predicted point cloud at previous timestep (closer to original data)
            pred_x0: Current prediction of x0
        """
        # Get diffusion parameters for current step
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]
        
        # If no predicted noise provided, use model to predict
        if pred_noise is None:
            assert context is not None, "Context must be provided to predict noise"
            pred_noise = self.predict_noise(x_t, t, context)
        
        # Construct x0 prediction (fully denoised prediction)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1) * pred_noise) / \
                torch.sqrt(alpha_cumprod).view(-1, 1, 1)
        
        # Calculate mean coefficients
        posterior_mean_coef1 = (torch.sqrt(alpha_cumprod_prev) * beta) / (1 - alpha_cumprod)
        posterior_mean_coef2 = (torch.sqrt(alpha) * (1 - alpha_cumprod_prev)) / (1 - alpha_cumprod)
        
        # Calculate posterior mean
        posterior_mean = posterior_mean_coef1.view(-1, 1, 1) * pred_x0 + \
                        posterior_mean_coef2.view(-1, 1, 1) * x_t
        
        # Get posterior variance
        posterior_var = self.posterior_variance[t]
        
        # Add noise only during training process
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        
        # Calculate x_{t-1}
        x_t_prev = posterior_mean + torch.sqrt(posterior_var).view(-1, 1, 1) * noise
        
        return x_t_prev, pred_x0
    
    @torch.no_grad()
    def reverse_diffusion(self, x_T, context, steps=None):
        """
        Complete reverse diffusion sampling process
        
        Args:
            x_T: (B, N, D) - Initial noise
            context: (B, M, C) - Conditional context features
            steps: Number of sampling steps, defaults to inference steps in config
            
        Returns:
            generated_samples: (B, N, D) - Generated point cloud samples
            intermediate_samples: All intermediate samples from the process
        """
        device = x_T.device
        batch_size = x_T.shape[0]
        steps = steps or cfg.diffusion_steps_infer
        
        # Store all intermediate results
        intermediate_samples = [x_T]
        
        # Current state
        x_t = x_T
        
        # Iteratively denoise
        for i in reversed(range(steps)):
            # Broadcast timestep to batch size
            timestep = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Execute single step of reverse diffusion
            x_t, pred_x0 = self.reverse_diffusion_step(x_t, timestep, context=context)
            
            # Save intermediate result
            intermediate_samples.append(x_t)
        
        return x_t, intermediate_samples
    
    @torch.no_grad()
    def sample(self, context, num_points, steps=None):
        """
        Sample points from the diffusion model.
        
        Args:
            context: (B, M, C) - Context features
            num_points: Number of points to generate per batch item
            steps: Number of sampling steps (default: cfg.diffusion_steps_infer)
            
        Returns:
            x_0: (B, num_points, 3) - Generated point cloud
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Use inference steps if not specified
        steps = steps or cfg.diffusion_steps_infer
        
        # Start with random noise
        x_T = torch.randn(batch_size, num_points, self.in_dim, device=device)
        
        # Sample using full reverse diffusion
        x_0, _ = self.reverse_diffusion(x_T, context, steps)
        
        return x_0
    
    @torch.no_grad()
def sample_with_ddim(self, context, num_points, steps=50, eta=0.0):
    """
    Accelerated sampling using DDIM (Denoising Diffusion Implicit Models)
    
    Args:
        context: (B, M, C) - Conditional context features
        num_points: Number of points to generate per sample
        steps: Number of DDIM sampling steps (typically fewer than DDPM)
        eta: 0 means fully deterministic, 1 means fully stochastic (DDPM)
        
    Returns:
        x_0: (B, num_points, D) - Generated point cloud
    """
    batch_size = context.shape[0]
    device = context.device
    
    # Create DDIM timestep schedule
    skip = self.steps // steps
    seq = list(range(0, self.steps, skip))
    
    # Start with random noise
    x_t = torch.randn(batch_size, num_points, self.in_dim, device=device)
    
    # Pre-compute alphas for efficiency
    alpha_cumprod_seq = self.alphas_cumprod[seq]
    
    # Iteratively denoise
    for i in range(len(seq) - 1, -1, -1):
        t = seq[i]
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        t_next = 0 if i == 0 else seq[i - 1]
        
        # Get alphas for current timestep (with pre-computed values)
        alpha_cumprod = alpha_cumprod_seq[i]
        alpha_cumprod_next = self.alphas_cumprod[t_next]
        
        # Predict noise
        pred_noise = self.predict_noise(x_t, t_tensor, context)
        
        # Predict x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1) * pred_noise) / \
                 torch.sqrt(alpha_cumprod).view(-1, 1, 1)
        
        # Variance scheduling (eta controls determinism between 0 and 1)
        c1 = eta * torch.sqrt((1 - alpha_cumprod / alpha_cumprod_next) * (1 - alpha_cumprod_next) / (1 - alpha_cumprod))
        c2 = torch.sqrt(1 - alpha_cumprod_next - c1**2)
        
        # Add noise only for non-final steps
        noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
        
        # Update x_t
        x_t = torch.sqrt(alpha_cumprod_next).view(-1, 1, 1) * pred_x0 + \
            c1.view(-1, 1, 1) * noise + \
            c2.view(-1, 1, 1) * pred_noise
    
    return x_t
    
    @torch.no_grad()
    def interpolate(self, context1, context2, num_points, steps=None, alpha=0.5):
        """
        Interpolate between two contexts to generate point clouds.
        
        Args:
            context1: (1, M, C) - First context
            context2: (1, M, C) - Second context
            num_points: Number of points to generate
            steps: Number of sampling steps
            alpha: Interpolation factor (0-1)
            
        Returns:
            x_0: (1, num_points, 3) - Interpolated point cloud
        """
        # Interpolate contexts
        context = alpha * context1 + (1 - alpha) * context2
        
        # Sample from interpolated context
        return self.sample(context, num_points, steps)