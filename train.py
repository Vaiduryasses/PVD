"""Training script with parallel GPU acceleration for PVD model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import os
import argparse
from datetime import datetime

from .models.model import PacoModel
from .dataset import PointCloudDataset
from .config import cfg, get_config

def setup(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args):
    """Training function for distributed training."""
    # Setup distributed environment
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # Create model
    model = PacoModel(
        transformer_dim=cfg.transformer_dim,
        transformer_depth=cfg.transformer_depth,
        transformer_heads=cfg.transformer_heads,
        transformer_window_size=cfg.transformer_window_size,
        dropout=cfg.dropout
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Create optimizer
    if cfg.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, 
                             momentum=0.9, weight_decay=cfg.weight_decay)
    
    # Create learning rate scheduler
    if cfg.scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.scheduler_gamma, patience=5)
    
    # Create datasets and dataloaders with distributed samplers
    train_dataset = PointCloudDataset(
        data_path=cfg.data_path, 
        split='train',
        max_points=cfg.max_points,
        augment=cfg.use_augmentation
    )
    
    val_dataset = PointCloudDataset(
        data_path=cfg.data_path, 
        split='val',
        max_points=cfg.max_points
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.max_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Get data
            input_pc = batch['input'].to(device)
            target_pc = batch['target'].to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast():
                loss = model(input_pc, target_pc, mode='train')
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}, skipping batch")
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss / cfg.accumulate_grad_batches).backward()
            
            if (i + 1) % cfg.accumulate_grad_batches == 0:
                # Additional gradient check before clipping
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Warning: Invalid gradients in {name}")
                            valid_gradients = False
                            break
                
                if not valid_gradients:
                    print("Skipping parameter update due to invalid gradients")
                    optimizer.zero_grad()
                    continue
                    
                # Clip gradients
                if cfg.gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                
                # Update parameters with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            
            # Log progress (only on master process)
            if rank == 0 and (i + 1) % cfg.log_every_n_steps == 0:
                print(f"Epoch [{epoch+1}/{cfg.max_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Synchronize losses across devices
        train_loss_tensor = torch.tensor(train_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                input_pc = batch['input'].to(device)
                target_pc = batch['target'].to(device)
                
                # Forward pass
                with autocast():
                    loss = model(input_pc, target_pc, mode='train')
                val_loss += loss.item()
        
        # Synchronize validation losses across devices
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        
        # Update scheduler
        if cfg.scheduler_type.lower() == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log epoch results (only on master process)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{cfg.max_epochs}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model (save unwrapped model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pt'))
                
                print(f"Best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
                # Save latest model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(cfg.checkpoint_dir, 'latest_model.pt'))
            
            # Early stopping
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Clean up distributed training
    cleanup()

def train(args):
    """Main training function with multi-GPU support."""
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        print(f"Training with {num_gpus} GPUs")
        # Launch distributed processes
        mp.spawn(
            train_distributed,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True
        )
    else:
        print("Training with single GPU or CPU")
        # Fall back to single device training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set seed for reproducibility
        torch.manual_seed(cfg.seed)
        
        # Create model
        model = PacoModel(
            transformer_dim=cfg.transformer_dim,
            transformer_depth=cfg.transformer_depth,
            transformer_heads=cfg.transformer_heads,
            transformer_window_size=cfg.transformer_window_size,
            dropout=cfg.dropout
        ).to(device)
        
        # Create optimizer
        if cfg.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, 
                                momentum=0.9, weight_decay=cfg.weight_decay)
        
        # Create learning rate scheduler
        if cfg.scheduler_type.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
        elif cfg.scheduler_type.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.max_epochs)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg.scheduler_gamma, patience=5)
        
        # Create datasets and dataloaders
        train_dataset = PointCloudDataset(
            data_path=cfg.data_path, 
            split='train',
            max_points=cfg.max_points,
            augment=cfg.use_augmentation
        )
        
        val_dataset = PointCloudDataset(
            data_path=cfg.data_path, 
            split='val',
            max_points=cfg.max_points
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()
        
        # Single-GPU training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(cfg.max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for i, batch in enumerate(train_loader):
                # Get data
                input_pc = batch['input'].to(device)
                target_pc = batch['target'].to(device)
                
                # Forward pass with mixed precision
                optimizer.zero_grad()
                
                with autocast():
                    loss = model(input_pc, target_pc, mode='train')
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value: {loss.item()}, skipping batch")
                    continue
                
                # Backward pass with gradient scaling
                scaler.scale(loss / cfg.accumulate_grad_batches).backward()
                
                if (i + 1) % cfg.accumulate_grad_batches == 0:
                    # Additional gradient check
                    valid_gradients = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Warning: Invalid gradients in {name}")
                                valid_gradients = False
                                break
                    
                    if not valid_gradients:
                        print("Skipping parameter update due to invalid gradients")
                        optimizer.zero_grad()
                        continue
                        
                    # Clip gradients
                    if cfg.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                    
                    # Update parameters with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item()
                
                # Log progress
                if (i + 1) % cfg.log_every_n_steps == 0:
                    print(f"Epoch [{epoch+1}/{cfg.max_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}")
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Get data
                    input_pc = batch['input'].to(device)
                    target_pc = batch['target'].to(device)
                    
                    # Forward pass
                    with autocast():
                        loss = model(input_pc, target_pc, mode='train')
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update scheduler
            if cfg.scheduler_type.lower() == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log epoch results
            print(f"Epoch [{epoch+1}/{cfg.max_epochs}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pt'))
                
                print(f"Best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
                # Save latest model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(cfg.checkpoint_dir, 'latest_model.pt'))
            
            # Early stopping
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PACO model")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to config file")
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        cfg = get_config().from_yaml(args.config)
    
    # Create output directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # Run training
    train(args)