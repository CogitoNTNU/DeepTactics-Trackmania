"""
Diagnostic script to analyze checkpoint sizes and replay buffer contents
"""
import torch
import os
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file and report what's saved."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / (1024**2):.2f} MB")
    print(f"{'='*70}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("Checkpoint keys and sizes:")
    print("-" * 70)
    
    total_size = 0
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            size_mb = value.numel() * value.element_size() / (1024**2)
            total_size += size_mb
            print(f"  {key:30s}: {str(value.shape):20s} {size_mb:10.2f} MB")
        
        elif isinstance(value, dict):
            # Check if it's a state_dict
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                size_mb = sum(v.numel() * v.element_size() for v in value.values()) / (1024**2)
                total_size += size_mb
                print(f"  {key:30s}: dict with {len(value)} tensors {size_mb:10.2f} MB")
            else:
                print(f"  {key:30s}: dict with {len(value)} entries")
        
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                size_mb = sum(v.numel() * v.element_size() for v in value) / (1024**2)
                total_size += size_mb
                print(f"  {key:30s}: list of {len(value)} tensors {size_mb:10.2f} MB")
            else:
                print(f"  {key:30s}: list with {len(value)} items")
        
        else:
            print(f"  {key:30s}: {type(value).__name__}")
    
    print("-" * 70)
    print(f"{'Estimated total size':30s}: {total_size:10.2f} MB")
    print(f"{'Actual file size':30s}: {os.path.getsize(checkpoint_path) / (1024**2):10.2f} MB")
    print()
    
    # Check for replay buffer data
    if 'replay_buffer_data' in checkpoint:
        rb_size = checkpoint['replay_buffer_data'].numel() * checkpoint['replay_buffer_data'].element_size() / (1024**2)
        rb_len = checkpoint.get('replay_buffer_len', 'unknown')
        print(f"[OK] Replay buffer data found: {rb_size:.2f} MB ({rb_len} transitions)")
    else:
        print(f"[MISSING] NO replay buffer data found!")
    
    # Check for PER weights
    if 'priority_weights' in checkpoint:
        pw_size = checkpoint['priority_weights'].numel() * checkpoint['priority_weights'].element_size() / (1024**2)
        print(f"[OK] Priority weights found: {pw_size:.2f} MB")
    else:
        print(f"[MISSING] NO priority weights found")
    
    print()

if __name__ == "__main__":
    checkpoint_dir = Path(r'c:\Users\Polar\Documents\GitHub\DeepTactics-Trackmania\checkpoints')
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.rglob("*.pt"))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
    else:
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Sort by modification time, most recent first
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Analyze the 5 most recent
        for checkpoint_path in checkpoint_files[:5]:
            analyze_checkpoint(str(checkpoint_path))
