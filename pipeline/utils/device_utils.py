"""
Device utility functions for cross-platform compatibility.
Detects and uses the best available device (MPS for macOS, CUDA for Linux, CPU as fallback).
"""
import torch

def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The best available device (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_device_map():
    """
    Get the device_map string for model loading.
    
    Returns:
        str: "mps" for macOS, "auto" for CUDA systems, "cpu" as fallback
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "auto"
    else:
        return "cpu"

def empty_cache():
    """Empty the cache for the current device."""
    if torch.backends.mps.is_available():
        # MPS doesn't have empty_cache, but we can try to trigger garbage collection
        import gc
        gc.collect()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def synchronize():
    """Synchronize the current device."""
    if torch.backends.mps.is_available():
        # MPS synchronizes automatically, but we can ensure completion
        pass
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

