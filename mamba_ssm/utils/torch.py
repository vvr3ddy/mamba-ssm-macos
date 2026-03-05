"""
PyTorch utility functions for device-agnostic operations.

This module provides device-aware AMP (Automatic Mixed Precision) decorators
that work correctly on CPU, CUDA, and MPS devices.
"""

from functools import partial
from typing import Callable, Optional

import torch


def get_autocast_device_type(device: Optional[str] = None) -> str:
    """
    Get the appropriate device type for autocast based on available hardware.
    
    Args:
        device: Optional explicit device string
        
    Returns:
        Device type string for autocast ("cuda", "mps", or "cpu")
    """
    if device is not None:
        return device
    
    # Auto-detect device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    """
    Create a device-aware AMP decorator.
    
    This decorator ensures that custom_fwd/custom_bwd work correctly
    with both CUDA and MPS devices.
    """
    def decorator(*args, device_type: Optional[str] = None, **kwargs):
        if cuda_amp_deprecated or device_type is None:
            # Use device_type if explicitly provided, otherwise default to cuda
            if device_type is None:
                device_type = get_autocast_device_type()
            kwargs["device_type"] = device_type
        return dec(*args, **kwargs)
    return decorator


# Handle PyTorch version differences
if hasattr(torch.amp, "custom_fwd"):  # PyTorch >= 2.0
    deprecated = True
    from torch.amp import custom_bwd, custom_fwd  # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_bwd, custom_fwd

# Create device-aware decorators
custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)


def autocast_context(
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
):
    """
    Create an autocast context manager for the given device.
    
    Args:
        device: Device for autocast ("auto", "cuda", "mps", "cpu")
        dtype: Target dtype for autocast
        enabled: Whether autocast is enabled
    
    Returns:
        Context manager for autocast
    
    Example:
        with autocast_context("mps", torch.float16):
            output = model(input)
    """
    if device == "auto":
        device = get_autocast_device_type()
    
    if device == "mps":
        # MPS doesn't support all dtypes
        if dtype is None:
            dtype = torch.float16
        return torch.amp.autocast(device_type=device, dtype=dtype, enabled=enabled)
    elif device == "cuda":
        if dtype is None:
            dtype = torch.bfloat16
        return torch.amp.autocast(device_type=device, dtype=dtype, enabled=enabled)
    else:
        # CPU doesn't support autocast in the same way
        return torch.amp.autocast(device_type="cpu", dtype=torch.float32, enabled=False)


def get_amp_dtype(device: str) -> torch.dtype:
    """
    Get the optimal AMP dtype for a given device.
    
    Args:
        device: Device string ("cuda", "mps", "cpu")
    
    Returns:
        Optimal dtype for AMP on that device
    """
    if device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        # bfloat16 support varies by chip generation
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            return torch.bfloat16
        except RuntimeError:
            return torch.float16
    else:
        return torch.float32
