"""
MPS-compatible causal 1D convolution operations.

This module provides fallback implementations for causal convolution
when CUDA kernels are not available (e.g., on Apple Silicon with MPS).
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


def causal_conv1d_fn_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    """
    MPS-compatible causal 1D depthwise convolution with optional fused activation.
    
    Args:
        x: Input tensor of shape (batch, dim, seqlen)
        weight: Convolution weights of shape (dim, width) for depthwise conv
        bias: Optional bias of shape (dim,)
        activation: Activation to apply ("silu", "swish", "relu", or None)
    
    Returns:
        Output tensor of shape (batch, dim, seqlen)
    """
    # Ensure input is contiguous for MPS
    if x.device.type == "mps" and not x.is_contiguous():
        x = x.contiguous()
    
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    
    # Causal convolution: pad on the left by (width - 1)
    # This ensures output at position t depends only on positions <= t
    x_padded = F.pad(x, (width - 1, 0))
    
    # Reshape weight for depthwise convolution: (dim, 1, width)
    weight_4d = weight.unsqueeze(1)
    
    # Apply depthwise convolution
    # groups=dim means each channel is convolved independently
    out = F.conv1d(x_padded, weight_4d, bias=bias, groups=dim)
    
    # Apply activation if specified
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation == "relu":
        out = F.relu(out)
    elif activation == "gelu":
        out = F.gelu(out)
    
    return out


def causal_conv1d_update_mps(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-step causal conv update for autoregressive decoding.
    
    This maintains a rolling buffer of the last `width` inputs and computes
    the convolution output efficiently for a single new token.
    
    Args:
        x: New input of shape (batch, dim)
        conv_state: Rolling buffer of shape (batch, dim, width)
        weight: Convolution weights of shape (dim, width)
        bias: Optional bias of shape (dim,)
        activation: Activation to apply
    
    Returns:
        Tuple of (output of shape (batch, dim), updated conv_state)
    """
    # Update rolling buffer: shift left and append new input
    # Use in-place operations where possible to avoid unnecessary allocations
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    
    # Compute convolution as dot product between state and weights
    # (batch, dim, width) @ (dim, width) -> (batch, dim)
    out = (conv_state * weight.unsqueeze(0)).sum(dim=-1)
    
    # Add bias if present
    if bias is not None:
        out = out + bias
    
    # Apply activation
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation == "relu":
        out = F.relu(out)
    elif activation == "gelu":
        out = F.gelu(out)
    
    return out, conv_state


class CausalConv1dMPS(torch.nn.Module):
    """
    Causal 1D convolution module optimized for MPS backend.
    
    This is a drop-in replacement for causal_conv1d when CUDA is not available.
    """
    
    def __init__(
        self,
        dim: int,
        width: int = 4,
        bias: bool = True,
        activation: str = "silu",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Depthwise convolution: (dim, width)
        self.weight = torch.nn.Parameter(torch.empty(dim, width, **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal convolution.
        
        Args:
            x: Input of shape (batch, dim, seqlen)
        
        Returns:
            Output of shape (batch, dim, seqlen)
        """
        return causal_conv1d_fn_mps(
            x, self.weight, self.bias, self.activation
        )
    
    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step update for autoregressive decoding.
        
        Args:
            x: New input of shape (batch, dim)
            conv_state: Rolling buffer of shape (batch, dim, width)
        
        Returns:
            Tuple of (output, updated conv_state)
        """
        return causal_conv1d_update_mps(
            x, conv_state, self.weight, self.bias, self.activation
        )
    
    def allocate_state(self, batch_size: int) -> torch.Tensor:
        """
        Allocate convolution state buffer for inference.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Zero-initialized state of shape (batch, dim, width)
        """
        return torch.zeros(
            batch_size,
            self.dim,
            self.width,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
