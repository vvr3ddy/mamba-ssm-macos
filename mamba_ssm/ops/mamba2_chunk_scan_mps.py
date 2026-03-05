"""
MPS-optimized chunk-based scan for Mamba2 (SSD architecture).

This module provides a correct implementation of the chunked state space model
scan algorithm for Apple Silicon MPS backend, replacing the approximate
fallback in the original mamba2.py.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional


def mamba_chunk_scan_mps(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int = 256,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dt_limit: tuple = (0.0, float("inf")),
) -> torch.Tensor:
    """
    Correct chunk-based SSM scan for Mamba2 on MPS.
    
    This implements the state space model recurrence:
        h_t = exp(dt_t * A) * h_{t-1} + dt_t * x_t * B_t
        y_t = C_t @ h_t
    
    The sequence is processed in chunks for memory efficiency. Within each chunk,
    we use a parallel scan algorithm. Between chunks, state is propagated sequentially.
    
    Args:
        x: Input of shape (B, L, H, P) - batch, seqlen, nheads, headdim
        dt: Timestep of shape (B, L, H)
        A: State matrix of shape (H,) - one per head
        B: Input projection of shape (B, L, G, N) - G groups, N state dim
        C: Output projection of shape (B, L, G, N)
        chunk_size: Number of timesteps per chunk
        D: Skip connection of shape (H,) or None
        z: Gating input of shape (B, L, H, P) or None
        seq_idx: Sequence indices for variable-length sequences or None
        initial_states: Initial hidden state of shape (B, H, P, N) or None
        dt_limit: Clamp dt values to this range
    
    Returns:
        Output of shape (B, L, H, P)
    """
    batch, seqlen, nheads, headdim = x.shape
    
    # Handle grouped attention (GQA)
    ngroups = B.shape[2] if B.ndim > 3 else 1
    dstate = B.shape[-1] if B.ndim > 3 else B.shape[-1]
    
    if ngroups < nheads:
        heads_per_group = nheads // ngroups
        B = repeat(B, "b l g d -> b l (g h) d", h=heads_per_group)
        C = repeat(C, "b l g d -> b l (g h) d", h=heads_per_group)
    
    # Clamp dt if limits specified
    if dt_limit != (0.0, float("inf")):
        dt = dt.clamp(dt_limit[0], dt_limit[1])
    
    # Compute dA = dt * A: (B, L, H)
    dA = torch.einsum("blh,h->blh", dt, A)
    
    # Compute dB_u = dt * x * B: (B, L, H, P, N)
    dB_u = torch.einsum("blh,blhp,blhn->blhpn", dt, x, B)
    
    # Process in chunks
    n_chunks = (seqlen + chunk_size - 1) // chunk_size
    
    # Initialize state
    if initial_states is not None:
        state = initial_states.clone()
    else:
        state = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=x.dtype)
    
    outputs = []
    
    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = min((c + 1) * chunk_size, seqlen)
        chunk_len = chunk_end - chunk_start
        
        if chunk_len == 0:
            continue
        
        # Extract chunk data
        dA_chunk = dA[:, chunk_start:chunk_end]
        dB_u_chunk = dB_u[:, chunk_start:chunk_end]
        C_chunk = C[:, chunk_start:chunk_end]
        
        # Compute cumulative decay within chunk
        dA_cumsum = dA_chunk.cumsum(dim=1)
        exp_dA_cumsum = torch.exp(dA_cumsum)
        
        # Weight dB_u by inverse cumulative decay
        inv_exp_dA_cumsum = torch.exp(-dA_cumsum).unsqueeze(-1).unsqueeze(-1)
        weighted_dB_u = dB_u_chunk * inv_exp_dA_cumsum
        
        # Cumulative sum of weighted inputs
        cum_weighted_dB_u = weighted_dB_u.cumsum(dim=1)
        
        # Add contribution from previous chunk's final state
        prev_state_contrib = state.unsqueeze(1) * exp_dA_cumsum.unsqueeze(-1).unsqueeze(-1)
        
        # Total state at each position in chunk
        all_states = prev_state_contrib + exp_dA_cumsum.unsqueeze(-1).unsqueeze(-1) * cum_weighted_dB_u
        
        # Compute outputs: y_t = C_t @ h_t
        y_chunk = torch.einsum("blhpn,blhn->blhp", all_states, C_chunk)
        outputs.append(y_chunk)
        
        # Update state for next chunk
        state = all_states[:, -1]
    
    # Concatenate all chunk outputs
    y = torch.cat(outputs, dim=1)
    
    # Add skip connection D * x
    if D is not None:
        y = y + x * D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    
    # Apply gating if present
    if z is not None:
        y = y * F.silu(z)
    
    return y


def mamba_chunk_scan_combined_mps(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int = 256,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dt_limit: tuple = (0.0, float("inf")),
) -> torch.Tensor:
    """
    Combined interface for Mamba2 chunk scan on MPS.
    """
    device_type = x.device.type
    
    if device_type == "mps":
        return mamba_chunk_scan_mps(
            x, dt, A, B, C, chunk_size, D, z, seq_idx, initial_states, dt_limit
        )
    else:
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            return mamba_chunk_scan_combined(
                x, dt, A, B, C, chunk_size, D, z, seq_idx, initial_states, **dt_limit
            )
        except ImportError:
            return mamba_chunk_scan_mps(
                x, dt, A, B, C, chunk_size, D, z, seq_idx, initial_states, dt_limit
            )


class MambaChunkScanMPS(torch.autograd.Function):
    """Autograd function for MPS chunk scan."""
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
        dt_limit: tuple = (0.0, float("inf")),
    ) -> torch.Tensor:
        ctx.save_for_backward(x, dt, A, B, C, D, z, seq_idx, initial_states)
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        
        return mamba_chunk_scan_mps(
            x, dt, A, B, C, chunk_size, D, z, seq_idx, initial_states, dt_limit
        )
    
    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x, dt, A, B, C, D, z, seq_idx, initial_states = ctx.saved_tensors
        
        with torch.enable_grad():
            x.requires_grad_(True)
            dt.requires_grad_(True)
            A.requires_grad_(True)
            B.requires_grad_(True)
            C.requires_grad_(True)
            
            out = mamba_chunk_scan_mps(
                x, dt, A, B, C, ctx.chunk_size, D, z, seq_idx, initial_states, ctx.dt_limit
            )
            
            grads = torch.autograd.grad(
                outputs=out,
                inputs=[x, dt, A, B, C],
                grad_outputs=dout,
                allow_unused=True,
            )
        
        return (
            grads[0], grads[1], grads[2], grads[3], grads[4],
            None, None, None, None, None, None
        )


def mamba_chunk_scan_fn(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int = 256,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dt_limit: tuple = (0.0, float("inf")),
) -> torch.Tensor:
    """Functional interface for MPS chunk scan."""
    return MambaChunkScanMPS.apply(
        x, dt, A, B, C, chunk_size, D, z, seq_idx, initial_states, dt_limit
    )
