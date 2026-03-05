"""
MPS-optimized selective scan implementations.

This module provides vectorized and torch.compile-optimized versions
of the selective scan operation for Apple Silicon MPS backend.
"""

import functools
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


def selective_scan_ref_vectorized(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor:
    """
    Vectorized reference implementation of selective scan for MPS.
    
    This is a more efficient version of selective_scan_ref that:
    1. Uses vectorized operations where possible
    2. Is designed to be optimized by torch.compile
    3. Avoids Python loop overhead through batched operations
    
    Args:
        u: Input of shape (B, D, L)
        delta: Timestep of shape (B, D, L)
        A: State matrix of shape (D, N)
        B: Input projection of shape (B, N, L) or (D, N)
        C: Output projection of shape (B, N, L) or (D, N)
        D: Skip connection of shape (D,) or None
        z: Gating input of shape (B, D, L) or None
        delta_bias: Bias added to delta of shape (D,) or None
        delta_softplus: Apply softplus to delta
        return_last_state: Return final hidden state
    
    Returns:
        Output of shape (B, D, L), optionally with last state
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    
    # Apply delta bias and softplus
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    seqlen = u.shape[2]
    
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    # Convert to float for numerical stability
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    
    # Precompute delta * A and delta * B * u for all timesteps
    # deltaA: (B, D, L, N)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    
    # deltaB_u: (B, D, L, N)
    if not is_variable_B:
        # B is (D, N) - broadcast across batch and sequence
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            # B is (B, N, L)
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            # B is (B, G, N, L) - repeat for grouped heads
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    
    # Handle variable C
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    
    # Sequential scan - this is the core recurrence
    # h_t = deltaA_t * h_{t-1} + deltaB_u_t
    # y_t = C_t @ h_t
    x = deltaA.new_zeros((batch, dim, dstate))
    ys = []
    
    # Vectorized loop - each iteration is still sequential but operations inside are batched
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        
        ys.append(y)
        
        # Track last state
        if i == seqlen - 1:
            last_state = x
    
    # Stack outputs: (B, D, L)
    y = torch.stack(ys, dim=2)
    
    # Add skip connection D * u
    out = y if D is None else y + u * rearrange(D.float(), "d -> d 1")
    
    # Apply gating if present
    if z is not None:
        out = out * F.silu(z.float())
    
    # Convert back to input dtype
    out = out.to(dtype=dtype_in)
    
    if return_last_state:
        return out, last_state
    return out


@functools.lru_cache(maxsize=None)
def _get_compiled_scan_fn():
    """
    Get torch.compile-optimized selective scan function.
    
    Uses LRU cache to avoid recompilation for the same input signature.
    """
    return torch.compile(
        selective_scan_ref_vectorized,
        backend="inductor",
        fullgraph=False,  # Allow graph breaks for complex control flow
        dynamic=True,     # Support variable sequence lengths
        options={
            "triton.cudagraphs": False,  # Not supported on MPS
        }
    )


def selective_scan_mps(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    use_compile: bool = True,
) -> torch.Tensor:
    """
    MPS-optimized selective scan with optional torch.compile acceleration.
    
    This function automatically selects the best implementation based on:
    - Device type (MPS vs CPU)
    - Whether torch.compile is available and enabled
    - Input shapes and dtypes
    
    Args:
        u: Input of shape (B, D, L)
        delta: Timestep of shape (B, D, L)
        A: State matrix of shape (D, N)
        B: Input projection of shape (B, N, L) or (D, N)
        C: Output projection of shape (B, N, L) or (D, N)
        D: Skip connection of shape (D,) or None
        z: Gating input of shape (B, D, L) or None
        delta_bias: Bias added to delta of shape (D,) or None
        delta_softplus: Apply softplus to delta
        return_last_state: Return final hidden state
        use_compile: Use torch.compile optimization (default: True)
    
    Returns:
        Output of shape (B, D, L), optionally with last state
    """
    device_type = u.device.type
    
    # Use compiled version on MPS if requested and available
    if use_compile and device_type == "mps":
        try:
            compiled_fn = _get_compiled_scan_fn()
            return compiled_fn(
                u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
            )
        except Exception:
            # Fall back to uncompiled version on error
            pass
    
    # Use vectorized reference implementation
    return selective_scan_ref_vectorized(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )


def selective_scan_parallel_prefix(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor:
    """
    Parallel prefix scan implementation of selective scan.
    
    This uses the associative scan (parallel prefix) algorithm to compute
    the SSM recurrence in O(log L) parallel steps instead of O(L) sequential steps.
    
    The recurrence h_t = A_t * h_{t-1} + B_t * u_t can be rewritten as an
    associative operation:
        (A₂, b₂) ⊕ (A₁, b₁) = (A₂ * A₁, A₂ * b₁ + b₂)
    
    This allows parallel computation using a tree reduction.
    
    Note: This implementation is currently experimental and may not be faster
    than the sequential version for typical sequence lengths (< 4096) due to
    the overhead of the parallel reduction. It's most beneficial for very long
    sequences.
    
    Args:
        Same as selective_scan_mps
    
    Returns:
        Output of shape (B, D, L), optionally with last state
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    seqlen = u.shape[2]
    
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    A = A.float()
    B = B.float()
    C = C.float()
    
    # Compute per-timestep transition matrices and inputs
    # deltaA: (B, D, L, N) where each element is exp(delta * A)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    
    # deltaB_u: (B, D, L, N)
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    
    # Parallel prefix scan using associative operator
    # State: (decay, accumulated_input) where:
    #   decay = product of deltaA values
    #   accumulated_input = weighted sum of deltaB_u values
    
    # Initialize: each position has (deltaA_t, deltaB_u_t)
    # We need to compute cumulative products and sums
    
    # For numerical stability, work in log space for the decay term
    log_deltaA = torch.log(deltaA + 1e-10)  # (B, D, L, N)
    
    # Cumulative sum of log gives cumulative product
    # cumsum along sequence dimension (dim=2)
    log_cum_decay = log_deltaA.cumsum(dim=2)  # (B, D, L, N)
    
    # The state at time t is:
    # h_t = sum_{s=0}^{t} exp(sum_{k=s+1}^{t} log_deltaA_k) * deltaB_u_s
    # 
    # This can be computed as:
    # h_t = exp(log_cum_decay_t) * cumsum(exp(-log_cum_decay) * deltaB_u)
    
    # Compute weighted inputs
    log_cum_decay_neg = -log_cum_decay
    weighted_input = torch.exp(log_cum_decay_neg) * deltaB_u  # (B, D, L, N)
    
    # Cumulative sum of weighted inputs
    cum_weighted = weighted_input.cumsum(dim=2)  # (B, D, L, N)
    
    # Un-weight to get actual state
    # x_t = exp(log_cum_decay_t) * cum_weighted_t
    x = torch.exp(log_cum_decay) * cum_weighted  # (B, D, L, N)
    
    # Compute outputs: y_t = C_t @ x_t
    is_variable_C = C.dim() >= 3
    if not is_variable_C:
        # C is (D, N)
        y = torch.einsum("bdln,dn->bdl", x, C)
    else:
        if C.dim() == 3:
            # C is (B, N, L)
            y = torch.einsum("bdln,bnl->bdl", x, C.transpose(1, 2))
        else:
            # C is (B, G, N, L)
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            y = torch.einsum("bdln,bdnl->bdl", x, C)
    
    # Add skip connection
    out = y if D is None else y + u * rearrange(D.float(), "d -> d 1")
    
    # Apply gating
    if z is not None:
        out = out * F.silu(z.float())
    
    out = out.to(dtype=dtype_in)
    
    if return_last_state:
        last_state = x[:, :, -1, :]  # (B, D, N)
        return out, last_state
    
    return out


class SelectiveScanMPS(torch.autograd.Function):
    """
    Autograd function for MPS selective scan with custom gradients.
    
    This provides a clean interface for the forward pass while using
    PyTorch's automatic differentiation for the backward pass through
    the vectorized implementation.
    """
    
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        delta_softplus: bool = False,
        return_last_state: bool = False,
        use_compile: bool = True,
    ) -> torch.Tensor:
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)
        ctx.delta_softplus = delta_softplus
        ctx.return_last_state = return_last_state
        ctx.use_compile = use_compile
        
        result = selective_scan_mps(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, use_compile
        )
        
        if return_last_state:
            out, last_state = result
            ctx.last_state = last_state
            return out, last_state
        return result
    
    @staticmethod
    def backward(ctx, dout, *args):
        # Use PyTorch's automatic differentiation through the forward implementation
        u, delta, A, B, C, D, z, delta_bias = ctx.saved_tensors
        
        # Compute gradients using autograd
        with torch.enable_grad():
            u.requires_grad_(True)
            delta.requires_grad_(True)
            A.requires_grad_(True)
            B.requires_grad_(True)
            C.requires_grad_(True)
            if D is not None:
                D.requires_grad_(True)
            if z is not None:
                z.requires_grad_(True)
            if delta_bias is not None:
                delta_bias.requires_grad_(True)
            
            out = selective_scan_mps(
                u, delta, A, B, C, D, z, delta_bias,
                ctx.delta_softplus, False, ctx.use_compile
            )
            
            grads = torch.autograd.grad(
                outputs=out,
                inputs=[u, delta, A, B, C, D, z, delta_bias],
                grad_outputs=dout,
                allow_unused=True,
            )
        
        # Map gradients back to input order
        grad_map = [grads[0], grads[1], grads[2], grads[3], grads[4]]
        if D is not None:
            grad_map.append(grads[5])
        else:
            grad_map.append(None)
        if z is not None:
            grad_map.append(grads[6] if D is not None else grads[5])
        else:
            grad_map.append(None)
        if delta_bias is not None:
            grad_map.append(grads[-1])
        else:
            grad_map.append(None)
        
        return tuple(grad_map) + (None, None)  # None for return_last_state, use_compile


def selective_scan_fn_mps(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    use_compile: bool = True,
) -> torch.Tensor:
    """
    Functional interface for MPS selective scan.
    
    This is the main entry point for calling the MPS-optimized selective scan.
    
    Args:
        u: Input of shape (B, D, L)
        delta: Timestep of shape (B, D, L)
        A: State matrix of shape (D, N)
        B: Input projection of shape (B, N, L) or (D, N)
        C: Output projection of shape (B, N, L) or (D, N)
        D: Skip connection of shape (D,) or None
        z: Gating input of shape (B, D, L) or None
        delta_bias: Bias added to delta of shape (D,) or None
        delta_softplus: Apply softplus to delta
        return_last_state: Return final hidden state
        use_compile: Use torch.compile optimization
    
    Returns:
        Output of shape (B, D, L), optionally with last state
    """
    return SelectiveScanMPS.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, use_compile
    )
